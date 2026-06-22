/**
 * organic-growth-animator.ts — M773: Organic Growth Entrance Animation
 * ─────────────────────────────────────────────────────────────────────────────
 * Animates cells growing into existence from a seed point when they first
 * appear or when a topology change introduces new cells.  The animation
 * combines three layered effects:
 *
 *   1. **Scale bloom** — uniform scale 0→1 from the cell's seed point
 *      (phyllotaxis-derived centre), with back-ease overshoot for organic feel.
 *
 *   2. **Fractal unfold** — an L-system string is iteratively revealed:
 *      the visible depth increases over time so branches appear to grow
 *      outward from the trunk, recapitulating morphogenesis.  The expansion
 *      progress drives per-branch opacity and stroke-dashoffset so the
 *      fractal skeleton "draws itself" across the cell body.
 *
 *   3. **Vein spread** — leaf-vein paths radiate from the seed point toward
 *      the cell boundary.  Each vein is a simplified L-system FERN variant
 *      whose arc-length reveal is staggered by golden-angle ordering so
 *      veins appear in the same phyllotactic spiral as sunflower seeds.
 *
 * The three layers compose multiplicatively: scale gates visibility,
 * fractal unfold provides structural skeleton, veins add surface detail.
 *
 * ── Integration ──────────────────────────────────────────────────────────────
 *
 *   import { OrganicGrowthAnimator } from '$lib/sph/organic-growth-animator';
 *
 *   const animator = new OrganicGrowthAnimator();
 *
 *   // When a new cell appears (topology change or initial load):
 *   animator.startGrowth({
 *     cellId: 'self_attn',
 *     seedX: 400, seedY: 300,
 *     targetWidth: 160, targetHeight: 80,
 *     species: 'cil-eye',
 *   });
 *
 *   // Each frame:
 *   animator.update(dt);
 *   const state = animator.getState('self_attn');
 *   if (state) {
 *     container.scale.set(state.scale);
 *     container.alpha = state.opacity;
 *     renderFractalOverlay(state.fractalPath, state.fractalOpacity);
 *     renderVeins(state.veins);
 *   }
 *
 * ── Topology change hook ─────────────────────────────────────────────────────
 *
 *   // Batch-start growth for all new cells in a topology diff:
 *   animator.startBatchGrowth(newCells, { staggerDelay: 0.08 });
 *
 * ── References ───────────────────────────────────────────────────────────────
 *   src/lib/sph/morphogenesis.ts      — L-system expansion + turtle interpreter
 *   src/lib/sph/phyllotaxis.ts        — golden-angle seed placement
 *   src/lib/sph/transition-system.ts  — TransitionSystem pattern (M748)
 *   src/lib/sph/differential-growth.ts — organic fractal fold simulation
 *   src/lib/sph/cell-body-bridge.ts   — CellPhysicsConfig, species mapping
 *   src/lib/tween-system.ts           — Easing functions
 *
 * [ASTRO-ORGANIC-GROWTH] debug prefix.
 */

import { Easing, type EasingFn } from '../tween-system';
import {
  Morphogenesis,
  type MorphogenesisConfig,
  type LSystemPreset,
} from './morphogenesis';
import { GOLDEN_ANGLE_RAD, cartesianAt } from './phyllotaxis';

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/** Default total growth duration in seconds */
const DEFAULT_DURATION = 0.85;

/** Scale bloom occupies the first portion of the timeline */
const SCALE_PHASE_RATIO = 0.45;

/** Fractal unfold starts slightly before scale ends and extends to ~80% */
const FRACTAL_START_RATIO = 0.2;
const FRACTAL_END_RATIO = 0.85;

/** Vein spread starts after the fractal skeleton appears */
const VEIN_START_RATIO = 0.35;
const VEIN_END_RATIO = 1.0;

/** Maximum number of veins to generate per cell (clamped by perf budget) */
const MAX_VEINS = 12;

/** Minimum scale to avoid degenerate transforms */
const SCALE_EPSILON = 0.001;

/** Number of L-system iterations for the fractal skeleton overlay */
const FRACTAL_ITERATIONS = 3;

/** Number of L-system iterations for individual vein strands */
const VEIN_ITERATIONS = 2;

/** Maximum concurrent growth animations */
const MAX_CONCURRENT = 64;

// ─────────────────────────────────────────────────────────────────────────────
// Deterministic LCG PRNG (matches morphogenesis.ts)
// ─────────────────────────────────────────────────────────────────────────────

class LCG {
  private s: number;
  constructor(seed: number) { this.s = seed >>> 0; }
  next(): number {
    this.s = Math.imul(1664525, this.s) + 1013904223;
    return (this.s >>> 0) / 0x100000000;
  }
  range(lo: number, hi: number): number { return lo + this.next() * (hi - lo); }
}

// ─────────────────────────────────────────────────────────────────────────────
// Species → L-system preset mapping
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Each cell species maps to an L-system preset that defines its growth
 * skeleton aesthetic.  The mapping encodes botanical metaphor:
 *   - Attention heads (cil-eye)      → FERN (branching fronds, visual scan)
 *   - FFN layers (cil-bolt)          → BUSH (dense dendritic network)
 *   - LayerNorm (cil-vector)         → ALGAE (linear regulated growth)
 *   - Residual add (cil-plus)        → SIERPINSKI (recursive self-similarity)
 *   - Projection (cil-arrow-right)   → DRAGON (directed energy flow)
 *   - Others                         → FERN (default organic)
 */
function speciesPreset(species: string): LSystemPreset {
  switch (species) {
    case 'cil-eye':          return 'FERN';
    case 'cil-bolt':         return 'BUSH';
    case 'cil-vector':       return 'ALGAE';
    case 'cil-plus':         return 'SIERPINSKI';
    case 'cil-arrow-right':  return 'DRAGON';
    case 'cil-filter':       return 'FERN';
    case 'cil-layers':       return 'BUSH';
    case 'cil-loop':         return 'FERN';
    case 'cil-code':         return 'DRAGON';
    case 'cil-graph':        return 'BUSH';
    default:                 return 'FERN';
  }
}

/**
 * Turn angle jitter per species — some species get tighter branching.
 */
function speciesAngleJitter(species: string): number {
  switch (species) {
    case 'cil-eye':       return 5;
    case 'cil-bolt':      return 8;
    case 'cil-vector':    return 2;
    case 'cil-plus':      return 0;
    case 'cil-arrow-right': return 3;
    default:              return 4;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Easing helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Organic back-ease with slight overshoot — models turgor pressure inflation */
const organicBackEase: EasingFn = (t: number): number => {
  // Custom back-ease with softer overshoot than standard (s = 1.2 vs 1.7)
  const s = 1.2;
  const s1 = s + 1;
  return s1 * t * t * t - s * t * t;
};

/** Smooth-start for fractal reveal — slow beginning, accelerating growth */
const fractalRevealEase: EasingFn = (t: number): number => {
  // Cubic ease-in-out biased toward ease-out for organic deceleration
  return t < 0.4
    ? 8 * t * t * t
    : 1 - Math.pow(-2 * t + 2, 3) / 2;
};

/** Staggered vein ease — each vein gets a delayed smooth-start */
function veinStaggerEase(t: number, delay: number): number {
  const adjusted = Math.max(0, (t - delay) / (1 - delay));
  // Quadratic ease-out for natural deceleration of tip growth
  return adjusted * (2 - adjusted);
}

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/** Lifecycle phase of a growth animation */
export type GrowthPhase = 'idle' | 'growing' | 'completed';

/** Configuration to start a single cell growth animation */
export interface GrowthConfig {
  /** Cell identifier (matches topology node id) */
  cellId: string;
  /** Seed point X — the origin from which the cell grows outward */
  seedX: number;
  /** Seed point Y */
  seedY: number;
  /** Final width of the cell bbox after growth completes */
  targetWidth: number;
  /** Final height of the cell bbox after growth completes */
  targetHeight: number;
  /** Cell species for L-system preset selection */
  species?: string;
  /** Total animation duration in seconds. Default: 0.85 */
  duration?: number;
  /** Pre-delay before animation begins (seconds). Default: 0 */
  delay?: number;
  /** Override the L-system preset. Default: derived from species */
  preset?: LSystemPreset;
  /** Number of veins. Default: 8 (clamped to MAX_VEINS) */
  veinCount?: number;
  /** Scale easing override. Default: organic back-ease */
  scaleEasing?: EasingFn;
  /** Callback when growth animation completes */
  onComplete?: (cellId: string) => void;
}

/** Configuration for batch growth (topology change) */
export interface BatchGrowthOptions {
  /** Stagger delay between successive cells (seconds). Default: 0.06 */
  staggerDelay?: number;
  /** Base duration for each cell. Default: DEFAULT_DURATION */
  duration?: number;
  /** Callback when ALL cells in the batch have finished growing */
  onBatchComplete?: () => void;
}

/**
 * A single vein strand — a simplified L-system path radiating from the
 * seed point toward the cell boundary.
 */
export interface VeinStrand {
  /** Phyllotaxis index determining angular placement */
  index: number;
  /** Angle from seed point (radians) — golden-angle ordered */
  angle: number;
  /** SVG path data for this vein's full extent */
  fullPath: string;
  /** Current reveal progress (0 = hidden, 1 = fully drawn) */
  revealProgress: number;
  /** Stagger delay (normalized 0..1 within the vein phase) */
  staggerDelay: number;
  /** Length of the vein path in pixels (for dashoffset calculation) */
  pathLength: number;
}

/**
 * Per-frame readable state of a growing cell.
 * Renderers consume this to apply visual transformations.
 */
export interface GrowthState {
  /** Unique animation handle */
  id: number;
  /** Cell being animated */
  cellId: string;
  /** Current lifecycle phase */
  phase: GrowthPhase;

  // ── Seed point ──
  /** Origin X from which growth radiates */
  seedX: number;
  /** Origin Y */
  seedY: number;

  // ── Scale bloom ──
  /** Current uniform scale factor (0..~1.05, includes back-ease overshoot) */
  scale: number;
  /** Current opacity (fades in during early scale phase) */
  opacity: number;

  // ── Fractal unfold ──
  /** SVG path string of the fractal skeleton at current reveal depth */
  fractalPath: string;
  /** Opacity of the fractal overlay (fades out as growth completes) */
  fractalOpacity: number;
  /** Current maximum visible branch depth (increases over time) */
  fractalDepth: number;
  /** Normalized progress of fractal reveal (0..1) */
  fractalProgress: number;

  // ── Vein spread ──
  /** Array of vein strands with individual reveal progress */
  veins: VeinStrand[];
  /** Global vein phase progress (0..1) */
  veinProgress: number;

  // ── Timing ──
  /** Total animation duration (seconds) */
  duration: number;
  /** Elapsed time including delay (seconds) */
  elapsed: number;
  /** Normalized overall progress (0..1) */
  overallProgress: number;

  /** Completion callback */
  onComplete: ((cellId: string) => void) | null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal: L-system string expansion (lightweight, no turtle/Bézier)
// ─────────────────────────────────────────────────────────────────────────────

/** Expand an L-system string without generating geometry. */
function expandLSystem(
  axiom: string,
  rules: Record<string, string>,
  iterations: number,
): string {
  let s = axiom;
  for (let i = 0; i < iterations; i++) {
    let next = '';
    for (const ch of s) {
      next += rules[ch] ?? ch;
    }
    s = next;
    if (s.length > 100_000) break;
  }
  return s;
}

/**
 * Lightweight turtle interpreter that generates SVG path data
 * up to a given maximum branch depth.  Used for the fractal
 * skeleton overlay — we don't need Bézier smoothing here,
 * just the raw polyline skeleton.
 */
function turtleToSVGPath(
  lstring: string,
  startX: number,
  startY: number,
  startAngle: number,
  stepLength: number,
  angleDeg: number,
  maxDepth: number,
): { path: string; totalLength: number } {
  const angleRad = (angleDeg * Math.PI) / 180;
  const stack: Array<{ x: number; y: number; angle: number; depth: number }> = [];
  let x = startX;
  let y = startY;
  let angle = startAngle;
  let depth = 0;
  let totalLen = 0;

  const parts: string[] = [];
  let penDown = true;
  let needsMoveTo = true;

  for (const ch of lstring) {
    switch (ch) {
      case 'F':
      case 'G': {
        if (depth <= maxDepth) {
          const nx = x + stepLength * Math.cos(angle);
          const ny = y + stepLength * Math.sin(angle);
          if (needsMoveTo) {
            parts.push(`M${x.toFixed(1)} ${y.toFixed(1)}`);
            needsMoveTo = false;
          }
          parts.push(`L${nx.toFixed(1)} ${ny.toFixed(1)}`);
          totalLen += stepLength;
          x = nx;
          y = ny;
        } else {
          x += stepLength * Math.cos(angle);
          y += stepLength * Math.sin(angle);
        }
        break;
      }
      case '+': angle -= angleRad; break;
      case '-': angle += angleRad; break;
      case '[': {
        stack.push({ x, y, angle, depth });
        depth += 1;
        if (depth > maxDepth) {
          // skip drawing in this branch
        }
        break;
      }
      case ']': {
        if (depth <= maxDepth) {
          needsMoveTo = true;
        }
        const popped = stack.pop();
        if (popped) {
          x = popped.x;
          y = popped.y;
          angle = popped.angle;
          depth = popped.depth;
        }
        break;
      }
      case '|': angle += Math.PI; break;
    }
  }

  return { path: parts.join(' '), totalLength: totalLen };
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal: Vein generation using simplified L-system FERN
// ─────────────────────────────────────────────────────────────────────────────

const VEIN_AXIOM = 'X';
const VEIN_RULES: Record<string, string> = {
  X: 'F-[[X]+X]+F[+FX]-X',
  F: 'FF',
};
const VEIN_ANGLE = 22;

function generateVeinStrands(
  seedX: number,
  seedY: number,
  targetW: number,
  targetH: number,
  count: number,
  rng: LCG,
): VeinStrand[] {
  const veins: VeinStrand[] = [];
  const maxRadius = Math.sqrt(targetW * targetW + targetH * targetH) * 0.45;
  const stepLen = maxRadius / Math.pow(2, VEIN_ITERATIONS + 1);

  const lstring = expandLSystem(VEIN_AXIOM, VEIN_RULES, VEIN_ITERATIONS);

  for (let i = 0; i < count; i++) {
    // Golden-angle ordering for phyllotactic spiral placement
    const angle = i * GOLDEN_ANGLE_RAD + rng.range(-0.1, 0.1);
    const jitteredAngle = VEIN_ANGLE + rng.range(-3, 3);

    const { path, totalLength } = turtleToSVGPath(
      lstring,
      seedX,
      seedY,
      angle,
      stepLen,
      jitteredAngle,
      2, // max branch depth for veins
    );

    // Stagger delay: earlier indices appear first (spiral outward)
    const staggerDelay = (i / count) * 0.6;

    veins.push({
      index: i,
      angle,
      fullPath: path,
      revealProgress: 0,
      staggerDelay,
      pathLength: totalLength,
    });
  }

  return veins;
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal: Fractal skeleton generation
// ─────────────────────────────────────────────────────────────────────────────

/** L-system grammars for each preset (duplicated from morphogenesis for decoupling) */
const PRESET_GRAMMARS: Record<string, { axiom: string; rules: Record<string, string>; angle: number }> = {
  FERN: { axiom: 'X', rules: { X: 'F+[[X]-X]-F[-FX]+X', F: 'FF' }, angle: 25 },
  ALGAE: { axiom: 'F', rules: { F: 'F[+F]F[-F]F' }, angle: 25.7 },
  BUSH: { axiom: 'F', rules: { F: 'FF+[+F-F-F]-[-F+F+F]' }, angle: 22.5 },
  SIERPINSKI: { axiom: 'F-G-G', rules: { F: 'F-G+F+G-F', G: 'GG' }, angle: 120 },
  DRAGON: { axiom: 'FX', rules: { X: 'X+YF+', Y: '-FX-Y' }, angle: 90 },
};

interface FractalSkeleton {
  /** The expanded L-system string */
  lstring: string;
  /** Grammar angle in degrees */
  angleDeg: number;
  /** Step length for turtle interpretation */
  stepLength: number;
  /** Maximum branch depth in the string (for progressive reveal) */
  maxBranchDepth: number;
}

function buildFractalSkeleton(
  preset: LSystemPreset,
  targetW: number,
  targetH: number,
  rng: LCG,
): FractalSkeleton {
  const key = preset === 'CUSTOM' ? 'FERN' : preset;
  const grammar = PRESET_GRAMMARS[key] ?? PRESET_GRAMMARS.FERN;

  const lstring = expandLSystem(grammar.axiom, grammar.rules, FRACTAL_ITERATIONS);

  // Compute max branch depth in the string
  let maxDepth = 0;
  let currentDepth = 0;
  for (const ch of lstring) {
    if (ch === '[') { currentDepth++; maxDepth = Math.max(maxDepth, currentDepth); }
    if (ch === ']') { currentDepth--; }
  }

  // Scale step length so the fractal roughly fits the cell bbox
  const extent = Math.max(targetW, targetH) * 0.4;
  const stepCount = (lstring.match(/[FG]/g) || []).length;
  const stepLength = stepCount > 0 ? extent / Math.sqrt(stepCount) : 5;

  return {
    lstring,
    angleDeg: grammar.angle + rng.range(-2, 2),
    stepLength,
    maxBranchDepth: maxDepth,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility: clamp & lerp (local to avoid cross-module coupling)
// ─────────────────────────────────────────────────────────────────────────────

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** Map a value from [inLo, inHi] → [0, 1], clamped. */
function remapClamp(v: number, inLo: number, inHi: number): number {
  if (inHi <= inLo) return v >= inHi ? 1 : 0;
  return clamp((v - inLo) / (inHi - inLo), 0, 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// OrganicGrowthAnimator — the orchestrator
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Manages organic growth entrance animations for cells.
 *
 * Each frame, call `update(dt)` with the frame delta in seconds.
 * Renderers query `getState(cellId)` to read interpolated values
 * (scale, opacity, fractal path, veins) and apply them to the
 * corresponding cell container / mesh / overlay.
 *
 * Growth animations are one-shot: once completed they are retained for
 * one additional frame (so renderers can read the final state) and then
 * removed.  The cell is then at full scale with no overlays.
 */
export class OrganicGrowthAnimator {
  /** Auto-incrementing animation ID */
  private _nextId = 1;

  /** Active growth animations keyed by cellId */
  private _active = new Map<string, GrowthState>();

  /** Just-completed animations retained for one frame */
  private _justCompleted = new Map<string, GrowthState>();

  /** Internal RNG for deterministic jitter */
  private _rng: LCG;

  /** Pre-built fractal skeletons keyed by cellId */
  private _skeletons = new Map<string, FractalSkeleton>();

  constructor(seed: number = 773) {
    this._rng = new LCG(seed);
  }

  // ── Public API: start / stop / query ────────────────────────────────────

  /**
   * Start an organic growth animation for a single cell.
   * If the cell already has an active growth animation, it is replaced.
   * Returns the animation ID.
   */
  startGrowth(config: GrowthConfig): number {
    if (this._active.size >= MAX_CONCURRENT) {
      console.warn(
        `[ASTRO-ORGANIC-GROWTH] max concurrent animations reached (${MAX_CONCURRENT}), ` +
        `skipping cell=${config.cellId}`
      );
      return -1;
    }

    const id = this._nextId++;
    const {
      cellId,
      seedX,
      seedY,
      targetWidth,
      targetHeight,
      species = 'cil-eye',
      duration = DEFAULT_DURATION,
      delay = 0,
      preset = speciesPreset(species),
      veinCount = 8,
      scaleEasing = organicBackEase,
      onComplete = null,
    } = config;

    // Build fractal skeleton
    const skeleton = buildFractalSkeleton(
      preset,
      targetWidth,
      targetHeight,
      this._rng,
    );
    this._skeletons.set(cellId, skeleton);

    // Generate vein strands
    const veins = generateVeinStrands(
      seedX,
      seedY,
      targetWidth,
      targetHeight,
      Math.min(veinCount, MAX_VEINS),
      this._rng,
    );

    const state: GrowthState = {
      id,
      cellId,
      phase: 'idle',
      seedX,
      seedY,

      scale: SCALE_EPSILON,
      opacity: 0,

      fractalPath: '',
      fractalOpacity: 0,
      fractalDepth: 0,
      fractalProgress: 0,

      veins,
      veinProgress: 0,

      duration,
      elapsed: 0,
      overallProgress: 0,

      onComplete,
    };

    this._active.set(cellId, state);

    console.log(
      `[ASTRO-ORGANIC-GROWTH] start id=${id} cell=${cellId} ` +
      `species=${species} preset=${preset} dur=${duration.toFixed(2)}s ` +
      `veins=${veins.length} delay=${delay.toFixed(2)}s`
    );

    // Store delay in elapsed as negative offset so timing works naturally
    state.elapsed = -delay;

    return id;
  }

  /**
   * Start growth animations for a batch of new cells (topology change).
   * Cells are staggered by golden-angle phyllotactic ordering for
   * a natural spiral-outward appearance sequence.
   */
  startBatchGrowth(
    cells: Array<{
      cellId: string;
      seedX: number;
      seedY: number;
      targetWidth: number;
      targetHeight: number;
      species?: string;
    }>,
    options: BatchGrowthOptions = {},
  ): number[] {
    const {
      staggerDelay = 0.06,
      duration = DEFAULT_DURATION,
      onBatchComplete,
    } = options;

    const ids: number[] = [];
    let completedCount = 0;

    // Sort by phyllotaxis-inspired ordering: radial distance from batch centroid
    const cx = cells.reduce((s, c) => s + c.seedX, 0) / (cells.length || 1);
    const cy = cells.reduce((s, c) => s + c.seedY, 0) / (cells.length || 1);

    const sorted = [...cells].sort((a, b) => {
      const da = Math.sqrt((a.seedX - cx) ** 2 + (a.seedY - cy) ** 2);
      const db = Math.sqrt((b.seedX - cx) ** 2 + (b.seedY - cy) ** 2);
      return da - db;
    });

    for (let i = 0; i < sorted.length; i++) {
      const cell = sorted[i];
      const delay = i * staggerDelay;

      const id = this.startGrowth({
        ...cell,
        duration,
        delay,
        onComplete: onBatchComplete
          ? (cellId) => {
              completedCount++;
              if (completedCount >= sorted.length) {
                onBatchComplete();
              }
            }
          : undefined,
      });
      ids.push(id);
    }

    console.log(
      `[ASTRO-ORGANIC-GROWTH] batch start: ${sorted.length} cells, ` +
      `stagger=${staggerDelay.toFixed(3)}s, totalSpan=${((sorted.length - 1) * staggerDelay + duration).toFixed(2)}s`
    );

    return ids;
  }

  /**
   * Cancel a cell's growth animation.  The cell retains its current
   * interpolated state (no snap-back to zero).
   */
  cancel(cellId: string): boolean {
    const removed = this._active.delete(cellId);
    this._skeletons.delete(cellId);
    if (removed) {
      console.log(`[ASTRO-ORGANIC-GROWTH] cancelled cell=${cellId}`);
    }
    return removed;
  }

  /** Cancel all active growth animations. */
  cancelAll(): void {
    this._active.clear();
    this._skeletons.clear();
    console.log('[ASTRO-ORGANIC-GROWTH] all animations cancelled');
  }

  /**
   * Read the current growth state for a cell.
   * Returns null if no growth animation is active for this cell.
   */
  getState(cellId: string): Readonly<GrowthState> | null {
    return this._active.get(cellId)
      ?? this._justCompleted.get(cellId)
      ?? null;
  }

  /** Check if a cell currently has an active growth animation. */
  isGrowing(cellId: string): boolean {
    const s = this._active.get(cellId);
    return s != null && s.phase !== 'completed';
  }

  /** Number of currently active growth animations. */
  get activeCount(): number { return this._active.size; }

  /** All cell IDs with active growth animations. */
  get activeCellIds(): string[] { return [...this._active.keys()]; }

  // ── Per-frame update ────────────────────────────────────────────────────

  /**
   * Advance all active growth animations by `dt` seconds.
   * Call once per frame from the main render loop.
   */
  update(dt: number): void {
    // Clear previous frame's completed set
    this._justCompleted.clear();

    for (const [cellId, state] of this._active) {
      state.elapsed += dt;

      // Handle pre-delay (elapsed starts negative when delayed)
      if (state.elapsed < 0) {
        state.phase = 'idle';
        continue;
      }

      state.phase = 'growing';

      // Overall normalized progress
      const t = clamp(state.elapsed / state.duration, 0, 1);
      state.overallProgress = t;

      // ── 1. Scale bloom ──────────────────────────────────────────────────
      const scaleT = remapClamp(t, 0, SCALE_PHASE_RATIO);
      state.scale = Math.max(
        SCALE_EPSILON,
        organicBackEase(scaleT),
      );
      // Opacity ramps up quickly during the first 30% of scale phase
      state.opacity = clamp(scaleT / 0.3, 0, 1);

      // ── 2. Fractal unfold ───────────────────────────────────────────────
      const fractalT = remapClamp(t, FRACTAL_START_RATIO, FRACTAL_END_RATIO);
      state.fractalProgress = fractalRevealEase(fractalT);

      const skeleton = this._skeletons.get(cellId);
      if (skeleton && fractalT > 0) {
        // Progressive depth reveal: increase visible branch depth over time
        const targetDepth = Math.floor(
          state.fractalProgress * (skeleton.maxBranchDepth + 1)
        );
        state.fractalDepth = targetDepth;

        // Generate SVG path at current reveal depth
        const { path } = turtleToSVGPath(
          skeleton.lstring,
          state.seedX,
          state.seedY,
          -Math.PI / 2, // grow upward by default
          skeleton.stepLength * state.scale, // scale with bloom
          skeleton.angleDeg,
          targetDepth,
        );
        state.fractalPath = path;

        // Fractal overlay fades in during reveal, then fades out once complete
        if (fractalT < 0.2) {
          state.fractalOpacity = fractalT / 0.2;
        } else if (fractalT > 0.8) {
          state.fractalOpacity = (1 - fractalT) / 0.2;
        } else {
          state.fractalOpacity = 1;
        }
        // Reduce opacity as cell becomes fully opaque
        state.fractalOpacity *= 0.6;
      } else {
        state.fractalPath = '';
        state.fractalOpacity = 0;
        state.fractalDepth = 0;
      }

      // ── 3. Vein spread ──────────────────────────────────────────────────
      const veinT = remapClamp(t, VEIN_START_RATIO, VEIN_END_RATIO);
      state.veinProgress = veinT;

      for (const vein of state.veins) {
        vein.revealProgress = veinStaggerEase(veinT, vein.staggerDelay);
      }

      // ── Check completion ────────────────────────────────────────────────
      if (t >= 1) {
        state.phase = 'completed';
        state.scale = 1;
        state.opacity = 1;
        state.fractalOpacity = 0;
        state.fractalPath = '';
        for (const vein of state.veins) {
          vein.revealProgress = 1;
        }

        this._justCompleted.set(cellId, state);
        this._active.delete(cellId);
        this._skeletons.delete(cellId);

        console.log(
          `[ASTRO-ORGANIC-GROWTH] completed cell=${cellId} id=${state.id} ` +
          `elapsed=${state.elapsed.toFixed(3)}s`
        );

        if (state.onComplete) {
          try {
            state.onComplete(cellId);
          } catch (err) {
            console.error(
              `[ASTRO-ORGANIC-GROWTH] onComplete error for ${cellId}:`, err
            );
          }
        }
      }
    }
  }

  // ── Utility: SVG rendering helpers ──────────────────────────────────────

  /**
   * Generate a complete SVG overlay string for a growing cell.
   * Combines the fractal skeleton and vein strands into a single
   * SVG group element ready for DOM insertion.
   *
   * The caller should apply `transform: scale(${state.scale})` and
   * `transform-origin: ${state.seedX}px ${state.seedY}px` to the
   * container element.
   *
   * @param cellId Cell to render overlay for
   * @param strokeColor CSS color for overlay paths. Default: 'rgba(120,200,140,0.4)'
   * @returns SVG `<g>` element string, or empty string if no animation is active
   */
  renderOverlaySVG(
    cellId: string,
    strokeColor = 'rgba(120,200,140,0.4)',
  ): string {
    const state = this.getState(cellId);
    if (!state || state.phase === 'completed') return '';

    const parts: string[] = [];
    parts.push(`<g class="organic-growth-overlay" data-cell="${cellId}">`);

    // Fractal skeleton
    if (state.fractalPath && state.fractalOpacity > 0.01) {
      parts.push(
        `<path d="${state.fractalPath}" ` +
        `fill="none" stroke="${strokeColor}" ` +
        `stroke-width="1.2" stroke-linecap="round" ` +
        `opacity="${state.fractalOpacity.toFixed(3)}" />`
      );
    }

    // Vein strands with stroke-dashoffset reveal
    for (const vein of state.veins) {
      if (vein.revealProgress <= 0.01 || !vein.fullPath) continue;

      const dashOffset = vein.pathLength * (1 - vein.revealProgress);
      const veinOpacity = Math.min(vein.revealProgress * 2, 1) * 0.35;

      parts.push(
        `<path d="${vein.fullPath}" ` +
        `fill="none" stroke="${strokeColor}" ` +
        `stroke-width="0.8" stroke-linecap="round" ` +
        `stroke-dasharray="${vein.pathLength.toFixed(1)}" ` +
        `stroke-dashoffset="${dashOffset.toFixed(1)}" ` +
        `opacity="${veinOpacity.toFixed(3)}" />`
      );
    }

    parts.push('</g>');
    return parts.join('\n');
  }

  // ── Dispose ─────────────────────────────────────────────────────────────

  /** Clean up all internal state. */
  dispose(): void {
    this._active.clear();
    this._justCompleted.clear();
    this._skeletons.clear();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience: preset growth configs for common species
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Pre-defined growth configurations tuned per species.
 * These can be spread into `startGrowth()` calls:
 *
 * @example
 * ```ts
 * animator.startGrowth({
 *   ...GROWTH_PRESETS['cil-eye'],
 *   cellId: 'self_attn',
 *   seedX: 400, seedY: 300,
 *   targetWidth: 160, targetHeight: 80,
 * });
 * ```
 */
export const GROWTH_PRESETS: Record<string, Partial<GrowthConfig>> = {
  'cil-eye': {
    species: 'cil-eye',
    preset: 'FERN',
    duration: 0.9,
    veinCount: 10,
  },
  'cil-bolt': {
    species: 'cil-bolt',
    preset: 'BUSH',
    duration: 0.75,
    veinCount: 8,
  },
  'cil-vector': {
    species: 'cil-vector',
    preset: 'ALGAE',
    duration: 0.7,
    veinCount: 6,
  },
  'cil-plus': {
    species: 'cil-plus',
    preset: 'SIERPINSKI',
    duration: 0.65,
    veinCount: 4,
  },
  'cil-arrow-right': {
    species: 'cil-arrow-right',
    preset: 'DRAGON',
    duration: 0.8,
    veinCount: 8,
  },
  'cil-filter': {
    species: 'cil-filter',
    preset: 'FERN',
    duration: 0.85,
    veinCount: 10,
  },
  'cil-layers': {
    species: 'cil-layers',
    preset: 'BUSH',
    duration: 0.8,
    veinCount: 8,
  },
  'cil-loop': {
    species: 'cil-loop',
    preset: 'FERN',
    duration: 0.85,
    veinCount: 8,
  },
  'cil-code': {
    species: 'cil-code',
    preset: 'DRAGON',
    duration: 0.8,
    veinCount: 6,
  },
  'cil-graph': {
    species: 'cil-graph',
    preset: 'BUSH',
    duration: 0.8,
    veinCount: 8,
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Singleton
// ─────────────────────────────────────────────────────────────────────────────

let _globalAnimator: OrganicGrowthAnimator | null = null;

/** Get the global OrganicGrowthAnimator singleton. */
export function getGlobalGrowthAnimator(): OrganicGrowthAnimator {
  if (!_globalAnimator) {
    _globalAnimator = new OrganicGrowthAnimator();
  }
  return _globalAnimator;
}

/** Replace the global OrganicGrowthAnimator (useful for testing). */
export function setGlobalGrowthAnimator(animator: OrganicGrowthAnimator): void {
  _globalAnimator = animator;
}
