/**
 * species-transition-effect.ts — M758: Species Transition Effect
 * ─────────────────────────────────────────────────────────────────────────────
 * Orchestrates the visual transition when a cell's species changes between
 * epochs. Three concurrent animation lanes run in parallel on every species
 * mutation event:
 *
 *   1. **Color Lerp** — Interpolates all palette channels (primary, glow,
 *      shadow, secondary) from the source species to the target species in
 *      perceptual Oklab space, avoiding the desaturated midpoint that naive
 *      sRGB lerp produces.  The glow channel uses a slightly faster ease so
 *      the luminous identity shifts before the body color catches up, giving
 *      a "soul-first, shell-second" feel.
 *
 *   2. **SDF Morph** — Cross-fades the signed distance field between source
 *      and target species shapes.  Rather than a simple linear blend (which
 *      creates intermediate blobs), we use a smooth-min union that preserves
 *      topological features of both shapes throughout the morph.  The SDF
 *      blend factor follows a back-ease curve so the source shape lingers
 *      briefly before accelerating into the target.
 *
 *   3. **Pattern Fade** — The species-specific decoration pattern (zigzag for
 *      cil-bolt, concentric rings for cil-eye, grid for cil-code, etc.) cross-
 *      fades with independent opacity envelopes: the old pattern dissolves out
 *      during the first 60% of the transition, and the new pattern materialises
 *      during the last 60%, creating a 20% overlap window where ghost traces of
 *      both patterns coexist.
 *
 * Integration points:
 *   - Triggered by cell-pubsub-loop when a cell's `species` field changes
 *     between consecutive epoch snapshots
 *   - Consumes `Color` and `lerp` from color-utils.ts
 *   - Consumes `SpeciesParams` from channels/rendering/species/params.json
 *     via the SPECIES_PALETTE in color-utils.ts
 *   - Feeds blended uniforms into CellMaterial.activateSpecies() transition
 *   - The SDF morph factor is consumed by sdf-species-filter.ts as a uniform
 *   - Pattern fade weights feed into pixi-cell-renderer's decoration layer
 *
 * Upstream references:
 *   src/lib/color-utils.ts                — Color, lerp, SPECIES_PALETTE
 *   src/lib/renderers/sdf-species-filter.ts — SDF shader filter
 *   src/lib/sph/transition-system.ts      — TransitionState pattern
 *   src/lib/sph/organic-sdf.ts            — flowerSDF, kochSDF, juliaSDF
 *   src/lib/sph/cell-visual-identity.ts   — Morphology, VisualProfile
 *   src/lib/tween-system.ts               — Easing functions
 *   src/lib/shaders/sdf-species-library.frag — speciesSDF() per-species shapes
 *   channels/rendering/species/params.json — species visual parameters
 *   channels/rendering/color/color_extra.py — _lerp_colour Python counterpart
 *
 * [ASTRO-SPECIES-FX] debug prefix.
 */

import {
  Color,
  lerp as colorLerp,
  getSpeciesPalette,
  type SpeciesPalette,
} from './color-utils';

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/** Default species transition duration in seconds */
const DEFAULT_DURATION = 0.65;

/** Minimum duration clamp to avoid division-by-zero and invisible transitions */
const MIN_DURATION = 0.05;

/**
 * Glow channel lead factor — the glow/emissive color transitions faster than
 * the body to create a "soul-first" effect.  At 1.35× the glow is ~35% ahead
 * of the body color.  Mirrors the Fresnel edge-glow lead in
 * shading_core.py apply_shading_energy_conservation().
 */
const GLOW_LEAD_FACTOR = 1.35;

/**
 * Pattern cross-fade overlap.  Both the old pattern fadeout and new pattern
 * fadein span this fraction of the total duration, creating an overlap zone.
 *   old pattern visible:   t ∈ [0.0, PATTERN_XFADE_SPAN]
 *   new pattern visible:   t ∈ [1.0 - PATTERN_XFADE_SPAN, 1.0]
 *   overlap window:        t ∈ [1.0 - PATTERN_XFADE_SPAN, PATTERN_XFADE_SPAN]
 * With SPAN = 0.60, the overlap is 0.20 (20% of the transition).
 */
const PATTERN_XFADE_SPAN = 0.60;

/**
 * SDF smooth-min blend radius.  Controls how rounded the union between
 * source and target SDF shapes is during the morph.  Larger values create
 * softer, more organic-feeling transitions but lose sharp features.
 * Mirrors the opSmoothUnion k parameter in lygia/sdf/opSmoothUnion.glsl.
 */
const SDF_SMOOTH_K = 0.15;

/**
 * Species SDF index mapping — must match sdf-species-library.frag species IDs
 * and _SPECIES_NAME_TO_INDEX in decoration_extra.py.
 */
const SPECIES_SDF_INDEX: Record<string, number> = {
  'cil-eye':         0,
  'cil-vector':      1,
  'cil-bolt':        2,
  'cil-plus':        3,
  'cil-arrow-right': 4,
  'cil-filter':      5,
  'cil-code':        6,
  'cil-layers':      7,
  'cil-loop':        8,
  'cil-graph':       9,
};

// ─────────────────────────────────────────────────────────────────────────────
// Easing functions
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Ease-in-out cubic — smooth acceleration / deceleration.
 * Used for the main color interpolation lane.
 */
function easeInOutCubic(t: number): number {
  return t < 0.5
    ? 4.0 * t * t * t
    : 1.0 - (-2.0 * t + 2.0) ** 3 / 2.0;
}

/**
 * Back-ease-out — overshoots the target then settles.
 * Used for SDF morph so the source shape lingers before snapping into the
 * target.  The overshoot magnitude (1.70158) matches the standard CSS
 * `cubic-bezier(0.175, 0.885, 0.32, 1.275)` back-ease.
 */
function easeOutBack(t: number): number {
  const c1 = 1.70158;
  const c3 = c1 + 1.0;
  return 1.0 + c3 * (t - 1.0) ** 3 + c1 * (t - 1.0) ** 2;
}

/**
 * Smooth-step (Hermite interpolation) for pattern fade envelopes.
 * Gives a natural fade without harsh edges.
 */
function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3.0 - 2.0 * t);
}

/** Clamp a value to [0, 1] */
function saturate(v: number): number {
  return Math.max(0.0, Math.min(1.0, v));
}

// ─────────────────────────────────────────────────────────────────────────────
// Oklab perceptual color space
// ─────────────────────────────────────────────────────────────────────────────
//
// Naive sRGB lerp desaturates midpoints (e.g. blue→yellow passes through grey).
// Oklab preserves perceived brightness and chroma throughout the interpolation.
// Reference: Björn Ottosson — https://bottosson.github.io/posts/oklab/

interface OklabColor { L: number; a: number; b: number; alpha: number }

/** Linear sRGB ← sRGB (gamma decode) */
function srgbToLinear(c: number): number {
  return c <= 0.04045
    ? c / 12.92
    : ((c + 0.055) / 1.055) ** 2.4;
}

/** sRGB ← Linear sRGB (gamma encode) */
function linearToSrgb(c: number): number {
  return c <= 0.0031308
    ? c * 12.92
    : 1.055 * (c ** (1.0 / 2.4)) - 0.055;
}

/** Oklab ← linear sRGB */
function linearSrgbToOklab(r: number, g: number, b: number): [number, number, number] {
  const l_ = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
  const m_ = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
  const s_ = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;

  const l = Math.cbrt(l_);
  const m = Math.cbrt(m_);
  const s = Math.cbrt(s_);

  return [
    0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
  ];
}

/** linear sRGB ← Oklab */
function oklabToLinearSrgb(L: number, a: number, b: number): [number, number, number] {
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.2914855480 * b;

  const l = l_ * l_ * l_;
  const m = m_ * m_ * m_;
  const s = s_ * s_ * s_;

  return [
     4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
  ];
}

/** Convert Color → Oklab */
function colorToOklab(c: Color): OklabColor {
  const lr = srgbToLinear(c.r);
  const lg = srgbToLinear(c.g);
  const lb = srgbToLinear(c.b);
  const [L, a, b] = linearSrgbToOklab(lr, lg, lb);
  return { L, a, b, alpha: c.a };
}

/** Convert Oklab → Color */
function oklabToColor(ok: OklabColor): Color {
  const [lr, lg, lb] = oklabToLinearSrgb(ok.L, ok.a, ok.b);
  return new Color(
    saturate(linearToSrgb(Math.max(0, lr))),
    saturate(linearToSrgb(Math.max(0, lg))),
    saturate(linearToSrgb(Math.max(0, lb))),
    saturate(ok.alpha),
  );
}

/** Interpolate two Colors in Oklab space at factor t ∈ [0, 1] */
function oklabLerp(a: Color, b: Color, t: number): Color {
  const okA = colorToOklab(a);
  const okB = colorToOklab(b);
  const tt = saturate(t);
  return oklabToColor({
    L:     okA.L     + (okB.L     - okA.L)     * tt,
    a:     okA.a     + (okB.a     - okA.a)     * tt,
    b:     okA.b     + (okB.b     - okA.b)     * tt,
    alpha: okA.alpha + (okB.alpha - okA.alpha) * tt,
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/** Lifecycle phase of a species transition */
export type TransitionPhase = 'idle' | 'running' | 'completed';

/**
 * Snapshot of all blended visual parameters at a given instant during
 * the transition.  Consumed by CellMaterial, pixi-cell-renderer, and the
 * SDF filter to render the in-between state.
 */
export interface TransitionFrame {
  /** Normalised progress [0, 1] — 0 = fully source, 1 = fully target */
  progress: number;

  /** Current lifecycle phase */
  phase: TransitionPhase;

  // ── Color Lerp lane ───────────────────────────────────────────────────
  /** Blended palette (all channels interpolated in Oklab) */
  palette: SpeciesPalette;

  /** Raw hex string for the blended primary/base color (convenience) */
  primaryHex: string;

  /** Raw hex string for the blended glow color (convenience) */
  glowHex: string;

  // ── SDF Morph lane ────────────────────────────────────────────────────
  /** Source species SDF index (for sdf-species-library.frag) */
  sdfIndexA: number;

  /** Target species SDF index */
  sdfIndexB: number;

  /**
   * SDF blend factor [0, 1] — drives the smooth-min union in the shader.
   * At 0.0 only the source SDF contributes; at 1.0 only the target.
   * May overshoot slightly due to back-ease (clamped in shader by saturate).
   */
  sdfBlend: number;

  /** Smooth-min k parameter for SDF union during morph */
  sdfSmoothK: number;

  // ── Pattern Fade lane ─────────────────────────────────────────────────
  /**
   * Old (source species) pattern opacity [1→0].
   * Feeds into the decoration overlay alpha for the departing pattern.
   */
  patternFadeOut: number;

  /**
   * New (target species) pattern opacity [0→1].
   * Feeds into the decoration overlay alpha for the arriving pattern.
   */
  patternFadeIn: number;

  // ── Metadata ──────────────────────────────────────────────────────────
  /** Source species name */
  fromSpecies: string;

  /** Target species name */
  toSpecies: string;

  /** Elapsed time in seconds since transition start */
  elapsed: number;

  /** Total transition duration in seconds */
  duration: number;
}

/**
 * Configuration for a species transition.
 * Sensible defaults allow fire-and-forget usage.
 */
export interface TransitionConfig {
  /** Transition duration in seconds (default: 0.65) */
  duration?: number;

  /**
   * Glow lead factor — how much faster the glow channel transitions
   * relative to the body.  1.0 = same speed, >1.0 = glow leads.
   * Default: 1.35.
   */
  glowLeadFactor?: number;

  /**
   * SDF smooth-min k parameter.
   * Larger = softer blend, smaller = sharper crossfade.
   * Default: 0.15.
   */
  sdfSmoothK?: number;

  /**
   * Pattern cross-fade span as a fraction of total duration.
   * Default: 0.60 (60%).
   */
  patternXfadeSpan?: number;

  /**
   * Optional callback invoked every frame with the blended TransitionFrame.
   * Use for driving external uniforms or DOM updates.
   */
  onFrame?: (frame: TransitionFrame) => void;

  /** Optional callback invoked once when the transition completes. */
  onComplete?: (frame: TransitionFrame) => void;
}

// ─────────────────────────────────────────────────────────────────────────────
// SpeciesTransitionEffect
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Manages a single species-change transition for one cell.
 *
 * Usage:
 * ```ts
 *   const fx = new SpeciesTransitionEffect('cil-eye', 'cil-bolt');
 *   fx.start();
 *
 *   // In your render loop:
 *   const frame = fx.tick(deltaSeconds);
 *   if (frame.phase === 'running') {
 *     material.setUniform('u_sdfBlend', frame.sdfBlend);
 *     material.setUniform('u_sdfIndexA', frame.sdfIndexA);
 *     material.setUniform('u_sdfIndexB', frame.sdfIndexB);
 *     material.setUniform('u_sdfSmoothK', frame.sdfSmoothK);
 *     container.tint = Color.fromHex(frame.primaryHex).toInt();
 *   }
 * ```
 */
export class SpeciesTransitionEffect {
  /** Source species name */
  readonly fromSpecies: string;

  /** Target species name */
  readonly toSpecies: string;

  /** Resolved configuration */
  private readonly _cfg: Required<TransitionConfig>;

  /** Cached source palette */
  private readonly _paletteA: SpeciesPalette;

  /** Cached target palette */
  private readonly _paletteB: SpeciesPalette;

  /** Source SDF index */
  private readonly _sdfA: number;

  /** Target SDF index */
  private readonly _sdfB: number;

  /** Elapsed time in seconds since start() */
  private _elapsed = 0;

  /** Current phase */
  private _phase: TransitionPhase = 'idle';

  /** Cached last frame to avoid recomputation on redundant reads */
  private _lastFrame: TransitionFrame | null = null;

  constructor(
    fromSpecies: string,
    toSpecies: string,
    config?: TransitionConfig,
  ) {
    this.fromSpecies = fromSpecies;
    this.toSpecies   = toSpecies;

    this._cfg = {
      duration:         Math.max(MIN_DURATION, config?.duration ?? DEFAULT_DURATION),
      glowLeadFactor:   config?.glowLeadFactor ?? GLOW_LEAD_FACTOR,
      sdfSmoothK:       config?.sdfSmoothK ?? SDF_SMOOTH_K,
      patternXfadeSpan: config?.patternXfadeSpan ?? PATTERN_XFADE_SPAN,
      onFrame:          config?.onFrame ?? (() => {}),
      onComplete:       config?.onComplete ?? (() => {}),
    };

    this._paletteA = getSpeciesPalette(fromSpecies);
    this._paletteB = getSpeciesPalette(toSpecies);
    this._sdfA     = SPECIES_SDF_INDEX[fromSpecies] ?? 0;
    this._sdfB     = SPECIES_SDF_INDEX[toSpecies]   ?? 0;
  }

  // ── Lifecycle ───────────────────────────────────────────────────────────

  /** Begin the transition.  Resets elapsed time and sets phase to 'running'. */
  start(): this {
    this._elapsed  = 0;
    this._phase    = 'running';
    this._lastFrame = null;
    return this;
  }

  /** Abort mid-flight — jumps to completed with the current blended state. */
  cancel(): this {
    this._phase = 'completed';
    return this;
  }

  /** Force-complete: jump to the final state instantly. */
  finish(): TransitionFrame {
    this._elapsed = this._cfg.duration;
    this._phase   = 'completed';
    return this._computeFrame(1.0);
  }

  /** Current lifecycle phase */
  get phase(): TransitionPhase { return this._phase; }

  /** Whether the transition is actively animating */
  get isRunning(): boolean { return this._phase === 'running'; }

  /** Normalised progress [0, 1] */
  get progress(): number {
    return saturate(this._elapsed / this._cfg.duration);
  }

  // ── Per-frame update ───────────────────────────────────────────────────

  /**
   * Advance the transition by `dt` seconds and return the blended frame.
   *
   * Call this once per render frame.  The returned `TransitionFrame` contains
   * all blended parameters needed to render the in-between state.
   *
   * @param dt  Delta time in seconds since last tick
   * @returns   Current blended state
   */
  tick(dt: number): TransitionFrame {
    if (this._phase === 'idle') {
      // Not started yet — return source state
      return this._computeFrame(0.0);
    }

    if (this._phase === 'completed') {
      // Already done — return cached final frame
      return this._lastFrame ?? this._computeFrame(1.0);
    }

    // Advance
    this._elapsed += dt;
    const rawProgress = this._elapsed / this._cfg.duration;

    if (rawProgress >= 1.0) {
      this._elapsed = this._cfg.duration;
      this._phase   = 'completed';
      const frame = this._computeFrame(1.0);
      this._lastFrame = frame;
      this._cfg.onComplete(frame);
      return frame;
    }

    const frame = this._computeFrame(rawProgress);
    this._lastFrame = frame;
    this._cfg.onFrame(frame);
    return frame;
  }

  // ── Core blending computation ──────────────────────────────────────────

  /**
   * Compute all three transition lanes at a given raw progress value.
   *
   * @param rawT  Unclamped progress [0, ∞); will be clamped per-lane
   */
  private _computeFrame(rawT: number): TransitionFrame {
    const t = saturate(rawT);

    // ── Lane 1: Color Lerp (Oklab) ────────────────────────────────────
    const colorT = easeInOutCubic(t);
    const glowT  = saturate(colorT * this._cfg.glowLeadFactor);

    const blendedPalette: SpeciesPalette = {
      base:   oklabLerp(this._paletteA.base,   this._paletteB.base,   colorT),
      border: oklabLerp(this._paletteA.border, this._paletteB.border, colorT),
      label:  oklabLerp(this._paletteA.label,  this._paletteB.label,  colorT),
      glow:   oklabLerp(this._paletteA.glow,   this._paletteB.glow,   glowT),
      dim:    oklabLerp(this._paletteA.dim,    this._paletteB.dim,    colorT),
    };

    // ── Lane 2: SDF Morph ─────────────────────────────────────────────
    // Back-ease so the source lingers then snaps to target
    const sdfBlend = saturate(easeOutBack(t));

    // Dynamic smooth-k: starts large (soft blend) and tightens as the morph
    // progresses, giving a jellyfish-like expansion then contraction.
    const dynamicK = this._cfg.sdfSmoothK * (1.0 - 0.6 * t);

    // ── Lane 3: Pattern Fade ──────────────────────────────────────────
    const span = this._cfg.patternXfadeSpan;

    // Old pattern fades out over [0, span]
    const patternFadeOut = 1.0 - smoothstep(0.0, span, t);

    // New pattern fades in over [1 - span, 1]
    const patternFadeIn  = smoothstep(1.0 - span, 1.0, t);

    // ── Assemble frame ────────────────────────────────────────────────
    const frame: TransitionFrame = {
      progress:        t,
      phase:           this._phase,

      palette:         blendedPalette,
      primaryHex:      blendedPalette.base.toHex(),
      glowHex:         blendedPalette.glow.toHex(),

      sdfIndexA:       this._sdfA,
      sdfIndexB:       this._sdfB,
      sdfBlend,
      sdfSmoothK:      dynamicK,

      patternFadeOut,
      patternFadeIn,

      fromSpecies:     this.fromSpecies,
      toSpecies:       this.toSpecies,
      elapsed:         this._elapsed,
      duration:        this._cfg.duration,
    };

    return frame;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// SpeciesTransitionManager — manages concurrent transitions across many cells
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Manages species transition effects for all cells in a scene.
 *
 * Typical integration:
 * ```ts
 *   const manager = new SpeciesTransitionManager();
 *
 *   // When epoch changes and species mutations are detected:
 *   for (const mutation of mutations) {
 *     manager.trigger(mutation.cellId, mutation.fromSpecies, mutation.toSpecies);
 *   }
 *
 *   // In render loop:
 *   const frames = manager.tickAll(delta);
 *   for (const [cellId, frame] of frames) {
 *     applyTransitionFrame(cellId, frame);
 *   }
 * ```
 */
export class SpeciesTransitionManager {
  /** Active transitions keyed by cell_id */
  private readonly _active = new Map<string, SpeciesTransitionEffect>();

  /** Default config applied to all transitions unless overridden per-cell */
  private readonly _defaultConfig: TransitionConfig;

  constructor(defaultConfig?: TransitionConfig) {
    this._defaultConfig = defaultConfig ?? {};
  }

  /**
   * Trigger a new species transition for a cell.
   * If the cell already has a running transition, it is cancelled and
   * replaced (the current blended state becomes the new source palette,
   * but for simplicity we re-start from the new fromSpecies).
   */
  trigger(
    cellId: string,
    fromSpecies: string,
    toSpecies: string,
    config?: TransitionConfig,
  ): SpeciesTransitionEffect {
    // Cancel any existing transition for this cell
    const existing = this._active.get(cellId);
    if (existing?.isRunning) {
      existing.cancel();
    }

    const merged: TransitionConfig = { ...this._defaultConfig, ...config };
    const fx = new SpeciesTransitionEffect(fromSpecies, toSpecies, merged);
    fx.start();
    this._active.set(cellId, fx);
    return fx;
  }

  /**
   * Advance all active transitions by `dt` seconds.
   *
   * Returns a Map of cellId → TransitionFrame for all cells that have
   * running transitions.  Completed transitions are pruned automatically.
   */
  tickAll(dt: number): Map<string, TransitionFrame> {
    const results = new Map<string, TransitionFrame>();

    for (const [cellId, fx] of this._active) {
      if (!fx.isRunning) {
        this._active.delete(cellId);
        continue;
      }

      const frame = fx.tick(dt);
      results.set(cellId, frame);

      // Prune completed transitions after delivering the final frame
      if (frame.phase === 'completed') {
        this._active.delete(cellId);
      }
    }

    return results;
  }

  /** Check if a cell currently has an active transition */
  hasTransition(cellId: string): boolean {
    return this._active.has(cellId) && (this._active.get(cellId)?.isRunning ?? false);
  }

  /** Get the current transition effect for a cell, if any */
  get(cellId: string): SpeciesTransitionEffect | undefined {
    return this._active.get(cellId);
  }

  /** Cancel all running transitions */
  cancelAll(): void {
    for (const fx of this._active.values()) {
      fx.cancel();
    }
    this._active.clear();
  }

  /** Number of currently active transitions */
  get activeCount(): number {
    return this._active.size;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// GLSL snippet for SDF morph — intended to be injected into
// sdf-species-filter or sdf-species-library consumers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * GLSL fragment for smooth-min SDF morph between two species shapes.
 *
 * Uniforms expected:
 *   uniform float u_sdfBlend;    // [0,1] from TransitionFrame.sdfBlend
 *   uniform float u_sdfSmoothK;  // from TransitionFrame.sdfSmoothK
 *   uniform int   u_sdfIndexA;   // source species SDF index
 *   uniform int   u_sdfIndexB;   // target species SDF index
 *
 * Requires speciesSDF(vec2 uv, int species) from sdf-species-library.frag.
 *
 * Usage in main():
 *   float d = morphSpeciesSDF(uv, u_sdfIndexA, u_sdfIndexB, u_sdfBlend, u_sdfSmoothK);
 */
export const SDF_MORPH_GLSL = /* glsl */`
// ── Smooth-min union (polynomial, degree 1) ──────────────────────────────
// Source: Inigo Quiles — https://iquilezles.org/articles/smin/
// Mirrors opSmoothUnion from lygia/sdf/opSmoothUnion.glsl
float opSmoothUnionPoly(float d1, float d2, float k) {
    float h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}

// ── Species SDF morph ────────────────────────────────────────────────────
// Cross-fades between two species SDF shapes using smooth-min union.
// At blend=0.0, only shape A contributes.
// At blend=1.0, only shape B contributes.
// In between, both shapes are present with soft union, creating organic
// intermediate forms rather than linear-blend blobs.
float morphSpeciesSDF(vec2 uv, int indexA, int indexB, float blend, float smoothK) {
    float dA = speciesSDF(uv, indexA);
    float dB = speciesSDF(uv, indexB);

    // Weight the SDF contributions by blend factor.
    // At blend=0 we want purely dA; at blend=1 purely dB.
    // We shift each SDF outward (positive = outside) proportionally to
    // how much it should disappear, then take the smooth union.
    float pushA = blend * 1.5;        // source recedes as blend grows
    float pushB = (1.0 - blend) * 1.5; // target recedes as blend shrinks

    float wA = dA + pushA;
    float wB = dB + pushB;

    // Smooth union of the two weighted SDFs
    return opSmoothUnionPoly(wA, wB, smoothK);
}
`;

/**
 * GLSL snippet for pattern cross-fade.
 *
 * Uniforms expected:
 *   uniform float u_patternFadeOut;  // old pattern opacity [1→0]
 *   uniform float u_patternFadeIn;   // new pattern opacity [0→1]
 *
 * Usage:
 *   vec4 oldPattern = ...; // old species decoration
 *   vec4 newPattern = ...; // new species decoration
 *   vec4 blended = blendPatterns(oldPattern, newPattern, u_patternFadeOut, u_patternFadeIn);
 */
export const PATTERN_FADE_GLSL = /* glsl */`
vec4 blendPatterns(vec4 oldP, vec4 newP, float fadeOut, float fadeIn) {
    vec4 a = oldP * fadeOut;
    vec4 b = newP * fadeIn;
    // Additive blend in the overlap zone, clamped to prevent overbright
    return clamp(a + b, 0.0, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Utility: detect species mutations between epoch snapshots
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A species mutation record emitted when comparing two epoch snapshots.
 */
export interface SpeciesMutation {
  cellId:      string;
  fromSpecies: string;
  toSpecies:   string;
}

/**
 * Compare two epoch cell registries and detect species changes.
 *
 * Both `prevCells` and `nextCells` are maps of cellId → species name.
 * Only cells present in both snapshots whose species differs are returned;
 * cells that appear or disappear between epochs are handled by the separate
 * enter/exit transition system (transition-system.ts).
 *
 * @param prevCells  cellId → species mapping from the previous epoch
 * @param nextCells  cellId → species mapping from the next epoch
 * @returns          Array of detected species mutations
 */
export function detectSpeciesMutations(
  prevCells: ReadonlyMap<string, string>,
  nextCells: ReadonlyMap<string, string>,
): SpeciesMutation[] {
  const mutations: SpeciesMutation[] = [];

  for (const [cellId, nextSpecies] of nextCells) {
    const prevSpecies = prevCells.get(cellId);
    if (prevSpecies !== undefined && prevSpecies !== nextSpecies) {
      mutations.push({ cellId, fromSpecies: prevSpecies, toSpecies: nextSpecies });
    }
  }

  return mutations;
}
