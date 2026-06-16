/**
 * pixi-blur-cell.ts
 *
 * Cell-pipeline blur module adapted from upstream/pixijs-engine/src/filters/defaults/blur/.
 * Provides the Gaussian blur primitive that powers bloom/glow pre-blur in the cell render pass.
 *
 * Design contract:
 *   - Wraps BlurFilter + BlurFilterPass from upstream pixijs-engine (native PixiJS v8 blur).
 *   - Exports CellBlurFilter: a thin, opinionated wrapper with cell-pipeline defaults.
 *   - Exports createCellBlurFilter(): factory keyed on bloom intent.
 *   - Exports applyBloomPreBlur(): attaches a two-pass Gaussian blur to any PixiJS Container
 *     before the bloom compositor reads it — matches the AdvancedBloomFilter internal pattern
 *     (extract → blur → composite) but exposes the blur step independently so pixi-cell-renderer
 *     can slot it into the glow layer pipeline.
 *   - Never generates SVG strings; never hardcodes geometry. JSON params in, PixiJS pipeline out.
 *
 * Upstream sources consumed:
 *   upstream/pixijs-engine/src/filters/defaults/blur/BlurFilter.ts
 *   upstream/pixijs-engine/src/filters/defaults/blur/BlurFilterPass.ts
 *   upstream/pixijs-engine/src/filters/defaults/blur/const.ts
 *   upstream/pixijs-engine/src/filters/defaults/blur/gl/generateBlurGlProgram.ts
 *   upstream/pixijs-engine/src/filters/defaults/blur/gpu/generateBlurProgram.ts
 *
 * Reference: pixi-filters-registry.ts (pattern for how filters integrate into this project)
 */

import { BlurFilter }     from '../../upstream/pixijs-engine/src/filters/defaults/blur/BlurFilter';
import { BlurFilterPass } from '../../upstream/pixijs-engine/src/filters/defaults/blur/BlurFilterPass';

import type { BlurFilterOptions }     from '../../upstream/pixijs-engine/src/filters/defaults/blur/BlurFilter';
import type { BlurFilterPassOptions } from '../../upstream/pixijs-engine/src/filters/defaults/blur/BlurFilterPass';
import type { Container }             from '../../upstream/pixijs-engine/src/scene/container/Container';

// ─── Re-export upstream primitives ───────────────────────────────────────────
// pixi-cell-renderer may want direct access to the raw PixiJS blur classes.
export { BlurFilter, BlurFilterPass };
export type { BlurFilterOptions, BlurFilterPassOptions };

// ─── Per-species bloom blur tuning ───────────────────────────────────────────
//
// These values calibrate the Gaussian blur that is applied to the glow sprite
// before the bloom compositor reads it.  Stronger / higher-quality for species
// whose glow should feel physically intense (cil-bolt, cil-eye); lighter for
// utility species that need glow but not at the expense of GPU budget.

export interface CellBlurPreset {
  /** Combined XY blur strength (pixels). */
  strength: number;
  /** Number of blur passes — higher = better quality, more expensive. */
  quality: number;
  /** Gaussian kernel footprint: 5 | 7 | 9 | 11 | 13 | 15. */
  kernelSize: 5 | 7 | 9 | 11 | 13 | 15;
  /**
   * When true, bloom radius is effectively symmetric; when false the blur is
   * slightly stronger on X to suggest horizontal light spread on the dark
   * background.  Implemented as a strengthX / strengthY ratio of 1.15.
   */
  anisotropic: boolean;
}

const CELL_BLUR_PRESETS: Record<string, CellBlurPreset> = {
  // High-energy: iris, bolt — wide, high-quality glow halo
  'cil-eye':         { strength: 14, quality: 4, kernelSize: 9,  anisotropic: false },
  'cil-bolt':        { strength: 16, quality: 5, kernelSize: 11, anisotropic: true  },

  // Medium glow: color signals, loops
  'cil-vector':      { strength: 10, quality: 3, kernelSize: 7,  anisotropic: false },
  'cil-loop':        { strength: 10, quality: 3, kernelSize: 7,  anisotropic: false },
  'cil-plus':        { strength: 10, quality: 3, kernelSize: 7,  anisotropic: false },

  // Subtle / utility: code, layers, filter, graph, arrow
  'cil-code':        { strength:  6, quality: 2, kernelSize: 5,  anisotropic: false },
  'cil-layers':      { strength:  7, quality: 2, kernelSize: 5,  anisotropic: false },
  'cil-filter':      { strength:  8, quality: 3, kernelSize: 5,  anisotropic: false },
  'cil-graph':       { strength:  6, quality: 2, kernelSize: 5,  anisotropic: false },
  'cil-arrow-right': { strength:  5, quality: 2, kernelSize: 5,  anisotropic: false },
};

/** Fallback preset for unknown species. */
const DEFAULT_CELL_BLUR_PRESET: CellBlurPreset = {
  strength: 8, quality: 3, kernelSize: 5, anisotropic: false,
};

/**
 * Return the blur preset for a cell species.
 * @param species - e.g. 'cil-eye', 'cil-bolt' …
 */
export function getCellBlurPreset(species: string): CellBlurPreset {
  return CELL_BLUR_PRESETS[species] ?? DEFAULT_CELL_BLUR_PRESET;
}

// ─── CellBlurFilter ───────────────────────────────────────────────────────────
//
// Thin opinionated wrapper around BlurFilter that hard-bakes cell-pipeline
// conventions:
//   - repeatEdgePixels = false (we want the natural edge feather for bloom halos)
//   - legacy = false            (use the optimised halving-scheme multi-pass)
//   - resolution = 'inherit'    (respect PixiJS device-pixel-ratio)

export interface CellBlurFilterOptions {
  /** Cell species key — drives preset lookup when explicit values are omitted. */
  species?: string;
  /** Override blur strength; defaults to species preset. */
  strength?: number;
  /** Override quality (pass count); defaults to species preset. */
  quality?: number;
  /** Override kernel size; defaults to species preset. */
  kernelSize?: 5 | 7 | 9 | 11 | 13 | 15;
  /**
   * When true, strength is split anisotropically (X × 1.15 : Y) to mimic
   * the horizontal light-spread characteristic of dark-panel UIs.
   * Overrides the preset's anisotropic flag.
   */
  anisotropic?: boolean;
}

/**
 * CellBlurFilter — species-aware PixiJS BlurFilter for the cell glow pipeline.
 *
 * Wraps upstream/pixijs-engine BlurFilter with per-species defaults and
 * exposes `setStrengthForBloom(scale)` so the Ticker can animate bloom pulse.
 *
 * @example
 * ```ts
 * const f = new CellBlurFilter({ species: 'cil-eye' });
 * glowSprite.filters = [f];
 *
 * // Animate bloom pulse in a Ticker callback:
 * const base = f.baseStrength;
 * app.ticker.add(t => f.setStrengthForBloom(1 + 0.25 * Math.sin(t.lastTime * 0.002)));
 * ```
 */
export class CellBlurFilter extends BlurFilter {
  /** Base (unscaled) strength sourced from species preset or constructor options. */
  public readonly baseStrength: number;
  /** Whether this instance uses anisotropic X/Y split. */
  public readonly anisotropic: boolean;

  constructor(options: CellBlurFilterOptions = {}) {
    const preset = getCellBlurPreset(options.species ?? '');

    const strength   = options.strength   ?? preset.strength;
    const quality    = options.quality    ?? preset.quality;
    const kernelSize = options.kernelSize ?? preset.kernelSize;
    const aniso      = options.anisotropic ?? preset.anisotropic;

    const strengthX = aniso ? strength * 1.15 : strength;
    const strengthY = strength;

    super({
      strengthX,
      strengthY,
      quality,
      kernelSize,
      legacy: false,
    } satisfies BlurFilterOptions);

    this.baseStrength = strength;
    this.anisotropic  = aniso;
  }

  /**
   * Scale blur strength relative to `baseStrength`.
   * Intended for Ticker-driven bloom pulse animation.
   *
   * @param scale - Multiplier applied to baseStrength (e.g. 0.8–1.3 for pulse).
   */
  setStrengthForBloom(scale: number): void {
    const s = this.baseStrength * Math.max(0, scale);
    this.strengthX = this.anisotropic ? s * 1.15 : s;
    this.strengthY = s;
  }
}

// ─── createCellBlurFilter ─────────────────────────────────────────────────────

/**
 * Factory: create a `CellBlurFilter` (or a raw `BlurFilter`) configured for a
 * specific bloom usage context within the cell pipeline.
 *
 * @param intent - Semantic blur role:
 *   - `'glow-halo'`   : soft wide halo around the glow sprite (default for bloom pre-blur)
 *   - `'focus-blur'`  : tighter blur preserving form; used for depth-of-field hint
 *   - `'motion-trail'`: very light single-pass blur for motion trail compositing
 * @param species - Optional cell species to drive preset selection.
 * @param overrides - Optional raw BlurFilterOptions to bypass preset.
 */
export function createCellBlurFilter(
  intent: 'glow-halo' | 'focus-blur' | 'motion-trail' = 'glow-halo',
  species?: string,
  overrides?: Partial<CellBlurFilterOptions>,
): CellBlurFilter {
  const intentMod: Partial<CellBlurFilterOptions> = (() => {
    switch (intent) {
      case 'glow-halo':
        // Full-quality wide Gaussian — canonical pre-bloom blur
        return {};
      case 'focus-blur':
        // Tighter blur; halve strength, keep quality
        return { strength: undefined }; // resolved below via scale
      case 'motion-trail':
        // Single cheap pass
        return { quality: 1, kernelSize: 5 };
    }
  })();

  const opts: CellBlurFilterOptions = { species, ...intentMod, ...overrides };

  // For focus-blur: halve the species-preset strength
  if (intent === 'focus-blur' && opts.strength === undefined) {
    const preset = getCellBlurPreset(species ?? '');
    opts.strength = Math.max(2, preset.strength * 0.5);
  }

  return new CellBlurFilter(opts);
}

// ─── applyBloomPreBlur ────────────────────────────────────────────────────────
//
// Attach a Gaussian blur to a PixiJS Container so the bloom compositor
// (AdvancedBloomFilter / GlowFilter) picks up a pre-blurred texture.
//
// Pattern mirrors AdvancedBloomFilter.apply():
//   1. Extract brightness  (done externally by AdvancedBloomFilter)
//   2. Blur the extracted  ← THIS IS WHAT WE SUPPLY
//   3. Composite           (done externally)
//
// In the cell renderer we want explicit control over step 2 so we can:
//   • Tune blur per species (cil-bolt gets a wider bloom halo)
//   • Animate bloom pulse via setStrengthForBloom()
//   • Pool/reuse blur instances across cells of the same species

/**
 * Options for `applyBloomPreBlur`.
 */
export interface BloomPreBlurOptions {
  /** Cell species to select preset. */
  species?: string;
  /**
   * Prepend the blur to any existing filters on the container rather than
   * replacing them.  Default: true.
   */
  prepend?: boolean;
  /** Direct overrides forwarded to CellBlurFilter constructor. */
  blurOverrides?: Partial<CellBlurFilterOptions>;
}

/**
 * Attach a species-tuned `CellBlurFilter` to `container.filters` as the
 * first (pre-bloom) stage in the filter chain.
 *
 * Returns the created `CellBlurFilter` so the caller can animate it.
 *
 * @example
 * ```ts
 * // In pixi-cell-renderer.ts createGlowSprite():
 * const blurFilter = applyBloomPreBlur(glowSprite, { species });
 * // Then attach the bloom compositor on top:
 * glowSprite.filters = [...(glowSprite.filters ?? []), bloomCompositor];
 * // Animate:
 * app.ticker.add(t => blurFilter.setStrengthForBloom(1 + 0.2 * Math.sin(t.lastTime * 0.001)));
 * ```
 */
export function applyBloomPreBlur(
  container: Container,
  options: BloomPreBlurOptions = {},
): CellBlurFilter {
  const { species, prepend = true, blurOverrides } = options;

  const blurFilter = createCellBlurFilter('glow-halo', species, blurOverrides);

  if (prepend) {
    const existing: unknown[] = (container.filters as unknown[] | null) ?? [];
    // Avoid stacking duplicate CellBlurFilters
    const withoutPrev = existing.filter(f => !(f instanceof CellBlurFilter));
    container.filters = [blurFilter, ...withoutPrev] as any;
  } else {
    container.filters = [blurFilter] as any;
  }

  return blurFilter;
}

// ─── BlurPassPool ─────────────────────────────────────────────────────────────
//
// Lightweight pool for BlurFilterPass instances (single-axis passes).
// The cell renderer creates O(N-cells) glow sprites; pooling single-axis
// passes avoids re-compiling the same GLSL programs repeatedly.
//
// Key: `${horizontal ? 'h' : 'v'}-k${kernelSize}`
//
// This intentionally does NOT pool full BlurFilter objects because those
// hold mutable strengthX/strengthY state that is per-cell.

const _passPool = new Map<string, BlurFilterPass[]>();

/**
 * Acquire a `BlurFilterPass` from the pool (or create a new one).
 * Caller is responsible for setting `.blur` (strength) and `.quality` after acquire.
 */
export function acquireBlurPass(
  horizontal: boolean,
  kernelSize: 5 | 7 | 9 | 11 | 13 | 15 = 5,
): BlurFilterPass {
  const key = `${horizontal ? 'h' : 'v'}-k${kernelSize}`;
  const pool = _passPool.get(key);
  if (pool && pool.length > 0) {
    return pool.pop()!;
  }
  return new BlurFilterPass({
    horizontal,
    kernelSize,
    strength: 8,
    quality: 3,
    legacy: false,
  } satisfies BlurFilterPassOptions);
}

/**
 * Return a `BlurFilterPass` to the pool after it is no longer needed.
 */
export function releaseBlurPass(pass: BlurFilterPass): void {
  const key = `${pass.horizontal ? 'h' : 'v'}-k5`; // kernelSize not exposed publicly; assume 5
  const pool = _passPool.get(key) ?? [];
  pool.push(pass);
  _passPool.set(key, pool);
}

// ─── Bloom-pipeline builder ───────────────────────────────────────────────────
//
// Convenience: build the full [CellBlurFilter, ...compositorFilters] array
// that pixi-cell-renderer slots onto a glow sprite.

/**
 * Build a filter array for use on a glow/bloom sprite:
 *   [ CellBlurFilter(species), ...extraFilters ]
 *
 * The caller supplies `extraFilters` (e.g. an AdvancedBloomFilter or GlowFilter)
 * that read the pre-blurred texture.  The blur sits first in the chain so PixiJS
 * processes it before the compositor sees the sprite.
 *
 * @returns { filters, blurFilter } — assign `filters` to `sprite.filters`;
 *          animate `blurFilter.setStrengthForBloom()` in the Ticker.
 *
 * @example
 * ```ts
 * // In pixi-cell-renderer createGlowSprite():
 * import { buildBloomFilterChain } from './pixi-blur-cell';
 * import { AdvancedBloomFilter } from '../../../upstream/pixijs-filters/src/advanced-bloom';
 *
 * const bloomScale = SPECIES_BLOOM_SCALE[species] ?? 1.5;
 * const bloom = new AdvancedBloomFilter({ bloomScale, threshold: 0.5, blur: 8, quality: 4 });
 *
 * const { filters, blurFilter } = buildBloomFilterChain(species, [bloom]);
 * glow.filters = filters;
 * // Store blurFilter on container for Ticker access:
 * (container as any).__bloomBlur = blurFilter;
 * ```
 */
export function buildBloomFilterChain(
  species: string,
  compositorFilters: unknown[],
  blurOverrides?: Partial<CellBlurFilterOptions>,
): { filters: unknown[]; blurFilter: CellBlurFilter } {
  const blurFilter = createCellBlurFilter('glow-halo', species, blurOverrides);
  const filters    = [blurFilter, ...compositorFilters];
  return { filters, blurFilter };
}
