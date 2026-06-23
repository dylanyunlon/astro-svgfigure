/**
 * pixi-filters-registry.ts
 * Auto-registered PixiJS filter registry sourced from upstream/pixijs-filters-v2.
 * AdvancedBloomFilter (AT HydraBloom 等价实现) 来自 upstream/pixijs-filters (v1).
 * Provides FILTER_REGISTRY (name → constructor) and createFilter factory.
 *
 * M031: GlowFilter (pixijs-filters-v2/src/glow) 已注册为 cell 选中高亮的外发光。
 *   - createCellGlow(color, opts)  — 按 hover/select 模式实例化 GlowFilter
 *   - GlowFilterOptions type 同步导出，供 pixi-cell-renderer.ts 的 setGlow() 使用
 */

// ─── advanced-bloom (upstream/pixijs-filters — AT HydraBloom 核心后处理) ────
import { AdvancedBloomFilter } from '../../../upstream/pixijs-filters/src/advanced-bloom';
export type { AdvancedBloomFilterOptions } from '../../../upstream/pixijs-filters/src/advanced-bloom/AdvancedBloomFilter';

// ─── blur ────────────────────────────────────────────────────────────────────
import { KawaseBlurFilter }    from '../../../upstream/pixijs-filters-v2/src/kawase-blur';
import { BackdropBlurFilter }  from '../../../upstream/pixijs-filters-v2/src/backdrop-blur';
import { RadialBlurFilter }    from '../../../upstream/pixijs-filters-v2/src/radial-blur';
import { ZoomBlurFilter }      from '../../../upstream/pixijs-filters-v2/src/zoom-blur';
import { TiltShiftFilter }     from '../../../upstream/pixijs-filters-v2/src/tilt-shift';
import { MotionBlurFilter }    from '../../../upstream/pixijs-filters-v2/src/motion-blur';

// ─── color ───────────────────────────────────────────────────────────────────
import { AdjustmentFilter }       from '../../../upstream/pixijs-filters-v2/src/adjustment';
import { ColorGradientFilter }    from '../../../upstream/pixijs-filters-v2/src/color-gradient';
import { ColorMapFilter }         from '../../../upstream/pixijs-filters-v2/src/color-map';
import { ColorOverlayFilter }     from '../../../upstream/pixijs-filters-v2/src/color-overlay';
import { ColorReplaceFilter }     from '../../../upstream/pixijs-filters-v2/src/color-replace';
import { GrayscaleFilter }        from '../../../upstream/pixijs-filters-v2/src/grayscale';
import { HslAdjustmentFilter }    from '../../../upstream/pixijs-filters-v2/src/hsl-adjustment';
import { MultiColorReplaceFilter }from '../../../upstream/pixijs-filters-v2/src/multi-color-replace';

// ─── style ───────────────────────────────────────────────────────────────────
import { AsciiFilter }       from '../../../upstream/pixijs-filters-v2/src/ascii';
import { CrossHatchFilter }  from '../../../upstream/pixijs-filters-v2/src/cross-hatch';
import { CRTFilter }         from '../../../upstream/pixijs-filters-v2/src/crt';
import { DotFilter }         from '../../../upstream/pixijs-filters-v2/src/dot';
import { EmbossFilter }      from '../../../upstream/pixijs-filters-v2/src/emboss';
import { OldFilmFilter }     from '../../../upstream/pixijs-filters-v2/src/old-film';
import { PixelateFilter }    from '../../../upstream/pixijs-filters-v2/src/pixelate';
import { SimplexNoiseFilter }from '../../../upstream/pixijs-filters-v2/src/simplex-noise';

// ─── distort ─────────────────────────────────────────────────────────────────
import { BulgePinchFilter } from '../../../upstream/pixijs-filters-v2/src/bulge-pinch';
import { ShockwaveFilter }  from '../../../upstream/pixijs-filters-v2/src/shockwave';
import { TwistFilter }      from '../../../upstream/pixijs-filters-v2/src/twist';
import { ReflectionFilter } from '../../../upstream/pixijs-filters-v2/src/reflection';

// ─── light ───────────────────────────────────────────────────────────────────
import { BloomFilter }          from '../../../upstream/pixijs-filters-v2/src/bloom';
import { GlowFilter }           from '../../../upstream/pixijs-filters-v2/src/glow';
export type { GlowFilterOptions } from '../../../upstream/pixijs-filters-v2/src/glow/GlowFilter';
import { GodrayFilter }         from '../../../upstream/pixijs-filters-v2/src/godray';
import { SimpleLightmapFilter } from '../../../upstream/pixijs-filters-v2/src/simple-lightmap';

// ─── edge ────────────────────────────────────────────────────────────────────
import { BevelFilter }      from '../../../upstream/pixijs-filters-v2/src/bevel';
import { OutlineFilter }    from '../../../upstream/pixijs-filters-v2/src/outline';
import { DropShadowFilter } from '../../../upstream/pixijs-filters-v2/src/drop-shadow';

// ─── fx ──────────────────────────────────────────────────────────────────────
import { GlitchFilter }      from '../../../upstream/pixijs-filters-v2/src/glitch';
import { RGBSplitFilter }    from '../../../upstream/pixijs-filters-v2/src/rgb-split';
import { ConvolutionFilter } from '../../../upstream/pixijs-filters-v2/src/convolution';

// ─────────────────────────────────────────────────────────────────────────────

export type FilterName =
  // advanced-bloom (AT HydraBloom 等价实现 — upstream/pixijs-filters)
  | 'advanced-bloom'
  // blur
  | 'kawase-blur' | 'backdrop-blur' | 'radial-blur'
  | 'zoom-blur'   | 'tilt-shift'    | 'motion-blur'
  // color
  | 'adjustment'       | 'color-gradient'     | 'color-map'
  | 'color-overlay'    | 'color-replace'      | 'grayscale'
  | 'hsl-adjustment'   | 'multi-color-replace'
  // style
  | 'ascii'     | 'cross-hatch' | 'crt'          | 'dot'
  | 'emboss'    | 'old-film'    | 'pixelate'      | 'simplex-noise'
  // distort
  | 'bulge-pinch' | 'shockwave' | 'twist' | 'reflection'
  // light
  | 'bloom' | 'glow' | 'godray' | 'simple-lightmap'
  // edge
  | 'bevel' | 'outline' | 'drop-shadow'
  // fx
  | 'glitch' | 'rgb-split' | 'convolution';

export type FilterCategory =
  | 'blur' | 'color' | 'style' | 'distort' | 'light' | 'edge' | 'fx';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type FilterConstructor = new (...args: any[]) => any;

interface FilterMeta {
  ctor: FilterConstructor;
  category: FilterCategory;
}

/**
 * Central registry mapping filter name → { ctor, category }.
 * Add new filters here; createFilter picks them up automatically.
 */
export const FILTER_REGISTRY: Record<FilterName, FilterMeta> = {
  // ── advanced-bloom (AT HydraBloom core — upstream/pixijs-filters) ─────────
  'advanced-bloom': { ctor: AdvancedBloomFilter, category: 'light' },

  // ── blur ──────────────────────────────────────────────────────────────────
  'kawase-blur':   { ctor: KawaseBlurFilter,    category: 'blur'    },
  'backdrop-blur': { ctor: BackdropBlurFilter,  category: 'blur'    },
  'radial-blur':   { ctor: RadialBlurFilter,    category: 'blur'    },
  'zoom-blur':     { ctor: ZoomBlurFilter,      category: 'blur'    },
  'tilt-shift':    { ctor: TiltShiftFilter,     category: 'blur'    },
  'motion-blur':   { ctor: MotionBlurFilter,    category: 'blur'    },

  // ── color ─────────────────────────────────────────────────────────────────
  'adjustment':          { ctor: AdjustmentFilter,        category: 'color' },
  'color-gradient':      { ctor: ColorGradientFilter,     category: 'color' },
  'color-map':           { ctor: ColorMapFilter,          category: 'color' },
  'color-overlay':       { ctor: ColorOverlayFilter,      category: 'color' },
  'color-replace':       { ctor: ColorReplaceFilter,      category: 'color' },
  'grayscale':           { ctor: GrayscaleFilter,         category: 'color' },
  'hsl-adjustment':      { ctor: HslAdjustmentFilter,     category: 'color' },
  'multi-color-replace': { ctor: MultiColorReplaceFilter, category: 'color' },

  // ── style ─────────────────────────────────────────────────────────────────
  'ascii':         { ctor: AsciiFilter,       category: 'style' },
  'cross-hatch':   { ctor: CrossHatchFilter,  category: 'style' },
  'crt':           { ctor: CRTFilter,         category: 'style' },
  'dot':           { ctor: DotFilter,         category: 'style' },
  'emboss':        { ctor: EmbossFilter,      category: 'style' },
  'old-film':      { ctor: OldFilmFilter,     category: 'style' },
  'pixelate':      { ctor: PixelateFilter,    category: 'style' },
  'simplex-noise': { ctor: SimplexNoiseFilter,category: 'style' },

  // ── distort ───────────────────────────────────────────────────────────────
  'bulge-pinch': { ctor: BulgePinchFilter, category: 'distort' },
  'shockwave':   { ctor: ShockwaveFilter,  category: 'distort' },
  'twist':       { ctor: TwistFilter,      category: 'distort' },
  'reflection':  { ctor: ReflectionFilter, category: 'distort' },

  // ── light ─────────────────────────────────────────────────────────────────
  'bloom':           { ctor: BloomFilter,          category: 'light' },
  'glow':            { ctor: GlowFilter,           category: 'light' },
  'godray':          { ctor: GodrayFilter,         category: 'light' },
  'simple-lightmap': { ctor: SimpleLightmapFilter, category: 'light' },

  // ── edge ──────────────────────────────────────────────────────────────────
  'bevel':       { ctor: BevelFilter,      category: 'edge' },
  'outline':     { ctor: OutlineFilter,    category: 'edge' },
  'drop-shadow': { ctor: DropShadowFilter, category: 'edge' },

  // ── fx ────────────────────────────────────────────────────────────────────
  'glitch':      { ctor: GlitchFilter,      category: 'fx' },
  'rgb-split':   { ctor: RGBSplitFilter,    category: 'fx' },
  'convolution': { ctor: ConvolutionFilter, category: 'fx' },
};

/**
 * Retrieve all filter names belonging to a given category.
 */
export function filtersByCategory(category: FilterCategory): FilterName[] {
  return (Object.entries(FILTER_REGISTRY) as [FilterName, FilterMeta][])
    .filter(([, meta]) => meta.category === category)
    .map(([name]) => name);
}

/**
 * Factory: instantiate any registered filter by name.
 *
 * @param name   - A key from FilterName
 * @param options - Constructor arguments forwarded verbatim
 * @returns      The instantiated filter
 *
 * @example
 * const blur  = createFilter('kawase-blur', { strength: 4, quality: 3 });
 * const glow  = createFilter('glow', { distance: 15, outerStrength: 2 });
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function createFilter<O = any>(name: FilterName, options?: O): InstanceType<FilterConstructor> {
  const entry = FILTER_REGISTRY[name];
  if (!entry) {
    throw new Error(`[pixi-filters-registry] Unknown filter: "${name}"`);
  }
  return options !== undefined ? new entry.ctor(options) : new entry.ctor();
}

// ─── Cell glow presets (M031) ────────────────────────────────────────────────

/**
 * Glow preset names used by setGlow() in pixi-cell-renderer.ts.
 *   'hover'  — soft cyan outer-glow applied on pointerover
 *   'select' — intense gold outer-glow applied on pointertap / programmatic select
 */
export type CellGlowMode = 'hover' | 'select';

/**
 * Per-mode GlowFilter configuration (M031).
 * distance/quality are compile-time constants baked into the GLSL shader;
 * only outerStrength, alpha, and color can be mutated after construction.
 *
 * hover  → cool cyan, moderate strength, no knockout
 * select → gold,      higher  strength, no knockout
 */
export const CELL_GLOW_PRESETS: Record<CellGlowMode, import('../../upstream/pixijs-filters-v2/src/glow/GlowFilter').GlowFilterOptions> = {
  hover: {
    distance:      12,
    outerStrength: 2.5,
    innerStrength: 0,
    color:         0x88CCFF,   // cyan-blue (matches HOVER_COLOR in cell-event-system)
    alpha:         0.9,
    quality:       0.15,
    knockout:      false,
  },
  select: {
    distance:      14,
    outerStrength: 4.0,
    innerStrength: 0.5,
    color:         0xFFD700,   // gold (matches SELECT_COLOR in cell-event-system)
    alpha:         1.0,
    quality:       0.15,
    knockout:      false,
  },
};

/**
 * createCellGlow — instantiate a GlowFilter for the given interaction mode.
 *
 * Uses CELL_GLOW_PRESETS so callers don't hard-code filter parameters.
 *
 * @example
 * // Applied by setGlow() internally; you can also call directly:
 * const glow = createCellGlow('hover');
 * container.filters = [glow];
 */
export function createCellGlow(mode: CellGlowMode): GlowFilter {
  return new GlowFilter(CELL_GLOW_PRESETS[mode]);
}

// ─── Per-cell filter chain (M132) ──────────────────────────────────────────
//
// buildCellFilterChain(container, species, bloomVariant)
//   → reads channels/physics/bloom_variants.json     for AdvancedBloomFilter params
//   → reads channels/physics/species_visual_traits.json for GlowFilter + GodrayFilter params
//   → assigns container.filters = [GlowFilter, AdvancedBloomFilter, GodrayFilter]
//
// This consolidates all three per-cell post-process filters into a single call,
// driven entirely by the physics JSON data rather than hard-coded inline values.

import type { Container } from '../../../upstream/pixijs-engine/src/scene/container/Container';

// ── JSON shape: channels/physics/bloom_variants.json ──────────────────────
interface BloomVariantEntry {
  bloomStrength: number;
  bloomRadius: number;
  luminosityThreshold: number;
}

/**
 * bloom_variants.json keyed by scene/variant name.
 * Underscore-prefixed keys (_source, _note) are metadata and excluded from lookup.
 */
type BloomVariantsMap = Record<string, BloomVariantEntry>;

// ── JSON shape: channels/physics/species_visual_traits.json ───────────────
interface SpeciesVisualColor {
  primary: string;
  secondary: string;
  glow: string;        // hex color string used for GlowFilter color
  hex_css: string;
}

interface SpeciesVisualTraitEntry {
  _role: string;
  color: SpeciesVisualColor;
  scale: {
    base_radius: number;
    min_radius: number;
    max_radius: number;
    aspect_ratio: number;
  };
  internal_structure: {
    count: number;
    type: string;
    description: string;
  };
  animation_hint: string;
}

type SpeciesVisualTraitsMap = Record<string, SpeciesVisualTraitEntry>;

// ── Static JSON imports (bundler-resolved, channels/physics/) ─────────────
// These are imported as JSON modules; the bundler (Vite/Astro) handles them.
import bloomVariantsData from '../../../channels/physics/bloom_variants.json';
import speciesVisualTraitsData from '../../../channels/physics/species_visual_traits.json';

const BLOOM_VARIANTS  = bloomVariantsData as BloomVariantsMap;
const SPECIES_TRAITS  = speciesVisualTraitsData as SpeciesVisualTraitsMap;

/**
 * Default bloom variant used when the caller doesn't specify one.
 * "home" is the strongest scene bloom (bloomStrength 3.82) — visually prominent.
 */
const DEFAULT_BLOOM_VARIANT = 'home';

/**
 * Godray presets per species, derived from animation_hint and internal_structure
 * in species_visual_traits.json. Each species gets a tuned GodrayFilter config
 * that complements its visual identity.
 */
function buildGodrayOptions(
  species: string,
  traits: SpeciesVisualTraitEntry,
): ConstructorParameters<typeof GodrayFilter>[0] {
  const structureCount = traits.internal_structure.count;
  const hint           = traits.animation_hint;

  // Base godray config — species-specific overrides below
  const base = {
    alpha: 0.55,
    time:  0,
  };

  // Radial / pulsing species get focal-point (non-parallel) rays
  if (hint === 'pulse_radial' || hint === 'spin_loop' || hint === 'pulse_nodes') {
    return {
      ...base,
      angle:      30,
      gain:       0.4 + structureCount * 0.02,   // more internal structures → slightly brighter
      lacunarity: 2.5,
      parallel:   false,
      center:     { x: 0, y: 0 },                // caller should reposition to cell centre
    };
  }

  // Directional / flow species get parallel rays
  if (hint === 'drift_horizontal' || hint === 'flow_right') {
    return {
      ...base,
      angle:      15,
      gain:       0.25,
      lacunarity: 3.5,
      parallel:   true,
    };
  }

  // Electric / flash species get pulsing burst from top
  if (hint === 'flash_angular') {
    return {
      ...base,
      angle:      0,
      gain:       0.55,
      lacunarity: 2.0,
      parallel:   false,
      center:     { x: 0, y: 0 },
      alpha:      0.7,
    };
  }

  // Stacking / converging species get moderate parallel rays
  if (hint === 'stack_rise' || hint === 'converge_merge') {
    return {
      ...base,
      angle:      45,
      gain:       0.3,
      lacunarity: 3.0,
      parallel:   true,
      alpha:      0.4,
    };
  }

  // Sweep / blink / other — moderate focal rays as fallback
  return {
    ...base,
    angle:      25,
    gain:       0.35,
    lacunarity: 2.8,
    parallel:   true,
  };
}

/**
 * Parse a CSS hex color string (#RRGGBB or #RGB) into a numeric 0xRRGGBB value.
 */
function hexToNumber(hex: string): number {
  const cleaned = hex.replace('#', '');
  if (cleaned.length === 3) {
    const r = cleaned[0], g = cleaned[1], b = cleaned[2];
    return parseInt(`${r}${r}${g}${g}${b}${b}`, 16);
  }
  return parseInt(cleaned, 16);
}

/**
 * Available bloom variant names (keys from bloom_variants.json, excluding metadata).
 */
export type BloomVariantName = string;

/**
 * buildCellFilterChain — construct and assign the per-cell PixiJS filter chain.
 *
 * Reads physics JSON data to configure three filters in a fixed order:
 *   1. GlowFilter          — species glow color from species_visual_traits.json
 *   2. AdvancedBloomFilter  — bloom params from bloom_variants.json
 *   3. GodrayFilter         — species-tuned godrays from species_visual_traits.json
 *
 * @param container     The PixiJS Container for the cell
 * @param species       Species key (e.g. 'cil-eye', 'cil-bolt') from cell_registry
 * @param bloomVariant  Key into bloom_variants.json (default: 'home')
 * @returns             The three filter instances for external animation/mutation
 *
 * @example
 * const { glowFilter, bloomFilter, godrayFilter } =
 *   buildCellFilterChain(cellContainer, 'cil-eye', 'homebloom');
 * // Animate godrayFilter.time in a Ticker callback
 */
export function buildCellFilterChain(
  container: Container,
  species: string,
  bloomVariant: string = DEFAULT_BLOOM_VARIANT,
): {
  glowFilter: GlowFilter;
  bloomFilter: AdvancedBloomFilter;
  godrayFilter: GodrayFilter;
} {
  // ── 1. GlowFilter — species glow color ──────────────────────────────────
  const traits = SPECIES_TRAITS[species];
  const glowColor  = traits ? hexToNumber(traits.color.glow) : 0xFFFFFF;
  const glowFilter = new GlowFilter({
    distance:      10,
    outerStrength: 3.0,
    innerStrength: 0.5,
    color:         glowColor,
    alpha:         0.85,
    quality:       0.15,
    knockout:      false,
  });

  // ── 2. AdvancedBloomFilter — bloom variant params ───────────────────────
  const bloom = BLOOM_VARIANTS[bloomVariant] ?? BLOOM_VARIANTS[DEFAULT_BLOOM_VARIANT];
  const bloomFilter = new AdvancedBloomFilter({
    threshold:  bloom.luminosityThreshold,
    bloomScale: bloom.bloomStrength,
    brightness: 1.0,
    blur:       bloom.bloomRadius * 4,   // bloomRadius (0–1 range) scaled to blur strength
    quality:    4,
  });

  // ── 3. GodrayFilter — species-tuned godrays ─────────────────────────────
  const godrayOpts   = traits
    ? buildGodrayOptions(species, traits)
    : { angle: 30, gain: 0.35, lacunarity: 2.5, parallel: true, alpha: 0.5, time: 0 };
  const godrayFilter = new GodrayFilter(godrayOpts);

  // ── Assign the ordered filter chain ─────────────────────────────────────
  container.filters = [glowFilter, bloomFilter, godrayFilter] as any[];

  return { glowFilter, bloomFilter, godrayFilter };
}

/**
 * getBloomVariantNames — list all available bloom variant keys.
 * Excludes metadata keys (_source, _note).
 */
export function getBloomVariantNames(): string[] {
  return Object.keys(BLOOM_VARIANTS).filter(k => !k.startsWith('_'));
}

/**
 * getSpeciesNames — list all species keys from species_visual_traits.json.
 * Excludes metadata keys (_comment).
 */
export function getSpeciesNames(): string[] {
  return Object.keys(SPECIES_TRAITS).filter(k => !k.startsWith('_'));
}

// ─── M390: Parametric filter chain from species_params ─────────────────────
//
// buildFilterChain(sp) reads glow_intensity, glow_color, bloom_strength
// directly from the species_params record. No species-name lookup involved.
// Returns only the filters whose corresponding parameter is > 0.

/**
 * Shape returned by buildFilterChain.
 * Each filter field is present only when its trigger parameter > 0.
 */
export interface ParametricFilterChainResult {
  glowFilter:  GlowFilter  | null;
  bloomFilter: BloomFilter  | null;
  /** Flat array of the instantiated filters (non-null), ready for container.filters. */
  filters: (GlowFilter | BloomFilter)[];
}

/**
 * buildFilterChain — construct a PixiJS filter chain purely from species_params.
 *
 * Reads numeric / color values directly from the params record:
 *   - glow_intensity > 0  → GlowFilter  (color = glow_color, outerStrength = glow_intensity)
 *   - bloom_strength > 0  → BloomFilter  (strength = bloom_strength)
 *
 * No species name is used for lookup — the caller passes the already-resolved
 * species_params object (e.g. from composite_params.json or params.json).
 *
 * @param sp  species_params record (Record<string, unknown>)
 * @returns   ParametricFilterChainResult with nullable filter refs + flat array
 *
 * @example
 * const { filters, glowFilter, bloomFilter } = buildFilterChain(cell.species_params);
 * container.filters = filters;
 */
export function buildFilterChain(
  sp: Record<string, unknown>,
): ParametricFilterChainResult {
  let glowFilter:  GlowFilter  | null = null;
  let bloomFilter: BloomFilter  | null = null;

  // ── GlowFilter — glow_intensity > 0 ──────────────────────────────────────
  const glowIntensity = (sp?.glow_intensity as number | undefined) ?? 0;
  if (glowIntensity > 0) {
    const glowColorRaw = sp?.glow_color as string | undefined;
    const glowColor    = glowColorRaw ? hexToNumber(glowColorRaw) : 0xFFFFFF;
    glowFilter = new GlowFilter({
      distance:      10,
      outerStrength: glowIntensity,
      innerStrength: 0,
      color:         glowColor,
      alpha:         0.85,
      quality:       0.15,
      knockout:      false,
    });
  }

  // ── BloomFilter — bloom_strength > 0 ──────────────────────────────────────
  const bloomStrength = (sp?.bloom_strength as number | undefined) ?? 0;
  if (bloomStrength > 0) {
    bloomFilter = new BloomFilter({
      strength: bloomStrength,
      quality:  4,
    });
  }

  // ── Collect non-null filters into a flat array ────────────────────────────
  const filters: (GlowFilter | BloomFilter)[] = [];
  if (glowFilter)  filters.push(glowFilter);
  if (bloomFilter) filters.push(bloomFilter);

  return { glowFilter, bloomFilter, filters };
}

// ─── Re-export all constructors for direct use ──────────────────────────────
export {
  // advanced-bloom (AT HydraBloom — upstream/pixijs-filters)
  AdvancedBloomFilter,
  // blur
  KawaseBlurFilter, BackdropBlurFilter, RadialBlurFilter,
  ZoomBlurFilter, TiltShiftFilter, MotionBlurFilter,
  // color
  AdjustmentFilter, ColorGradientFilter, ColorMapFilter,
  ColorOverlayFilter, ColorReplaceFilter, GrayscaleFilter,
  HslAdjustmentFilter, MultiColorReplaceFilter,
  // style
  AsciiFilter, CrossHatchFilter, CRTFilter, DotFilter,
  EmbossFilter, OldFilmFilter, PixelateFilter, SimplexNoiseFilter,
  // distort
  BulgePinchFilter, ShockwaveFilter, TwistFilter, ReflectionFilter,
  // light
  BloomFilter, GlowFilter, GodrayFilter, SimpleLightmapFilter,
  // edge
  BevelFilter, OutlineFilter, DropShadowFilter,
  // fx
  GlitchFilter, RGBSplitFilter, ConvolutionFilter,
};
