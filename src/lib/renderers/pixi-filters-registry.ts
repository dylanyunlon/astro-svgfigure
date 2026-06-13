/**
 * pixi-filters-registry.ts
 * Auto-registered PixiJS filter registry sourced from upstream/pixijs-filters-v2.
 * Provides FILTER_REGISTRY (name → constructor) and createFilter factory.
 */

// ─── blur ────────────────────────────────────────────────────────────────────
import { KawaseBlurFilter }    from '../../upstream/pixijs-filters-v2/src/kawase-blur';
import { BackdropBlurFilter }  from '../../upstream/pixijs-filters-v2/src/backdrop-blur';
import { RadialBlurFilter }    from '../../upstream/pixijs-filters-v2/src/radial-blur';
import { ZoomBlurFilter }      from '../../upstream/pixijs-filters-v2/src/zoom-blur';
import { TiltShiftFilter }     from '../../upstream/pixijs-filters-v2/src/tilt-shift';
import { MotionBlurFilter }    from '../../upstream/pixijs-filters-v2/src/motion-blur';

// ─── color ───────────────────────────────────────────────────────────────────
import { AdjustmentFilter }       from '../../upstream/pixijs-filters-v2/src/adjustment';
import { ColorGradientFilter }    from '../../upstream/pixijs-filters-v2/src/color-gradient';
import { ColorMapFilter }         from '../../upstream/pixijs-filters-v2/src/color-map';
import { ColorOverlayFilter }     from '../../upstream/pixijs-filters-v2/src/color-overlay';
import { ColorReplaceFilter }     from '../../upstream/pixijs-filters-v2/src/color-replace';
import { GrayscaleFilter }        from '../../upstream/pixijs-filters-v2/src/grayscale';
import { HslAdjustmentFilter }    from '../../upstream/pixijs-filters-v2/src/hsl-adjustment';
import { MultiColorReplaceFilter }from '../../upstream/pixijs-filters-v2/src/multi-color-replace';

// ─── style ───────────────────────────────────────────────────────────────────
import { AsciiFilter }       from '../../upstream/pixijs-filters-v2/src/ascii';
import { CrossHatchFilter }  from '../../upstream/pixijs-filters-v2/src/cross-hatch';
import { CRTFilter }         from '../../upstream/pixijs-filters-v2/src/crt';
import { DotFilter }         from '../../upstream/pixijs-filters-v2/src/dot';
import { EmbossFilter }      from '../../upstream/pixijs-filters-v2/src/emboss';
import { OldFilmFilter }     from '../../upstream/pixijs-filters-v2/src/old-film';
import { PixelateFilter }    from '../../upstream/pixijs-filters-v2/src/pixelate';
import { SimplexNoiseFilter }from '../../upstream/pixijs-filters-v2/src/simplex-noise';

// ─── distort ─────────────────────────────────────────────────────────────────
import { BulgePinchFilter } from '../../upstream/pixijs-filters-v2/src/bulge-pinch';
import { ShockwaveFilter }  from '../../upstream/pixijs-filters-v2/src/shockwave';
import { TwistFilter }      from '../../upstream/pixijs-filters-v2/src/twist';
import { ReflectionFilter } from '../../upstream/pixijs-filters-v2/src/reflection';

// ─── light ───────────────────────────────────────────────────────────────────
import { BloomFilter }          from '../../upstream/pixijs-filters-v2/src/bloom';
import { GlowFilter }           from '../../upstream/pixijs-filters-v2/src/glow';
import { GodrayFilter }         from '../../upstream/pixijs-filters-v2/src/godray';
import { SimpleLightmapFilter } from '../../upstream/pixijs-filters-v2/src/simple-lightmap';

// ─── edge ────────────────────────────────────────────────────────────────────
import { BevelFilter }      from '../../upstream/pixijs-filters-v2/src/bevel';
import { OutlineFilter }    from '../../upstream/pixijs-filters-v2/src/outline';
import { DropShadowFilter } from '../../upstream/pixijs-filters-v2/src/drop-shadow';

// ─── fx ──────────────────────────────────────────────────────────────────────
import { GlitchFilter }      from '../../upstream/pixijs-filters-v2/src/glitch';
import { RGBSplitFilter }    from '../../upstream/pixijs-filters-v2/src/rgb-split';
import { ConvolutionFilter } from '../../upstream/pixijs-filters-v2/src/convolution';

// ─────────────────────────────────────────────────────────────────────────────

export type FilterName =
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

// ─── Re-export all constructors for direct use ──────────────────────────────
export {
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
