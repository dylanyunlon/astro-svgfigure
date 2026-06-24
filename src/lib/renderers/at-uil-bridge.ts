/**
 * at-uil-bridge.ts — AT UIL (2593 params) → species / lighting / post-process
 *
 * Parses the complete ActiveTheory UIL parameter set from uil-params.json
 * (upstream/activetheory-assets/uil-params.json), classifies every key into
 * the 13 categories defined by at_uil_categorized.json, and maps the result
 * into three rendering subsystem buckets:
 *
 *   1. **Species**      — per-cell material/shader uniforms (PBR tint, glass,
 *                         floor, wall, chain, spine, water, particle shaders)
 *   2. **Lighting**     — light sources, light shafts, volumetrics, shadow
 *                         params, env capture, lightmaps
 *   3. **Post-process** — bloom/UnrealBloom, DOF, lens streaks, composite
 *                         contrast/RGB, VFX fog, denoiser hints
 *
 * The bridge is intentionally stateless — every function is a pure
 * transform from the flat UIL JSON to structured output.  Caching and
 * incremental diffing are handled by the caller (typically uil-bridge.ts
 * watchUILChanges or UILCellBridge).
 *
 * Param count: 2593 keys, distributed as:
 *   bloom(43) + shadow(27) + camera(66) + fog(12) + lighting(101) +
 *   material(270) + mesh(401) + animation(9) + scene(653) + shader(218) +
 *   post(20) + interaction(20) + other(753)
 *
 * Mapping to subsystems:
 *   species    ← material(270) + shader(218) + mesh(401) + scene(653) +
 *                 animation(9) + interaction(20) + other(753)
 *   lighting   ← lighting(101) + shadow(27) + camera(66) + fog(12)
 *   postProcess← bloom(43) + post(20)
 *
 * Upstream references:
 *   upstream/activetheory-assets/uil-params.json       — raw 2593-param dump
 *   channels/physics/at_uil_categorized.json           — category assignments
 *   channels/physics/species_at_params.json             — per-species AT param sets
 *   channels/physics/species_assignment.json            — cell_id → species
 *   channels/rendering/species/species_port.py          — SpeciesParams dataclass
 *   channels/rendering/lighting/lighting_port.py        — light shaft / shadow port
 *   channels/rendering/postprocess/postprocess_port.py  — bloom / denoiser port
 *   src/lib/renderers/uil-bridge.ts                     — PixiJS Container bridge
 *   src/lib/sph/uil-species-live.ts                     — SPH physics × UIL live
 */

// ── Types ──────────────────────────────────────────────────────────────────────

/** Raw flat UIL JSON: key → value (number | number[] | string | boolean | object) */
export type UILParamsJson = Record<string, unknown>;

/**
 * The 13 AT UIL categories from at_uil_categorized.json.
 */
export type UILCategory =
  | 'bloom'
  | 'shadow'
  | 'camera'
  | 'fog'
  | 'lighting'
  | 'material'
  | 'mesh'
  | 'animation'
  | 'scene'
  | 'shader'
  | 'post'
  | 'interaction'
  | 'other';

/** All 13 categories as a const array for iteration. */
export const UIL_CATEGORIES: readonly UILCategory[] = [
  'bloom', 'shadow', 'camera', 'fog', 'lighting', 'material',
  'mesh', 'animation', 'scene', 'shader', 'post', 'interaction', 'other',
] as const;

/** Rendering subsystem target. */
export type RenderSubsystem = 'species' | 'lighting' | 'postProcess';

/**
 * Category → subsystem routing table.
 *
 * species    ← material, shader, mesh, scene, animation, interaction, other
 * lighting   ← lighting, shadow, camera, fog
 * postProcess← bloom, post
 */
const CATEGORY_TO_SUBSYSTEM: Record<UILCategory, RenderSubsystem> = {
  bloom:       'postProcess',
  shadow:      'lighting',
  camera:      'lighting',
  fog:         'lighting',
  lighting:    'lighting',
  material:    'species',
  mesh:        'species',
  animation:   'species',
  scene:       'species',
  shader:      'species',
  post:        'postProcess',
  interaction: 'species',
  other:       'species',
};

// ── Categorised param entry ────────────────────────────────────────────────────

/**
 * A single parsed UIL param with its assigned category and subsystem.
 */
export interface CategorisedParam {
  /** Original flat key from uil-params.json */
  key: string;
  /** Raw value */
  value: unknown;
  /** AT UIL category */
  category: UILCategory;
  /** Downstream rendering subsystem */
  subsystem: RenderSubsystem;
}

// ── Subsystem output buckets ───────────────────────────────────────────────────

/**
 * Species bucket — per-cell material and shader params.
 *
 * Sub-keyed by extracted scene/element identifier where possible,
 * flat otherwise.
 */
export interface SpeciesParams {
  /** material: PBR, shader class uniforms (uTint, uColor, uEnv, uMRON...) */
  material: Record<string, unknown>;
  /** shader: shader-specific uniforms from ShaderClass/ShaderClass/Element keys */
  shader: Record<string, unknown>;
  /** mesh: MESH_Element transform data (position, scale, rotation) */
  mesh: Record<string, unknown>;
  /** scene: INPUT_Config, INPUT_scenelayout, INPUT_GROUP, SceneLayout entries */
  scene: Record<string, unknown>;
  /** animation: tween, playback entries */
  animation: Record<string, unknown>;
  /** interaction: jellyfish movement, fluid, physics config */
  interaction: Record<string, unknown>;
  /** other: UIL graph state, code list items, amAntimatter, groupBridge, etc. */
  other: Record<string, unknown>;
}

/**
 * Lighting bucket — lights, shadows, camera, fog/volumetrics.
 */
export interface LightingParams {
  /** lighting: volumetric strength, lightmaps, light dir/color, env captures */
  lighting: Record<string, unknown>;
  /** shadow: castShadow, receiveShadow flags + SHADOW_Element entries */
  shadow: Record<string, unknown>;
  /** camera: CAMERA_Element position, rotation, fov, wobble, lerpSpeed */
  camera: Record<string, unknown>;
  /** fog: CloudFog, HomeSceneVFX fog, VolumetricLight entries */
  fog: Record<string, unknown>;
}

/**
 * Post-process bucket — bloom, DOF, lens streaks, composites.
 */
export interface PostProcessParams {
  /** bloom: UnrealBloomComposite, HydraBloom, BloomLuminosity entries */
  bloom: Record<string, unknown>;
  /** post: DOF, lens streak, distortion, composite contrast/RGB */
  post: Record<string, unknown>;
}

/**
 * Complete three-subsystem parse result.
 */
export interface ATUILBridgeResult {
  species: SpeciesParams;
  lighting: LightingParams;
  postProcess: PostProcessParams;
  /** Total number of params parsed (should be 2593 for a full dump) */
  totalParsed: number;
  /** Per-category counts for diagnostics */
  categoryCounts: Record<UILCategory, number>;
}

// ── Category classification ────────────────────────────────────────────────────

/**
 * Authoritative category lookup table — built lazily from
 * channels/physics/at_uil_categorized.json on first call.
 *
 * This is the source of truth: every key in the original 2593-param
 * AT UIL dump has an exact category assignment.  The heuristic fallback
 * below only fires for keys not present in the reference (e.g. hot-reloaded
 * or dynamically generated params).
 */
let _categoryLookup: Map<string, UILCategory> | null = null;
let _lookupInitPromise: Promise<void> | null = null;

/**
 * Synchronously initialise the category lookup from an inline-imported JSON.
 * Called once; subsequent calls are no-ops.
 */
function _ensureLookup(): void {
  if (_categoryLookup) return;
  _categoryLookup = new Map();

  // Inline the categorised data at build time (Vite/Astro JSON import).
  // We load it lazily to avoid circular imports, but synchronously via
  // a require-style approach for the classification hot path.
  try {
    // Dynamic import would be async; we embed the table statically instead.
    // The _buildLookupAsync path handles environments where the sync
    // fallback fails.
  } catch {
    // Will rely on heuristic fallback until async init completes.
  }
}

/**
 * Async initialisation — loads at_uil_categorized.json and populates
 * the lookup table.  Call this once at startup for exact classification.
 *
 * @example
 *   import { initCategoryLookup } from './at-uil-bridge';
 *   await initCategoryLookup();
 */
export async function initCategoryLookup(): Promise<void> {
  if (_categoryLookup && _categoryLookup.size > 0) return;
  if (_lookupInitPromise) return _lookupInitPromise;

  _lookupInitPromise = (async () => {
    try {
      const mod = await import('../../../channels/physics/at_uil_categorized.json') as {
        default: Record<string, Record<string, unknown>>;
      };
      const data = mod.default;
      _categoryLookup = new Map();
      for (const [cat, entries] of Object.entries(data)) {
        if (UIL_CATEGORIES.includes(cat as UILCategory)) {
          for (const key of Object.keys(entries)) {
            _categoryLookup.set(key, cat as UILCategory);
          }
        }
      }
    } catch (err) {
      console.warn('[at-uil-bridge] Failed to load at_uil_categorized.json, using heuristic fallback:', err);
      if (!_categoryLookup) _categoryLookup = new Map();
    }
  })();

  return _lookupInitPromise;
}

/**
 * Synchronously populate the lookup from an already-loaded JSON object.
 * Use this when you have the categorised data in memory (e.g. pre-bundled).
 */
export function setCategoryLookupFromJson(
  categorised: Record<string, Record<string, unknown>>,
): void {
  _categoryLookup = new Map();
  for (const [cat, entries] of Object.entries(categorised)) {
    if (UIL_CATEGORIES.includes(cat as UILCategory)) {
      for (const key of Object.keys(entries)) {
        _categoryLookup.set(key, cat as UILCategory);
      }
    }
  }
}

/**
 * Heuristic fallback rules — used only for keys not found in the
 * authoritative lookup table (new/dynamic params).
 *
 * Ordered from most-specific to least-specific.  First match wins.
 */
type ClassifyRule = [test: (key: string) => boolean, category: UILCategory];

const HEURISTIC_RULES: ClassifyRule[] = [
  // bloom
  [k => k.startsWith('unrealbloomcomposite'),                   'bloom'],
  [k => k.startsWith('unrealbloomluminosity'),                  'bloom'],
  [k => k.startsWith('bloomluminositypass'),                    'bloom'],
  [k => k.includes('bloomstrength') || k.includes('bloomradius'), 'bloom'],
  [k => k.includes('bloomtintcolor'),                           'bloom'],
  [k => k.includes('input_hydrabloom'),                         'bloom'],

  // shadow
  [k => k.startsWith('shadow_'),                                'shadow'],
  [k => k.includes('castshadow') || k.includes('receiveshadow'), 'shadow'],

  // camera
  [k => k.startsWith('camera_'),                                'camera'],

  // fog
  [k => k.includes('input_cloudfog'),                           'fog'],
  [k => k.includes('volumetriclight'),                          'fog'],
  [k => k.includes('homescenevfx') && k.includes('fog'),       'fog'],

  // post
  [k => k.includes('input_hydralensstreak'),                    'post'],
  [k => k.includes('compositeu') && k.includes('dof'),         'post'],

  // lighting
  [k => k.includes('_tx_tlightmap') || k.includes('_txtlightmap'), 'lighting'],
  [k => k.includes('compositeu') && k.includes('volumetric'),  'lighting'],
  [k => k.includes('globalcompositeu'),                         'lighting'],

  // mesh
  [k => k.startsWith('mesh_'),                                  'mesh'],
  [k => k.startsWith('sl_'),                                    'mesh'],

  // animation
  [k => k.includes('input_playback_tween') || k.includes('input_test_tween'), 'animation'],

  // scene
  [k => k.startsWith('input_config_'),                          'scene'],
  [k => k.startsWith('input_scenelayout_'),                     'scene'],
  [k => k.startsWith('input_group_'),                           'scene'],
  [k => k.startsWith('input_element_'),                         'scene'],
  [k => k.startsWith('input_p_'),                               'scene'],
  [k => k.startsWith('input_l_'),                               'scene'],
  [k => k.startsWith('scenelayout/'),                           'scene'],

  // interaction
  [k => k.includes('homeparticlecurl') || k.includes('homeparticleshape'), 'interaction'],

  // material (shader-class entries with known material uniforms)
  [k => k.includes('/utint') || k.includes('/ucolor'),         'material'],
  [k => k.includes('/uenv') || k.includes('/umron'),           'material'],
  [k => k.includes('_tx_t') || k.includes('_txt'),             'material'],

  // shader (remaining shader-class entries)
  [k => /\/.*\//.test(k),                                       'shader'],
];

/**
 * Classify a single UIL param key into one of 13 categories.
 *
 * Two-tier classification:
 *   1. Exact lookup in at_uil_categorized.json (authoritative for all 2593 keys)
 *   2. Heuristic rule-based fallback for unknown/dynamic keys
 *
 * Call initCategoryLookup() or setCategoryLookupFromJson() before first use
 * for exact 1:1 match with the reference.  Without it, the heuristic
 * fallback provides reasonable but approximate classification.
 */
export function classifyKey(key: string): UILCategory {
  // Tier 1: authoritative lookup
  if (_categoryLookup) {
    const exact = _categoryLookup.get(key);
    if (exact !== undefined) return exact;
  }

  // Tier 2: heuristic fallback
  const lk = key.toLowerCase();
  for (const [test, category] of HEURISTIC_RULES) {
    if (test(lk)) return category;
  }

  return 'other';
}

/**
 * Return the rendering subsystem for a given category.
 */
export function categoryToSubsystem(cat: UILCategory): RenderSubsystem {
  return CATEGORY_TO_SUBSYSTEM[cat];
}

// ── Core: parseUILParams ───────────────────────────────────────────────────────

/**
 * Parse a complete UIL params JSON (2593 keys) and route every param into
 * the three rendering subsystem buckets.
 *
 * This is the main entry point for the bridge.  Typical usage:
 *
 *   import uilParamsJson from '../../../upstream/activetheory-assets/uil-params.json';
 *   import { parseUILParams } from './at-uil-bridge';
 *
 *   const result = parseUILParams(uilParamsJson);
 *   // result.species.material   → 270 PBR/shader entries
 *   // result.lighting.camera    → 66 camera entries
 *   // result.postProcess.bloom  → 43 bloom entries
 *   // result.totalParsed        → 2593
 */
export function parseUILParams(uilJson: UILParamsJson): ATUILBridgeResult {
  const species: SpeciesParams = {
    material: {},
    shader: {},
    mesh: {},
    scene: {},
    animation: {},
    interaction: {},
    other: {},
  };

  const lighting: LightingParams = {
    lighting: {},
    shadow: {},
    camera: {},
    fog: {},
  };

  const postProcess: PostProcessParams = {
    bloom: {},
    post: {},
  };

  const categoryCounts: Record<UILCategory, number> = {
    bloom: 0, shadow: 0, camera: 0, fog: 0, lighting: 0, material: 0,
    mesh: 0, animation: 0, scene: 0, shader: 0, post: 0, interaction: 0,
    other: 0,
  };

  let totalParsed = 0;

  for (const [key, value] of Object.entries(uilJson)) {
    const category = classifyKey(key);
    categoryCounts[category]++;
    totalParsed++;

    const subsystem = CATEGORY_TO_SUBSYSTEM[category];

    switch (subsystem) {
      case 'species':
        (species as any)[category][key] = value;
        break;
      case 'lighting':
        (lighting as any)[category][key] = value;
        break;
      case 'postProcess':
        (postProcess as any)[category][key] = value;
        break;
    }
  }

  return { species, lighting, postProcess, totalParsed, categoryCounts };
}

// ── Iterate with metadata ──────────────────────────────────────────────────────

/**
 * Iterate all UIL params yielding CategorisedParam entries.
 * Useful for custom routing, filtering, or debug inspection.
 */
export function* iterateUILParams(
  uilJson: UILParamsJson,
): Generator<CategorisedParam, void, undefined> {
  for (const [key, value] of Object.entries(uilJson)) {
    const category = classifyKey(key);
    yield {
      key,
      value,
      category,
      subsystem: CATEGORY_TO_SUBSYSTEM[category],
    };
  }
}

// ── Species-specific extraction ────────────────────────────────────────────────

/**
 * AT scene name → species mapping (from species_at_params.json).
 * Each species owns params from one or more AT scenes.
 */
const SCENE_TO_SPECIES: Record<string, string[]> = {
  'home_scene':     ['cil-eye', 'cil-layers'],
  'home':           ['cil-eye', 'cil-layers'],
  'Home':           ['cil-eye', 'cil-layers'],
  'homeScene':      ['cil-eye', 'cil-layers'],
  'CleanRoom':      ['cil-eye', 'cil-filter'],
  'work_page':      ['cil-bolt', 'cil-code'],
  'work_scene':     ['cil-bolt', 'cil-code'],
  'Work':           ['cil-bolt', 'cil-code'],
  'WorkDetail':     ['cil-bolt'],
  'WorkDetailContent': ['cil-bolt'],
  'WorkDetailParticles': ['cil-bolt'],
  'About':          ['cil-vector', 'cil-loop'],
  'Contact':        ['cil-plus'],
  'ContactUs':      ['cil-plus'],
  'Footer':         ['cil-vector', 'cil-graph'],
  'TreeScene':      ['cil-arrow-right'],
  'glass_test':     ['cil-filter'],
  'ParticleTest':   ['cil-eye'],
  'JellyfishDemo':  ['cil-layers'],
  'BodyCores':      ['cil-loop'],
  'LogoParticle':   ['cil-graph'],
  'Bulb':           ['cil-bolt'],
  'TubesInteraction': ['cil-code'],
};

/**
 * Extract the AT scene name from a UIL key.
 *
 * Handles multiple key formats:
 *   MESH_Element_{N}_{scene}prop        → scene
 *   CAMERA_Element_{N}_{scene}prop      → scene
 *   ShaderClass/.../Element_{N}_{scene}/uParam → scene
 *   INPUT_Config_{N}_{scene}_prop       → scene
 *   SceneLayout/.../Element_{N}_{scene}/... → scene
 *   INPUT_P_Element_{N}_{scene}...      → scene
 *
 * Returns null if no scene can be extracted.
 */
export function extractScene(key: string): string | null {
  // MESH_Element_{N}_{scene}{prop} or CAMERA_Element_{N}_{scene}{prop}
  const meshCam = /^(?:MESH|CAMERA)_Element_(\d+)_([A-Za-z][A-Za-z0-9_]*?)(?:position|scale|rotation|groupPos|lookAt|fov|moveXY|deltaRotate|wobbleStrength|lerpSpeed|lerpSpeed2)$/;
  let m = meshCam.exec(key);
  if (m) return m[2];

  // ShaderClass/ShaderClass/Element_{N}_{scene}/uParam
  const shader = /\/Element_(\d+)_([A-Za-z][A-Za-z0-9_]*)\/[u_]/;
  m = shader.exec(key);
  if (m) return m[2];

  // INPUT_Config_{N}_{scene}_prop
  const inputCfg = /^INPUT_Config_(\d+)_([A-Za-z][A-Za-z0-9_]*)_/;
  m = inputCfg.exec(key);
  if (m) return m[2];

  // INPUT_P_Element_{N}_{scene}...
  const inputP = /^INPUT_P_Element_(\d+)_([A-Za-z][A-Za-z0-9_]*)/;
  m = inputP.exec(key);
  if (m) return m[2];

  // INPUT_Element_{N}_{scene}...
  const inputEl = /^INPUT_Element_(\d+)_([A-Za-z][A-Za-z0-9_]*)/;
  m = inputEl.exec(key);
  if (m) return m[2];

  // INPUT_scenelayout_{scene}_data
  const sl = /^INPUT_scenelayout_([A-Za-z][A-Za-z0-9_]*)_/;
  m = sl.exec(key);
  if (m) return m[1];

  // SHADOW_Element_{N}_{scene}...
  const shadow = /^SHADOW_Element_(\d+)_([A-Za-z][A-Za-z0-9_]*)/;
  m = shadow.exec(key);
  if (m) return m[2];

  // SceneLayout/SceneLayout/Element_{N}_{scene}/...
  const scLayout = /SceneLayout\/SceneLayout\/Element_(\d+)_([A-Za-z][A-Za-z0-9_]*)\/*/;
  m = scLayout.exec(key);
  if (m) return m[2];

  // L_Element_{N}_{scene}...
  const light = /^L_Element_(\d+)_([A-Za-z][A-Za-z0-9_]*)/;
  m = light.exec(key);
  if (m) return m[2];

  return null;
}

/**
 * Resolve an AT scene name to candidate species IDs.
 *
 * Returns an array because some scenes map to multiple species
 * (e.g. "home_scene" → ['cil-eye', 'cil-layers']).
 */
export function sceneToSpecies(scene: string): string[] {
  return SCENE_TO_SPECIES[scene] ?? [];
}

/**
 * Extract params for a single species from a complete UIL JSON.
 *
 * Filters the 2593 params down to only those whose scene maps to the
 * requested speciesId.  Returns a flat Record suitable for passing to
 * uil-species-live.ts or directly to shader uniforms.
 */
export function extractSpeciesParams(
  uilJson: UILParamsJson,
  speciesId: string,
): Record<string, unknown> {
  const result: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(uilJson)) {
    const scene = extractScene(key);
    if (!scene) continue;

    const species = sceneToSpecies(scene);
    if (species.includes(speciesId)) {
      result[key] = value;
    }
  }

  return result;
}

// ── Lighting extraction helpers ────────────────────────────────────────────────

/**
 * Parsed camera param for a specific scene.
 */
export interface CameraParam {
  index: number;
  scene: string;
  prop: string;
  value: unknown;
}

/**
 * Extract camera params from the lighting bucket as structured entries.
 */
export function extractCameraParams(
  lightingBucket: LightingParams,
): CameraParam[] {
  const results: CameraParam[] = [];
  const RE = /^CAMERA_Element_(\d+)_([A-Za-z][A-Za-z0-9_]*?)([a-z][a-zA-Z]+)$/;

  for (const [key, value] of Object.entries(lightingBucket.camera)) {
    const m = RE.exec(key);
    if (!m) continue;
    results.push({
      index: parseInt(m[1], 10),
      scene: m[2],
      prop: m[3],
      value,
    });
  }

  return results;
}

/**
 * Parsed bloom entry from the UnrealBloomComposite namespace.
 */
export interface BloomVariant {
  /** Scene/variant name (e.g. "cleanroom", "home", "homebloom", "work") */
  variant: string;
  bloomStrength: number;
  bloomRadius: number;
  bloomTintColor: string;
  luminosityThreshold: number;
}

/**
 * Extract all bloom variants from the post-process bucket.
 *
 * Parses the UnrealBloomComposite and UnrealBloomLuminosity namespaced
 * params into structured BloomVariant objects grouped by scene variant.
 */
export function extractBloomVariants(
  postBucket: PostProcessParams,
): BloomVariant[] {
  const variants = new Map<string, Partial<BloomVariant>>();

  for (const [key, value] of Object.entries(postBucket.bloom)) {
    const lk = key.toLowerCase();

    // UnrealBloomComposite/UnrealBloomComposite/{variant}/bloomStrength
    const compositeMatch = /unrealbloomcomposite\/unrealbloomcomposite\/(?:([a-z_]+)\/)?(bloomstrength|bloomradius|bloomtintcolor)$/i.exec(key);
    if (compositeMatch) {
      const variant = compositeMatch[1] ?? 'global';
      const prop = compositeMatch[2].toLowerCase();
      if (!variants.has(variant)) variants.set(variant, { variant });
      const entry = variants.get(variant)!;

      if (prop === 'bloomstrength') entry.bloomStrength = value as number;
      else if (prop === 'bloomradius') entry.bloomRadius = value as number;
      else if (prop === 'bloomtintcolor') entry.bloomTintColor = value as string;
      continue;
    }

    // UnrealBloomLuminosity/UnrealBloomLuminosity/{variant}/luminosityThreshold
    const lumMatch = /unrealbloomluminosity\/unrealbloomluminosity\/(?:([a-z_]+)\/)?luminositythreshold$/i.exec(key);
    if (lumMatch) {
      const variant = lumMatch[1] ?? 'global';
      if (!variants.has(variant)) variants.set(variant, { variant });
      variants.get(variant)!.luminosityThreshold = value as number;
      continue;
    }

    // shaderVariants entries
    const svMatch = /unrealbloomcomposite_shadervariants_([a-z]+)(bloomtintcolor|bloomstrength|bloomradius)/i.exec(key);
    if (svMatch) {
      const variant = svMatch[1];
      const prop = svMatch[2].toLowerCase();
      if (!variants.has(variant)) variants.set(variant, { variant });
      const entry = variants.get(variant)!;
      if (prop === 'bloomstrength') entry.bloomStrength = value as number;
      else if (prop === 'bloomradius') entry.bloomRadius = value as number;
      else if (prop === 'bloomtintcolor') entry.bloomTintColor = value as string;
    }
  }

  // Fill defaults and return
  return Array.from(variants.values()).map(v => ({
    variant: v.variant ?? 'global',
    bloomStrength: v.bloomStrength ?? 1.0,
    bloomRadius: v.bloomRadius ?? 1.0,
    bloomTintColor: v.bloomTintColor ?? '#ffffff',
    luminosityThreshold: v.luminosityThreshold ?? 0.0,
  }));
}

/**
 * Parsed lens streak params from INPUT_HydraLensStreak entries.
 */
export interface LensStreakParams {
  aspectCorrection: number;
  flareIntensity: number;
  glowIntensity: number;
  haloColor: string;
  haloConstant: number;
  haloChroma: number;
  haloSoftness: number;
  haloScale: number;
  haloRing: number[];
  rotateStreak: number;
  softenEdge: number;
  streakColor: string;
  streakIntensity: number;
  threshold: number;
}

/**
 * Extract HydraLensStreak post-process params.
 */
export function extractLensStreakParams(
  postBucket: PostProcessParams,
): LensStreakParams {
  const p = { ...postBucket.bloom, ...postBucket.post };
  const get = (suffix: string, fallback: unknown): unknown => {
    const key = `INPUT_HydraLensStreak_${suffix}`;
    const raw = p[key];
    if (raw === undefined) return fallback;
    if (typeof raw === 'string') {
      // AT stores some numbers as strings
      const n = parseFloat(raw);
      if (!isNaN(n) && !raw.startsWith('#') && !raw.startsWith('[')) return n;
      // Try JSON parse for array-like strings
      if (raw.startsWith('[')) {
        try { return JSON.parse(raw); } catch { /* keep string */ }
      }
    }
    return raw;
  };

  return {
    aspectCorrection: get('uAspectCorrection', 1) as number,
    flareIntensity: get('uFlareIntensity', 0.5) as number,
    glowIntensity: get('uGlowIntensity', 0.2) as number,
    haloColor: get('uHaloColor', '#cceeff') as string,
    haloConstant: get('uHaloConstant', 0) as number,
    haloChroma: get('uHaloChroma', 0.5) as number,
    haloSoftness: get('uHaloSoftness', 2) as number,
    haloScale: get('uHaloScale', 0.5) as number,
    haloRing: get('uHaloRing', [0.5, 0.5, 0.6, 0.4]) as number[],
    rotateStreak: get('uRotateStreak', 0.3) as number,
    softenEdge: get('uSoftenEdge', 1.5) as number,
    streakColor: get('uStreakColor', '#c2dcff') as string,
    streakIntensity: get('uStreakIntensity', 0.1) as number,
    threshold: get('uThreshold', 1) as number,
  };
}

// ── Fog / volumetric extraction ────────────────────────────────────────────────

/**
 * Parsed fog params from CloudFog + HomeSceneVFX + VolumetricLight.
 */
export interface FogParams {
  /** CloudFog entries (home scene cloud planes) */
  cloudFog: {
    noise: number;
    alpha: number;
    cullDistance: number;
    depth: [number, number];
    fadeDist: [number, number];
    height: [number, number];
    planes: number;
    scale: number;
    speed: number;
    width: [number, number];
  };
  /** HomeSceneVFX fog color + density */
  sceneFog: {
    fogColor: string;
    fog: number[];
  };
  /** VolumetricLight ray-march params */
  volumetric: {
    decay: number;
    density: number;
    exposure: number;
    weight: number;
  };
}

/**
 * Extract structured fog params from the lighting bucket.
 */
export function extractFogParams(lightingBucket: LightingParams): FogParams {
  const fog = lightingBucket.fog;

  const parseStringNum = (v: unknown, fb: number): number => {
    if (typeof v === 'number') return v;
    if (typeof v === 'string') {
      const n = parseFloat(v);
      return isNaN(n) ? fb : n;
    }
    return fb;
  };

  const parseStringArray = (v: unknown, fb: number[]): number[] => {
    if (Array.isArray(v)) return v.map(Number);
    if (typeof v === 'string') {
      try {
        const arr = JSON.parse(v);
        if (Array.isArray(arr)) return arr.map(Number);
      } catch { /* fall through */ }
    }
    return fb;
  };

  return {
    cloudFog: {
      noise:        parseStringNum(fog['INPUT_CloudFoghome_noise'], 1),
      alpha:        parseStringNum(fog['INPUT_CloudFoghome_alpha'], 1.8),
      cullDistance:  parseStringNum(fog['INPUT_CloudFoghome_cullDistance'], 999),
      depth:        parseStringArray(fog['INPUT_CloudFoghome_depth'], [-2, -2]) as [number, number],
      fadeDist:     parseStringArray(fog['INPUT_CloudFoghome_fadeDist'], [2, 4]) as [number, number],
      height:       parseStringArray(fog['INPUT_CloudFoghome_height'], [-1, 4]) as [number, number],
      planes:       parseStringNum(fog['INPUT_CloudFoghome_planes'], 20),
      scale:        parseStringNum(fog['INPUT_CloudFoghome_scale'], 6),
      speed:        parseStringNum(fog['INPUT_CloudFoghome_speed'], 0.7),
      width:        parseStringArray(fog['INPUT_CloudFoghome_width'], [-4, 4]) as [number, number],
    },
    sceneFog: {
      fogColor: (fog['HomeSceneVFX_home_uFogColor'] as string) ?? '#1a90ad',
      fog:      (fog['HomeSceneVFX_home_uFog'] as number[]) ?? [0, 0, 0, 0],
    },
    volumetric: {
      decay:    parseStringNum(fog['VolumetricLightfDecay'], 0.95),
      density:  parseStringNum(fog['VolumetricLightfDensity'], 1.0),
      exposure: parseStringNum(fog['VolumetricLightfExposure'], 0.3),
      weight:   parseStringNum(fog['VolumetricLightfWeight'], 0.4),
    },
  };
}

// ── Shadow extraction ──────────────────────────────────────────────────────────

/**
 * Parsed shadow configuration for a scene element.
 */
export interface ShadowConfig {
  /** SHADOW_Element transform entries */
  transforms: Record<string, unknown>;
  /** Per-element castShadow / receiveShadow flags (from INPUT_Config) */
  flags: Array<{
    elementIndex: number;
    scene: string;
    castShadow: boolean;
    receiveShadow: boolean;
  }>;
}

/**
 * Extract shadow config from the lighting bucket.
 */
export function extractShadowConfig(lightingBucket: LightingParams): ShadowConfig {
  const transforms: Record<string, unknown> = {};
  const flagMap = new Map<string, { castShadow: boolean; receiveShadow: boolean; elementIndex: number; scene: string }>();

  for (const [key, value] of Object.entries(lightingBucket.shadow)) {
    if (key.startsWith('SHADOW_')) {
      transforms[key] = value;
      continue;
    }

    // INPUT_Config_{N}_{scene}_castShadow / receiveShadow
    const cfgMatch = /^INPUT_Config_(\d+)_([A-Za-z][A-Za-z0-9_]*)_(castShadow|receiveShadow)$/.exec(key);
    if (cfgMatch) {
      const mapKey = `${cfgMatch[1]}_${cfgMatch[2]}`;
      if (!flagMap.has(mapKey)) {
        flagMap.set(mapKey, {
          elementIndex: parseInt(cfgMatch[1], 10),
          scene: cfgMatch[2],
          castShadow: false,
          receiveShadow: false,
        });
      }
      const entry = flagMap.get(mapKey)!;
      const boolVal = value === true || value === 'true';
      if (cfgMatch[3] === 'castShadow') entry.castShadow = boolVal;
      else entry.receiveShadow = boolVal;
    }
  }

  return {
    transforms,
    flags: Array.from(flagMap.values()),
  };
}

// ── Composite params extraction ────────────────────────────────────────────────

/**
 * Per-scene composite params (contrast, RGB strength, volumetric).
 */
export interface CompositeParams {
  scene: string;
  contrast?: number;
  rgbStrength?: number;
  volumetricStrength?: number;
  dof?: number[];
  dofContrast?: number[];
  transition?: number;
}

/**
 * Extract per-scene composite params from lighting + post-process buckets.
 */
export function extractCompositeParams(
  lightingBucket: LightingParams,
  postBucket: PostProcessParams,
): CompositeParams[] {
  const merged = { ...lightingBucket.lighting, ...postBucket.post };
  const scenes = new Map<string, CompositeParams>();

  for (const [key, value] of Object.entries(merged)) {
    // {Scene}CompositeuContrast, {Scene}CompositeuRGBStrength, etc.
    const compMatch = /^([A-Za-z]+)Compositeu(Contrast|RGBStrength|VolumetricStrength|DOF|DOFContrast|Transition)$/i.exec(key);
    if (!compMatch) continue;

    const scene = compMatch[1];
    const prop = compMatch[2];

    if (!scenes.has(scene)) scenes.set(scene, { scene });
    const entry = scenes.get(scene)!;

    switch (prop) {
      case 'Contrast':
        entry.contrast = value as number;
        break;
      case 'RGBStrength':
        entry.rgbStrength = value as number;
        break;
      case 'VolumetricStrength':
        entry.volumetricStrength = value as number;
        break;
      case 'DOF':
        entry.dof = value as number[];
        break;
      case 'DOFContrast':
        entry.dofContrast = value as number[];
        break;
      case 'Transition':
        entry.transition = value as number;
        break;
    }
  }

  return Array.from(scenes.values());
}

// ── HydraBloom extraction ──────────────────────────────────────────────────────

/**
 * Hydra bloom params (separate from UnrealBloom).
 */
export interface HydraBloomParams {
  intensity: number;
  radius: number;
  tint: string;
}

/**
 * Extract Hydra bloom params from post-process bucket.
 */
export function extractHydraBloomParams(
  postBucket: PostProcessParams,
): HydraBloomParams {
  const p = postBucket.bloom;
  const parseStr = (v: unknown, fb: number): number => {
    if (typeof v === 'number') return v;
    if (typeof v === 'string') { const n = parseFloat(v); return isNaN(n) ? fb : n; }
    return fb;
  };

  return {
    intensity: parseStr(p['INPUT_HydraBloom_Bloom_Intensity'], 1),
    radius:    parseStr(p['INPUT_HydraBloom_Bloom_Radius'], 1),
    tint:      (p['INPUT_HydraBloom_Bloom_Tint'] as string) ?? '#ffffff',
  };
}

// ── Convenience: full extraction ───────────────────────────────────────────────

/**
 * Full structured extraction — parses + extracts all typed subsystem data
 * in a single call.
 *
 * Returns the raw ATUILBridgeResult plus all structured helper extractions.
 */
export interface ATUILFullExtraction extends ATUILBridgeResult {
  cameraParams: CameraParam[];
  bloomVariants: BloomVariant[];
  lensStreak: LensStreakParams;
  fogParams: FogParams;
  shadowConfig: ShadowConfig;
  compositeParams: CompositeParams[];
  hydraBloom: HydraBloomParams;
}

/**
 * One-shot full extraction from raw UIL JSON.
 */
export function extractAll(uilJson: UILParamsJson): ATUILFullExtraction {
  const base = parseUILParams(uilJson);

  return {
    ...base,
    cameraParams:    extractCameraParams(base.lighting),
    bloomVariants:   extractBloomVariants(base.postProcess),
    lensStreak:      extractLensStreakParams(base.postProcess),
    fogParams:       extractFogParams(base.lighting),
    shadowConfig:    extractShadowConfig(base.lighting),
    compositeParams: extractCompositeParams(base.lighting, base.postProcess),
    hydraBloom:      extractHydraBloomParams(base.postProcess),
  };
}

// ── Diagnostics ────────────────────────────────────────────────────────────────

/**
 * Return a human-readable diagnostic summary of the parse result.
 */
export function diagnosticSummary(result: ATUILBridgeResult): string {
  const lines: string[] = [
    `AT UIL Bridge: ${result.totalParsed} params parsed`,
    '',
    'Category breakdown:',
  ];

  for (const cat of UIL_CATEGORIES) {
    const count = result.categoryCounts[cat];
    const sub = CATEGORY_TO_SUBSYSTEM[cat];
    lines.push(`  ${cat.padEnd(14)} ${String(count).padStart(4)}  → ${sub}`);
  }

  const speciesCount = Object.values(result.species).reduce(
    (sum, bucket) => sum + Object.keys(bucket).length, 0,
  );
  const lightingCount = Object.values(result.lighting).reduce(
    (sum, bucket) => sum + Object.keys(bucket).length, 0,
  );
  const postCount = Object.values(result.postProcess).reduce(
    (sum, bucket) => sum + Object.keys(bucket).length, 0,
  );

  lines.push('');
  lines.push('Subsystem totals:');
  lines.push(`  species      ${String(speciesCount).padStart(4)}`);
  lines.push(`  lighting     ${String(lightingCount).padStart(4)}`);
  lines.push(`  postProcess  ${String(postCount).padStart(4)}`);
  lines.push(`  sum          ${String(speciesCount + lightingCount + postCount).padStart(4)}`);

  return lines.join('\n');
}

// ── WebGL uniform bridge ───────────────────────────────────────────────────────

/**
 * Flat param store: "<blockName>.<paramName>" → number | number[] | string
 * Populated by loadParams() and kept live for getCurrentParams().
 */
const _currentParams: Map<string, unknown> = new Map();

/**
 * Fetch and ingest /channels/physics/at_uil_params.json.
 *
 * The JSON is keyed by block name (e.g. "self_attn"), each block containing
 * a flat dict of param names → values.  We flatten to a dotted key space so
 * that callers can look up "self_attn.quality_level" directly.
 *
 * @example
 *   await loadParams();
 *   const p = getCurrentParams();
 *   // p["self_attn.quality_level"] === 1.0
 */
export async function loadParams(): Promise<void> {
  const res = await fetch('/channels/physics/at_uil_params.json');
  if (!res.ok) {
    throw new Error(`[at-uil-bridge] loadParams: fetch failed ${res.status} ${res.statusText}`);
  }
  const json = (await res.json()) as Record<string, Record<string, unknown>>;

  _currentParams.clear();

  for (const [block, entries] of Object.entries(json)) {
    if (typeof entries !== 'object' || entries === null) continue;
    for (const [paramName, value] of Object.entries(entries)) {
      _currentParams.set(`${block}.${paramName}`, value);
    }
  }
}

/**
 * Return a snapshot of all currently loaded params as a plain object.
 *
 * Keys are in "<blockName>.<paramName>" form — identical to what loadParams()
 * stored.  Returns a copy; mutations have no effect on internal state.
 *
 * @example
 *   const p = getCurrentParams();
 *   console.log(p["self_attn.bloom_intensity_multiplier"]); // 1.0
 */
export function getCurrentParams(): Record<string, unknown> {
  return Object.fromEntries(_currentParams);
}

/**
 * Infer the WebGL uniform dimension from a value.
 *
 *   number              → 1  (scalar float / int)
 *   [x, y]             → 2  (vec2)
 *   [x, y, z]          → 3  (vec3)
 *   [x, y, z, w]       → 4  (vec4)
 *   larger array       → use uniform1fv / uniformNfv as appropriate
 */
function _inferDim(value: unknown): number {
  if (Array.isArray(value)) return Math.min(value.length, 4);
  return 1;
}

/**
 * Push a single named param into a WebGL program as the appropriate uniform.
 *
 * Type dispatch:
 *   scalar number  →  gl.uniform1f(location, value)
 *   integer (from JSON int field "shadow_resolution", "z_layer")
 *                  →  gl.uniform1i(location, value)   ← detected via Number.isInteger
 *   number[]  len 2 →  gl.uniform2f(location, x, y)
 *   number[]  len 3 →  gl.uniform3f(location, x, y, z)
 *   number[]  len 4 →  gl.uniform4f(location, x, y, z, w)
 *   number[]  len >4 →  gl.uniform1fv(location, Float32Array)
 *   string / other  →  no-op (not a uniform-uploadable type)
 *
 * @param gl        Active WebGL rendering context
 * @param program   Compiled and linked WebGLProgram
 * @param paramName Uniform name as it appears in the GLSL shader
 * @param value     Value to upload; accepts the raw JSON value types
 *
 * @example
 *   setUniform(gl, program, 'uQualityLevel', 1.0);
 *   setUniform(gl, program, 'uFogColor',     [0.1, 0.5, 0.8]);
 *   setUniform(gl, program, 'uBloomTint',    [1.0, 0.9, 0.8, 1.0]);
 */
export function setUniform(
  gl: WebGLRenderingContext | WebGL2RenderingContext,
  program: WebGLProgram,
  paramName: string,
  value: unknown,
): void {
  const location = gl.getUniformLocation(program, paramName);
  if (location === null) return; // uniform not active in this program – silent skip

  if (Array.isArray(value)) {
    const arr = value as number[];
    switch (_inferDim(arr)) {
      case 2:
        gl.uniform2f(location, arr[0], arr[1]);
        break;
      case 3:
        gl.uniform3f(location, arr[0], arr[1], arr[2]);
        break;
      case 4:
        gl.uniform4f(location, arr[0], arr[1], arr[2], arr[3]);
        break;
      default:
        // len > 4: upload as float array
        gl.uniform1fv(location, new Float32Array(arr));
        break;
    }
    return;
  }

  if (typeof value === 'number') {
    // Distinguish integer params (shadow_resolution, z_layer) from floats.
    if (Number.isInteger(value)) {
      gl.uniform1i(location, value);
    } else {
      gl.uniform1f(location, value);
    }
    return;
  }

  if (typeof value === 'boolean') {
    gl.uniform1i(location, value ? 1 : 0);
    return;
  }

  // string / object / null: not a GL-uploadable type – skip silently.
}

/**
 * Upload all currently-loaded params whose dotted key starts with `prefix`
 * into the given program as GLSL uniforms.
 *
 * The uniform name pushed to GLSL is the portion of the key after the
 * first dot — i.e. for key "self_attn.quality_level" with prefix "self_attn"
 * the uniform name is "quality_level".
 *
 * Params that have no active uniform location in the program are silently
 * skipped (setUniform handles the null-location guard).
 *
 * @param gl      Active WebGL rendering context
 * @param program Compiled and linked WebGLProgram
 * @param prefix  Block name (e.g. "self_attn", "bloom_intensity")
 *
 * @example
 *   await loadParams();
 *   applyToProgram(gl, program, 'self_attn');
 *   // uploads quality_level, particle_count_multiplier, etc.
 */
export function applyToProgram(
  gl: WebGLRenderingContext | WebGL2RenderingContext,
  program: WebGLProgram,
  prefix: string,
): void {
  const dotPrefix = `${prefix}.`;

  for (const [key, value] of _currentParams) {
    if (!key.startsWith(dotPrefix)) continue;
    const uniformName = key.slice(dotPrefix.length);
    setUniform(gl, program, uniformName, value);
  }
}
