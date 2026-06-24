/**
 * at-uil-live-panel.ts — M828: AT UIL Live Panel
 * ─────────────────────────────────────────────────────────────────────────────
 * Realtime HTML control panel that exposes the full AT UIL parameter set
 * (upstream/activetheory-assets/uil-params.json, 2593 entries) for live
 * inspection and tweaking during development.
 *
 * Architecture
 * ─────────────────────────────────────────────────────────────────────────────
 *  1. Load uil-params.json and classify every entry by prefix into named
 *     sections:  CAMERA · POST_PROCESS · VOLUMETRIC_LIGHT · LIGHTS · SHADERS ·
 *                PARTICLES · SHADOWS · MESH · MISC
 *
 *  2. For each live-controllable entry (numeric scalar, numeric vec2/3/4,
 *     hex colour, boolean) render the appropriate UIL-inspired widget:
 *       • number  → range slider + numeric text input (two-way)
 *       • vec2/3/4 → N linked numeric inputs
 *       • color   → <input type="color"> swatch
 *       • bool    → checkbox toggle
 *
 *  3. Two-way binding:
 *       panel → params: onChange fires an `ATUILParamChange` CustomEvent
 *                       carrying { key, value, section }
 *       params → panel: ATUILLivePanel.set(key, value) updates the widget DOM
 *                       without triggering a change event (prevents loops).
 *
 *  4. Preset save / load:
 *       savePreset(name)  → serialises current param snapshot to
 *                           localStorage key `at-uil-preset:<name>`
 *       loadPreset(name)  → restores values + refreshes all widgets
 *       listPresets()     → string[] of stored preset names
 *       exportPreset(name)→ downloads a JSON file via anchor click
 *       importPreset(json)→ parses JSON string and loads preset
 *
 *  5. Section filtering:
 *       showSection(name) / hideSection(name)
 *       filterByKey(query) — real-time substring search across all keys
 *
 *  6. Panel lifecycle:
 *       mount(container)  → injects DOM into a container element
 *       unmount()         → removes DOM and all event listeners
 *       toggle()          → show/hide the entire panel
 *
 * Usage
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   const panel = new ATUILLivePanel();
 *   await panel.init('/assets/uil-params.json');  // or pass params directly
 *   panel.mount(document.body);
 *
 *   // listen for changes
 *   panel.on('change', ({ key, value }) => {
 *     myRenderer.setUniform(key, value);
 *   });
 *
 *   // push programmatic update (e.g. from physics sim)
 *   panel.set('VolumetricLight_home_fExposure', 1.2);
 *
 *   // save the current state
 *   panel.savePreset('night-mode');
 *   panel.loadPreset('night-mode');
 *
 * Research: xiaodi #M828 — cell-pubsub-loop
 */

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/** Raw value types that can appear in uil-params.json. */








export type RawValue = number | boolean | string | number[];

/** Value types the live panel can control (texture dicts and str lists excluded). */
export type LiveValue = number | boolean | string | number[];

/** Control widget type inferred from value shape. */
export type WidgetType = 'slide' | 'vec' | 'color' | 'bool';

/** One classified param entry. */
export interface UILParamEntry {
  key: string;
  section: PanelSection;
  widgetType: WidgetType;
  value: LiveValue;
  /** For vec widgets: dimensionality (2–4). */
  vecLen?: number;
  /** Minimum/maximum hints (auto-derived or user-supplied). */
  min?: number;
  max?: number;
  step?: number;
}

/** Named panel sections — maps to top-level grouping in the DOM. */
export type PanelSection =
  | 'CAMERA'
  | 'POST_PROCESS'
  | 'VOLUMETRIC_LIGHT'
  | 'LIGHTS'
  | 'SHADERS'
  | 'PARTICLES'
  | 'SHADOWS'
  | 'MESH'
  | 'MISC';

// ─────────────────────────────────────────────────────────────────────────────
// Module classifications (UIL 2593 params → named modules)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Fine-grained module within a section.
 * 2593 AT UIL params are distributed across these modules.
 *
 * Distribution (approximate):
 *   CAMERA      : 89  params  — per-element camera position/fov/lerp settings
 *   LIGHTING    : 35  params  — L_Element lights + INPUT_L config
 *   BLOOM       : 62  params  — UnrealBloom strength/radius/threshold
 *   VOLUMETRIC  : 14  params  — VolumetricLight fDecay/fDensity/fExposure
 *   COMPOSITE   : 40  params  — HomeComposite / WorkComposite / CleanRoom RGB + DOF
 *   PARTICLE    : 434 params  — am_ behaviour uniforms + INPUT_P config
 *   SCENE       : 966 params  — INPUT_Config scene-layout entries
 *   SHADER      : 395 params  — Shader material uniforms (Glass, PBR, Floor, Wall…)
 *   MESH        : 333 params  — MESH_ position/rotation/scale per element
 *   SHADOW      : 10  params  — SHADOW_ light configs
 *   MISC        : 215 params  — UIL_graph, GROUP, INPUT_GROUP, etc.
 */
export type UILModule =
  | 'CAMERA'
  | 'LIGHTING'
  | 'BLOOM'
  | 'VOLUMETRIC'
  | 'COMPOSITE'
  | 'PARTICLE'
  | 'SCENE'
  | 'SHADER'
  | 'MESH'
  | 'SHADOW'
  | 'MISC';

/** Mapping from UILModule to its PanelSection. */
export const MODULE_TO_SECTION: Record<UILModule, PanelSection> = {
  CAMERA:     'CAMERA',
  LIGHTING:   'LIGHTS',
  BLOOM:      'POST_PROCESS',
  VOLUMETRIC: 'VOLUMETRIC_LIGHT',
  COMPOSITE:  'POST_PROCESS',
  PARTICLE:   'PARTICLES',
  SCENE:      'MISC',
  SHADER:     'SHADERS',
  MESH:       'MESH',
  SHADOW:     'SHADOWS',
  MISC:       'MISC',
};

/** Classify a raw UIL key into its fine-grained module. */
export function classifyModule(key: string): UILModule {
  if (key.startsWith('CAMERA_'))                                              return 'CAMERA';
  if (key.startsWith('VolumetricLight'))                                      return 'VOLUMETRIC';
  if (/UnrealBloom|BloomLuminosity|INPUT_HydraBloom/i.test(key))             return 'BLOOM';
  if (/Composite|HomeSceneVFX|homeParticle/i.test(key))                      return 'COMPOSITE';
  if (key.startsWith('L_') || key.startsWith('INPUT_L_'))                    return 'LIGHTING';
  if (key.startsWith('SHADOW_'))                                              return 'SHADOW';
  if (key.startsWith('MESH_'))                                                return 'MESH';
  if (key.startsWith('am_') || key.startsWith('INPUT_P_') ||
      /Proton|Spline|Antimatter|Particle.*config/i.test(key))                 return 'PARTICLE';
  if (key.startsWith('INPUT_Config') || key.startsWith('INPUT_scenelayout'))  return 'SCENE';
  if (key.includes('Shader') || key.includes('PBR') || key.startsWith('PhysicalShader')) {
    return 'SHADER';
  }
  return 'MISC';
}

/** Module display metadata. */
export const MODULE_META: Record<UILModule, { label: string; icon: string; approxCount: number }> = {
  CAMERA:     { label: 'Camera',      icon: '📷', approxCount:  89 },
  LIGHTING:   { label: 'Lighting',    icon: '💡', approxCount:  35 },
  BLOOM:      { label: 'Bloom',       icon: '✨', approxCount:  62 },
  VOLUMETRIC: { label: 'Volumetric',  icon: '🌫', approxCount:  14 },
  COMPOSITE:  { label: 'Composite',   icon: '🎨', approxCount:  40 },
  PARTICLE:   { label: 'Particles',   icon: '🌊', approxCount: 434 },
  SCENE:      { label: 'Scene',       icon: '🏛',  approxCount: 966 },
  SHADER:     { label: 'Shaders',     icon: '🔮', approxCount: 395 },
  MESH:       { label: 'Mesh',        icon: '🧊', approxCount: 333 },
  SHADOW:     { label: 'Shadows',     icon: '🌑', approxCount:  10 },
  MISC:       { label: 'Misc',        icon: '⚙️',  approxCount: 215 },
};

// ─────────────────────────────────────────────────────────────────────────────
// Species → UIL preset system
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Known cell species identifiers.
 * Maps to CoreML cell icon names from CIL.
 */
export type SpeciesId =
  | 'cil-eye'
  | 'cil-bolt'
  | 'cil-vector'
  | 'cil-plus'
  | 'cil-arrow-right'
  | 'cil-filter'
  | 'cil-code'
  | 'cil-layers'
  | 'cil-loop'
  | 'cil-graph';

/**
 * A species UIL preset: a sparse map of param key → override value.
 * Only params that differ from the global AT UIL defaults need be listed.
 * Values are applied via `ATUILLivePanel.setBatch()` when the species activates.
 */
export interface SpeciesUILPreset {
  /** Human-readable description of this species' visual character. */
  description: string;
  /** Params overriding the AT UIL baseline. */
  params: Partial<Record<string, LiveValue>>;
  /** Module tags that this preset primarily affects (for UI grouping). */
  modules: UILModule[];
}

/**
 * SPECIES_UIL_PRESETS
 * ─────────────────────────────────────────────────────────────────────────────
 * Each species maps to a curated subset of the 2593 AT UIL params that defines
 * its unique visual identity.  Values are sourced directly from:
 *   channels/physics/species_at_params.json  (AT artist-tuned values)
 *
 * Rationale per species:
 *   cil-eye         → Multi-Head Attention — focal, calm, wide FOV, soft bloom
 *   cil-bolt        → FFN / activation — sharp, high-energy, tight bloom + fast lerp
 *   cil-vector      → Embedding — diffuse, warm, mid-range bloom, balanced PBR
 *   cil-plus        → LayerNorm / residual — clean, additive, cool-toned
 *   cil-arrow-right → Output projection — directional, structured, tree-scene env
 *   cil-filter      → Attention mask — filtered, selective, clean-room mood
 *   cil-code        → Token / positional — technical, dark, sparse bloom
 *   cil-layers      → Stack / depth — layered, rich PBR, deep home scene
 *   cil-loop        → Recurrent / loop — cyclic, pulsing, about-scene bloom
 *   cil-graph       → Dependency graph — structural, footer-scene, low bloom
 */
export const SPECIES_UIL_PRESETS: Record<SpeciesId, SpeciesUILPreset> = {
  'cil-eye': {
    description: 'Multi-Head Attention — focal perception, soft wide-angle look',
    modules: ['CAMERA', 'BLOOM', 'VOLUMETRIC', 'LIGHTING', 'SHADER'],
    params: {
      // Camera — wide, slow lerp, gentle wobble
      'CAMERA_Element_3_home_scenefov':          30,
      'CAMERA_Element_3_home_scenewobbleStrength': 0.1,
      'CAMERA_Element_1_Homefov':                30,
      'CAMERA_Element_1_HomelerpSpeed':          0.1,
      'CAMERA_Element_1_HomelerpSpeed2':         1,
      'CAMERA_Element_1_homeScenefov':           20,
      // Bloom — soft home bloom
      'UnrealBloomComposite/UnrealBloomComposite/home/bloomStrength':  1.2,
      'UnrealBloomComposite/UnrealBloomComposite/home/bloomRadius':    1.0,
      'UnrealBloomComposite_shaderVariants_homebloomStrength':         0.6,
      'UnrealBloomComposite_shaderVariants_homebloomRadius':           0.8,
      'UnrealBloomLuminosity/UnrealBloomLuminosity/home/luminosityThreshold': 0,
      // Volumetric — mild exposure
      'VolumetricLight_home_fExposure': 0.86,
      'VolumetricLight_home_fDensity':  0.22,
      'VolumetricLight_home_fDecay':    0.80,
      'VolumetricLight_home_fWeight':   0.34,
      // Lighting — comfortable intensity
      'L_Element_10_home_sceneintensity': 2.19,
      'L_Element_11_home_sceneintensity': 3.44,
      // Glass — clear refractive, gentle distort
      'GlassCubeShader/GlassCubeShader/Element_0_home_scene/uDistortStrength': 8.06,
      'GlassCubeShader/GlassCubeShader/Element_0_home_scene/uReflectScale':    1.0,
      'GlassCubeShader/GlassCubeShader/Element_0_home_scene/uFresnelPow':      1.5,
    },
  },

  'cil-bolt': {
    description: 'FFN / activation — high-energy, sharp contrast, fast transitions',
    modules: ['CAMERA', 'BLOOM', 'SHADER', 'PARTICLE'],
    params: {
      // Camera — tighter FOV work scene, quick lerp
      'CAMERA_Element_2_Workfov':         35,
      'CAMERA_Element_2_WorklerpSpeed':   0.07,
      'CAMERA_Element_2_WorklerpSpeed2':  1.0,
      'CAMERA_Element_1_WorkDetaillerpSpeed': 0.07,
      // Bloom — strong work bloom
      'UnrealBloomComposite_shaderVariants_workbloomStrength': 0.5,
      'UnrealBloomComposite_shaderVariants_workbloomRadius':   0.5,
      'UnrealBloomComposite/UnrealBloomComposite/workbloom/bloomStrength': 1.0,
      'UnrealBloomComposite/UnrealBloomComposite/workbloom/bloomRadius':   1.0,
      'UnrealBloomLuminosity/UnrealBloomLuminosity/workbloom/luminosityThreshold': 0,
      // Work shaders — energetic distortion
      'WorkDetailCube/WorkDetailCube/Element_0_WorkDetail/uDistortStrength': 5.0,
      'WorkDetailCube/WorkDetailCube/Element_0_WorkDetail/uNormalScale':     6.0,
      'WorkDetailCube/WorkDetailCube/Element_0_WorkDetail/uFresnelPow':      0.1,
      'WorkDetailCube/WorkDetailCube/Element_0_WorkDetail/uParticleDarken':  0.6,
      // SpineShader — visible normal structure
      'SpineShader/SpineShader/Element_5_Work/uNormalStrength': 0.19,
      // Particle — fast curl noise
      'am_ProtonAntimatter_P_Element_0_particleTestuCurlNoiseSpeed': 0.74,
      'am_ProtonAntimatter_P_Element_0_particleTestuCurlNoiseScale': 7.76,
    },
  },

  'cil-vector': {
    description: 'Embedding / representation — warm diffuse, balanced PBR',
    modules: ['CAMERA', 'BLOOM', 'SHADER', 'LIGHTING'],
    params: {
      // Camera — home + cleanroom dual context
      'CAMERA_Element_3_home_scenefov':            30,
      'CAMERA_Element_3_home_scenewobbleStrength':  0.1,
      'CAMERA_Element_10_CleanRoomfov':            30,
      'CAMERA_Element_10_CleanRoomlerpSpeed':      0.08,
      // Bloom — about scene + footer (embedding = broad coverage)
      'UnrealBloomComposite_shaderVariants_aboutbloomStrength': 1.0,
      'UnrealBloomComposite_shaderVariants_aboutbloomRadius':   1.0,
      'UnrealBloomComposite_shaderVariants_footerbloomStrength': 0.7,
      'UnrealBloomComposite_shaderVariants_footerbloomRadius':   0.5,
      // Logo shader — subtle normal
      'AboutLogoShader/AboutLogoShader/Element_2_About/uNormalStrength': 0.24,
      // PBR — warm home scene
      'ATPBR/ATPBR/Element_6_homeScene/uEnv': [1.5, 1],
      'ATPBR/ATPBR/Element_6_homeScene/uMRON': [1, 1.3, 1, 1],
    },
  },

  'cil-plus': {
    description: 'LayerNorm / residual connection — clean, additive, cool-toned',
    modules: ['CAMERA', 'BLOOM', 'COMPOSITE', 'LIGHTING'],
    params: {
      // Camera — cleanroom (normalisation = clean environment)
      'CAMERA_Element_10_CleanRoomfov':       30,
      'CAMERA_Element_10_CleanRoomlerpSpeed': 0.08,
      'CAMERA_Element_1_Homefov':             30,
      'CAMERA_Element_1_HomelerpSpeed':       0.1,
      // Bloom — contact (residual = touch point)
      'UnrealBloomComposite_shaderVariants_contactbloomStrength':   0.8,
      'UnrealBloomComposite_shaderVariants_contactbloomRadius':     0.5,
      'UnrealBloomComposite_shaderVariants_contactbloomTintColor':  '#ffffff',
      // Cleanroom composite — balanced VFX
      'CleanRoomCompositeuRGBStrength':        0.3,
      'CleanRoomCompositeuVolumetricStrength': 0.3,
      // Volumetric — cleanroom light
      'VolumetricLight_cleanroom_fExposure': 0.62,
      'VolumetricLight_cleanroom_fDensity':  0.29,
      'VolumetricLight_cleanroom_fDecay':    0.865,
    },
  },

  'cil-arrow-right': {
    description: 'Output projection — directional, tree-scene, flowing water',
    modules: ['CAMERA', 'BLOOM', 'SHADER', 'COMPOSITE'],
    params: {
      // Tree scene dominates
      'UnrealBloomComposite_shaderVariants_treebloomStrength': 0.8,
      'UnrealBloomComposite_shaderVariants_treebloomRadius':   0.7,
      'UnrealBloomComposite/UnrealBloomComposite/treescene/bloomStrength': 1.0,
      'UnrealBloomComposite/UnrealBloomComposite/treescene/bloomRadius':   1.0,
      'UnrealBloomLuminosity/UnrealBloomLuminosity/treescene/luminosityThreshold': 0,
      // TreeScene composite — rich contrast
      'TreeSceneCompositeuRGBStrength': 0,
      'TreeSceneCompositeuContrast':    [1, 1.5],
      // Tree PBR materials
      'TreeFBR/TreeFBR/Element_0_TreeScene/uNormalStrength':  1.0,
      'TreeFBR/TreeFBR/Element_16_TreeScene/uNormalStrength': 1.0,
      'TreeFBR/TreeFBR/Element_1_TreeScene/uNormalStrength':  1.0,
      'TreeFBR/TreeFBR/Element_7_TreeScene/uNormalStrength':  1.0,
      'TreeFBR/TreeFBR/Element_5_TreeScene/uNormalStrength':  0.4,
      // Water — directional flow
      'TreeWaterShader/TreeWaterShader/uSpeed':              0.12,
      'TreeWaterShader/TreeWaterShader/uNormalStrength':     0.67,
      'TreeWaterShader/TreeWaterShader/uMouseUVStrength':    0,
    },
  },

  'cil-filter': {
    description: 'Attention mask / filter — selective, clean-room environment',
    modules: ['CAMERA', 'BLOOM', 'SHADER', 'COMPOSITE', 'VOLUMETRIC'],
    params: {
      // Cleanroom camera — filtered view
      'CAMERA_Element_10_CleanRoomfov':       30,
      'CAMERA_Element_10_CleanRoomlerpSpeed': 0.08,
      // Cleanroom bloom
      'UnrealBloomComposite/UnrealBloomComposite/cleanroom/bloomStrength': 1.0,
      'UnrealBloomComposite/UnrealBloomComposite/cleanroom/bloomRadius':   1.0,
      'UnrealBloomLuminosity/UnrealBloomLuminosity/cleanroom/luminosityThreshold': 0.2,
      // Cleanroom composite — volumetric atmosphere
      'CleanRoomCompositeuRGBStrength':        0.3,
      'CleanRoomCompositeuVolumetricStrength': 0.3,
      // Glass — selective refraction (filter metaphor)
      'CleanRoomGlass/CleanRoomGlass/Element_4_CleanRoom/uDistortStrength':  -1.0,
      'CleanRoomGlass/CleanRoomGlass/Element_4_CleanRoom/uFresnelPow':       -0.03,
      'CleanRoomGlass/CleanRoomGlass/Element_4_CleanRoom/uRefractionRatio':  0.34,
      // Floor reflections
      'FloorShader/FloorShader/Element_0_CleanRoom/uDistortStrength': 0.44,
      'FloorShader/FloorShader/Element_0_CleanRoom/uMirrorStrength':  0.55,
      'FloorShader/FloorShader/Element_0_CleanRoom/uNormalStrength':  1.0,
      // Volumetric — cleanroom
      'VolumetricLight_cleanroom_fExposure': 0.62,
      'VolumetricLight_cleanroom_fDensity':  0.29,
      'VolumetricLight_cleanroom_fDecay':    0.865,
      'VolumetricLight_cleanroom_fWeight':   1.0,
    },
  },

  'cil-code': {
    description: 'Token / positional encoding — technical, sparse, dark work-scene',
    modules: ['CAMERA', 'BLOOM', 'SHADER'],
    params: {
      // Work camera — sharp focus
      'CAMERA_Element_2_Workfov':                35,
      'CAMERA_Element_2_WorklerpSpeed':          0.07,
      'CAMERA_Element_2_WorklerpSpeed2':         1.0,
      'CAMERA_Element_1_WorkDetaillerpSpeed':    0.07,
      // Work bloom — constrained
      'UnrealBloomComposite_shaderVariants_workbloomStrength': 0.5,
      'UnrealBloomComposite_shaderVariants_workbloomRadius':   0.5,
      'WorkCompositeuRGBStrength': 0,
      'WorkCompositeuTransition':  0,
      // Chain / spine shaders (code = structure)
      'SpineShader/SpineShader/Element_5_Work/uNormalStrength': 0.19,
      'SpineShader/SpineShader/Element_5_Work/uReflection':     [2.7, 0.85],
      // Work detail cube
      'WorkDetailCube/WorkDetailCube/Element_0_WorkDetail/uDistortStrength': 5.0,
      'WorkDetailCube/WorkDetailCube/Element_0_WorkDetail/uNormalScale':     6.0,
    },
  },

  'cil-layers': {
    description: 'Transformer stack depth — layered PBR, deep home atmosphere',
    modules: ['CAMERA', 'BLOOM', 'VOLUMETRIC', 'SHADER', 'LIGHTING'],
    params: {
      // Same base as cil-eye but richer PBR
      'CAMERA_Element_3_home_scenefov':          30,
      'CAMERA_Element_3_home_scenewobbleStrength': 0.1,
      'CAMERA_Element_1_Homefov':                30,
      'CAMERA_Element_1_HomelerpSpeed':          0.1,
      'CAMERA_Element_1_HomelerpSpeed2':         1,
      'CAMERA_Element_1_homeScenefov':           20,
      // Bloom — strong home bloom (depth = brightness)
      'UnrealBloomComposite/UnrealBloomComposite/home/bloomStrength': 3.82,
      'UnrealBloomComposite/UnrealBloomComposite/home/bloomRadius':   1.0,
      'UnrealBloomComposite/UnrealBloomComposite/homebloom/bloomStrength': 1.2,
      'UnrealBloomComposite/UnrealBloomComposite/homebloom/bloomRadius':   1.0,
      'UnrealBloomComposite_shaderVariants_homebloomStrength': 0.6,
      'UnrealBloomComposite_shaderVariants_homebloomRadius':   0.8,
      // Volumetric — layered fog
      'VolumetricLight_home_fExposure': 0.86,
      'VolumetricLight_home_fDensity':  0.22,
      'VolumetricLight_home_fDecay':    0.80,
      'VolumetricLight_home_fWeight':   0.34,
      // Home composite — rich atmosphere
      'HomeCompositeuVolumetricStrength': 1.1,
      'HomeCompositeuRGBStrength':        0,
      // Lighting — deep scene
      'L_Element_10_home_sceneintensity': 2.19,
      'L_Element_10_home_scenedistance':  60,
      'L_Element_11_home_sceneintensity': 3.44,
      // Glass cube — deep reflections
      'GlassCubeShader/GlassCubeShader/Element_0_home_scene/uDistortStrength': 8.06,
      'GlassCubeShader/GlassCubeShader/Element_0_home_scene/uReflectScale':    1.0,
    },
  },

  'cil-loop': {
    description: 'Recurrent / cyclic — pulsing about-scene bloom, cyclical motion',
    modules: ['CAMERA', 'BLOOM', 'SHADER'],
    params: {
      // About + cleanroom cameras (loop = cross-scene)
      'CAMERA_Element_3_home_scenefov':          30,
      'CAMERA_Element_3_home_scenewobbleStrength': 0.1,
      'CAMERA_Element_10_CleanRoomfov':          30,
      'CAMERA_Element_10_CleanRoomlerpSpeed':    0.08,
      'CAMERA_Element_1_Homefov':                30,
      'CAMERA_Element_1_HomelerpSpeed':          0.1,
      // Bloom — about scene (cyclic = complete loop)
      'UnrealBloomComposite_shaderVariants_aboutbloomStrength': 1.0,
      'UnrealBloomComposite_shaderVariants_aboutbloomRadius':   1.0,
      // Logo shader — looping normals
      'AboutLogoShader/AboutLogoShader/Element_2_About/uNormalStrength': 0.24,
    },
  },

  'cil-graph': {
    description: 'Dependency graph — structural, footer scene, minimal bloom',
    modules: ['CAMERA', 'BLOOM', 'SHADER'],
    params: {
      // Footer + cleanroom cameras
      'CAMERA_Element_3_home_scenefov':          30,
      'CAMERA_Element_3_home_scenewobbleStrength': 0.1,
      'CAMERA_Element_10_CleanRoomfov':          30,
      'CAMERA_Element_10_CleanRoomlerpSpeed':    0.08,
      'CAMERA_Element_1_Homefov':                30,
      'CAMERA_Element_1_HomelerpSpeed':          0.1,
      // Bloom — footer (graph = low-level, structural)
      'UnrealBloomComposite_shaderVariants_footerbloomStrength': 0.7,
      'UnrealBloomComposite_shaderVariants_footerbloomRadius':   0.5,
      'UnrealBloomComposite_shaderVariants_footerbloomTintColor': '#ffffff',
      // Jelly shaders (graph nodes = organic connections)
      'JellyShader/JellyShader/Element_2_Footer/uReflection': [1, 0.15],
    },
  },
};

/** Payload emitted on every widget interaction. */
export interface ParamChangeEvent {
  key: string;
  value: LiveValue;
  section: PanelSection;
  /** Previous value before this change. */
  prev: LiveValue;
}

/** Serialised preset snapshot (key → value). */
export type PresetSnapshot = Record<string, LiveValue>;

/** Event handler for param changes. */
export type ChangeHandler = (evt: ParamChangeEvent) => void;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

const SECTION_ORDER: PanelSection[] = [
  'CAMERA',
  'POST_PROCESS',
  'VOLUMETRIC_LIGHT',
  'LIGHTS',
  'SHADERS',
  'PARTICLES',
  'SHADOWS',
  'MESH',
  'MISC',
];

const SECTION_LABELS: Record<PanelSection, string> = {
  CAMERA:           '📷 Camera',
  POST_PROCESS:     '✨ Post Process',
  VOLUMETRIC_LIGHT: '💡 Volumetric Light',
  LIGHTS:           '🔆 Lights',
  SHADERS:          '🎨 Shaders',
  PARTICLES:        '🌊 Particles',
  SHADOWS:          '🌑 Shadows',
  MESH:             '🧊 Mesh',
  MISC:             '⚙️ Misc',
};

const PRESET_STORAGE_PREFIX = 'at-uil-preset:';

/**
 * Sensible default slider ranges by last token in key name.
 * Pairs are [min, max, step].
 */
const RANGE_HINTS: Record<string, [number, number, number]> = {
  fov:              [10,  120, 0.5],
  intensity:        [0,   5,   0.01],
  exposure:         [0,   3,   0.01],
  density:          [0,   2,   0.01],
  decay:            [0.5, 1,   0.001],
  strength:         [0,   3,   0.01],
  scale:            [0,   10,  0.05],
  speed:            [0,   10,  0.01],
  threshold:        [0,   1,   0.01],
  radius:           [0,   5,   0.01],
  roughness:        [0,   1,   0.01],
  metallic:         [0,   1,   0.01],
  alpha:            [0,   1,   0.01],
  opacity:          [0,   1,   0.01],
  contrast:         [0,   2,   0.01],
  brightness:       [0,   2,   0.01],
  lerpSpeed:        [0,   1,   0.001],
  lerpSpeed2:       [0,   2,   0.01],
  wobbleStrength:   [0,   1,   0.001],
  deltaRotate:      [-180, 180, 0.5],
  distance:         [0,   200, 0.1],
  far:              [0,   2000, 1],
  near:             [0,   10,  0.01],
  bounce:           [0,   1,   0.01],
  weight:           [0,   2,   0.01],
  uPointSize:       [0,   10,  0.1],
  uNormalStrength:  [0,   3,   0.01],
  uAlpha:           [0,   1,   0.01],
  uFresnelPow:      [0,   8,   0.01],
  uShininess:       [0,   200, 0.5],
  uEnvBlend:        [0,   1,   0.01],
  uDistortStrength: [0,   2,   0.01],
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Decide which section a param key belongs to. */
function classifySection(key: string): PanelSection {
  if (key.startsWith('CAMERA_'))                                  return 'CAMERA';
  if (key.startsWith('VolumetricLight'))                          return 'VOLUMETRIC_LIGHT';
  if (/Bloom|Composite|Luminosity/i.test(key))                    return 'POST_PROCESS';
  if (key.startsWith('L_'))                                       return 'LIGHTS';
  if (key.startsWith('SHADOW_'))                                  return 'SHADOWS';
  if (key.startsWith('MESH_'))                                    return 'MESH';
  if (key.startsWith('am_') || key.startsWith('homeParticle'))    return 'PARTICLES';
  if (key.includes('Shader') || key.includes('PBR'))              return 'SHADERS';
  return 'MISC';
}

/** Decide the widget type from a value. */
function classifyWidget(value: RawValue): WidgetType | null {
  if (typeof value === 'boolean')                           return 'bool';
  if (typeof value === 'string' && value.startsWith('#'))  return 'color';
  if (typeof value === 'number')                           return 'slide';
  if (Array.isArray(value) && value.length >= 2 && value.length <= 4
      && value.every(x => typeof x === 'number'))          return 'vec';
  return null; // not live-controllable (dict/string list)
}

/** Infer slider range from the last meaningful token of a param key. */
function inferRange(key: string): [number, number, number] {
  const tokens = key.replace(/[/_]/g, ' ').split(' ');
  for (let i = tokens.length - 1; i >= 0; i--) {
    const t = tokens[i];
    if (RANGE_HINTS[t]) return RANGE_HINTS[t];
    // partial suffix match
    for (const hint of Object.keys(RANGE_HINTS)) {
      if (t.toLowerCase().endsWith(hint.toLowerCase())) return RANGE_HINTS[hint];
    }
  }
  return [-10, 10, 0.01]; // safe default
}

/** Round a number to `digits` decimal places. */
function round(n: number, digits = 4): number {
  const f = 10 ** digits;
  return Math.round(n * f) / f;
}

/** Deep-clone a LiveValue (handles array case). */
function cloneValue(v: LiveValue): LiveValue {
  return Array.isArray(v) ? [...v] : v;
}

/** CSS for the entire panel, injected once as a <style> tag. */
const PANEL_CSS = `
.at-uil-panel {
  position: fixed; top: 10px; right: 10px; z-index: 99999;
  width: 320px; max-height: calc(100vh - 20px);
  background: #0d0d0f; color: #c8c8d0;
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 11px; border: 1px solid #2a2a38;
  border-radius: 6px; box-shadow: 0 8px 32px rgba(0,0,0,0.7);
  display: flex; flex-direction: column; overflow: hidden;
  user-select: none;
}
.at-uil-panel.hidden { display: none; }

.at-uil-panel-header {
  padding: 8px 10px; background: #13131a;
  border-bottom: 1px solid #2a2a38;
  display: flex; align-items: center; gap: 6px; flex-shrink: 0;
}
.at-uil-panel-title {
  flex: 1; font-size: 12px; font-weight: 700; color: #9090ff;
  letter-spacing: 0.05em;
}
.at-uil-panel-toggle-btn {
  background: #222230; border: 1px solid #333348; color: #9090ff;
  border-radius: 3px; padding: 2px 7px; cursor: pointer; font-size: 10px;
}
.at-uil-panel-toggle-btn:hover { background: #2a2a45; }

.at-uil-toolbar {
  padding: 5px 8px; background: #101016;
  border-bottom: 1px solid #1e1e2a;
  display: flex; gap: 5px; flex-wrap: wrap; flex-shrink: 0;
}
.at-uil-toolbar input[type=text] {
  flex: 1; background: #1a1a24; border: 1px solid #2a2a38;
  color: #c8c8d0; border-radius: 3px; padding: 3px 6px; font-size: 10px;
  outline: none;
}
.at-uil-toolbar input[type=text]:focus { border-color: #6060cc; }
.at-uil-btn {
  background: #1a1a26; border: 1px solid #2d2d44; color: #9090cc;
  border-radius: 3px; padding: 3px 7px; cursor: pointer; font-size: 10px;
  transition: background 0.1s;
}
.at-uil-btn:hover { background: #252536; color: #b0b0ff; }

.at-uil-preset-row {
  padding: 4px 8px; background: #0e0e18;
  border-bottom: 1px solid #1e1e2a;
  display: flex; gap: 4px; align-items: center; flex-shrink: 0;
}
.at-uil-preset-row select {
  flex: 1; background: #1a1a24; border: 1px solid #2a2a38;
  color: #c8c8d0; border-radius: 3px; padding: 3px 4px; font-size: 10px;
  outline: none;
}

.at-uil-scroll {
  overflow-y: auto; flex: 1;
  scrollbar-width: thin; scrollbar-color: #2a2a48 #0d0d0f;
}
.at-uil-scroll::-webkit-scrollbar { width: 5px; }
.at-uil-scroll::-webkit-scrollbar-thumb { background: #2a2a48; border-radius: 3px; }

.at-uil-section { margin-bottom: 1px; }

.at-uil-section-header {
  padding: 5px 8px; background: #16161f;
  border-top: 1px solid #202030; border-bottom: 1px solid #202030;
  cursor: pointer; display: flex; align-items: center; gap: 6px;
  color: #7878c8; font-size: 10px; font-weight: 700; letter-spacing: 0.08em;
  text-transform: uppercase;
}
.at-uil-section-header:hover { background: #1c1c28; }
.at-uil-section-arrow { transition: transform 0.15s; color: #4444aa; }
.at-uil-section-header.collapsed .at-uil-section-arrow { transform: rotate(-90deg); }
.at-uil-section-count {
  margin-left: auto; font-size: 9px; color: #4444aa; font-weight: 400;
}

.at-uil-section-body { padding: 0; }
.at-uil-section-body.collapsed { display: none; }

.at-uil-row {
  padding: 3px 8px 3px 10px; border-bottom: 1px solid #141420;
  display: flex; align-items: center; gap: 5px;
  transition: background 0.05s;
}
.at-uil-row:hover { background: #131320; }
.at-uil-row.hidden { display: none; }

.at-uil-label {
  width: 130px; flex-shrink: 0; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap;
  color: #808098; font-size: 10px; cursor: default;
}
.at-uil-label:hover { color: #a0a0c0; }

.at-uil-widget { flex: 1; display: flex; align-items: center; gap: 3px; }

/* Slide (range) */
.at-uil-slide { flex: 1; display: flex; align-items: center; gap: 3px; }
.at-uil-slide input[type=range] {
  flex: 1; height: 3px; cursor: pointer; accent-color: #6060cc;
  background: #2a2a40; border-radius: 2px;
  -webkit-appearance: none; appearance: none; outline: none;
}
.at-uil-slide input[type=range]::-webkit-slider-thumb {
  -webkit-appearance: none; width: 8px; height: 8px;
  border-radius: 50%; background: #8080dd; cursor: pointer;
}
.at-uil-num-input {
  width: 48px; background: #1a1a28; border: 1px solid #282840;
  color: #b0b0d0; border-radius: 2px; padding: 1px 3px;
  font-size: 10px; font-family: inherit; text-align: right; outline: none;
}
.at-uil-num-input:focus { border-color: #5050aa; }

/* Vec */
.at-uil-vec { display: flex; gap: 3px; flex-wrap: wrap; }
.at-uil-vec .at-uil-num-input { width: 56px; }

/* Color */
.at-uil-color { display: flex; align-items: center; gap: 5px; }
.at-uil-color input[type=color] {
  width: 22px; height: 22px; padding: 0; border: none;
  background: none; cursor: pointer; border-radius: 3px;
}
.at-uil-color-hex {
  flex: 1; background: #1a1a28; border: 1px solid #282840;
  color: #b0b0d0; border-radius: 2px; padding: 1px 4px;
  font-size: 10px; font-family: inherit; outline: none;
}
.at-uil-color-hex:focus { border-color: #5050aa; }

/* Bool */
.at-uil-bool input[type=checkbox] { accent-color: #6060cc; cursor: pointer; }

.at-uil-status {
  padding: 3px 8px; font-size: 9px; color: #4444aa;
  border-top: 1px solid #1e1e2a; text-align: right; flex-shrink: 0;
  background: #0a0a12;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// ATUILLivePanel
// ─────────────────────────────────────────────────────────────────────────────

export class ATUILLivePanel {

  // ── Public state ───────────────────────────────────────────────────────────

  /** All classified entries, keyed by UIL param key. */
  public params: Map<string, UILParamEntry> = new Map();

  /** Current live values (starts as a copy of the parsed defaults). */
  public values: Map<string, LiveValue> = new Map();

  // ── Private state ──────────────────────────────────────────────────────────

  private _root: HTMLElement | null = null;
  private _mounted = false;
  private _visible = true;
  private _filterQuery = '';

  /** section → collapsed state */
  private _collapsed: Map<PanelSection, boolean> = new Map();

  /** widget element references: key → { root, inputs[], etc. } */
  private _widgets: Map<string, WidgetRefs> = new Map();

  /** Registered change handlers. */
  private _handlers: Set<ChangeHandler> = new Set();

  /** Status bar element. */
  private _statusEl: HTMLElement | null = null;

  /** Preset select element. */
  private _presetSelect: HTMLSelectElement | null = null;

  /** Internal flag to suppress change events during programmatic set(). */
  private _suppressEvents = false;

  // ── Init ───────────────────────────────────────────────────────────────────

  /**
   * Initialise the panel from a URL or a pre-loaded params object.
   *
   * @param source URL string → fetch + parse JSON  |  plain object → use directly
   */
  async init(source: string | Record<string, RawValue>): Promise<void> {
    let raw: Record<string, RawValue>;
    if (typeof source === 'string') {
      const res = await fetch(source);
      raw = await res.json();
    } else {
      raw = source;
    }
    this._classifyParams(raw);
    if (this._mounted && this._root) {
      // Re-render if already mounted
      this._renderPanelBody();
    }
  }

  /**
   * Classify all raw params into UILParamEntry records.
   */
  private _classifyParams(raw: Record<string, RawValue>): void {
    this.params.clear();
    this.values.clear();

    for (const [key, rawValue] of Object.entries(raw)) {
      const wt = classifyWidget(rawValue);
      if (wt === null) continue; // skip textures, string lists, dicts

      const section = classifySection(key);
      const [min, max, step] = inferRange(key);
      const vecLen = (wt === 'vec') ? (rawValue as number[]).length : undefined;

      const entry: UILParamEntry = {
        key,
        section,
        widgetType: wt,
        value: rawValue as LiveValue,
        min,
        max,
        step,
        vecLen,
      };
      this.params.set(key, entry);
      this.values.set(key, cloneValue(rawValue as LiveValue));
    }
  }

  // ── Mount / Unmount ────────────────────────────────────────────────────────

  /**
   * Inject the panel DOM into `container`.  Can be called before `init()`;
   * the body will be populated once params are available.
   */
  mount(container: HTMLElement = document.body): void {
    if (this._mounted) return;

    // Inject CSS once
    if (!document.getElementById('at-uil-panel-style')) {
      const style = document.createElement('style');
      style.id = 'at-uil-panel-style';
      style.textContent = PANEL_CSS;
      document.head.appendChild(style);
    }

    this._root = this._buildShell();
    container.appendChild(this._root);
    this._mounted = true;

    if (this.params.size > 0) {
      this._renderPanelBody();
    }
  }

  /** Remove the panel from the DOM and clean up. */
  unmount(): void {
    if (!this._mounted || !this._root) return;
    this._root.remove();
    this._root = null;
    this._mounted = false;
    this._widgets.clear();
    this._statusEl = null;
    this._presetSelect = null;
  }

  // ── Visibility ─────────────────────────────────────────────────────────────

  toggle(): void {
    this._visible = !this._visible;
    if (this._root) {
      this._root.classList.toggle('hidden', !this._visible);
    }
  }

  show(): void {
    this._visible = true;
    this._root?.classList.remove('hidden');
  }

  hide(): void {
    this._visible = false;
    this._root?.classList.add('hidden');
  }

  // ── Two-way binding ────────────────────────────────────────────────────────

  /**
   * Programmatically update a param value and refresh its widget.
   * Does NOT fire a change event.
   */
  set(key: string, value: LiveValue): void {
    const entry = this.params.get(key);
    if (!entry) return;

    this._suppressEvents = true;
    this.values.set(key, cloneValue(value));
    this._refreshWidget(key, value);
    this._suppressEvents = false;
  }

  /**
   * Read the current live value of a param.
   */
  get(key: string): LiveValue | undefined {
    return this.values.get(key);
  }

  /**
   * Batch-set many values at once.  Triggers one refresh cycle but no
   * individual change events.
   */
  setBatch(updates: Record<string, LiveValue>): void {
    this._suppressEvents = true;
    for (const [key, val] of Object.entries(updates)) {
      this.values.set(key, cloneValue(val));
      this._refreshWidget(key, val);
    }
    this._suppressEvents = false;
  }

  // ── Events ─────────────────────────────────────────────────────────────────

  /** Register a change handler.  Returns an unsubscribe function. */
  on(event: 'change', handler: ChangeHandler): () => void {
    this._handlers.add(handler);
    return () => this._handlers.delete(handler);
  }

  /** Remove a previously registered handler. */
  off(_event: 'change', handler: ChangeHandler): void {
    this._handlers.delete(handler);
  }

  private _emit(evt: ParamChangeEvent): void {
    if (this._suppressEvents) return;
    for (const h of this._handlers) {
      try { h(evt); } catch { /* handler errors must not break the panel */ }
    }
    // Also dispatch a DOM CustomEvent for non-module consumers
    if (this._root) {
      this._root.dispatchEvent(new CustomEvent('ATUILParamChange', {
        bubbles: true,
        detail: evt,
      }));
    }
  }

  // ── Sections ───────────────────────────────────────────────────────────────

  showSection(name: PanelSection): void {
    this._collapsed.set(name, false);
    this._applyCollapseState(name);
  }

  hideSection(name: PanelSection): void {
    this._collapsed.set(name, true);
    this._applyCollapseState(name);
  }

  collapseAll(): void {
    for (const s of SECTION_ORDER) this._collapsed.set(s, true);
    for (const s of SECTION_ORDER) this._applyCollapseState(s);
  }

  expandAll(): void {
    for (const s of SECTION_ORDER) this._collapsed.set(s, false);
    for (const s of SECTION_ORDER) this._applyCollapseState(s);
  }

  private _applyCollapseState(section: PanelSection): void {
    if (!this._root) return;
    const body = this._root.querySelector<HTMLElement>(`[data-section-body="${section}"]`);
    const header = this._root.querySelector<HTMLElement>(`[data-section-header="${section}"]`);
    const collapsed = this._collapsed.get(section) ?? false;
    body?.classList.toggle('collapsed', collapsed);
    header?.classList.toggle('collapsed', collapsed);
  }

  // ── Search / filter ────────────────────────────────────────────────────────

  filterByKey(query: string): void {
    this._filterQuery = query.toLowerCase();
    this._applyFilter();
  }

  private _applyFilter(): void {
    if (!this._root) return;
    const q = this._filterQuery;
    let visible = 0;
    for (const [key, refs] of this._widgets) {
      const show = q === '' || key.toLowerCase().includes(q);
      refs.rowEl.classList.toggle('hidden', !show);
      if (show) visible++;
    }
    if (this._statusEl) {
      this._statusEl.textContent = q
        ? `Showing ${visible} / ${this.params.size} params matching "${q}"`
        : `${this.params.size} params across ${SECTION_ORDER.length} sections`;
    }
    // Also ensure section headers are visible if any child is visible
    for (const section of SECTION_ORDER) {
      const body = this._root.querySelector<HTMLElement>(`[data-section-body="${section}"]`);
      if (!body) continue;
      const anyVisible = Array.from(body.querySelectorAll('.at-uil-row'))
        .some(r => !r.classList.contains('hidden'));
      const sectionEl = body.closest('.at-uil-section') as HTMLElement | null;
      if (sectionEl) sectionEl.style.display = anyVisible || q === '' ? '' : 'none';
    }
  }

  // ── Presets ────────────────────────────────────────────────────────────────

  /**
   * Save current values as a named preset in localStorage.
   */
  savePreset(name: string): void {
    if (!name.trim()) return;
    const snapshot: PresetSnapshot = {};
    for (const [key, val] of this.values) {
      snapshot[key] = cloneValue(val);
    }
    localStorage.setItem(PRESET_STORAGE_PREFIX + name, JSON.stringify(snapshot));
    this._refreshPresetList();
    this._setStatus(`Preset "${name}" saved  (${Object.keys(snapshot).length} params)`);
  }

  /**
   * Load a named preset from localStorage and apply it to the panel.
   */
  loadPreset(name: string): boolean {
    const raw = localStorage.getItem(PRESET_STORAGE_PREFIX + name);
    if (!raw) {
      this._setStatus(`Preset "${name}" not found`);
      return false;
    }
    try {
      const snap: PresetSnapshot = JSON.parse(raw);
      this.setBatch(snap);
      this._setStatus(`Preset "${name}" loaded`);
      return true;
    } catch {
      this._setStatus(`Preset "${name}" parse error`);
      return false;
    }
  }

  /**
   * Delete a preset from localStorage.
   */
  deletePreset(name: string): void {
    localStorage.removeItem(PRESET_STORAGE_PREFIX + name);
    this._refreshPresetList();
    this._setStatus(`Preset "${name}" deleted`);
  }

  /** Return all stored preset names. */
  listPresets(): string[] {
    const names: string[] = [];
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (k && k.startsWith(PRESET_STORAGE_PREFIX)) {
        names.push(k.slice(PRESET_STORAGE_PREFIX.length));
      }
    }
    return names.sort();
  }

  /**
   * Export the current values (or a named preset) as a downloadable JSON file.
   */
  exportPreset(name = 'at-uil-export'): void {
    const snapshot: PresetSnapshot = {};
    for (const [key, val] of this.values) snapshot[key] = cloneValue(val);
    const blob = new Blob([JSON.stringify(snapshot, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${name}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    this._setStatus(`Exported "${name}.json"  (${Object.keys(snapshot).length} params)`);
  }

  /**
   * Import a preset from a JSON string.
   */
  importPreset(jsonStr: string, saveName?: string): boolean {
    try {
      const snap: PresetSnapshot = JSON.parse(jsonStr);
      this.setBatch(snap);
      if (saveName) {
        localStorage.setItem(PRESET_STORAGE_PREFIX + saveName, jsonStr);
        this._refreshPresetList();
      }
      this._setStatus(`Imported ${Object.keys(snap).length} params`);
      return true;
    } catch {
      this._setStatus('Import failed — invalid JSON');
      return false;
    }
  }

  /**
   * Reset all values to their defaults from the original uil-params.json.
   */
  resetToDefaults(): void {
    this._suppressEvents = true;
    for (const [key, entry] of this.params) {
      const def = cloneValue(entry.value);
      this.values.set(key, def);
      this._refreshWidget(key, def);
    }
    this._suppressEvents = false;
    this._setStatus('Reset to defaults');
  }

  // ── DOM building ───────────────────────────────────────────────────────────

  private _buildShell(): HTMLElement {
    const panel = document.createElement('div');
    panel.className = 'at-uil-panel';

    // ── Header ───────────────────────────────────────────────────────────────
    const header = document.createElement('div');
    header.className = 'at-uil-panel-header';

    const title = document.createElement('span');
    title.className = 'at-uil-panel-title';
    title.textContent = 'AT UIL LIVE PANEL';

    const toggleBtn = document.createElement('button');
    toggleBtn.className = 'at-uil-panel-toggle-btn';
    toggleBtn.textContent = '▼';
    toggleBtn.addEventListener('click', () => {
      const scroll = panel.querySelector('.at-uil-scroll') as HTMLElement | null;
      if (!scroll) return;
      const hidden = scroll.style.display === 'none';
      scroll.style.display = hidden ? '' : 'none';
      toggleBtn.textContent = hidden ? '▼' : '▶';
    });

    header.append(title, toggleBtn);
    panel.appendChild(header);

    // ── Toolbar ───────────────────────────────────────────────────────────────
    const toolbar = document.createElement('div');
    toolbar.className = 'at-uil-toolbar';

    const searchInput = document.createElement('input');
    searchInput.type = 'text';
    searchInput.placeholder = 'filter params…';
    searchInput.addEventListener('input', () => this.filterByKey(searchInput.value));

    const collapseAllBtn = document.createElement('button');
    collapseAllBtn.className = 'at-uil-btn';
    collapseAllBtn.textContent = '⊟';
    collapseAllBtn.title = 'Collapse all';
    collapseAllBtn.addEventListener('click', () => this.collapseAll());

    const expandAllBtn = document.createElement('button');
    expandAllBtn.className = 'at-uil-btn';
    expandAllBtn.textContent = '⊞';
    expandAllBtn.title = 'Expand all';
    expandAllBtn.addEventListener('click', () => this.expandAll());

    const resetBtn = document.createElement('button');
    resetBtn.className = 'at-uil-btn';
    resetBtn.textContent = '↺';
    resetBtn.title = 'Reset to defaults';
    resetBtn.addEventListener('click', () => this.resetToDefaults());

    const exportBtn = document.createElement('button');
    exportBtn.className = 'at-uil-btn';
    exportBtn.textContent = '⬇';
    exportBtn.title = 'Export preset JSON';
    exportBtn.addEventListener('click', () => this.exportPreset());

    toolbar.append(searchInput, collapseAllBtn, expandAllBtn, resetBtn, exportBtn);
    panel.appendChild(toolbar);

    // ── Preset row ────────────────────────────────────────────────────────────
    const presetRow = document.createElement('div');
    presetRow.className = 'at-uil-preset-row';

    const presetSelect = document.createElement('select');
    presetSelect.title = 'Saved presets';
    this._presetSelect = presetSelect;
    this._refreshPresetList();

    const savePresetBtn = document.createElement('button');
    savePresetBtn.className = 'at-uil-btn';
    savePresetBtn.textContent = 'Save';
    savePresetBtn.title = 'Save current preset';
    savePresetBtn.addEventListener('click', () => {
      const name = prompt('Preset name:', 'my-preset');
      if (name) this.savePreset(name);
    });

    const loadPresetBtn = document.createElement('button');
    loadPresetBtn.className = 'at-uil-btn';
    loadPresetBtn.textContent = 'Load';
    loadPresetBtn.title = 'Load selected preset';
    loadPresetBtn.addEventListener('click', () => {
      const name = presetSelect.value;
      if (name) this.loadPreset(name);
    });

    const delPresetBtn = document.createElement('button');
    delPresetBtn.className = 'at-uil-btn';
    delPresetBtn.textContent = '✕';
    delPresetBtn.title = 'Delete selected preset';
    delPresetBtn.addEventListener('click', () => {
      const name = presetSelect.value;
      if (name && confirm(`Delete preset "${name}"?`)) this.deletePreset(name);
    });

    presetRow.append(presetSelect, savePresetBtn, loadPresetBtn, delPresetBtn);
    panel.appendChild(presetRow);

    // ── Scroll area (populated later) ─────────────────────────────────────────
    const scroll = document.createElement('div');
    scroll.className = 'at-uil-scroll';
    scroll.id = 'at-uil-scroll';
    panel.appendChild(scroll);

    // ── Status bar ────────────────────────────────────────────────────────────
    const status = document.createElement('div');
    status.className = 'at-uil-status';
    status.textContent = 'Initialising…';
    this._statusEl = status;
    panel.appendChild(status);

    return panel;
  }

  /** (Re)render the scrollable section body after params are classified. */
  private _renderPanelBody(): void {
    if (!this._root) return;
    const scroll = this._root.querySelector('#at-uil-scroll') as HTMLElement;
    scroll.innerHTML = '';
    this._widgets.clear();

    // Group entries by section
    const bySection = new Map<PanelSection, UILParamEntry[]>();
    for (const s of SECTION_ORDER) bySection.set(s, []);

    for (const entry of this.params.values()) {
      bySection.get(entry.section)!.push(entry);
    }

    for (const section of SECTION_ORDER) {
      const entries = bySection.get(section)!;
      if (entries.length === 0) continue;

      const sectionEl = this._buildSection(section, entries);
      scroll.appendChild(sectionEl);
    }

    this._setStatus(`${this.params.size} params across ${SECTION_ORDER.length} sections`);
  }

  private _buildSection(section: PanelSection, entries: UILParamEntry[]): HTMLElement {
    const wrap = document.createElement('div');
    wrap.className = 'at-uil-section';

    // Header
    const header = document.createElement('div');
    header.className = 'at-uil-section-header';
    header.dataset.sectionHeader = section;

    const arrow = document.createElement('span');
    arrow.className = 'at-uil-section-arrow';
    arrow.textContent = '▾';

    const label = document.createElement('span');
    label.textContent = SECTION_LABELS[section];

    const count = document.createElement('span');
    count.className = 'at-uil-section-count';
    count.textContent = `${entries.length}`;

    header.append(arrow, label, count);
    header.addEventListener('click', () => {
      const col = this._collapsed.get(section) ?? false;
      this._collapsed.set(section, !col);
      this._applyCollapseState(section);
    });
    wrap.appendChild(header);

    // Body
    const body = document.createElement('div');
    body.className = 'at-uil-section-body';
    body.dataset.sectionBody = section;

    // Sort entries alphabetically
    entries.sort((a, b) => a.key.localeCompare(b.key));

    for (const entry of entries) {
      const row = this._buildRow(entry);
      body.appendChild(row);
    }

    wrap.appendChild(body);
    return wrap;
  }

  private _buildRow(entry: UILParamEntry): HTMLElement {
    const row = document.createElement('div');
    row.className = 'at-uil-row';
    row.dataset.paramKey = entry.key;

    // Short label: last 2 tokens
    const labelText = entry.key.split(/[_/]/).slice(-2).join('_');

    const labelEl = document.createElement('span');
    labelEl.className = 'at-uil-label';
    labelEl.textContent = labelText;
    labelEl.title = entry.key;
    row.appendChild(labelEl);

    const widget = document.createElement('div');
    widget.className = 'at-uil-widget';

    const currentVal = this.values.get(entry.key) ?? entry.value;
    const refs: WidgetRefs = { rowEl: row, inputs: [] };

    switch (entry.widgetType) {
      case 'slide':
        this._buildSlideWidget(entry, widget, currentVal as number, refs);
        break;
      case 'vec':
        this._buildVecWidget(entry, widget, currentVal as number[], refs);
        break;
      case 'color':
        this._buildColorWidget(entry, widget, currentVal as string, refs);
        break;
      case 'bool':
        this._buildBoolWidget(entry, widget, currentVal as boolean, refs);
        break;
    }

    row.appendChild(widget);
    this._widgets.set(entry.key, refs);
    return row;
  }

  // ── Widget builders ────────────────────────────────────────────────────────

  private _buildSlideWidget(
    entry: UILParamEntry,
    container: HTMLElement,
    value: number,
    refs: WidgetRefs,
  ): void {
    const wrap = document.createElement('div');
    wrap.className = 'at-uil-slide';

    const min = entry.min ?? -10;
    const max = entry.max ?? 10;
    const step = entry.step ?? 0.01;

    const clampedVal = Math.min(max, Math.max(min, value));

    const range = document.createElement('input');
    range.type = 'range';
    range.min = String(min);
    range.max = String(max);
    range.step = String(step);
    range.value = String(clampedVal);

    const num = document.createElement('input');
    num.type = 'number';
    num.className = 'at-uil-num-input';
    num.step = String(step);
    num.value = String(round(value));

    range.addEventListener('input', () => {
      const v = parseFloat(range.value);
      num.value = String(round(v));
      this._onWidgetChange(entry.key, v);
    });

    num.addEventListener('change', () => {
      const v = parseFloat(num.value);
      if (!isNaN(v)) {
        const clamped = Math.min(max, Math.max(min, v));
        range.value = String(clamped);
        num.value = String(round(v));
        this._onWidgetChange(entry.key, v);
      }
    });

    refs.inputs = [range, num];

    wrap.append(range, num);
    container.appendChild(wrap);
  }

  private _buildVecWidget(
    entry: UILParamEntry,
    container: HTMLElement,
    value: number[],
    refs: WidgetRefs,
  ): void {
    const wrap = document.createElement('div');
    wrap.className = 'at-uil-vec';
    const inputs: HTMLInputElement[] = [];

    for (let i = 0; i < value.length; i++) {
      const num = document.createElement('input');
      num.type = 'number';
      num.className = 'at-uil-num-input';
      num.step = String(entry.step ?? 0.01);
      num.value = String(round(value[i]));
      num.placeholder = ['x', 'y', 'z', 'w'][i] ?? String(i);

      num.addEventListener('change', () => {
        const current = (this.values.get(entry.key) as number[]).slice();
        const v = parseFloat(num.value);
        if (!isNaN(v)) current[i] = v;
        this._onWidgetChange(entry.key, current);
      });

      inputs.push(num);
      wrap.appendChild(num);
    }

    refs.inputs = inputs;
    container.appendChild(wrap);
  }

  private _buildColorWidget(
    entry: UILParamEntry,
    container: HTMLElement,
    value: string,
    refs: WidgetRefs,
  ): void {
    const wrap = document.createElement('div');
    wrap.className = 'at-uil-color';

    const swatch = document.createElement('input');
    swatch.type = 'color';
    swatch.value = value.length === 7 ? value : '#ffffff';

    const hexInput = document.createElement('input');
    hexInput.type = 'text';
    hexInput.className = 'at-uil-color-hex';
    hexInput.value = value;
    hexInput.maxLength = 7;

    swatch.addEventListener('input', () => {
      hexInput.value = swatch.value;
      this._onWidgetChange(entry.key, swatch.value);
    });

    hexInput.addEventListener('change', () => {
      const v = hexInput.value;
      if (/^#[0-9a-fA-F]{6}$/.test(v)) {
        swatch.value = v;
        this._onWidgetChange(entry.key, v);
      }
    });

    refs.inputs = [swatch, hexInput];
    wrap.append(swatch, hexInput);
    container.appendChild(wrap);
  }

  private _buildBoolWidget(
    entry: UILParamEntry,
    container: HTMLElement,
    value: boolean,
    refs: WidgetRefs,
  ): void {
    const wrap = document.createElement('div');
    wrap.className = 'at-uil-bool';

    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = value;

    cb.addEventListener('change', () => {
      this._onWidgetChange(entry.key, cb.checked);
    });

    refs.inputs = [cb];
    wrap.appendChild(cb);
    container.appendChild(wrap);
  }

  // ── Widget refresh (programmatic set) ─────────────────────────────────────

  private _refreshWidget(key: string, value: LiveValue): void {
    const refs = this._widgets.get(key);
    const entry = this.params.get(key);
    if (!refs || !entry) return;

    switch (entry.widgetType) {
      case 'slide': {
        const v = value as number;
        const [range, num] = refs.inputs as HTMLInputElement[];
        if (range) range.value = String(v);
        if (num) num.value = String(round(v));
        break;
      }
      case 'vec': {
        const arr = value as number[];
        for (let i = 0; i < arr.length; i++) {
          const inp = refs.inputs[i] as HTMLInputElement | undefined;
          if (inp) inp.value = String(round(arr[i]));
        }
        break;
      }
      case 'color': {
        const hex = value as string;
        const [swatch, hexInput] = refs.inputs as HTMLInputElement[];
        if (swatch && /^#[0-9a-fA-F]{6}$/.test(hex)) swatch.value = hex;
        if (hexInput) hexInput.value = hex;
        break;
      }
      case 'bool': {
        const cb = refs.inputs[0] as HTMLInputElement | undefined;
        if (cb) cb.checked = value as boolean;
        break;
      }
    }
  }

  // ── Change dispatch ────────────────────────────────────────────────────────

  private _onWidgetChange(key: string, newValue: LiveValue): void {
    const prev = cloneValue(this.values.get(key) ?? (this.params.get(key)?.value ?? 0));
    this.values.set(key, cloneValue(newValue));
    const section = this.params.get(key)?.section ?? 'MISC';
    this._emit({ key, value: newValue, section, prev });
    this._setStatus(`${key.split(/[_/]/).slice(-2).join('/')} → ${_valueToString(newValue)}`);
  }

  // ── Preset list refresh ────────────────────────────────────────────────────

  private _refreshPresetList(): void {
    if (!this._presetSelect) return;
    const names = this.listPresets();
    this._presetSelect.innerHTML = '';
    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = names.length ? '— select preset —' : '(no presets)';
    this._presetSelect.appendChild(placeholder);
    for (const name of names) {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      this._presetSelect.appendChild(opt);
    }
  }

  // ── Status bar ─────────────────────────────────────────────────────────────

  private _setStatus(msg: string): void {
    if (this._statusEl) this._statusEl.textContent = msg;
  }

  // ── Snapshot utils ─────────────────────────────────────────────────────────

  /** Return a plain snapshot of all current values. */
  snapshot(): PresetSnapshot {
    const out: PresetSnapshot = {};
    for (const [k, v] of this.values) out[k] = cloneValue(v);
    return out;
  }

  /** Return only the params belonging to a given section. */
  sectionSnapshot(section: PanelSection): PresetSnapshot {
    const out: PresetSnapshot = {};
    for (const [k, entry] of this.params) {
      if (entry.section === section) out[k] = cloneValue(this.values.get(k) ?? entry.value);
    }
    return out;
  }

  /** Return params matching a substring query. */
  searchParams(query: string): UILParamEntry[] {
    const q = query.toLowerCase();
    return Array.from(this.params.values()).filter(e => e.key.toLowerCase().includes(q));
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Refs to DOM nodes for a single param row. */
interface WidgetRefs {
  rowEl: HTMLElement;
  inputs: (HTMLInputElement | HTMLElement)[];
}

/** Compact string representation for status bar display. */
function _valueToString(v: LiveValue): string {
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (typeof v === 'string')  return v;
  if (typeof v === 'number')  return round(v, 3).toString();
  return `[${(v as number[]).map(n => round(n, 3)).join(', ')}]`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Species ↔ UIL preset binding API
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Apply a species UIL preset to an existing ATUILLivePanel instance.
 *
 * @param panel   Initialised ATUILLivePanel
 * @param species The species identifier
 * @param blend   0–1 interpolation factor (1 = full preset, 0 = no change)
 *
 * @example
 *   applySpeciesPreset(panel, 'cil-bolt', 1.0);
 *   applySpeciesPreset(panel, 'cil-eye',  0.5); // half-blend
 */
export function applySpeciesPreset(
  panel: ATUILLivePanel,
  species: SpeciesId,
  blend = 1.0,
): void {
  const preset = SPECIES_UIL_PRESETS[species];
  if (!preset) {
    console.warn(`[ATUILLivePanel] Unknown species: "${species}"`);
    return;
  }

  const updates: Record<string, LiveValue> = {};

  for (const [key, targetValue] of Object.entries(preset.params)) {
    if (targetValue === undefined) continue;

    if (blend >= 1) {
      updates[key] = targetValue;
      continue;
    }

    // Blend with current value
    const currentValue = panel.get(key);
    if (currentValue === undefined) {
      updates[key] = targetValue;
      continue;
    }

    if (typeof targetValue === 'number' && typeof currentValue === 'number') {
      updates[key] = currentValue + (targetValue - currentValue) * blend;
    } else if (
      Array.isArray(targetValue) && Array.isArray(currentValue) &&
      targetValue.length === currentValue.length &&
      targetValue.every((x: unknown) => typeof x === 'number')
    ) {
      updates[key] = (currentValue as number[]).map(
        (c, i) => c + ((targetValue as number[])[i] - c) * blend,
      );
    } else {
      // Non-numeric (color, bool) — no blend, hard-switch
      updates[key] = blend >= 0.5 ? targetValue : currentValue;
    }
  }

  panel.setBatch(updates);
}

/**
 * Get the UIL params snapshot for a species without applying them.
 * Useful for preview or diff computation.
 */
export function getSpeciesPresetParams(
  species: SpeciesId,
): Record<string, LiveValue> | null {
  const preset = SPECIES_UIL_PRESETS[species];
  if (!preset) return null;
  const out: Record<string, LiveValue> = {};
  for (const [k, v] of Object.entries(preset.params)) {
    if (v !== undefined) out[k] = v;
  }
  return out;
}

/**
 * Diff two species presets: returns keys that differ and their values from each.
 */
export function diffSpeciesPresets(
  a: SpeciesId,
  b: SpeciesId,
): Array<{ key: string; a: LiveValue | undefined; b: LiveValue | undefined }> {
  const pA = SPECIES_UIL_PRESETS[a]?.params ?? {};
  const pB = SPECIES_UIL_PRESETS[b]?.params ?? {};
  const keys = new Set([...Object.keys(pA), ...Object.keys(pB)]);
  const diffs: Array<{ key: string; a: LiveValue | undefined; b: LiveValue | undefined }> = [];

  for (const key of keys) {
    const va = pA[key];
    const vb = pB[key];
    const same =
      va === vb ||
      (Array.isArray(va) && Array.isArray(vb) &&
       va.length === vb.length &&
       (va as number[]).every((x, i) => x === (vb as number[])[i]));
    if (!same) diffs.push({ key, a: va as LiveValue | undefined, b: vb as LiveValue | undefined });
  }
  return diffs;
}

// ─────────────────────────────────────────────────────────────────────────────
// UILModulePanel — module-aware sub-panel for fine-grained param control
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A lightweight wrapper that augments ATUILLivePanel with module-aware
 * filtering and species preset switching in the browser DevTools panel.
 *
 * Usage:
 *   const modPanel = new UILModulePanel(panel);
 *   modPanel.mount(document.body);
 *   modPanel.activateSpecies('cil-bolt');
 *   modPanel.filterModule('BLOOM');
 */
export class UILModulePanel {

  private _panel: ATUILLivePanel;
  private _activeSpecies: SpeciesId | null = null;
  private _activeModule: UILModule | null = null;
  private _moduleEl: HTMLElement | null = null;

  constructor(panel: ATUILLivePanel) {
    this._panel = panel;
  }

  /**
   * Apply a species preset to the underlying ATUILLivePanel.
   * Optionally filter the panel view to show only that species' modules.
   */
  activateSpecies(species: SpeciesId, blend = 1.0, autoFilter = true): void {
    this._activeSpecies = species;
    applySpeciesPreset(this._panel, species, blend);

    if (autoFilter) {
      const preset = SPECIES_UIL_PRESETS[species];
      if (preset?.modules.length) {
        const sectionsToShow = new Set<PanelSection>(
          preset.modules.map(m => MODULE_TO_SECTION[m]),
        );
        const ALL_SECTIONS: PanelSection[] = [
          'CAMERA', 'POST_PROCESS', 'VOLUMETRIC_LIGHT', 'LIGHTS',
          'SHADERS', 'PARTICLES', 'SHADOWS', 'MESH', 'MISC',
        ];
        for (const section of ALL_SECTIONS) {
          if (sectionsToShow.has(section)) {
            this._panel.showSection(section);
          } else {
            this._panel.hideSection(section);
          }
        }
      }
    }

    this._updateModuleDisplay();
  }

  /** Filter panel to show only params belonging to a specific module. */
  filterModule(module: UILModule | null): void {
    this._activeModule = module;
    if (module === null) {
      this._panel.filterByKey('');
    } else {
      const keyFragment = _moduleKeyFragment(module);
      this._panel.filterByKey(keyFragment);
    }
    this._updateModuleDisplay();
  }

  /** Get the active species. */
  get activeSpecies(): SpeciesId | null { return this._activeSpecies; }

  /** Get the active module filter. */
  get activeModule(): UILModule | null { return this._activeModule; }

  /** Render a small species/module badge HUD near the main panel. */
  mount(container: HTMLElement = document.body): void {
    const el = document.createElement('div');
    el.id = 'at-uil-module-panel';
    el.style.cssText = [
      'position:fixed;top:10px;right:340px;z-index:99999',
      'background:#0d0d0f;border:1px solid #2a2a38;border-radius:6px',
      'padding:8px;font:11px "SF Mono",monospace;color:#c8c8d0',
      'min-width:160px;box-shadow:0 4px 16px rgba(0,0,0,0.6)',
    ].join(';');
    this._moduleEl = el;
    container.appendChild(el);
    this._updateModuleDisplay();
  }

  unmount(): void {
    this._moduleEl?.remove();
    this._moduleEl = null;
  }

  private _updateModuleDisplay(): void {
    if (!this._moduleEl) return;
    const sp = this._activeSpecies;
    const mod = this._activeModule;
    const preset = sp ? SPECIES_UIL_PRESETS[sp] : null;

    const lines: string[] = [
      `<div style="color:#9090ff;font-weight:700;margin-bottom:4px">UIL Modules</div>`,
    ];

    if (sp) {
      lines.push(`<div style="color:#60c060;margin-bottom:4px">🔵 ${sp}</div>`);
    }

    for (const [m, meta] of Object.entries(MODULE_META) as [UILModule, typeof MODULE_META[UILModule]][]) {
      const active = sp && preset?.modules.includes(m);
      const current = mod === m;
      const color = current ? '#ffffa0' : active ? '#a0c0ff' : '#404060';
      lines.push(
        `<div style="color:${color};cursor:pointer;padding:1px 0"` +
        ` onclick="window._uilModPanel?.filterModule('${m}')"` +
        `>${meta.icon} ${meta.label} <span style="opacity:0.5">(~${meta.approxCount})</span></div>`,
      );
    }

    if (mod) {
      lines.push(
        `<div style="margin-top:4px;color:#80a0ff;cursor:pointer"` +
        ` onclick="window._uilModPanel?.filterModule(null)">↩ Clear filter</div>`,
      );
    }

    this._moduleEl.innerHTML = lines.join('');
    (window as unknown as Record<string, unknown>)._uilModPanel = this;
  }
}

/** Map UILModule to a substring fragment useful for filterByKey(). */
function _moduleKeyFragment(module: UILModule): string {
  switch (module) {
    case 'CAMERA':     return 'CAMERA_';
    case 'LIGHTING':   return 'L_Element';
    case 'BLOOM':      return 'Bloom';
    case 'VOLUMETRIC': return 'VolumetricLight';
    case 'COMPOSITE':  return 'Composite';
    case 'PARTICLE':   return 'am_';
    case 'SCENE':      return 'INPUT_Config';
    case 'SHADER':     return 'Shader';
    case 'MESH':       return 'MESH_';
    case 'SHADOW':     return 'SHADOW_';
    case 'MISC':       return 'UIL_';
    default:           return '';
  }
}
