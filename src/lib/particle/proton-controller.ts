/**
 * proton-controller.ts — AT Proton 行为控制器
 *
 * 管理 Antimatter+ behaviors，支持 AT UIL INPUT_P 的 preset 模式。
 *
 * AT INPUT_P preset 类型 (来源: at_uil_params.json, 182 keys, 12 particle systems):
 *   curl       — curl noise displacement (#require(curl.glsl))
 *   fluid      — mouse/screen-space fluid interaction (#require(glscreenprojection.glsl))
 *   pointcloud — sample positions from tPointCloud texture
 *   spline     — spline path following (#require(splineparticles.fs))
 *
 * AT INPUT_P 架构:
 *   behavior.executionOrder 控制 code block 拼接顺序
 *   每个 code block 有 preset + GLSL snippet + uniforms
 *   多个 preset 可组合 (e.g. home_scene: planeshape → curl → fluid → lerp)
 *
 * ProtonController 将 AT 的 behavior 层抽象为运行时可切换的 preset 配置:
 *   - 每个 preset 封装 AT 的 uniform 默认值 + GLSL code pattern
 *   - 支持单个 preset 或多 preset 组合 (behavior chain)
 *   - 通过 AntimatterParticleSystem 驱动 GPU ping-pong
 *   - uniform 可通过 setUniform() / tweenUniform() 实时调节
 *
 * AT parameter sources (at_uil_params.json):
 *   curl:
 *     uCurlNoiseScale: 1       — spatial frequency of curl noise
 *     uCurlTimeScale: 0        — time evolution rate
 *     uCurlNoiseSpeed: 0       — curl displacement strength
 *     uSpatialHashScale: 0.1   — per-particle spatial jitter (BodyCores)
 *     uTemporalHashScale: 0.1  — per-particle temporal jitter (BodyCores)
 *
 *   fluid:
 *     uMouseStrength: 1        — fluid interaction strength
 *     uProjMatrix: Cmat4       — projection matrix (computed)
 *     uProjNormalMatrix: Cmat4 — projection normal matrix (computed)
 *     uModelMatrix: Cmat4      — model matrix (computed)
 *     tFluidMask: Csampler2D   — fluid mask texture (computed)
 *     tFluid: Csampler2D       — fluid velocity texture (computed)
 *
 *   pointcloud:
 *     tPointCloud: Csampler2D  — point cloud positions texture (computed)
 *
 *   spline:
 *     uSplineThickness: 1      — thickness of spline tube
 *     uThicknessStep: [1, 1]   — thickness step range
 *     uThicknessSpeed: 0       — thickness animation speed
 *     uRangeThickness: 0       — range-based thickness
 *     uRangeScale: 1           — range scale
 *     uDistribution: 1         — distribution mode
 *     uDistributionRange: [1,1]— distribution range
 *     uExtrudeRandom: 1        — extrusion randomness
 *     uSCurlNoiseScale: 1      — spline curl scale (AT: WorkDetailParticles)
 *     uSCurlTimeScale: 0       — spline curl time scale
 *     uSCurlNoiseSpeed: 0      — spline curl speed
 *
 * References:
 *   src/lib/particle/particle-behavior.ts   — ParticleBehavior, BehaviorPreset types
 *   src/lib/particle/antimatter.ts          — AntimatterParticleSystem, AntimatterUniforms
 *   src/lib/renderers/proton-particles.ts   — ProtonSystem (PixiJS-level proton presets)
 *   channels/physics/at_uil_params.json     — raw AT UIL parameter dump
 */

import type {
  AntimatterParticleSystem,
  AntimatterUniforms,
} from './antimatter.js';

import type {
  BehaviorPreset,
  ParticleUniformValue,
} from './particle-behavior.js';

// ── Preset Uniform Defaults ─────────────────────────────────────────────────
// Direct transcription of AT's INPUT_P code block uniform defaults.
// Each preset's defaults come from the most representative AT system:
//   curl       → Element_19_home_scene / Element_0_BodyCores
//   fluid      → Element_19_home_scene / Element_0_WorkDetailParticles
//   pointcloud → Element_0_ParticleTest / Element_21_TreeScene
//   spline     → Element_0_WorkDetailParticles

/**
 * Curl preset uniforms — AT curl.glsl pattern.
 *
 * AT code (home_scene):
 *   #require(curl.glsl)
 *   vec3 curl = curlNoise(pos * uCurlNoiseScale*0.1 + (time * uCurlTimeScale * 0.1));
 *   target += curl * uCurlNoiseSpeed * 0.01 * HZ;
 *
 * BodyCores variant adds spatialHash + temporalHash for per-particle phase offset.
 */
export interface CurlPresetUniforms {
  /** Curl noise spatial frequency (AT: uCurlNoiseScale, default 1) */
  uCurlNoiseScale: number;
  /** Curl noise time evolution rate (AT: uCurlTimeScale, default 0) */
  uCurlTimeScale: number;
  /** Curl noise displacement strength (AT: uCurlNoiseSpeed, default 0) */
  uCurlNoiseSpeed: number;
  /** Per-particle spatial jitter scale (AT BodyCores: uSpatialHashScale, default 0.1) */
  uSpatialHashScale?: number;
  /** Per-particle temporal jitter scale (AT BodyCores: uTemporalHashScale, default 0.1) */
  uTemporalHashScale?: number;
}

/**
 * Fluid preset uniforms — AT screen-space fluid interaction.
 *
 * AT code:
 *   #require(glscreenprojection.glsl)
 *   vec3 mpos = vec3(uModelMatrix * vec4(pos, 1.0));
 *   vec2 screenUV = getProjection(mpos, uProjMatrix);
 *   vec3 flow = vec3(texture2D(tFluid, screenUV).xy, 0.0);
 *   applyNormal(flow, uProjNormalMatrix);
 *   target += flow * 0.0001 * uMouseStrength * HZ * texture2D(tFluidMask, screenUV).r;
 */
export interface FluidPresetUniforms {
  /** Fluid interaction strength (AT: uMouseStrength, default 1) */
  uMouseStrength: number;
}

/**
 * Pointcloud preset uniforms — AT point cloud texture sampling.
 *
 * AT code:
 *   vec3 pointShape = texture2D(tPointCloud, uv).xyz;
 *   vec3 target = pointShape;
 *
 * The tPointCloud texture is a computed sampler2D (provided at runtime).
 */
export interface PointcloudPresetUniforms {
  // No numeric uniforms — the only input is the tPointCloud texture (runtime).
  // Placeholder for consistency and future extension.
}

/**
 * Spline preset uniforms — AT spline path following.
 *
 * AT code (WorkDetailParticles):
 *   #require(curl.glsl)
 *   #require(splineparticles.fs)
 *   sRandom = random; sOrigin = origin;
 *   float travel = texture2D(tLife, vUv).z;
 *   vec3 target = getSplinePos(travel);
 *   if (uSetup > 0.5 || travel < 0.001) pos = target;
 *   pos += (target - pos) * 0.07 * HZ;
 */
export interface SplinePresetUniforms {
  /** Spline tube thickness (AT: uSplineThickness, default 1) */
  uSplineThickness: number;
  /** Thickness step range (AT: uThicknessStep, default [1, 1]) */
  uThicknessStep: [number, number];
  /** Thickness animation speed (AT: uThicknessSpeed, default 0) */
  uThicknessSpeed: number;
  /** Range-based thickness (AT: uRangeThickness, default 0) */
  uRangeThickness: number;
  /** Range scale (AT: uRangeScale, default 1) */
  uRangeScale: number;
  /** Distribution mode (AT: uDistribution, default 1) */
  uDistribution: number;
  /** Distribution range (AT: uDistributionRange, default [1, 1]) */
  uDistributionRange: [number, number];
  /** Extrusion randomness (AT: uExtrudeRandom, default 1) */
  uExtrudeRandom: number;
  /** Spline-local curl noise scale (AT: uSCurlNoiseScale, default 1) */
  uSCurlNoiseScale: number;
  /** Spline-local curl time scale (AT: uSCurlTimeScale, default 0) */
  uSCurlTimeScale: number;
  /** Spline-local curl speed (AT: uSCurlNoiseSpeed, default 0) */
  uSCurlNoiseSpeed: number;
}

/** Union of all preset uniform interfaces */
export type PresetUniforms =
  | CurlPresetUniforms
  | FluidPresetUniforms
  | PointcloudPresetUniforms
  | SplinePresetUniforms;

// ── Preset Default Values ───────────────────────────────────────────────────

const CURL_DEFAULTS: Readonly<CurlPresetUniforms> = {
  uCurlNoiseScale:    1,
  uCurlTimeScale:     0,
  uCurlNoiseSpeed:    0,
  uSpatialHashScale:  0.1,
  uTemporalHashScale: 0.1,
};

const FLUID_DEFAULTS: Readonly<FluidPresetUniforms> = {
  uMouseStrength: 1,
};

const POINTCLOUD_DEFAULTS: Readonly<PointcloudPresetUniforms> = {};

const SPLINE_DEFAULTS: Readonly<SplinePresetUniforms> = {
  uSplineThickness:    1,
  uThicknessStep:      [1, 1],
  uThicknessSpeed:     0,
  uRangeThickness:     0,
  uRangeScale:         1,
  uDistribution:       1,
  uDistributionRange:  [1, 1],
  uExtrudeRandom:      1,
  uSCurlNoiseScale:    1,
  uSCurlTimeScale:     0,
  uSCurlNoiseSpeed:    0,
};

/** Preset name → default uniforms map */
const PRESET_DEFAULTS: Record<string, Readonly<Record<string, unknown>>> = {
  curl:       CURL_DEFAULTS,
  fluid:      FLUID_DEFAULTS,
  pointcloud: POINTCLOUD_DEFAULTS,
  spline:     SPLINE_DEFAULTS,
};

// ── Preset Metadata ─────────────────────────────────────────────────────────

/**
 * Metadata for each AT behavior preset.
 *
 * AT's INPUT_P code blocks each have:
 *   - preset name → determines GLSL template and #require dependencies
 *   - GLSL code snippet → injected into the behavior shader main()
 *   - uniforms → per-block uniform declarations
 *
 * This table captures the structural metadata (not the GLSL itself,
 * which lives in particle-behavior.ts compileParticleShader).
 */
export interface PresetMeta {
  /** Preset identifier */
  name: BehaviorPreset;
  /** Human-readable display name */
  displayName: string;
  /** #require dependencies needed by this preset */
  requires: string[];
  /**
   * AT systems that use this preset (for reference).
   * Format: "Element_{N}_{SystemName}"
   */
  usedBy: string[];
  /** Default uniform names and types (for UI generation / validation) */
  uniformSchema: Record<string, {
    type: ParticleUniformValue['type'];
    default: number | string | number[];
  }>;
}

const PRESET_META: Record<string, PresetMeta> = {
  curl: {
    name: 'curl',
    displayName: 'Curl Noise',
    requires: ['curl.glsl'],
    usedBy: [
      'Element_0_BodyCores',
      'Element_0_LogoParticle',
      'Element_0_TubesInteraction',
      'Element_0_particleTest',
      'Element_19_home_scene',
      'Element_4_work_page',
    ],
    uniformSchema: {
      uCurlNoiseScale:    { type: 'number', default: 1 },
      uCurlTimeScale:     { type: 'number', default: 0 },
      uCurlNoiseSpeed:    { type: 'number', default: 0 },
      uSpatialHashScale:  { type: 'number', default: 0.1 },
      uTemporalHashScale: { type: 'number', default: 0.1 },
    },
  },

  fluid: {
    name: 'fluid',
    displayName: 'Mouse Fluid',
    requires: ['glscreenprojection.glsl'],
    usedBy: [
      'Element_0_LogoParticle',
      'Element_0_ParticleTest',
      'Element_0_WorkDetailParticles',
      'Element_19_home_scene',
      'Element_4_work_page',
      'Element_6_Work',
    ],
    uniformSchema: {
      uMouseStrength:    { type: 'number',   default: 1 },
      uProjMatrix:       { type: 'computed', default: 'Cmat4' },
      uProjNormalMatrix: { type: 'computed', default: 'Cmat4' },
      uModelMatrix:      { type: 'computed', default: 'Cmat4' },
      tFluidMask:        { type: 'computed', default: 'Csampler2D' },
      tFluid:            { type: 'computed', default: 'Csampler2D' },
    },
  },

  pointcloud: {
    name: 'pointcloud',
    displayName: 'Point Cloud',
    requires: [],
    usedBy: [
      'Element_0_ParticleTest',
      'Element_21_TreeScene',
      'Element_6_Work',
    ],
    uniformSchema: {
      tPointCloud: { type: 'computed', default: 'Csampler2D' },
    },
  },

  spline: {
    name: 'spline',
    displayName: 'Spline Path',
    requires: ['curl.glsl', 'splineparticles.fs'],
    usedBy: [
      'Element_0_WorkDetailParticles',
    ],
    uniformSchema: {
      uSplineThickness:   { type: 'number', default: 1 },
      uThicknessStep:     { type: 'array',  default: [1, 1] },
      uThicknessSpeed:    { type: 'number', default: 0 },
      uRangeThickness:    { type: 'number', default: 0 },
      uRangeScale:        { type: 'number', default: 1 },
      uDistribution:      { type: 'number', default: 1 },
      uDistributionRange: { type: 'array',  default: [1, 1] },
      uExtrudeRandom:     { type: 'number', default: 1 },
      uSCurlNoiseScale:   { type: 'number', default: 1 },
      uSCurlTimeScale:    { type: 'number', default: 0 },
      uSCurlNoiseSpeed:   { type: 'number', default: 0 },
    },
  },
};

// ── Behavior Chain ──────────────────────────────────────────────────────────

/**
 * A single entry in the behavior chain — one preset with its overrides.
 *
 * AT behavior.executionOrder controls the order code blocks are concatenated.
 * Each block maps to a preset with per-block uniform overrides.
 *
 * Example (home_scene): planeshape → curl → fluid → lerp
 * In our model: [
 *   { preset: 'curl', uniforms: { uCurlNoiseSpeed: 0.5 } },
 *   { preset: 'fluid', uniforms: { uMouseStrength: 2 } },
 * ]
 */
export interface BehaviorEntry {
  /** Preset type */
  preset: BehaviorPreset;
  /** Uniform overrides (merged with preset defaults) */
  uniforms?: Record<string, number | number[]>;
  /** Whether this entry is currently enabled */
  enabled?: boolean;
}

// ── Tween State ─────────────────────────────────────────────────────────────

interface ActiveTween {
  key: string;
  fromValue: number;
  toValue: number;
  duration: number;
  elapsed: number;
  easing: (t: number) => number;
  onComplete?: () => void;
}

// ── Easing functions ────────────────────────────────────────────────────────

/** Built-in easing functions matching AT's common tween patterns */
export const Easing = {
  linear:      (t: number) => t,
  easeInQuad:  (t: number) => t * t,
  easeOutQuad: (t: number) => t * (2 - t),
  easeInOut:   (t: number) => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t,
  easeOutCubic:(t: number) => (--t) * t * t + 1,
} as const;

export type EasingName = keyof typeof Easing;

// ── ProtonController ────────────────────────────────────────────────────────

/**
 * Configuration for creating a ProtonController.
 */
export interface ProtonControllerConfig {
  /** The AntimatterParticleSystem to drive */
  antimatter: AntimatterParticleSystem;
  /** Initial behavior chain (preset sequence with optional overrides) */
  behaviors?: BehaviorEntry[];
  /** Whether to auto-apply preset uniforms to antimatter on construction (default true) */
  autoApply?: boolean;
}

/**
 * ProtonController — AT Proton 行为控制器
 *
 * Manages Antimatter+ particle behaviors through the AT INPUT_P preset model.
 * Supports the four core presets (curl, fluid, pointcloud, spline) with
 * composable behavior chains, runtime uniform tweaking, and smooth transitions.
 *
 * The controller acts as a bridge between the high-level behavior configuration
 * (preset selection + uniform overrides) and the low-level AntimatterParticleSystem
 * (GPU ping-pong FBO uniforms).
 *
 * @example
 * ```ts
 * import { AntimatterParticleSystem } from '$lib/particle/antimatter';
 * import { ProtonController } from '$lib/particle/proton-controller';
 *
 * const am = new AntimatterParticleSystem({ gl, particleCount: 16384 });
 *
 * // Single preset
 * const ctrl = new ProtonController({
 *   antimatter: am,
 *   behaviors: [
 *     { preset: 'curl', uniforms: { uCurlNoiseSpeed: 0.5, uCurlNoiseScale: 2 } },
 *   ],
 * });
 *
 * // Composable chain (AT home_scene pattern: curl → fluid)
 * const ctrl2 = new ProtonController({
 *   antimatter: am,
 *   behaviors: [
 *     { preset: 'curl',  uniforms: { uCurlNoiseSpeed: 0.3, uCurlTimeScale: 0.1 } },
 *     { preset: 'fluid', uniforms: { uMouseStrength: 2 } },
 *   ],
 * });
 *
 * // Runtime tweaking
 * ctrl.setUniform('uCurlNoiseSpeed', 1.5);
 * ctrl.tweenUniform('uCurlNoiseScale', 3, 1000);
 *
 * // Switch presets at runtime
 * ctrl.setPreset('spline');
 *
 * // Per-frame update (drives tweens)
 * ctrl.update(delta);
 * ```
 */
export class ProtonController {
  /** The controlled AntimatterParticleSystem */
  readonly antimatter: AntimatterParticleSystem;

  /** Current behavior chain */
  private _behaviors: BehaviorEntry[];

  /** Merged uniform state (preset defaults + overrides + tweened values) */
  private _uniforms: Record<string, number | number[]>;

  /** Active uniform tweens */
  private _tweens: ActiveTween[] = [];

  /** Listener callbacks for behavior changes */
  private _listeners: Array<(behaviors: ReadonlyArray<BehaviorEntry>) => void> = [];

  constructor(config: ProtonControllerConfig) {
    this.antimatter = config.antimatter;
    this._behaviors = config.behaviors ? [...config.behaviors] : [];
    this._uniforms = {};

    // Build merged uniform state from behavior chain
    this._rebuildUniforms();

    // Auto-apply to antimatter system
    if (config.autoApply !== false) {
      this._applyToAntimatter();
    }
  }

  // ── Preset management ───────────────────────────────────────────────────

  /**
   * Set a single preset (replaces entire behavior chain).
   *
   * @param preset - Preset name ('curl' | 'fluid' | 'pointcloud' | 'spline')
   * @param uniforms - Optional uniform overrides
   *
   * @example
   * ```ts
   * ctrl.setPreset('curl', { uCurlNoiseSpeed: 0.8 });
   * ```
   */
  setPreset(
    preset: BehaviorPreset,
    uniforms?: Record<string, number | number[]>,
  ): void {
    this._behaviors = [{ preset, uniforms, enabled: true }];
    this._tweens.length = 0; // Cancel active tweens on preset change
    this._rebuildUniforms();
    this._applyToAntimatter();
    this._notifyListeners();
  }

  /**
   * Set the full behavior chain (replaces existing).
   *
   * @param behaviors - Array of behavior entries in execution order
   *
   * @example
   * ```ts
   * // AT home_scene pattern: curl → fluid
   * ctrl.setBehaviors([
   *   { preset: 'curl',  uniforms: { uCurlNoiseSpeed: 0.3 } },
   *   { preset: 'fluid', uniforms: { uMouseStrength: 2 } },
   * ]);
   * ```
   */
  setBehaviors(behaviors: BehaviorEntry[]): void {
    this._behaviors = [...behaviors];
    this._tweens.length = 0;
    this._rebuildUniforms();
    this._applyToAntimatter();
    this._notifyListeners();
  }

  /**
   * Add a behavior entry to the end of the chain.
   *
   * @param entry - Behavior entry to append
   */
  addBehavior(entry: BehaviorEntry): void {
    this._behaviors.push({ ...entry });
    this._rebuildUniforms();
    this._applyToAntimatter();
    this._notifyListeners();
  }

  /**
   * Remove a behavior entry by index.
   *
   * @param index - Index in the behavior chain to remove
   * @returns The removed entry, or undefined if index was out of range
   */
  removeBehavior(index: number): BehaviorEntry | undefined {
    if (index < 0 || index >= this._behaviors.length) return undefined;
    const [removed] = this._behaviors.splice(index, 1);
    this._rebuildUniforms();
    this._applyToAntimatter();
    this._notifyListeners();
    return removed;
  }

  /**
   * Enable or disable a behavior entry in the chain.
   *
   * @param index - Index in the behavior chain
   * @param enabled - Whether the behavior is active
   */
  setBehaviorEnabled(index: number, enabled: boolean): void {
    if (index < 0 || index >= this._behaviors.length) return;
    this._behaviors[index].enabled = enabled;
    this._rebuildUniforms();
    this._applyToAntimatter();
    this._notifyListeners();
  }

  /**
   * Get current behavior chain (read-only snapshot).
   */
  get behaviors(): ReadonlyArray<Readonly<BehaviorEntry>> {
    return this._behaviors;
  }

  /**
   * Get the list of active preset names in execution order.
   */
  get activePresets(): BehaviorPreset[] {
    return this._behaviors
      .filter(b => b.enabled !== false)
      .map(b => b.preset);
  }

  // ── Uniform management ──────────────────────────────────────────────────

  /**
   * Set a uniform value immediately.
   * Cancels any active tween on this uniform.
   *
   * This applies to the merged uniform state and also propagates to the
   * AntimatterParticleSystem's uniform set where the name matches
   * an AntimatterUniforms key.
   *
   * @param name - Uniform name (e.g. 'uCurlNoiseSpeed')
   * @param value - Numeric value or numeric array
   */
  setUniform(name: string, value: number | number[]): void {
    // Cancel any active tween on this key
    this._tweens = this._tweens.filter(tw => tw.key !== name);

    this._uniforms[name] = value;
    this._applyUniformToAntimatter(name, value);
  }

  /**
   * Get the current value of a uniform.
   *
   * @param name - Uniform name
   * @returns Current value, or undefined if not set
   */
  getUniform(name: string): number | number[] | undefined {
    return this._uniforms[name];
  }

  /**
   * Get all current uniforms as a read-only snapshot.
   */
  get uniforms(): Readonly<Record<string, number | number[]>> {
    return { ...this._uniforms };
  }

  /**
   * Tween a numeric uniform from its current value to a target over duration.
   *
   * @param name     - Uniform name (must be a scalar number uniform)
   * @param target   - Target value
   * @param duration - Duration in milliseconds
   * @param easing   - Easing function name or custom function (default: 'easeInOut')
   * @returns A Promise that resolves when the tween completes
   *
   * @example
   * ```ts
   * await ctrl.tweenUniform('uCurlNoiseSpeed', 2.0, 500, 'easeOutCubic');
   * ```
   */
  tweenUniform(
    name: string,
    target: number,
    duration: number,
    easing: EasingName | ((t: number) => number) = 'easeInOut',
  ): Promise<void> {
    // Cancel existing tween on this key
    this._tweens = this._tweens.filter(tw => tw.key !== name);

    const current = this._uniforms[name];
    const fromValue = typeof current === 'number' ? current : 0;
    const easingFn = typeof easing === 'function' ? easing : Easing[easing];

    return new Promise<void>((resolve) => {
      this._tweens.push({
        key: name,
        fromValue,
        toValue: target,
        duration: duration / 1000, // Convert ms to seconds for delta-based update
        elapsed: 0,
        easing: easingFn,
        onComplete: resolve,
      });
    });
  }

  // ── Per-frame update ────────────────────────────────────────────────────

  /**
   * Update active tweens. Call once per frame.
   *
   * @param delta - Time delta in seconds (same as AntimatterParticleSystem.update)
   */
  update(delta: number): void {
    if (this._tweens.length === 0) return;

    const completed: ActiveTween[] = [];

    for (const tw of this._tweens) {
      tw.elapsed += delta;
      const t = Math.min(tw.elapsed / tw.duration, 1);
      const eased = tw.easing(t);
      const value = tw.fromValue + (tw.toValue - tw.fromValue) * eased;

      this._uniforms[tw.key] = value;
      this._applyUniformToAntimatter(tw.key, value);

      if (t >= 1) {
        completed.push(tw);
      }
    }

    // Remove completed tweens and fire callbacks
    if (completed.length > 0) {
      this._tweens = this._tweens.filter(tw => !completed.includes(tw));
      for (const tw of completed) {
        tw.onComplete?.();
      }
    }
  }

  // ── Change listeners ────────────────────────────────────────────────────

  /**
   * Register a callback for behavior chain changes.
   *
   * @param callback - Called with the new behavior chain whenever it changes
   * @returns Unsubscribe function
   */
  onChange(
    callback: (behaviors: ReadonlyArray<BehaviorEntry>) => void,
  ): () => void {
    this._listeners.push(callback);
    return () => {
      this._listeners = this._listeners.filter(l => l !== callback);
    };
  }

  // ── Preset metadata queries ─────────────────────────────────────────────

  /**
   * Get metadata for a preset (display name, dependencies, AT systems).
   *
   * @param preset - Preset name
   * @returns PresetMeta or undefined if unknown
   */
  static getPresetMeta(preset: string): PresetMeta | undefined {
    return PRESET_META[preset];
  }

  /**
   * Get all available preset names.
   */
  static get availablePresets(): string[] {
    return Object.keys(PRESET_META);
  }

  /**
   * Get default uniform values for a preset.
   *
   * @param preset - Preset name
   * @returns Default uniforms or empty object for unknown presets
   */
  static getPresetDefaults(preset: string): Readonly<Record<string, unknown>> {
    return PRESET_DEFAULTS[preset] ?? {};
  }

  /**
   * Get the uniform schema for a preset (for UI generation / validation).
   *
   * @param preset - Preset name
   * @returns Schema record or undefined for unknown presets
   */
  static getUniformSchema(
    preset: string,
  ): Record<string, { type: ParticleUniformValue['type']; default: number | string | number[] }> | undefined {
    return PRESET_META[preset]?.uniformSchema;
  }

  // ── Snapshot / serialisation ────────────────────────────────────────────

  /**
   * Export the current controller state as a serialisable snapshot.
   * Useful for saving presets to UIL JSON or persisting state.
   */
  toSnapshot(): ProtonControllerSnapshot {
    return {
      behaviors: this._behaviors.map(b => ({
        preset: b.preset,
        uniforms: b.uniforms ? { ...b.uniforms } : undefined,
        enabled: b.enabled,
      })),
      uniforms: { ...this._uniforms },
    };
  }

  /**
   * Restore controller state from a snapshot.
   *
   * @param snapshot - Previously exported snapshot
   */
  fromSnapshot(snapshot: ProtonControllerSnapshot): void {
    this._behaviors = snapshot.behaviors.map(b => ({
      preset: b.preset,
      uniforms: b.uniforms ? { ...b.uniforms } : undefined,
      enabled: b.enabled,
    }));
    this._tweens.length = 0;
    this._rebuildUniforms();

    // Apply snapshot uniforms on top (they may include tweened state)
    if (snapshot.uniforms) {
      for (const [key, val] of Object.entries(snapshot.uniforms)) {
        this._uniforms[key] = val;
      }
    }

    this._applyToAntimatter();
    this._notifyListeners();
  }

  // ── Cleanup ─────────────────────────────────────────────────────────────

  /**
   * Cancel all active tweens and clear listeners.
   * Does NOT dispose the AntimatterParticleSystem (caller manages that).
   */
  dispose(): void {
    this._tweens.length = 0;
    this._listeners.length = 0;
  }

  // ── Private: rebuild merged uniforms from behavior chain ────────────────

  /**
   * Rebuild the merged uniform state from the behavior chain.
   *
   * Order: preset defaults (first → last in chain), then per-entry overrides
   * (first → last). Later entries take priority, matching AT's
   * behavior.executionOrder semantics.
   */
  private _rebuildUniforms(): void {
    const merged: Record<string, number | number[]> = {};

    for (const entry of this._behaviors) {
      if (entry.enabled === false) continue;

      // 1. Apply preset defaults
      const defaults = PRESET_DEFAULTS[entry.preset];
      if (defaults) {
        for (const [key, val] of Object.entries(defaults)) {
          if (typeof val === 'number') {
            merged[key] = val;
          } else if (Array.isArray(val)) {
            merged[key] = [...val] as number[];
          }
        }
      }

      // 2. Apply per-entry overrides
      if (entry.uniforms) {
        for (const [key, val] of Object.entries(entry.uniforms)) {
          if (typeof val === 'number') {
            merged[key] = val;
          } else if (Array.isArray(val)) {
            merged[key] = [...val];
          }
        }
      }
    }

    this._uniforms = merged;
  }

  // ── Private: apply uniforms to AntimatterParticleSystem ─────────────────

  /**
   * Map merged uniforms to AntimatterUniforms and push to the system.
   *
   * AT uniform mapping (ProtonController → AntimatterParticleSystem):
   *   uCurlNoiseScale  → uCurlScale     (spatial frequency)
   *   uCurlNoiseSpeed  → uCurlStrength  (displacement force)
   *   uCurlTimeScale   → uCurlSpeed     (time evolution)
   *
   * Antimatter-native uniforms (decay, decayRandom, originStrength, etc.)
   * are passed through directly if present in the merged state.
   */
  private _applyToAntimatter(): void {
    const am = this.antimatter;

    for (const [key, value] of Object.entries(this._uniforms)) {
      this._applyUniformToAntimatter(key, value);
    }

    // Ensure antimatter reads fresh uniform state
    void am;
  }

  /**
   * Apply a single uniform to the AntimatterParticleSystem.
   *
   * Handles the AT → Antimatter naming transform:
   *   INPUT_P curl preset uses uCurlNoiseScale/Speed/TimeScale
   *   AntimatterParticleSystem uses uCurlScale/Strength/Speed
   */
  private _applyUniformToAntimatter(
    key: string,
    value: number | number[],
  ): void {
    const am = this.antimatter;

    // Map AT INPUT_P names to AntimatterUniforms names
    const mapping = UNIFORM_MAPPING[key];
    const targetKey = mapping ?? key;

    // Only set if the key is a known AntimatterUniforms field
    if (isAntimatterUniformKey(targetKey) && typeof value === 'number') {
      am.setUniform(
        targetKey as keyof AntimatterUniforms,
        value as never,
      );
    }

    // Handle array uniforms that map to tuple fields
    if (targetKey === 'decayRandom' && Array.isArray(value) && value.length === 2) {
      am.setUniform('decayRandom', [value[0], value[1]] as [number, number]);
    }
  }

  private _notifyListeners(): void {
    for (const listener of this._listeners) {
      listener(this._behaviors);
    }
  }
}

// ── Snapshot type ───────────────────────────────────────────────────────────

export interface ProtonControllerSnapshot {
  behaviors: BehaviorEntry[];
  uniforms: Record<string, number | number[]>;
}

// ── Uniform name mapping (AT INPUT_P → AntimatterUniforms) ──────────────────

/**
 * AT INPUT_P preset uniforms don't always map 1:1 to AntimatterUniforms.
 * This table resolves the naming differences.
 *
 * AT INPUT_P (particle-behavior.ts)     → Antimatter (antimatter.ts)
 *   uCurlNoiseScale                      → uCurlScale
 *   uCurlNoiseSpeed                      → uCurlStrength
 *   uCurlTimeScale                       → uCurlSpeed
 *
 * The spline-local variants (uSCurlNoiseScale etc.) are not directly
 * mapped to Antimatter because they're used inside the behavior shader
 * code blocks, not as Antimatter-system-level uniforms.
 */
const UNIFORM_MAPPING: Record<string, string> = {
  uCurlNoiseScale: 'uCurlScale',
  uCurlNoiseSpeed: 'uCurlStrength',
  uCurlTimeScale:  'uCurlSpeed',
};

/** All known AntimatterUniforms keys (for type-safe setUniform calls) */
const ANTIMATTER_UNIFORM_KEYS = new Set<string>([
  'uTime', 'uDelta', 'uMaxCount',
  'decay', 'decayRandom',
  'uCurlScale', 'uCurlSpeed', 'uCurlStrength',
  'uOriginStrength',
  'HZ', 'uDPR', 'uSetup',
]);

function isAntimatterUniformKey(key: string): boolean {
  return ANTIMATTER_UNIFORM_KEYS.has(key);
}

// ── Factory functions ───────────────────────────────────────────────────────

/**
 * Create a ProtonController with a single curl noise preset.
 *
 * AT reference: Element_19_home_scene, Element_0_BodyCores
 *
 * @param antimatter - AntimatterParticleSystem instance
 * @param overrides  - Optional uniform overrides on top of curl defaults
 *
 * @example
 * ```ts
 * const ctrl = createCurlController(am, {
 *   uCurlNoiseScale: 2,
 *   uCurlNoiseSpeed: 0.5,
 *   uCurlTimeScale: 0.1,
 * });
 * ```
 */
export function createCurlController(
  antimatter: AntimatterParticleSystem,
  overrides?: Partial<CurlPresetUniforms>,
): ProtonController {
  return new ProtonController({
    antimatter,
    behaviors: [{
      preset: 'curl',
      uniforms: overrides as Record<string, number | number[]> | undefined,
      enabled: true,
    }],
  });
}

/**
 * Create a ProtonController with a single fluid preset.
 *
 * AT reference: Element_19_home_scene code_3, Element_0_WorkDetailParticles code_2
 *
 * @param antimatter - AntimatterParticleSystem instance
 * @param overrides  - Optional uniform overrides on top of fluid defaults
 */
export function createFluidController(
  antimatter: AntimatterParticleSystem,
  overrides?: Partial<FluidPresetUniforms>,
): ProtonController {
  return new ProtonController({
    antimatter,
    behaviors: [{
      preset: 'fluid',
      uniforms: overrides as Record<string, number | number[]> | undefined,
      enabled: true,
    }],
  });
}

/**
 * Create a ProtonController with a single pointcloud preset.
 *
 * AT reference: Element_0_ParticleTest code_1, Element_21_TreeScene code_1
 *
 * @param antimatter - AntimatterParticleSystem instance
 * @param overrides  - Optional uniform overrides
 */
export function createPointcloudController(
  antimatter: AntimatterParticleSystem,
  overrides?: Partial<PointcloudPresetUniforms>,
): ProtonController {
  return new ProtonController({
    antimatter,
    behaviors: [{
      preset: 'pointcloud',
      uniforms: overrides as Record<string, number | number[]> | undefined,
      enabled: true,
    }],
  });
}

/**
 * Create a ProtonController with a single spline preset.
 *
 * AT reference: Element_0_WorkDetailParticles code_1
 *
 * @param antimatter - AntimatterParticleSystem instance
 * @param overrides  - Optional uniform overrides on top of spline defaults
 *
 * @example
 * ```ts
 * const ctrl = createSplineController(am, {
 *   uSCurlNoiseSpeed: 5,
 *   uSCurlNoiseScale: 2,
 *   uThicknessSpeed: 1,
 * });
 * ```
 */
export function createSplineController(
  antimatter: AntimatterParticleSystem,
  overrides?: Partial<SplinePresetUniforms>,
): ProtonController {
  return new ProtonController({
    antimatter,
    behaviors: [{
      preset: 'spline',
      uniforms: overrides as Record<string, number | number[]> | undefined,
      enabled: true,
    }],
  });
}

/**
 * Create a ProtonController with a composable behavior chain.
 *
 * This mirrors AT's behavior.executionOrder pattern where multiple presets
 * are chained together (e.g. home_scene: planeshape → curl → fluid → lerp).
 *
 * @param antimatter - AntimatterParticleSystem instance
 * @param chain      - Array of [preset, overrides?] pairs
 *
 * @example
 * ```ts
 * // AT home_scene pattern (simplified)
 * const ctrl = createCompositeController(am, [
 *   ['curl',  { uCurlNoiseSpeed: 0.3, uCurlTimeScale: 0.1 }],
 *   ['fluid', { uMouseStrength: 2 }],
 * ]);
 * ```
 */
export function createCompositeController(
  antimatter: AntimatterParticleSystem,
  chain: Array<[BehaviorPreset, Record<string, number | number[]>?]>,
): ProtonController {
  return new ProtonController({
    antimatter,
    behaviors: chain.map(([preset, uniforms]) => ({
      preset,
      uniforms,
      enabled: true,
    })),
  });
}
