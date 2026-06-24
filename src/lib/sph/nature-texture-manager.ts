// src/lib/sph/nature-texture-manager.ts
//
// M753: Unified registry and lifecycle manager for all nature-texture generators.
//
// The codebase contains seven independent natural-texture systems, each with
// its own initialisation dance, GPU resource lifecycle, and tick/step/generate
// calling convention:
//
//   ReactionDiffusionSim  — Gray-Scott reaction-diffusion (constructor + async init + step)
//   PhysarumSimulation    — slime-mould trail map          (static create + tick)
//   DifferentialGrowth    — organic fractal folds           (constructor + tick, CPU-only)
//   BoidsCompute          — flocking simulation             (constructor + tick + readback)
//   TuringPatternGenerator— Turing-pattern texture          (constructor + async generate)
//   NaturalPatternGenerator— Voronoi/Worley cell textures  (constructor + async generate)
//   Morphogenesis         — L-system plant growth           (constructor + generate, CPU-only)
//
// NatureTextureManager provides:
//   • A typed registry so callers refer to generators by a NatureTextureKind
//     string rather than importing seven different modules.
//   • Unified async createGenerator() / destroyGenerator() that handles each
//     generator's unique initialisation and teardown.
//   • destroyAll() for bulk cleanup (e.g. on scene unload).
//   • A has() / get() / list() introspection API so the render pipeline can
//     query which generators are currently alive.
//   • Type-safe narrow accessors (getReactionDiffusion(), getPhysarum(), etc.)
//     for callers that need the full per-generator API without casting.
//
// Design constraints:
//   • The manager does NOT own the GPUDevice — the caller passes it in.
//   • Each generator kind is a singleton within one manager instance; calling
//     createGenerator() a second time for the same kind destroys the old one
//     first (preventing GPU resource leaks).
//   • CPU-only generators (DifferentialGrowth, Morphogenesis) don't require a
//     GPUDevice and can be created even when WebGPU is unavailable.







import {
} from './reaction-diffusion';

} from './physarum-sim';
} from './differential-growth';
} from './boids-compute';
} from './turing-pattern';
} from './natural-patterns';
} from './morphogenesis';

  ReactionDiffusionSim,
  type RDSimConfig,

  PhysarumSimulation,
  type PhysarumParams,

  DifferentialGrowth,
  type DifferentialGrowthConfig,

  BoidsCompute,
  type BoidsParams,

  TuringPatternGenerator,
  type TuringPatternParams,

  NaturalPatternGenerator,
  type NaturalPatternParams,

  Morphogenesis,
  type MorphogenesisConfig,

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/** Discriminated union of all supported nature-texture generator kinds. */
export type NatureTextureKind =
  | 'reaction-diffusion'
  | 'physarum'
  | 'differential-growth'
  | 'boids'
  | 'turing-pattern'
  | 'natural-pattern'
  | 'morphogenesis';

/** Union of all generator instance types managed by the registry. */
export type NatureTextureInstance =
  | ReactionDiffusionSim
  | PhysarumSimulation
  | DifferentialGrowth
  | BoidsCompute
  | TuringPatternGenerator
  | NaturalPatternGenerator
  | Morphogenesis;

/**
 * Per-kind configuration for createGenerator().
 *
 * Each key maps a NatureTextureKind to the config object its constructor or
 * factory method expects.  GPU-based generators also accept width/height
 * overrides via their native config types.
 */
export interface NatureTextureConfigMap {
  'reaction-diffusion': RDSimConfig;
  'physarum':           PhysarumCreateConfig;
  'differential-growth': DifferentialGrowthConfig;
  'boids':             BoidsParams;
  'turing-pattern':    TuringPatternParams;
  'natural-pattern':   NaturalPatternParams;
  'morphogenesis':     MorphogenesisConfig;
}

/**
 * PhysarumSimulation.create() takes positional args rather than a config
 * object, so we wrap them here for uniform ergonomics.
 */
export interface PhysarumCreateConfig {
  width?:      number;
  height?:     number;
  agentCount?: number;
  params?:     Partial<PhysarumParams>;
}

/**
 * Map from NatureTextureKind → concrete class type.
 * Used by the narrow get*() accessors for compile-time safety.
 */
export interface NatureTextureInstanceMap {
  'reaction-diffusion':  ReactionDiffusionSim;
  'physarum':            PhysarumSimulation;
  'differential-growth': DifferentialGrowth;
  'boids':               BoidsCompute;
  'turing-pattern':      TuringPatternGenerator;
  'natural-pattern':     NaturalPatternGenerator;
  'morphogenesis':       Morphogenesis;
}

/** Metadata returned by list() for each active generator. */
export interface GeneratorEntry {
  kind:      NatureTextureKind;
  instance:  NatureTextureInstance;
  /** Whether this generator requires a GPUDevice. */
  gpuBased:  boolean;
  /** Timestamp (ms) when the generator was created. */
  createdAt: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Kinds that run on the GPU and require a device. */
const GPU_KINDS = new Set<NatureTextureKind>([
  'reaction-diffusion',
  'physarum',
  'boids',
  'turing-pattern',
  'natural-pattern',
]);

/** Kinds that are CPU-only (no GPUDevice needed). */
const CPU_KINDS = new Set<NatureTextureKind>([
  'differential-growth',
  'morphogenesis',
]);

/** All valid kinds — used for input validation. */
const ALL_KINDS = new Set<NatureTextureKind>([...GPU_KINDS, ...CPU_KINDS]);

// ─────────────────────────────────────────────────────────────────────────────
// NatureTextureManager
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Unified registry and lifecycle manager for nature-texture generators.
 *
 * Usage:
 * ```ts
 * const mgr = new NatureTextureManager(device);
 *
 * // Create generators on demand
 * await mgr.createGenerator('reaction-diffusion', { width: 512, height: 512 });
 * await mgr.createGenerator('physarum', { width: 1024, height: 1024, agentCount: 500_000 });
 * mgr.createGenerator('differential-growth', { width: 800, height: 600 });
 *
 * // Query
 * mgr.has('physarum');                      // true
 * mgr.get('physarum');                      // PhysarumSimulation
 * mgr.getPhysarum();                        // PhysarumSimulation (narrow-typed)
 * mgr.list();                               // GeneratorEntry[]
 *
 * // Tear down one
 * mgr.destroyGenerator('physarum');
 *
 * // Tear down all (scene unload)
 * mgr.destroyAll();
 * ```
 */
export class NatureTextureManager {
  /** The WebGPU device shared by all GPU-based generators.  May be null if
   *  only CPU generators will be used. */
  private readonly device: GPUDevice | null;

  /** Active generator instances keyed by kind (at most one per kind). */
  private readonly registry = new Map<NatureTextureKind, {
    instance:  NatureTextureInstance;
    createdAt: number;
  }>();

  /**
   * @param device  WebGPU device used by GPU-based generators.
   *                Pass `null` if you only intend to use CPU generators
   *                (DifferentialGrowth, Morphogenesis).
   */
  constructor(device: GPUDevice | null) {
    this.device = device;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Factory — unified async creation
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Create and register a generator.
   *
   * If a generator of the same kind already exists it is destroyed first,
   * preventing GPU resource leaks from double-creation.
   *
   * @param kind    Generator kind string.
   * @param config  Kind-specific configuration (see NatureTextureConfigMap).
   * @returns       The created generator instance, narrowly typed.
   */
  async createGenerator<K extends NatureTextureKind>(
    kind:    K,
    config?: NatureTextureConfigMap[K],
  ): Promise<NatureTextureInstanceMap[K]> {
    if (!ALL_KINDS.has(kind)) {
      throw new Error(`NatureTextureManager: unknown kind "${kind}"`);
    }

    // Destroy existing instance of the same kind (singleton-per-kind)
    if (this.registry.has(kind)) {
      this.destroyGenerator(kind);
    }

    // Ensure GPU device is available for GPU-based generators
    if (GPU_KINDS.has(kind) && !this.device) {
      throw new Error(
        `NatureTextureManager: "${kind}" requires a GPUDevice but none was provided`,
      );
    }

    const instance = await this._instantiate(kind, config);
    this.registry.set(kind, { instance, createdAt: Date.now() });

    return instance as NatureTextureInstanceMap[K];
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Internal — per-kind instantiation
  // ─────────────────────────────────────────────────────────────────────────

  private async _instantiate<K extends NatureTextureKind>(
    kind:   K,
    config?: NatureTextureConfigMap[K],
  ): Promise<NatureTextureInstance> {
    const dev = this.device!;

    switch (kind) {

      // ── Reaction-Diffusion ────────────────────────────────────────────
      // constructor(device, cfg) + async init()
      case 'reaction-diffusion': {
        const cfg = (config ?? {}) as RDSimConfig;
        const sim = new ReactionDiffusionSim(dev, cfg);
        await sim.init();
        return sim;
      }

      // ── Physarum ──────────────────────────────────────────────────────
      // static async create(device, width, height, agentCount, params)
      case 'physarum': {
        const cfg = (config ?? {}) as PhysarumCreateConfig;
        return PhysarumSimulation.create(
          dev,
          cfg.width      ?? 1024,
          cfg.height     ?? 1024,
          cfg.agentCount ?? 1_000_000,
          cfg.params,
        );
      }

      // ── Differential Growth (CPU-only) ────────────────────────────────
      // constructor(cfg) — synchronous, no GPU
      case 'differential-growth': {
        const cfg = (config ?? {}) as DifferentialGrowthConfig;
        return new DifferentialGrowth(cfg);
      }

      // ── Boids ─────────────────────────────────────────────────────────
      // constructor(device, params) — synchronous init, pipelines built inline
      case 'boids': {
        const cfg = (config ?? { count: 4096 }) as BoidsParams;
        return new BoidsCompute(dev, cfg);
      }

      // ── Turing Pattern ────────────────────────────────────────────────
      // constructor(device) — generate() is called separately
      case 'turing-pattern': {
        return new TuringPatternGenerator(dev);
      }

      // ── Natural Pattern ───────────────────────────────────────────────
      // constructor(device) — generate() is called separately
      case 'natural-pattern': {
        return new NaturalPatternGenerator(dev);
      }

      // ── Morphogenesis (CPU-only) ──────────────────────────────────────
      // constructor(cfg)
      case 'morphogenesis': {
        const cfg = (config ?? {}) as MorphogenesisConfig;
        return new Morphogenesis(cfg);
      }

      default: {
        // Exhaustiveness guard
        const _exhaustive: never = kind;
        throw new Error(`NatureTextureManager: unhandled kind "${_exhaustive}"`);
      }
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Destroy — per-generator and bulk
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Destroy a single generator and remove it from the registry.
   *
   * Calls the generator's destroy() method if it has one, releasing GPU
   * buffers, textures, and pipelines.  CPU-only generators without a
   * destroy() method are simply dropped.
   *
   * No-op if the kind is not currently registered.
   */
  destroyGenerator(kind: NatureTextureKind): void {
    const entry = this.registry.get(kind);
    if (!entry) return;

    this._teardown(entry.instance);
    this.registry.delete(kind);
  }

  /**
   * Destroy all registered generators and clear the registry.
   * Intended for scene-unload or hot-module-replacement cleanup.
   */
  destroyAll(): void {
    for (const entry of this.registry.values()) {
      this._teardown(entry.instance);
    }
    this.registry.clear();
  }

  /** Call destroy() on a generator instance if the method exists. */
  private _teardown(instance: NatureTextureInstance): void {
    if ('destroy' in instance && typeof instance.destroy === 'function') {
      try {
        instance.destroy();
      } catch (err) {
        console.warn('NatureTextureManager: destroy() threw:', err);
      }
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Introspection
  // ─────────────────────────────────────────────────────────────────────────

  /** Whether a generator of the given kind is currently registered. */
  has(kind: NatureTextureKind): boolean {
    return this.registry.has(kind);
  }

  /**
   * Get the generator instance for a given kind, or `undefined` if not
   * registered.  For type-safe access prefer the narrow getters below.
   */
  get<K extends NatureTextureKind>(kind: K): NatureTextureInstanceMap[K] | undefined {
    const entry = this.registry.get(kind);
    return entry?.instance as NatureTextureInstanceMap[K] | undefined;
  }

  /** Number of currently active generators. */
  get size(): number {
    return this.registry.size;
  }

  /** List all active generators with metadata. */
  list(): GeneratorEntry[] {
    const entries: GeneratorEntry[] = [];
    for (const [kind, { instance, createdAt }] of this.registry) {
      entries.push({
        kind,
        instance,
        gpuBased: GPU_KINDS.has(kind),
        createdAt,
      });
    }
    return entries;
  }

  /** All registered kinds. */
  get activeKinds(): NatureTextureKind[] {
    return [...this.registry.keys()];
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Narrow typed accessors
  //
  // Each returns the concrete class type (or undefined) so callers can use
  // the full generator-specific API without casting.
  // ─────────────────────────────────────────────────────────────────────────

  getReactionDiffusion(): ReactionDiffusionSim | undefined {
    return this.get('reaction-diffusion');
  }

  getPhysarum(): PhysarumSimulation | undefined {
    return this.get('physarum');
  }

  getDifferentialGrowth(): DifferentialGrowth | undefined {
    return this.get('differential-growth');
  }

  getBoids(): BoidsCompute | undefined {
    return this.get('boids');
  }

  getTuringPattern(): TuringPatternGenerator | undefined {
    return this.get('turing-pattern');
  }

  getNaturalPattern(): NaturalPatternGenerator | undefined {
    return this.get('natural-pattern');
  }

  getMorphogenesis(): Morphogenesis | undefined {
    return this.get('morphogenesis');
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Static helpers
  // ─────────────────────────────────────────────────────────────────────────

  /** All valid NatureTextureKind values. */
  static readonly ALL_KINDS: readonly NatureTextureKind[] = [...ALL_KINDS];

  /** Kinds that require a GPUDevice. */
  static readonly GPU_KINDS: readonly NatureTextureKind[] = [...GPU_KINDS];

  /** Kinds that are CPU-only. */
  static readonly CPU_KINDS: readonly NatureTextureKind[] = [...CPU_KINDS];

  /** Check whether a kind requires a GPU device. */
  static isGpuBased(kind: NatureTextureKind): boolean {
    return GPU_KINDS.has(kind);
  }
}
