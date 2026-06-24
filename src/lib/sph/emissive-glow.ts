/**
 * emissive-glow.ts — M791: Self-Emissive Glow System
 * ─────────────────────────────────────────────────────────────────────────────
 * Species-driven self-emission system: each Cell emits coloured light based on
 * its species identity, and that light is projected onto surrounding particles.
 *
 * Two emission modes mapped to Transformer-analogy species:
 *
 *   softmax  → CONTINUOUS PULSE  (持续脉动发光)
 *     Low-frequency sinusoidal intensity modulation that breathes like a star.
 *     The softmax concentrates probability mass → sustained energy radiation.
 *     Pulse frequency driven by kinetic energy (peaked dist = faster breathing).
 *     Colour temperature: white-hot core → amber corona → deep red at low energy.
 *
 *   attention → HIGH-FREQ FLICKER  (高频闪烁)
 *     Multi-head attention distributes across the sequence in rapid bursts.
 *     Stochastic on/off flicker at 8–24 Hz (above saccade perception threshold),
 *     modulated by an attention "confidence" signal derived from local density.
 *     Colour: cyan-violet iridescent band matching the iridescence material.
 *
 * Other species receive a default AMBIENT GLOW mode — gentle, steady-state
 * emission at low intensity (no pulse, no flicker) so they participate in
 * the coloured light field without dominating it.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Architecture
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *  ┌─ Per cell ──────────────────────────────────────────────────────────────┐
 *  │  EmissiveProfile  (from species-shader-registry + cell-material-system) │
 *  │  → GlowMode (continuous_pulse | high_freq_flicker | ambient)          │
 *  │  → base emissive colour RGB (HDR, values > 1.0 allowed)              │
 *  │  → pulse/flicker parameters                                          │
 *  │  → light projection radius + falloff exponent                        │
 *  └─────────────────────────────────────────────────────────────────────────┘
 *             │
 *             ▼
 *  ┌─ EmissiveGlowSystem.tick(dt, world) ────────────────────────────────────┐
 *  │  1. Update per-cell emissive intensity (pulse / flicker / ambient)     │
 *  │  2. Project coloured light onto neighbour particles (spatial query)    │
 *  │  3. Write emissive buffer (Float32Array: RGBA per particle)           │
 *  │  4. Accumulate per-cell HDR contribution for bloom extraction          │
 *  └─────────────────────────────────────────────────────────────────────────┘
 *             │ emissiveBuffer (Float32Array)
 *             ▼
 *  ┌─ Bloom pipeline integration ────────────────────────────────────────────┐
 *  │  • emissiveBuffer is sampled by ParticleInstancer fragment shader      │
 *  │    as an additive HDR colour contribution (output.rgb += emissive.rgb) │
 *  │  • Pixels with luminance > bloom threshold feed ATBloomPostProcess     │
 *  │    Pass 10 in the RenderCompositor pipeline (threshold → blur → comp)  │
 *  │  • Per-cell bloom params (bloomScale, bloomRadius) are modulated by    │
 *  │    the instantaneous emissive intensity for temporal coherence          │
 *  └─────────────────────────────────────────────────────────────────────────┘
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Light projection algorithm
 * ─────────────────────────────────────────────────────────────────────────────
 * For each emitting cell C:
 *   1. Query spatial hash for particles within C's projection radius.
 *   2. For each neighbour particle P:
 *      a. Distance d = |P.pos - C.pos|
 *      b. Attenuation A = max(0, 1 - (d / radius)^falloff)
 *      c. Received light = C.emissiveColor * C.intensity * A
 *      d. Accumulate into P's emissive buffer slot (additive blending)
 *   3. Species-specific colour modulation:
 *      - softmax projects warm (amber–white) light
 *      - attention projects cool (cyan–violet) light
 *      - Colour mixing at overlap creates natural interference tones
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Upstream references
 * ─────────────────────────────────────────────────────────────────────────────
 *   src/lib/sph/cell-material-system.ts    — CellSpecies, SOFTMAX_MATERIAL, ATTENTION_MATERIAL
 *   src/lib/sph/species-shader-registry.ts — SpeciesShaderConfig, bloomStrength/Threshold/Radius
 *   src/lib/sph/physics-uniform-bridge.ts  — PhysicsUniforms, samplePhysicsForBody
 *   src/lib/sph/at-bloom-postprocess.ts    — ATBloomPostProcess, ATBloomParams
 *   src/lib/sph/world-stepper.ts           — World, Particle
 *   src/lib/sph/spatial-hash.ts            — SpatialHashGrid, buildNeighborLists
 *   src/lib/sph/particle-instancing.ts     — ParticleInstancer, INSTANCE_STRIDE
 *   src/lib/sph/render-compositor.ts       — Pass 10 bloom, Pass 3 geometry
 *   src/lib/sph/color-palette.ts           — RGB
 *   src/lib/sph/types.ts                   — MAX_PARTICLES
 *
 * Research: xiaodi #M791 — cell-pubsub-loop
 */




// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/** Floats per particle in the emissive buffer: R, G, B, A (HDR, linear). */



import type { World, Particle }           from './world-stepper';
import type { PhysicsUniforms }            from './physics-uniform-bridge';
import type { ATBloomParams }              from './at-bloom-postprocess';
import type { SpeciesShaderConfig }        from './species-shader-registry';
import type { CellSpecies }                from './cell-material-system';
import type { RGB }                        from './color-palette';
import { MAX_PARTICLES }                   from './types';
import { getSpeciesShaderConfig }          from './species-shader-registry';
import { samplePhysicsForBody }            from './physics-uniform-bridge';

<<<<<<< HEAD
// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

// [orphan-precise] /** Floats per particle in the emissive buffer: R, G, B, A (HDR, linear). */
=======
>>>>>>> ecb00e743307774715a4cdccaff74dfb0983baea
export const EMISSIVE_STRIDE = 4;

/** Maximum projection radius (world units) — clamp to prevent O(N²) blow-up. */
const MAX_PROJECTION_RADIUS = 6.0;

/** Default spatial hash cell size for neighbour queries. */
const HASH_CELL_SIZE = 2.0;

/** Lygia random scale constants (sin-less hash, from contact-sparks.ts). */
const RS_X = 0.1031;
const RS_Y = 0.1030;
const RS_Z = 0.0973;

// ─────────────────────────────────────────────────────────────────────────────
// Glow mode — the three emission behaviours
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Emission mode for a cell's self-glow.
 *
 *   continuous_pulse  — sinusoidal breathing (softmax: sustained energy)
 *   high_freq_flicker — stochastic rapid on/off (attention: burst distribution)
 *   ambient           — steady low glow (all other species: background light)
 */
export type GlowMode = 'continuous_pulse' | 'high_freq_flicker' | 'ambient';

// ─────────────────────────────────────────────────────────────────────────────
// Emissive profile — per-cell emission configuration
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Complete emission profile for a single cell.
 * Derived from species identity + physics state at registration time,
 * then modulated live by PhysicsUniforms each tick.
 */
export interface EmissiveProfile {
  /** Canonical cell identifier. */
  cellId: string;

  /** Cell species string (e.g. 'cil-eye', 'cil-bolt'). */
  species: string;

  /** Material species type ('attention', 'softmax', etc.). */
  materialSpecies: CellSpecies;

  /** Emission mode determined by species. */
  glowMode: GlowMode;

  // ── Base emission colour (HDR, linear) ────────────────────────────────────

  /** Base emissive colour (R, G, B). Values > 1.0 produce HDR bloom. */
  baseColor: [number, number, number];

  /** Peak emissive intensity multiplier (modulated by pulse/flicker). */
  peakIntensity: number;

  // ── Pulse params (continuous_pulse mode) ──────────────────────────────────

  /** Pulse frequency in Hz (softmax default: 0.8 Hz — slow stellar breathing). */
  pulseFreqHz: number;

  /** Pulse depth: 0 = no modulation, 1 = full on/off. Default 0.4. */
  pulseDepth: number;

  /**
   * Phase offset (radians) — prevents all softmax cells from pulsing in sync.
   * Derived from cellId hash at registration.
   */
  phaseOffset: number;

  // ── Flicker params (high_freq_flicker mode) ──────────────────────────────

  /** Flicker base rate in Hz (attention default: 12 Hz). */
  flickerRateHz: number;

  /** Flicker duty cycle: fraction of time the light is "on" (0–1). Default 0.6. */
  flickerDuty: number;

  /**
   * Confidence threshold for flicker suppression (0–1).
   * When local density is below this normalised threshold, flicker pauses
   * (attention head has no context to attend over).
   */
  flickerConfidenceThreshold: number;

  // ── Light projection ─────────────────────────────────────────────────────

  /** Radius (world units) within which this cell projects coloured light. */
  projectionRadius: number;

  /** Falloff exponent for distance attenuation: A = (1 - d/r)^falloff. */
  falloffExponent: number;

  // ── Live state (mutated each tick) ────────────────────────────────────────

  /** Current instantaneous emissive intensity [0, peakIntensity]. */
  currentIntensity: number;

  /** Accumulated phase for pulse oscillation (radians). */
  accumulatedPhase: number;

  /** Flicker state: is the light currently on? */
  flickerOn: boolean;

  /** Time since last flicker toggle (seconds). */
  flickerTimer: number;

  /** Current flicker interval (seconds) — randomised each toggle. */
  flickerInterval: number;

  /** Cell position (updated from world each tick). */
  posX: number;
  posY: number;

  // ── Bloom feedback ───────────────────────────────────────────────────────

  /** Runtime bloom scale modulation driven by emissive intensity. */
  bloomScaleMod: number;

  /** Runtime bloom radius modulation. */
  bloomRadiusMod: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Species → emission config mapping
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Species-to-glow-mode mapping.
 *
 * Design rationale:
 *   softmax  = sustained energy radiator → continuous pulse
 *   attention = rapid burst distribution  → high-frequency flicker
 *   all others = background participants  → gentle ambient glow
 */
const SPECIES_GLOW_MAP: Record<string, {
  materialSpecies: CellSpecies;
  mode: GlowMode;
}> = {
  // ── Transformer-analogy species ──────────────────────────────────────────
  'cil-eye':         { materialSpecies: 'attention',  mode: 'high_freq_flicker' },
  'cil-bolt':        { materialSpecies: 'ffn',        mode: 'ambient' },
  'cil-vector':      { materialSpecies: 'embedding',  mode: 'ambient' },
  'cil-plus':        { materialSpecies: 'layernorm',  mode: 'ambient' },
  'cil-arrow-right': { materialSpecies: 'ffn',        mode: 'ambient' },
  'cil-filter':      { materialSpecies: 'attention',  mode: 'high_freq_flicker' },
  'cil-code':        { materialSpecies: 'softmax',    mode: 'continuous_pulse' },
  'cil-layers':      { materialSpecies: 'layernorm',  mode: 'ambient' },
  'cil-loop':        { materialSpecies: 'softmax',    mode: 'continuous_pulse' },
  'cil-graph':       { materialSpecies: 'embedding',  mode: 'ambient' },
};

/**
 * Emissive colour palettes per mode (HDR linear RGB).
 *
 * Softmax: black-body gradient — white-hot centre (1.8, 1.7, 1.4) at peak,
 *   amber corona (1.2, 0.7, 0.15) at half, deep red (0.6, 0.12, 0.02) at low.
 *   Interpolated by current energy level.
 *
 * Attention: iridescent band — cyan-violet shimmer (0.5, 0.9, 1.6) at peak,
 *   violet (0.7, 0.3, 1.2) at half, deep blue (0.15, 0.1, 0.6) at low.
 *   Matches the thin-film interference colours from ATTENTION_MATERIAL.
 *
 * Ambient: warm neutral — (0.8, 0.75, 0.65) at peak, very subtle.
 */
const EMISSIVE_COLOR_RAMP_SOFTMAX: [number, number, number][] = [
  [0.60, 0.12, 0.02],   // low energy: deep red ember
  [1.20, 0.70, 0.15],   // mid energy: amber corona
  [1.80, 1.70, 1.40],   // high energy: white-hot core
];

const EMISSIVE_COLOR_RAMP_ATTENTION: [number, number, number][] = [
  [0.15, 0.10, 0.60],   // low confidence: deep blue
  [0.70, 0.30, 1.20],   // mid confidence: violet
  [0.50, 0.90, 1.60],   // high confidence: cyan-white
];

const EMISSIVE_COLOR_AMBIENT: [number, number, number] = [0.80, 0.75, 0.65];

// ─────────────────────────────────────────────────────────────────────────────
// Utility: deterministic hash for phase offset
// ─────────────────────────────────────────────────────────────────────────────

/**
 * FNV-1a-inspired string hash → [0, 2π) phase offset.
 * Ensures different cells don't pulse / flicker in lock-step.
 */
function hashToPhase(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return ((h >>> 0) / 0xffffffff) * Math.PI * 2;
}

/**
 * Lygia sin-less random (scalar → [0, 1)).
 * Ported from contact-sparks.ts / random.wgsl.
 */
function randomHash(seed: number): number {
  let p = seed * RS_X;
  p = p - Math.floor(p);
  p *= p + 33.33;
  p *= p + p;
  return p - Math.floor(p);
}

/**
 * Lerp between two RGB triplets.
 */
function lerpRGB(
  a: [number, number, number],
  b: [number, number, number],
  t: number,
): [number, number, number] {
  const s = 1 - t;
  return [a[0] * s + b[0] * t, a[1] * s + b[1] * t, a[2] * s + b[2] * t];
}

/**
 * Sample a 3-stop colour ramp at parameter t ∈ [0, 1].
 */
function sampleRamp(
  ramp: [number, number, number][],
  t: number,
): [number, number, number] {
  const clamped = Math.max(0, Math.min(1, t));
  if (clamped <= 0.5) {
    return lerpRGB(ramp[0], ramp[1], clamped * 2);
  }
  return lerpRGB(ramp[1], ramp[2], (clamped - 0.5) * 2);
}

// ─────────────────────────────────────────────────────────────────────────────
// EmissiveGlowSystem
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Configuration for the emissive glow system.
 */
export interface EmissiveGlowConfig {
  /** Maximum particles the emissive buffer can hold. Default: MAX_PARTICLES. */
  maxParticles?: number;

  /** Spatial hash cell size for neighbour lookup. Default: 2.0. */
  hashCellSize?: number;

  /**
   * Global emissive intensity multiplier.
   * Scales all emission before bloom. Useful for performance/aesthetic tuning.
   * @default 1.0
   */
  globalIntensity?: number;

  /**
   * Maximum number of light-projecting cells processed per tick.
   * Limits compute cost on dense scenes. Default: 64.
   */
  maxProjectors?: number;

  /**
   * Bloom feedback: multiply instantaneous emissive intensity into the
   * ATBloomPostProcess bloomScale uniform.
   * @default true
   */
  bloomFeedback?: boolean;
}

/** Defaults for EmissiveGlowConfig. */
export const EMISSIVE_GLOW_DEFAULTS: Required<EmissiveGlowConfig> = {
  maxParticles:    MAX_PARTICLES,
  hashCellSize:    HASH_CELL_SIZE,
  globalIntensity: 1.0,
  maxProjectors:   64,
  bloomFeedback:   true,
};

/**
 * Per-cell bloom feedback snapshot — consumed by render-compositor
 * to dynamically modulate ATBloomPostProcess params.
 */
export interface CellBloomFeedback {
  cellId: string;
  /** Emissive-driven bloom scale modifier (1.0 = no change). */
  bloomScaleMod: number;
  /** Emissive-driven bloom radius modifier (1.0 = no change). */
  bloomRadiusMod: number;
  /** Instantaneous emissive intensity [0, peak]. */
  intensity: number;
  /** Current emissive colour (HDR linear RGB). */
  color: [number, number, number];
}

/**
 * EmissiveGlowSystem
 *
 * Manages the self-emission lifecycle for all registered cells:
 *   1. Register cells → derive EmissiveProfile from species
 *   2. tick(dt, world) → update intensities, project light, write buffer
 *   3. Consumers read emissiveBuffer + bloomFeedback for rendering
 *
 * The system is designed to be plugged into the RenderCompositor pipeline
 * between Pass 3 (Cell Materials) and Pass 10 (Bloom):
 *
 *   compositor.tick() →
 *     ... Pass 3 (cell geo) ...
 *     emissiveGlow.tick(dt, world)  ← HERE
 *     ... Pass 6 (particle compositor reads emissiveBuffer) ...
 *     ... Pass 10 (bloom uses per-cell bloomScaleMod) ...
 */
export class EmissiveGlowSystem {

  // ── Configuration ──────────────────────────────────────────────────────────
  private readonly _cfg: Required<EmissiveGlowConfig>;

  // ── Cell registry ──────────────────────────────────────────────────────────
  private readonly _profiles = new Map<string, EmissiveProfile>();

  // ── Emissive buffer ────────────────────────────────────────────────────────
  /**
   * RGBA Float32 buffer: one vec4 per particle.
   * Layout: [R0, G0, B0, A0, R1, G1, B1, A1, …]
   * Values are HDR-linear, clamped to [0, 8.0] to stay within bloom range.
   * The A channel stores the total received light intensity (for debug/UI).
   */
  private _emissiveBuffer: Float32Array;

  /**
   * Public read-only view of the emissive buffer.
   * Updated each tick — consumers should read after tick() returns.
   */
  get emissiveBuffer(): Float32Array { return this._emissiveBuffer; }

  // ── Bloom feedback ─────────────────────────────────────────────────────────
  private readonly _bloomFeedback: CellBloomFeedback[] = [];

  /** Per-cell bloom feedback array — updated each tick. */
  get bloomFeedback(): readonly CellBloomFeedback[] { return this._bloomFeedback; }

  // ── Spatial hash (lightweight, rebuilt each tick) ─────────────────────────
  private _hashBuckets = new Map<number, number[]>();
  private _hashInvCell: number;

  // ── Timing ──────────────────────────────────────────────────────────────────
  private _elapsedTime = 0;

  // ── Stats ──────────────────────────────────────────────────────────────────
  private _lastProjectorCount = 0;
  private _lastLitParticleCount = 0;

  /** Number of cells that projected light last tick. */
  get projectorCount(): number { return this._lastProjectorCount; }
  /** Number of particles that received projected light last tick. */
  get litParticleCount(): number { return this._lastLitParticleCount; }

  // ─────────────────────────────────────────────────────────────────────────
  // Constructor
  // ─────────────────────────────────────────────────────────────────────────

  constructor(config?: EmissiveGlowConfig) {
    this._cfg = { ...EMISSIVE_GLOW_DEFAULTS, ...config };
    this._emissiveBuffer = new Float32Array(this._cfg.maxParticles * EMISSIVE_STRIDE);
    this._hashInvCell = 1.0 / this._cfg.hashCellSize;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Cell registration
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Register a cell for emissive glow tracking.
   *
   * Derives the EmissiveProfile from the species identity:
   *   species → glow mode + colour palette + pulse/flicker params
   *
   * @param cellId   Unique cell identifier.
   * @param species  Species string (e.g. 'cil-eye', 'cil-code').
   * @param posX     Initial X position (world units).
   * @param posY     Initial Y position (world units).
   */
  addCell(cellId: string, species: string, posX = 0, posY = 0): void {
    if (this._profiles.has(cellId)) return;

    const speciesInfo = SPECIES_GLOW_MAP[species] ?? {
      materialSpecies: 'embedding' as CellSpecies,
      mode: 'ambient' as GlowMode,
    };

    const shaderCfg = getSpeciesShaderConfig(species);
    const phaseOffset = hashToPhase(cellId);

    const profile: EmissiveProfile = {
      cellId,
      species,
      materialSpecies: speciesInfo.materialSpecies,
      glowMode:        speciesInfo.mode,

      // ── Base colour (derived below based on mode) ──────────────────────
      baseColor:       [0, 0, 0],
      peakIntensity:   0,

      // ── Pulse (continuous_pulse) ───────────────────────────────────────
      pulseFreqHz:     0.8,
      pulseDepth:      0.40,
      phaseOffset,

      // ── Flicker (high_freq_flicker) ────────────────────────────────────
      flickerRateHz:        12.0,
      flickerDuty:          0.60,
      flickerConfidenceThreshold: 0.25,

      // ── Light projection ───────────────────────────────────────────────
      projectionRadius: 3.0,
      falloffExponent:  2.0,

      // ── Live state ─────────────────────────────────────────────────────
      currentIntensity: 0,
      accumulatedPhase: phaseOffset,
      flickerOn:        true,
      flickerTimer:     0,
      flickerInterval:  1 / 12,
      posX,
      posY,

      // ── Bloom feedback ─────────────────────────────────────────────────
      bloomScaleMod:  1.0,
      bloomRadiusMod: 1.0,
    };

    // ── Mode-specific parameter overrides ────────────────────────────────

    switch (speciesInfo.mode) {
      case 'continuous_pulse': {
        // Softmax: HDR white-hot core, wide projection, slow pulse
        profile.baseColor      = [...EMISSIVE_COLOR_RAMP_SOFTMAX[2]];
        profile.peakIntensity  = shaderCfg.bloomStrength * 1.5;
        profile.pulseFreqHz    = shaderCfg.bloomPulseFrequency;
        profile.pulseDepth     = shaderCfg.bloomPulseAmplitude * 2.0;
        profile.projectionRadius = Math.min(
          shaderCfg.bloomRadius * 5.0,
          MAX_PROJECTION_RADIUS,
        );
        profile.falloffExponent = 1.8;
        break;
      }

      case 'high_freq_flicker': {
        // Attention: cyan-violet iridescence, tight projection, fast flicker
        profile.baseColor      = [...EMISSIVE_COLOR_RAMP_ATTENTION[2]];
        profile.peakIntensity  = shaderCfg.bloomStrength * 1.2;
        profile.flickerRateHz  = 12.0 + shaderCfg.bloomPulseFrequency * 6.0;
        profile.flickerDuty    = 0.55 + shaderCfg.bloomPulseAmplitude * 0.3;
        profile.projectionRadius = Math.min(
          shaderCfg.bloomRadius * 3.5,
          MAX_PROJECTION_RADIUS,
        );
        profile.falloffExponent = 2.5;
        // Initialise first flicker interval
        profile.flickerInterval = 1.0 / profile.flickerRateHz;
        break;
      }

      case 'ambient':
      default: {
        // Gentle steady glow, minimal projection
        const albedo = shaderCfg.materialParams.albedo ?? [0.5, 0.5, 0.5];
        profile.baseColor = [
          albedo[0] * 0.6 + EMISSIVE_COLOR_AMBIENT[0] * 0.4,
          albedo[1] * 0.6 + EMISSIVE_COLOR_AMBIENT[1] * 0.4,
          albedo[2] * 0.6 + EMISSIVE_COLOR_AMBIENT[2] * 0.4,
        ];
        profile.peakIntensity = shaderCfg.bloomStrength * 0.3;
        profile.projectionRadius = Math.min(
          shaderCfg.bloomRadius * 2.0,
          MAX_PROJECTION_RADIUS * 0.5,
        );
        profile.falloffExponent = 3.0;
        break;
      }
    }

    this._profiles.set(cellId, profile);
  }

  /**
   * Unregister a cell.
   */
  removeCell(cellId: string): void {
    this._profiles.delete(cellId);
  }

  /**
   * Returns whether a cell is registered.
   */
  hasCell(cellId: string): boolean {
    return this._profiles.has(cellId);
  }

  /**
   * Get the live EmissiveProfile for a cell (read-only snapshot).
   */
  getProfile(cellId: string): Readonly<EmissiveProfile> | undefined {
    return this._profiles.get(cellId);
  }

  /**
   * Get all registered cell IDs.
   */
  getCellIds(): string[] {
    return [...this._profiles.keys()];
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Main tick
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Advance the emissive glow system by one timestep.
   *
   * Pipeline:
   *   1. Clear emissive buffer
   *   2. Update each cell's intensity (pulse / flicker / ambient)
   *   3. Update cell positions from world particle data
   *   4. Build lightweight spatial hash over particles
   *   5. Project coloured light from emitting cells onto neighbours
   *   6. Compute per-cell bloom feedback
   *
   * @param dt     Delta time in seconds.
   * @param world  Current SPH World snapshot.
   */
  tick(dt: number, world: World): void {
    this._elapsedTime += dt;
    const particles = world.particles;
    const count = particles.length;

    // ── 1. Clear emissive buffer ─────────────────────────────────────────
    this._emissiveBuffer.fill(0);

    // ── 2 & 3. Update intensities + positions ────────────────────────────
    for (const profile of this._profiles.values()) {
      this._updateCellPosition(profile, particles);
      this._updateIntensity(profile, dt, world);
    }

    // ── 4. Build spatial hash over particles ─────────────────────────────
    this._buildSpatialHash(particles, count);

    // ── 5. Project light ─────────────────────────────────────────────────
    let projectorCount = 0;
    let litCount = 0;

    // Sort by peakIntensity descending so the budget goes to brightest cells
    const sortedProfiles = [...this._profiles.values()]
      .filter(p => p.currentIntensity > 0.01)
      .sort((a, b) => b.currentIntensity - a.currentIntensity);

    const maxProjectors = this._cfg.maxProjectors;

    for (const profile of sortedProfiles) {
      if (projectorCount >= maxProjectors) break;

      const lit = this._projectLight(profile, particles, count);
      if (lit > 0) {
        projectorCount++;
        litCount += lit;
      }
    }

    this._lastProjectorCount = projectorCount;
    this._lastLitParticleCount = litCount;

    // ── 6. Bloom feedback ────────────────────────────────────────────────
    this._updateBloomFeedback();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Intensity update (per-cell, per-tick)
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Update the cell's position from the nearest rigid body / particle.
   * Cells are rigid bodies in the SPH world — we find the particle whose
   * species matches and use the cluster centroid as the cell position.
   */
  private _updateCellPosition(profile: EmissiveProfile, particles: Particle[]): void {
    // Find particles matching this cell's species and compute centroid
    let cx = 0, cy = 0, n = 0;
    for (let i = 0, len = particles.length; i < len; i++) {
      const p = particles[i];
      if (p.species === profile.species) {
        cx += p.x;
        cy += p.y;
        n++;
      }
    }
    if (n > 0) {
      profile.posX = cx / n;
      profile.posY = cy / n;
    }
  }

  /**
   * Update the instantaneous emissive intensity based on glow mode.
   */
  private _updateIntensity(
    profile: EmissiveProfile,
    dt: number,
    world: World,
  ): void {
    const globalMul = this._cfg.globalIntensity;

    switch (profile.glowMode) {

      // ── CONTINUOUS PULSE (softmax) ────────────────────────────────────
      case 'continuous_pulse': {
        profile.accumulatedPhase += dt * profile.pulseFreqHz * Math.PI * 2;
        // Keep phase bounded to avoid floating-point drift
        if (profile.accumulatedPhase > 1e6) {
          profile.accumulatedPhase -= Math.floor(profile.accumulatedPhase / (Math.PI * 2)) * Math.PI * 2;
        }

        // Sinusoidal pulse: intensity oscillates between
        //   peak * (1 - depth) … peak * 1.0
        const sinVal = Math.sin(profile.accumulatedPhase);
        const pulse = 1.0 - profile.pulseDepth * 0.5 * (1.0 - sinVal);

        // Energy-level modulation: sample average kinetic energy of
        // nearby same-species particles to scale colour temperature
        const energyLevel = this._sampleLocalEnergy(profile, world.particles);
        const energyT = Math.max(0, Math.min(1, energyLevel));

        // Colour temperature shift along black-body ramp
        profile.baseColor = sampleRamp(EMISSIVE_COLOR_RAMP_SOFTMAX, energyT);

        profile.currentIntensity = profile.peakIntensity * pulse * globalMul;
        break;
      }

      // ── HIGH-FREQ FLICKER (attention) ─────────────────────────────────
      case 'high_freq_flicker': {
        profile.flickerTimer += dt;

        if (profile.flickerTimer >= profile.flickerInterval) {
          // Toggle on/off
          profile.flickerOn = !profile.flickerOn;
          profile.flickerTimer -= profile.flickerInterval;

          // Randomise next interval: jitter ±30% around base period
          const basePeriod = 1.0 / profile.flickerRateHz;
          const jitter = randomHash(this._elapsedTime * 1000 + hashToPhase(profile.cellId));
          profile.flickerInterval = basePeriod * (0.7 + jitter * 0.6);
        }

        // Confidence gate: suppress flicker if local density is too low
        const confidence = this._sampleLocalDensity(profile, world.particles);
        const gated = confidence > profile.flickerConfidenceThreshold;

        if (profile.flickerOn && gated) {
          // Duty cycle shapes the on-intensity (soft ramp, not hard step)
          const dutyRamp = Math.min(1.0, profile.flickerTimer / (profile.flickerInterval * profile.flickerDuty));
          const rampShape = 1.0 - Math.pow(1.0 - dutyRamp, 2.0); // ease-out

          // Colour shifts with confidence level
          const confT = Math.max(0, Math.min(1, confidence));
          profile.baseColor = sampleRamp(EMISSIVE_COLOR_RAMP_ATTENTION, confT);

          profile.currentIntensity = profile.peakIntensity * rampShape * globalMul;
        } else {
          // Off state — very dim residual (not full black, preserves spatial coherence)
          profile.currentIntensity = profile.peakIntensity * 0.05 * globalMul;
        }
        break;
      }

      // ── AMBIENT ───────────────────────────────────────────────────────
      case 'ambient':
      default: {
        // Steady glow with very gentle sine modulation (barely perceptible)
        const breathe = 1.0 + 0.08 * Math.sin(this._elapsedTime * 0.5 + profile.phaseOffset);
        profile.currentIntensity = profile.peakIntensity * breathe * globalMul;
        break;
      }
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Spatial hash
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Build a lightweight spatial hash over all particles.
   * The hash maps (cellX, cellY) → particle index list.
   * Used for O(1) neighbour queries during light projection.
   */
  private _buildSpatialHash(particles: Particle[], count: number): void {
    this._hashBuckets.clear();
    const inv = this._hashInvCell;

    for (let i = 0; i < count; i++) {
      const p = particles[i];
      const key = this._hashKey(Math.floor(p.x * inv), Math.floor(p.y * inv));
      let bucket = this._hashBuckets.get(key);
      if (!bucket) {
        bucket = [];
        this._hashBuckets.set(key, bucket);
      }
      bucket.push(i);
    }
  }

  /**
   * Spatial hash key: Cantor pairing function on signed grid coords.
   */
  private _hashKey(cx: number, cy: number): number {
    // Shift to unsigned for pairing
    const a = cx >= 0 ? cx * 2 : -cx * 2 - 1;
    const b = cy >= 0 ? cy * 2 : -cy * 2 - 1;
    return ((a + b) * (a + b + 1)) / 2 + b;
  }

  /**
   * Query spatial hash for particle indices within a radius around (px, py).
   */
  private _queryRadius(
    px: number,
    py: number,
    radius: number,
    particles: Particle[],
    out: number[],
  ): void {
    out.length = 0;
    const inv = this._hashInvCell;
    const cellRadius = Math.ceil(radius * inv);

    const cx0 = Math.floor(px * inv);
    const cy0 = Math.floor(py * inv);
    const r2 = radius * radius;

    for (let dx = -cellRadius; dx <= cellRadius; dx++) {
      for (let dy = -cellRadius; dy <= cellRadius; dy++) {
        const key = this._hashKey(cx0 + dx, cy0 + dy);
        const bucket = this._hashBuckets.get(key);
        if (!bucket) continue;

        for (let k = 0; k < bucket.length; k++) {
          const idx = bucket[k];
          const p = particles[idx];
          const ddx = p.x - px;
          const ddy = p.y - py;
          if (ddx * ddx + ddy * ddy <= r2) {
            out.push(idx);
          }
        }
      }
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Light projection
  // ─────────────────────────────────────────────────────────────────────────

  /** Reusable array for query results (avoids GC pressure). */
  private readonly _queryBuf: number[] = [];

  /**
   * Project coloured light from a single emitting cell onto nearby particles.
   *
   * For each neighbour within projectionRadius:
   *   received = emissiveColor × intensity × attenuation(distance)
   *   emissiveBuffer[idx] += received  (additive blending)
   *
   * @returns Number of particles that received light.
   */
  private _projectLight(
    profile: EmissiveProfile,
    particles: Particle[],
    count: number,
  ): number {
    if (profile.currentIntensity <= 0.001) return 0;

    const buf = this._emissiveBuffer;
    const radius = profile.projectionRadius;
    const falloff = profile.falloffExponent;
    const invR = 1.0 / Math.max(0.001, radius);
    const intensity = profile.currentIntensity;
    const [cr, cg, cb] = profile.baseColor;

    // Query neighbours
    this._queryRadius(
      profile.posX,
      profile.posY,
      radius,
      particles,
      this._queryBuf,
    );

    let litCount = 0;

    for (let k = 0; k < this._queryBuf.length; k++) {
      const idx = this._queryBuf[k];
      if (idx >= this._cfg.maxParticles) continue;

      const p = particles[idx];
      const dx = p.x - profile.posX;
      const dy = p.y - profile.posY;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const normDist = dist * invR;

      // Distance attenuation: A = max(0, (1 - normDist))^falloff
      if (normDist >= 1.0) continue;
      const attenuation = Math.pow(1.0 - normDist, falloff);

      // Species affinity: same-species particles receive +20% bonus
      const affinityMul = p.species === profile.species ? 1.2 : 1.0;

      // Compute received light (HDR additive)
      const rx = cr * intensity * attenuation * affinityMul;
      const gy = cg * intensity * attenuation * affinityMul;
      const bz = cb * intensity * attenuation * affinityMul;

      // Accumulate into emissive buffer (additive, clamped at write)
      const off = idx * EMISSIVE_STRIDE;
      buf[off]     = Math.min(8.0, buf[off]     + rx);
      buf[off + 1] = Math.min(8.0, buf[off + 1] + gy);
      buf[off + 2] = Math.min(8.0, buf[off + 2] + bz);
      buf[off + 3] = Math.min(8.0, buf[off + 3] + intensity * attenuation);

      litCount++;
    }

    return litCount;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Bloom feedback
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Compute per-cell bloom modulation from current emissive intensity.
   *
   * Emissive-driven bloom scaling:
   *   bloomScaleMod  = lerp(1.0, 2.0, intensity / peakIntensity)
   *   bloomRadiusMod = lerp(1.0, 1.5, intensity / peakIntensity)
   *
   * This ensures that during pulse peaks / flicker-on phases, the bloom
   * post-process widens and brightens, creating temporal coherence between
   * the emissive light and the final screen glow.
   */
  private _updateBloomFeedback(): void {
    this._bloomFeedback.length = 0;

    if (!this._cfg.bloomFeedback) return;

    for (const profile of this._profiles.values()) {
      const normIntensity = profile.peakIntensity > 0
        ? profile.currentIntensity / profile.peakIntensity
        : 0;

      const bloomScaleMod  = 1.0 + normIntensity * 1.0;  // 1.0 → 2.0
      const bloomRadiusMod = 1.0 + normIntensity * 0.5;   // 1.0 → 1.5

      profile.bloomScaleMod  = bloomScaleMod;
      profile.bloomRadiusMod = bloomRadiusMod;

      this._bloomFeedback.push({
        cellId:        profile.cellId,
        bloomScaleMod,
        bloomRadiusMod,
        intensity:     profile.currentIntensity,
        color:         [...profile.baseColor],
      });
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Local physics sampling helpers
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Sample average kinetic energy of particles near a cell.
   * Returns normalised [0, 1] value (0 = stationary, 1 = fast).
   */
  private _sampleLocalEnergy(profile: EmissiveProfile, particles: Particle[]): number {
    const sampleRadius = profile.projectionRadius * 0.5;
    this._queryRadius(profile.posX, profile.posY, sampleRadius, particles, this._queryBuf);

    if (this._queryBuf.length === 0) return 0;

    let totalKE = 0;
    for (let k = 0; k < this._queryBuf.length; k++) {
      const p = particles[this._queryBuf[k]];
      totalKE += p.vx * p.vx + p.vy * p.vy;
    }

    const avgKE = totalKE / this._queryBuf.length;
    // Normalise to [0, 1] with a reference velocity of 10 units/s
    const MAX_SPEED_SQ = 100.0;
    return Math.min(1.0, avgKE / MAX_SPEED_SQ);
  }

  /**
   * Sample local particle density (normalised count within sample radius).
   * Returns [0, 1] where 1 = densely packed neighbourhood.
   */
  private _sampleLocalDensity(profile: EmissiveProfile, particles: Particle[]): number {
    const sampleRadius = profile.projectionRadius * 0.6;
    this._queryRadius(profile.posX, profile.posY, sampleRadius, particles, this._queryBuf);

    // Normalise: 50 neighbours in sample radius → density = 1.0
    const REF_COUNT = 50;
    return Math.min(1.0, this._queryBuf.length / REF_COUNT);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Integration helpers
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Get the ATBloomParams modulation for a specific cell.
   *
   * Returns a partial ATBloomParams that should be multiplied into the
   * base bloom configuration when rendering this cell's bloom contribution.
   *
   * Usage in render-compositor:
   *   const baseCfg = getSpeciesShaderConfig(species);
   *   const emissiveMod = glowSystem.getBloomModulation(cellId);
   *   bloom.setParams({
   *     bloomScale: baseCfg.bloomStrength * emissiveMod.bloomScale,
   *     blurRadius: baseCfg.bloomRadius * 16 * emissiveMod.blurRadius,
   *   });
   */
  getBloomModulation(cellId: string): { bloomScale: number; blurRadius: number } {
    const profile = this._profiles.get(cellId);
    if (!profile) return { bloomScale: 1.0, blurRadius: 1.0 };
    return {
      bloomScale: profile.bloomScaleMod,
      blurRadius: profile.bloomRadiusMod,
    };
  }

  /**
   * Aggregate bloom params: returns a single ATBloomParams partial
   * representing the scene-wide emissive contribution.
   *
   * Used when only one ATBloomPostProcess instance serves the whole scene.
   * The aggregation takes the max intensity across all cells, weighted by
   * their projection coverage.
   */
  getAggregateBloomParams(): Partial<ATBloomParams> {
    let maxScale = 1.0;
    let maxRadius = 1.0;

    for (const profile of this._profiles.values()) {
      if (profile.bloomScaleMod > maxScale) maxScale = profile.bloomScaleMod;
      if (profile.bloomRadiusMod > maxRadius) maxRadius = profile.bloomRadiusMod;
    }

    return {
      bloomScale: maxScale,
      blurRadius: maxRadius,
    };
  }

  /**
   * Read emissive colour for a specific particle index.
   * Returns [R, G, B, A] where A = total received intensity.
   */
  readParticleEmissive(particleIndex: number): [number, number, number, number] {
    const off = particleIndex * EMISSIVE_STRIDE;
    if (off + 3 >= this._emissiveBuffer.length) return [0, 0, 0, 0];
    return [
      this._emissiveBuffer[off],
      this._emissiveBuffer[off + 1],
      this._emissiveBuffer[off + 2],
      this._emissiveBuffer[off + 3],
    ];
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Dispose
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Release all resources.
   */
  dispose(): void {
    this._profiles.clear();
    this._hashBuckets.clear();
    this._bloomFeedback.length = 0;
    this._emissiveBuffer = new Float32Array(0);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Factory
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Create an EmissiveGlowSystem pre-configured for the cell-pubsub-loop pipeline.
 *
 * Usage:
 *   const glow = createEmissiveGlowSystem();
 *   glow.addCell('cell-0', 'cil-code', 5, 5);
 *   glow.addCell('cell-1', 'cil-eye', 10, 10);
 *
 *   // render loop:
 *   glow.tick(dt, world);
 *   const buf = glow.emissiveBuffer;   // → Float32Array RGBA per particle
 *   const bloom = glow.getAggregateBloomParams();
 *
 * @param config  Optional overrides.
 */
export function createEmissiveGlowSystem(
  config?: EmissiveGlowConfig,
): EmissiveGlowSystem {
  return new EmissiveGlowSystem(config);
}

// ─────────────────────────────────────────────────────────────────────────────
// WGSL emissive buffer sampling snippet
// ─────────────────────────────────────────────────────────────────────────────

/**
 * WGSL snippet for sampling the emissive buffer in the particle fragment shader.
 *
 * The emissive buffer is uploaded as a storage buffer (read-only) and sampled
 * by particle index.  The result is added to the fragment output before the
 * bloom luminance threshold pass extracts bright pixels.
 *
 * Integration point: ParticleInstancer fragment shader (particle-instancing.ts)
 *   or ATPBRMaterial fragment shader (at-pbr-material.ts).
 *
 * Usage:
 *   // In WGSL module:
 *   ${EMISSIVE_BUFFER_WGSL}
 *
 *   // In fragment main:
 *   let emissive = sampleEmissive(particleIndex);
 *   color = vec4f(color.rgb + emissive.rgb, color.a);
 */
export const EMISSIVE_BUFFER_WGSL = /* wgsl */`
// ── Emissive buffer (M791) ──────────────────────────────────────────────────
// Read-only storage buffer: vec4<f32> per particle (R, G, B, totalIntensity).
// Written by EmissiveGlowSystem.tick() on CPU, uploaded to GPU each frame.

struct EmissiveEntry {
  r : f32,
  g : f32,
  b : f32,
  a : f32,  // total received intensity (for debug / UI)
}

@group(2) @binding(0) var<storage, read> emissiveBuffer : array<EmissiveEntry>;

/// Sample the emissive contribution for a particle.
/// Returns HDR linear RGB + total received intensity in alpha.
fn sampleEmissive(particleIndex: u32) -> vec4f {
  let e = emissiveBuffer[particleIndex];
  return vec4f(e.r, e.g, e.b, e.a);
}

/// Apply emissive to a fragment colour (additive blend).
/// The result may exceed 1.0 — this is intentional for HDR bloom extraction.
fn applyEmissive(baseColor: vec4f, particleIndex: u32) -> vec4f {
  let e = sampleEmissive(particleIndex);
  return vec4f(baseColor.rgb + e.rgb, baseColor.a);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Singleton (optional — matches pattern in other sph modules)
// ─────────────────────────────────────────────────────────────────────────────

let _globalGlow: EmissiveGlowSystem | null = null;

/** Get the global EmissiveGlowSystem instance (if set). */
export function getGlobalEmissiveGlow(): EmissiveGlowSystem | null {
  return _globalGlow;
}

/** Set the global EmissiveGlowSystem instance. */
export function setGlobalEmissiveGlow(system: EmissiveGlowSystem | null): void {
  _globalGlow = system;
}
