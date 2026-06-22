/**
 * species-visual-dna.ts — M733: Species Visual DNA
 * ─────────────────────────────────────────────────────────────────────────────
 * Single-call facade that concatenates the three visual identity modules:
 *
 *   1. species-shader-registry  → static shader stack (SDF, material, pattern, bloom)
 *   2. physics-uniform-bridge   → live SPH physics sampling for a rigid body
 *   3. uil-species-live         → AT UIL params modulated by physics state
 *
 * The result, VisualDNA, is the complete render-time visual configuration for
 * one cell: everything a renderer needs to draw, shade, and animate it in a
 * single frame.
 *
 * Usage
 * ─────────────────────────────────────────────────────────────────────────────
 *   import { getVisualDNA, initVisualDNA } from './species-visual-dna';
 *
 *   // once at startup (pre-loads UIL baselines; optional — auto-inits lazily)
 *   await initVisualDNA();
 *
 *   // per-frame, per-cell
 *   const dna = getVisualDNA('cil-eye', world, '42');
 *   // dna.shader   → SpeciesShaderConfig   (SDF shape, material, pattern, bloom)
 *   // dna.physics  → PhysicsUniforms       (raw SPH neighbourhood sample)
 *   // dna.resolved → resolved binding vals  (physics × shader bindings evaluated)
 *   // dna.uilBag   → SpeciesUniformBag     (AT UIL params modulated by physics)
 *   // dna.bloom    → runtime bloom config   (strength, radius, threshold, pulse)
 *   // dna.material → runtime material overrides (fresnel, irid, opacity, envScale)
 *   // dna.pattern  → runtime pattern params (speed, contrast)
 *   // dna.sdf      → runtime SDF overrides  (distort amount)
 *
 * Upstream
 * ─────────────────────────────────────────────────────────────────────────────
 *   src/lib/sph/species-shader-registry.ts  — SpeciesShaderConfig, resolvePhysicsBindings
 *   src/lib/sph/physics-uniform-bridge.ts   — samplePhysicsForBody, PhysicsUniforms
 *   src/lib/sph/uil-species-live.ts         — getSpeciesUniforms, physicsUniformsToState
 */

import type { World } from './world-stepper';

import {
  getSpeciesShaderConfig,
  resolvePhysicsBindings,
  type SpeciesShaderConfig,
  type PhysicsBindings,
} from './species-shader-registry';

import {
  samplePhysicsForBody,
  type PhysicsUniforms,
} from './physics-uniform-bridge';

import {
  initSpeciesLive,
  getSpeciesUniforms,
  physicsUniformsToState,
  type SpeciesUniformBag,
} from './uil-species-live';

// ─── Output types ────────────────────────────────────────────────────────────

/** Runtime bloom parameters, fully resolved against physics state. */
export interface RuntimeBloom {
  /** Effective bloom strength (base × physics binding). */
  strength: number;
  /** Effective bloom radius (base × physics binding). */
  radius: number;
  /** Bloom luminance threshold (static from shader config). */
  threshold: number;
  /** Pulse amplitude (static from shader config). */
  pulseAmplitude: number;
  /** Effective pulse frequency (base × physics binding). */
  pulseFrequency: number;
}

/** Runtime material overrides derived from physics bindings. */
export interface RuntimeMaterial {
  /** Fresnel rim power (base × physics binding, or static default). */
  fresnelStrength: number;
  /** Iridescence film thickness in nm (base + physics binding delta). */
  iridThickness: number;
  /** Overall opacity (base × physics binding). */
  opacity: number;
  /** PBR environment scale (base × physics binding). */
  envScale: number;
}

/** Runtime pattern shader parameters. */
export interface RuntimePattern {
  /** Pattern animation speed multiplier. */
  speed: number;
  /** Pattern contrast multiplier. */
  contrast: number;
}

/** Runtime SDF overrides. */
export interface RuntimeSdf {
  /** SDF distortion amount (additive from physics binding). */
  distort: number;
}

/**
 * VisualDNA — the complete visual identity for one cell at one moment in time.
 *
 * Combines the static declarative shader stack from the registry with live
 * physics-driven modulation values.  A renderer receiving this struct has
 * everything it needs to configure SDF, material, pattern, bloom, and
 * post-processing for the cell — no further lookups required.
 */
export interface VisualDNA {
  /** Species identifier (e.g. 'cil-eye'). */
  speciesId: string;

  /** Body identifier within the SPH world. */
  bodyId: string;

  /** Static shader stack definition from species-shader-registry. */
  shader: SpeciesShaderConfig;

  /** Raw physics uniforms sampled from the SPH neighbourhood. */
  physics: PhysicsUniforms;

  /**
   * Physics bindings evaluated against live physics state.
   * Keys are visual targets (bloomStrength, sdfDistort, …); values are the
   * final numbers after scale + clamp + mode application.
   */
  resolved: Partial<Record<keyof PhysicsBindings, number>>;

  /**
   * Full AT UIL uniform bag, modulated by physics state.
   * Contains all 2593+ AT scene params for this species, live-adjusted.
   */
  uilBag: SpeciesUniformBag;

  /** Runtime bloom parameters (convenience extraction from resolved). */
  bloom: RuntimeBloom;

  /** Runtime material overrides (convenience extraction from resolved). */
  material: RuntimeMaterial;

  /** Runtime pattern parameters (convenience extraction from resolved). */
  pattern: RuntimePattern;

  /** Runtime SDF overrides (convenience extraction from resolved). */
  sdf: RuntimeSdf;
}

// ─── Default sample radius ───────────────────────────────────────────────────

/**
 * Default SPH sample radius multiplier.
 * Applied to world.config.smoothingRadius to compute the neighbourhood query
 * radius in physics-uniform-bridge.  3× smoothing radius captures the full
 * SPH kernel support while keeping the neighbour list reasonable.
 */
const SAMPLE_RADIUS_FACTOR = 3;

// ─── Initialisation ──────────────────────────────────────────────────────────

/**
 * Pre-load UIL species baselines.  Optional — getVisualDNA() auto-inits
 * lazily on first call, but calling this at startup avoids a blank first frame.
 */
export async function initVisualDNA(): Promise<void> {
  await initSpeciesLive();
}

// ─── Core API ────────────────────────────────────────────────────────────────

/**
 * Build the complete VisualDNA for a single cell.
 *
 * Orchestration flow:
 *   1. Look up the static SpeciesShaderConfig from the registry.
 *   2. Sample live PhysicsUniforms for the body from the SPH world.
 *   3. Evaluate physics bindings (shader config × physics → resolved values).
 *   4. Convert PhysicsUniforms → PhysicsState, then query UIL modulation.
 *   5. Assemble convenience sub-structs (bloom, material, pattern, sdf).
 *
 * @param speciesId     Species identifier (e.g. 'cil-eye').
 * @param world         Live SPH World instance (from world-stepper).
 * @param bodyId        String-form RigidBody.id within the world.
 * @param sampleRadius  Optional override for the SPH neighbour query radius.
 *                      Defaults to SAMPLE_RADIUS_FACTOR × smoothingRadius.
 */
export function getVisualDNA(
  speciesId: string,
  world: World,
  bodyId: string,
  sampleRadius?: number,
): VisualDNA {
  // 1. Static shader stack
  const shader = getSpeciesShaderConfig(speciesId);

  // 2. Live physics sampling
  const radius = sampleRadius ?? world.config.smoothingRadius * SAMPLE_RADIUS_FACTOR;
  const physics = samplePhysicsForBody(world, bodyId, radius);

  // 3. Evaluate physics → visual bindings
  const resolved = resolvePhysicsBindings(speciesId, physics);

  // 4. UIL modulation (AT scene params × physics)
  const physState = physicsUniformsToState(physics, world.config.restDensity);
  const uilBag = getSpeciesUniforms(speciesId, physState);

  // 5. Extract convenience sub-structs from resolved bindings + shader defaults
  const bloom: RuntimeBloom = {
    strength:       resolved.bloomStrength    ?? shader.bloomStrength,
    radius:         resolved.bloomRadius      ?? shader.bloomRadius,
    threshold:      shader.bloomThreshold,
    pulseAmplitude: shader.bloomPulseAmplitude,
    pulseFrequency: resolved.pulseFrequency   ?? shader.bloomPulseFrequency,
  };

  const material: RuntimeMaterial = {
    fresnelStrength: resolved.fresnelStrength  ?? (shader.materialParams.fresnelPower ?? 2.0),
    iridThickness:   resolved.iridThickness    ?? (shader.materialParams.iridThickness ?? 400.0),
    opacity:         resolved.opacity           ?? 1.0,
    envScale:        resolved.materialEnvScale  ?? 1.0,
  };

  const pattern: RuntimePattern = {
    speed:    resolved.patternSpeed    ?? 1.0,
    contrast: resolved.patternContrast ?? 1.0,
  };

  const sdf: RuntimeSdf = {
    distort: resolved.sdfDistort ?? 0.0,
  };

  return {
    speciesId,
    bodyId,
    shader,
    physics,
    resolved,
    uilBag,
    bloom,
    material,
    pattern,
    sdf,
  };
}

// ─── Batch helpers ───────────────────────────────────────────────────────────

/**
 * Build VisualDNA for every rigid body in the world.
 *
 * Requires a mapping from bodyId → speciesId. Typically obtained from
 * cell-body-bridge (CellPhysicsConfig[]) or the cell_registry.
 *
 * @param speciesMap    Map<bodyId (string), speciesId (string)>
 * @param world         Live SPH World.
 * @param sampleRadius  Optional override for the SPH neighbour query radius.
 * @returns             Map<bodyId, VisualDNA>
 */
export function getAllVisualDNA(
  speciesMap: Map<string, string>,
  world: World,
  sampleRadius?: number,
): Map<string, VisualDNA> {
  const result = new Map<string, VisualDNA>();

  for (const [bodyId, speciesId] of speciesMap) {
    result.set(bodyId, getVisualDNA(speciesId, world, bodyId, sampleRadius));
  }

  return result;
}

/**
 * Lightweight update: re-evaluate only the physics-driven portions of an
 * existing VisualDNA without re-looking-up the shader config.
 *
 * Useful in tight render loops where the species assignment doesn't change
 * frame-to-frame but the physics state does.
 *
 * @param prev          Previous VisualDNA from a prior frame.
 * @param world         Current SPH World state.
 * @param sampleRadius  Optional override.
 * @returns             Fresh VisualDNA with updated physics + resolved values.
 */
export function updateVisualDNA(
  prev: VisualDNA,
  world: World,
  sampleRadius?: number,
): VisualDNA {
  return getVisualDNA(prev.speciesId, world, prev.bodyId, sampleRadius);
}
