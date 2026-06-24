/**
 * uil-species-live.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * UIL AT params × SPH physics live interpolation
 *
 * AT ships 2593 UIL parameters that control every visual effect — bloom
 * strength, glass distortion, fog density, PBR env intensity, camera wobble,
 * and so on. These are static design-time values tuned by artists.
 *
 * This module makes them dynamic: each scalar / vector UIL param becomes a
 * function of the local SPH physics state (density, velocity, pressure,
 * vorticity). The result is a "live uniform bag" that can be handed directly
 * to any AT/Three.js pass or material.
 *
 * Architecture
 * ─────────────────────────────────────────────────────────────────────────────
 *  1. On first access, load + merge species_at_params.json & species_physics.json
 *     into a per-species baseline table.
 *  2. Classify every numeric / vec param by "modulation role" (bloom, glass,
 *     fog, env, wobble, …).
 *  3. At runtime, accept a PhysicsState snapshot and return a mutated copy of
 *     the baseline where modulatable params are scaled by physics-derived
 *     factors (densityRatio, speed, pressure, vorticity, kineticEnergy).
 *
 * Physics → visual mappings (canonical, extend freely):
 * ┌─────────────────────────┬────────────────────────────────────────────────┐
 * │ Param family            │ Modulation                                     │
 * ├─────────────────────────┼────────────────────────────────────────────────┤
 * │ bloomStrength           │ base × densityRatio  (dense → brighter glow)  │
 * │ bloomRadius             │ base × lerp(0.8, 1.4, speed)                  │
 * │ uDistortStrength        │ base × (1 + pressure × 0.15)                  │
 * │ uAlpha (glass)          │ base × clamp(1 - pressure×0.3, 0.1, 1)        │
 * │ uEnv[0] (PBR exposure)  │ base × densityRatio                           │
 * │ uFog / CloudFog alpha   │ base × lerp(0.6, 1.5, speed)                  │
 * │ wobbleStrength          │ base × (1 + vorticity × 0.4)                  │
 * │ lerpSpeed / lerpSpeed2  │ base × clamp(speed × 2, 0.5, 3)               │
 * │ fov (scene cameras)     │ base + vorticity × 2  (clamped ±10°)          │
 * └─────────────────────────┴────────────────────────────────────────────────┘
 *
 * Usage
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   // once at startup (or lazy — getSpeciesUniforms auto-inits)
 *   await initSpeciesLive();
 *
 *   // per-frame, per-cell
 *   const uniforms = getSpeciesUniforms('cil-eye', {
 *     density:       particleDensity,
 *     velocity:      [vx, vy],
 *     pressure:      particlePressure,
 *     vorticity:     localCurl,
 *     kineticEnergy: ke,
 *     time:          world.time,
 *     restDensity:   world.config.restDensity,
 *   });
 *   // uniforms is a flat Record — set on AT material / post pass directly.
 */

// ─── Imports ──────────────────────────────────────────────────────────────────

import type { PhysicsUniforms } from './physics-uniform-bridge';

// ─── Types ────────────────────────────────────────────────────────────────────

// [orphan-precise] /** Scalar or numeric-array uniform value (string/texture paths are dropped). */
export type UniformValue = number | number[];

/** Complete live uniform bag for one species at one physics moment. */
export type SpeciesUniformBag = Record<string, UniformValue>;

/**
 * Minimal physics snapshot required for interpolation.
 * Compatible with PhysicsUniforms from physics-uniform-bridge.ts.
 */
export interface PhysicsState {
  /** Local SPH density (kg/m³ or sim units). */
  density: number;
  /** SPH rest density from world.config (used to form dimensionless ratio). */
  restDensity: number;
  /** Local velocity vector [vx, vy]. */
  velocity: [number, number];
  /** Local pressure. */
  pressure: number;
  /**
   * Local vorticity (curl of velocity).  Positive = CCW.
   * From physics-uniform-bridge: clamped to [-100, 100].
   */
  vorticity: number;
  /** Normalised local kinetic energy (½ Σm|v|² / count). */
  kineticEnergy: number;
  /** World time (seconds). */
  time: number;
}

/** Per-species static baseline (AT UIL + physics constants merged). */
interface SpeciesBaseline {
  /** Raw numeric scalars from species_at_params.json. */
  atScalars: Record<string, number>;
  /** Raw numeric-array params from species_at_params.json. */
  atVectors: Record<string, number[]>;
  /** Physics constants from species_physics.json. */
  physics: SpeciesPhysicsRow;
}

interface SpeciesAtParamsFile {
  [speciesId: string]: {
    description?: string;
    at_scene_source?: string;
    assigned_params: Record<string, unknown>;
  };
}

interface SpeciesPhysicsRow {
  mass: number;
  friction: number;
  restitution: number;
  buoyancy: number;
}

interface SpeciesPhysicsFile {
  [speciesId: string]: SpeciesPhysicsRow;
}

// ─── Module-level cache ───────────────────────────────────────────────────────

let _baseline: Map<string, SpeciesBaseline> | null = null;
let _initPromise: Promise<void> | null = null;

// ─── Param classification helpers ────────────────────────────────────────────

/**
 * Classify a UIL param key into a modulation role.
 * Returns null for params we intentionally leave static (flags, indices, etc.).
 */
type ModRole =
  | 'bloomStrength'
  | 'bloomRadius'
  | 'glassDistort'
  | 'glassAlpha'
  | 'pbrEnv'
  | 'fogAlpha'
  | 'wobble'
  | 'lerpSpeed'
  | 'fov'
  | null;

function classifyKey(key: string): ModRole {
  const k = key.toLowerCase();

  // Bloom
  if (k.includes('bloomstrength'))                         return 'bloomStrength';
  if (k.includes('bloomradius'))                           return 'bloomRadius';

  // Glass / distortion
  if (k.includes('udistortstrength') || k.includes('udistort'))
                                                           return 'glassDistort';
  if (k.includes('ualpha') && !k.includes('camera'))      return 'glassAlpha';

  // PBR environment intensity (uEnv[0] = exposure)
  if (k.endsWith('/uenv') || k.endsWith('_uenv'))          return 'pbrEnv';

  // Fog / volumetric
  if (k.includes('fog') && (k.includes('alpha') || k.includes('_alpha') ||
      k.includes('noise') || k.includes('intensity')))    return 'fogAlpha';
  if (k.includes('ufog') && k.includes('home'))           return 'fogAlpha';

  // Camera wobble
  if (k.includes('wobblestrength'))                        return 'wobble';

  // Camera interpolation speed
  if (k.includes('lerpspeed'))                             return 'lerpSpeed';

  // Field of view
  if (k.includes('scenefov') || k.endsWith('fov'))        return 'fov';

  return null;
}

// ─── Physics-to-modifier functions ───────────────────────────────────────────

/** Clamp a value to [lo, hi]. */
function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

/** Linear interpolation. */
function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * clamp(t, 0, 1);
}

/**
 * Derive a normalised speed scalar [0…1] from velocity + kinetic energy.
 * Blended so a fast-moving but low-density cell still registers.
 */
function speedNorm(state: PhysicsState): number {
  const [vx, vy] = state.velocity;
  const speed = Math.sqrt(vx * vx + vy * vy);
  // Normalise against a sensible reference (~50 sim units/s)
  return clamp(speed / 50, 0, 1);
}

/** Density ratio ρ_local / ρ_rest.  1 = at-rest fluid. */
function densityRatio(state: PhysicsState): number {
  if (state.restDensity <= 0) return 1;
  return clamp(state.density / state.restDensity, 0.1, 3.0);
}

/** Normalised pressure [0…1], clamped.  Reference: 500 Pa. */
function pressureNorm(state: PhysicsState): number {
  return clamp(state.pressure / 500, 0, 1);
}

/** Normalised vorticity [-1…1].  AT range: [-100, 100]. */
function vorticityNorm(state: PhysicsState): number {
  return clamp(state.vorticity / 100, -1, 1);
}

// ─── Per-role modulation ─────────────────────────────────────────────────────

/**
 * Apply physics-based modulation to a single scalar param.
 * Returns the mutated value.
 */
function modulateScalar(
  role: ModRole,
  base: number,
  state: PhysicsState,
): number {
  const dr   = densityRatio(state);
  const spd  = speedNorm(state);
  const pn   = pressureNorm(state);
  const vn   = vorticityNorm(state);

  switch (role) {
    // Dense regions glow harder: bloom ∝ ρ/ρ₀
    case 'bloomStrength':
      return base * dr;

    // Fast-moving regions spread bloom wider
    case 'bloomRadius':
      return base * lerp(0.8, 1.4, spd);

    // High-pressure compresses / warps glass surface
    case 'glassDistort':
      return base * (1 + pn * 0.15);

    // Pressure crowds glass opacity (compression = less transparent)
    case 'glassAlpha':
      return clamp(base * (1 - pn * 0.3), 0.05, 1.0);

    // PBR env exposure tracks density (denser = more absorbed light)
    case 'pbrEnv':
      return base * dr;

    // Turbulent speed increases fog / cloud opacity
    case 'fogAlpha':
      return base * lerp(0.6, 1.5, spd);

    // Vorticity drives camera wobble
    case 'wobble':
      return base * (1 + Math.abs(vn) * 0.4);

    // Fast local flow → snappier camera transitions
    case 'lerpSpeed':
      return base * clamp(1 + spd * 2, 0.5, 3.0);

    // Vortex pulls FOV slightly wider (vertigo effect, clamped ±10°)
    case 'fov':
      return clamp(base + vn * 2, base - 10, base + 10);

    default:
      return base;
  }
}

/**
 * Apply modulation to a numeric-array param.
 * uEnv, uMRON, uFog, etc.
 */
function modulateVector(
  role: ModRole,
  base: number[],
  state: PhysicsState,
): number[] {
  if (role === null) return base;

  const dr  = densityRatio(state);
  const spd = speedNorm(state);

  // uEnv = [exposure, intensity]: scale exposure (index 0) by density
  if (role === 'pbrEnv') {
    return base.map((v, i) => (i === 0 ? v * dr : v));
  }

  // uFog = [near, far, density, alpha]: modulate density+alpha by speed
  if (role === 'fogAlpha') {
    return base.map((v, i) => (i >= 2 ? v * lerp(0.6, 1.5, spd) : v));
  }

  // For other vector roles, scale the entire vector uniformly
  return base.map(v => modulateScalar(role, v, state));
}

// ─── Data loading ─────────────────────────────────────────────────────────────

/**
 * Parse raw assigned_params from species_at_params.json into typed buckets.
 */
function parseAssignedParams(raw: Record<string, unknown>): {
  scalars: Record<string, number>;
  vectors: Record<string, number[]>;
} {
  const scalars: Record<string, number> = {};
  const vectors: Record<string, number[]> = {};

  for (const [k, v] of Object.entries(raw)) {
    if (typeof v === 'number') {
      scalars[k] = v;
    } else if (typeof v === 'string') {
      // Attempt numeric coercion for string-encoded numbers
      const n = parseFloat(v);
      if (!isNaN(n) && String(n) === v.trim()) {
        scalars[k] = n;
      }
      // Otherwise skip (texture paths, hex colors, JSON blobs)
    } else if (Array.isArray(v)) {
      // Numeric arrays only; skip mixed or nested structures
      if (v.every(x => typeof x === 'number')) {
        vectors[k] = v as number[];
      }
    }
    // booleans / objects / null → not modulated as uniforms, skip
  }

  return { scalars, vectors };
}

/**
 * Load and merge species_at_params.json + species_physics.json into the
 * baseline map.  Called once, result cached in _baseline.
 */
async function loadBaseline(): Promise<Map<string, SpeciesBaseline>> {
  // Dynamic imports work in both Vite/Astro (JSON modules) and Node test env.
  // The paths resolve relative to the project root.
  const [atRaw, physRaw] = await Promise.all([
    import('../../../channels/physics/species_at_params.json') as Promise<{ default: SpeciesAtParamsFile }>,
    import('../../../channels/physics/species_physics.json')   as Promise<{ default: SpeciesPhysicsFile }>,
  ]);

  const atData   = atRaw.default   as SpeciesAtParamsFile;
  const physData = physRaw.default as SpeciesPhysicsFile;

  const map = new Map<string, SpeciesBaseline>();

  for (const [speciesId, body] of Object.entries(atData)) {
    const { scalars, vectors } = parseAssignedParams(body.assigned_params ?? {});

    const physRow: SpeciesPhysicsRow = physData[speciesId] ?? {
      mass: 80, friction: 0.5, restitution: 0.3, buoyancy: 0.5,
    };

    map.set(speciesId, {
      atScalars: scalars,
      atVectors: vectors,
      physics:   physRow,
    });
  }

  return map;
}

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Pre-load species data.  Safe to call multiple times; subsequent calls are
 * no-ops once loading completes.
 */
export async function initSpeciesLive(): Promise<void> {
  if (_baseline !== null) return;
  if (_initPromise)       return _initPromise;

  _initPromise = loadBaseline().then(map => {
    _baseline = map;
  });

  return _initPromise;
}

/**
 * Return the complete live uniform bag for `speciesId` at the given physics
 * moment.  Merges AT UIL baseline with physics-derived multipliers.
 *
 * @param speciesId  e.g. 'cil-eye' | 'cil-bolt' | …
 * @param state      Live SPH physics snapshot for this cell.
 * @returns          Flat Record<string, number|number[]> ready for shader use.
 *
 * Auto-inits synchronously if baseline already loaded, or returns a static
 * zero-state fallback if called before initSpeciesLive() resolves — so it is
 * safe to call on the first rendered frame without awaiting init.
 */
export function getSpeciesUniforms(
  speciesId: string,
  state: PhysicsState,
): SpeciesUniformBag {
  // Trigger async load if needed (fire-and-forget — next frame will have data)
  if (_baseline === null) {
    initSpeciesLive().catch(console.error);
    return {};
  }

  const baseline = _baseline.get(speciesId);
  if (!baseline) {
    // Unknown species — try a graceful degradation rather than throwing
    console.warn(`[uil-species-live] unknown speciesId "${speciesId}"`);
    return {};
  }

  const result: SpeciesUniformBag = {};

  // ── Scalars ──────────────────────────────────────────────────────────────
  for (const [k, base] of Object.entries(baseline.atScalars)) {
    const role = classifyKey(k);
    result[k] = role !== null ? modulateScalar(role, base, state) : base;
  }

  // ── Vectors ──────────────────────────────────────────────────────────────
  for (const [k, base] of Object.entries(baseline.atVectors)) {
    const role = classifyKey(k);
    result[k] = role !== null ? modulateVector(role, base, state) : base;
  }

  // ── Physics-derived extras (not in AT UIL but useful for downstream) ─────
  //
  // Expose core physics scalars as u_* uniforms so custom shaders can read
  // them without reimporting physics-uniform-bridge separately.
  result['u_sph_densityRatio']  = densityRatio(state);
  result['u_sph_speedNorm']     = speedNorm(state);
  result['u_sph_pressureNorm']  = pressureNorm(state);
  result['u_sph_vorticityNorm'] = vorticityNorm(state);
  result['u_sph_buoyancy']      = baseline.physics.buoyancy;
  result['u_sph_mass']          = baseline.physics.mass;
  result['u_sph_time']          = state.time;

  return result;
}

/**
 * Convenience: build a PhysicsState from a PhysicsUniforms bag
 * (as returned by physics-uniform-bridge.ts).
 *
 * @param pu         Output of samplePhysicsUniforms() from physics-uniform-bridge.
 * @param restDensity  world.config.restDensity
 */
export function physicsUniformsToState(
  pu: PhysicsUniforms,
  restDensity: number,
): PhysicsState {
  // PhysicsUniforms.u_density is already normalised (ρ/ρ₀) — undo that.
  return {
    density:      pu.u_density * restDensity,
    restDensity,
    velocity:     pu.u_velocity,
    pressure:     pu.u_pressure,
    vorticity:    pu.u_vorticity,
    kineticEnergy: pu.u_kineticEnergy,
    time:         pu.u_time,
  };
}

/**
 * Return the list of species IDs present in the loaded baseline.
 * Useful for iterating all cells or building debug UIs.
 */
export function getLoadedSpecies(): string[] {
  return _baseline ? Array.from(_baseline.keys()) : [];
}
