/**
 * BloomVariants.ts
 * Per-species bloom parameter lookup.
 * Fetches bloom_variants.json at runtime (no static import),
 * exposes getBloomParams(species) for use in the Three.js FX pipeline.
 *
 * Usage:
 *   import { initBloomVariants, getBloomParams, BloomParams } from '$lib/BloomVariants';
 *   await initBloomVariants();            // call once at startup
 *   const params = getBloomParams('cil-eye');
 *   // → { bloomStrength: 2.0, bloomRadius: 0.8, luminosityThreshold: 0.0 }
 */

export interface BloomParams {
  bloomStrength: number;
  bloomRadius: number;
  luminosityThreshold: number;
}

type BloomVariantMap = Record<string, BloomParams>;

// Module-level cache — populated by initBloomVariants()
let BLOOM_VARIANTS: BloomVariantMap = {};
let _initialized = false;

/**
 * Fetch bloom_variants.json at runtime and populate the module cache.
 * Safe to call multiple times — subsequent calls are no-ops.
 */
export async function initBloomVariants(): Promise<void> {
  if (_initialized) return;
  const res = await fetch('/channels/physics/bloom_variants.json');
  BLOOM_VARIANTS = (await res.json()) as BloomVariantMap;
  _initialized = true;
}

/**
 * Default fallback bloom params for unknown species.
 * Conservative values so unknown cells don't blow out the scene.
 */
const DEFAULT_BLOOM_PARAMS: BloomParams = {
  bloomStrength: 0.5,
  bloomRadius: 0.3,
  luminosityThreshold: 0.1,
};

/**
 * Returns bloom parameters for a given species key.
 * Falls back to DEFAULT_BLOOM_PARAMS if the species is not registered.
 *
 * @param species - e.g. 'cil-eye', 'cil-bolt', 'cil-vector', ...
 * @returns BloomParams { bloomStrength, bloomRadius, luminosityThreshold }
 */
export function getBloomParams(species: string): BloomParams {
  return BLOOM_VARIANTS[species] ?? DEFAULT_BLOOM_PARAMS;
}

/**
 * Returns all registered species keys.
 */
export function getRegisteredSpecies(): string[] {
  return Object.keys(BLOOM_VARIANTS);
}

/**
 * Returns the full bloom variant map (read-only reference).
 */
export function getAllBloomVariants(): Readonly<BloomVariantMap> {
  return BLOOM_VARIANTS;
}
