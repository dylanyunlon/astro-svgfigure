/**
 * BloomVariants.ts
 * Per-species bloom parameter lookup.
 * Reads bloom_variants.json at build time (Astro/Vite static import),
 * exposes getBloomParams(species) for use in the Three.js FX pipeline.
 *
 * Usage:
 *   import { getBloomParams, BloomParams } from '$lib/BloomVariants';
 *   const params = getBloomParams('cil-eye');
 *   // → { bloomStrength: 2.0, bloomRadius: 0.8, luminosityThreshold: 0.0 }
 */

import bloomVariantsRaw from '../../channels/physics/bloom_variants.json';

export interface BloomParams {
  bloomStrength: number;
  bloomRadius: number;
  luminosityThreshold: number;
}

type BloomVariantMap = Record<string, BloomParams>;

// Cast the imported JSON to the typed map
const BLOOM_VARIANTS: BloomVariantMap = bloomVariantsRaw as BloomVariantMap;

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
