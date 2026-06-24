/**
 * cell-visual-identity.ts — M731: Cell Visual Identity from Physics
 * ─────────────────────────────────────────────────────────────────────────────
 * Every cell's visual appearance is *derived* from its physical attributes —
 * nothing is hardcoded per cellId.  Two cells with the same species, QoS,
 * force field, and contact state will look identical; change any input and
 * the visual shifts accordingly.
 *
 * Derivation chain:
 *
 *   species          → base morphology (jellyfish / petal / coral / mycelium / crystal)
 *   QoS reliability  → border sharpness   (RELIABLE = crisp membrane, BEST_EFFORT = fuzzy)
 *   QoS mps          → internal flow speed (high bandwidth = fast particle stream)
 *   force_field      → decoration particle direction & intensity
 *   collision contacts → contact-point spark/ripple intensity
 *
 * The output VisualProfile is a plain data bag designed to be consumed
 * directly by ATSceneCompositor, pixi-cell-renderer, or any other
 * rendering pipeline without further physics lookups.
 *
 * Upstream references:
 *   channels/physics/cell_registry.json    — cell bbox, species, z
 *   channels/physics/force_field.json      — per-cell force vectors
 *   src/lib/sph/qosSpatial.ts              — QoSProfile, QOS_PRESETS
 *   src/lib/sph/types.ts                   — QoSProfile, ContactConstraint
 *   src/lib/sph/species-shader-registry.ts — SpeciesShaderConfig, getSpeciesShaderConfig
 *   src/lib/sph/physics-uniform-bridge.ts  — PhysicsUniforms
 *   src/lib/sph/color-palette.ts           — RGB
 */


import type { QoSProfile }        from './types';
import type { MaterialType, PatternShader, SdfShape }
import { getSpeciesShaderConfig }  from './species-shader-registry';
import { qosToSpatial }           from './qosSpatial';

// [orphan-precise]                                    from './species-shader-registry';

// ─────────────────────────────────────────────────────────────────────────────
// Morphology — the five base morphologies derived from species semantics
// ─────────────────────────────────────────────────────────────────────────────

// [orphan-precise] /**
// [orphan-precise]  * Base morphology archetype.  Each species string maps to exactly one
// [orphan-precise]  * morphology through semantic analysis of the species role, not through
// [orphan-precise]  * a per-cellId lookup table.
// [orphan-precise]  *
// [orphan-precise]  *   jellyfish  — translucent, pulsing, trailing tentacles (attention / sensory)
// [orphan-precise]  *   petal      — radially symmetric, soft edges, breathing (embedding / encoding)
// [orphan-precise]  *   coral      — branching, rigid skeleton, textured surface (structural / norm)
// [orphan-precise]  *   mycelium   — networked filaments, spreading, organic flow (routing / skip)
// [orphan-precise]  *   crystal    — faceted, sharp geometry, internal refraction (computation / FFN)
// [orphan-precise]  */
export type Morphology =
  | 'jellyfish'
  | 'petal'
  | 'coral'
  | 'mycelium'
  | 'crystal';

// ─────────────────────────────────────────────────────────────────────────────
// Force input — simplified force vector for visual derivation
// ─────────────────────────────────────────────────────────────────────────────

/** Per-cell force vector from channels/physics/force_field.json */
export interface ForceInput {
  dx: number;
  dy: number;
  dz: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Contact input — collision contact summary
// ─────────────────────────────────────────────────────────────────────────────

/** Summary of active collision contacts for a single cell */
export interface ContactSummary {
  /** Number of active contact points */
  count: number;
  /** Average normal impulse across active contacts (0 if none) */
  avgImpulse: number;
  /** Average contact normal direction (unit vector; [0,0] if none) */
  avgNormal: [number, number];
}

// ─────────────────────────────────────────────────────────────────────────────
// VisualProfile — the output consumed by the rendering pipeline
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Complete visual profile for a single cell.  Every field is derived from
 * physics — none require knowledge of the cellId itself.
 *
 * Consumption:
 *   ATSceneCompositor reads materialType + particleDensity + flowSpeed
 *   pixi-cell-renderer reads borderSharpness + glowIntensity + colorPalette
 *   contact-sparks reads sparkIntensity + sparkDirection
 *   particle-compositor reads decorationDirection + decorationSpeed
 */
export interface VisualProfile {
  // ── Morphology ──────────────────────────────────────────────────────────────

  /** Base morphology archetype derived from species. */
  morphology: Morphology;

  /** SDF shape from the species shader registry (passthrough for compositor). */
  sdfShape: SdfShape;

  // ── Material ────────────────────────────────────────────────────────────────

  /** Surface material model: matcap | pbr | iridescence. */
  materialType: MaterialType;

  /** Pattern shader applied inside the SDF mask. */
  patternShader: PatternShader;

  // ── Particle density & flow ─────────────────────────────────────────────────

  /**
   * Internal particle density factor (0–1 normalised).
   * Derived from QoS historyDepth and durability — cells that retain more
   * state have denser internal particle fields.
   */
  particleDensity: number;

  /**
   * Internal flow speed multiplier (≥ 0).
   * Derived from QoS mps: high message throughput → fast particle streams
   * inside the cell body.  Zero mps → gentle ambient drift (0.1 base).
   */
  flowSpeed: number;

  // ── Border ──────────────────────────────────────────────────────────────────

  /**
   * Border sharpness (0–1).
   * 1.0 = crisp, well-defined membrane (RELIABLE QoS)
   * 0.0 = soft, fuzzy, dissipating edge (BEST_EFFORT QoS)
   *
   * Maps to SDF anti-aliasing width and boundary stiffness in the renderer.
   */
  borderSharpness: number;

  // ── Glow / Bloom ────────────────────────────────────────────────────────────

  /**
   * Base glow intensity multiplier (≥ 0).
   * Derived from species bloomStrength modulated by QoS bandwidth.
   * High-throughput cells glow brighter because more energy flows through them.
   */
  glowIntensity: number;

  // ── Color ───────────────────────────────────────────────────────────────────

  /**
   * Three-colour palette derived from species + QoS interaction.
   *   base      — primary fill colour (species identity)
   *   accent    — highlights and active-state tint (QoS-modulated)
   *   rim       — Fresnel rim / edge glow colour (force-field influenced)
   */
  colorPalette: {
    base:   [number, number, number];
    accent: [number, number, number];
    rim:    [number, number, number];
  };

  // ── Force-field decoration ──────────────────────────────────────────────────

  /**
   * Direction of external decoration particles (unit vector in world space).
   * Derived from the cell's force_field vector.  Particles orbiting or
   * trailing the cell body follow this direction.
   */
  decorationDirection: [number, number];

  /**
   * Speed of decoration particles (world units / s, ≥ 0).
   * Proportional to force_field magnitude — strong forces produce fast,
   * energetic orbiting particles.
   */
  decorationSpeed: number;

  // ── Contact effects ─────────────────────────────────────────────────────────

  /**
   * Spark / ripple intensity at contact points (0–1).
   * 0 = no active contacts (dormant cell)
   * 1 = heavy collision with high impulse (maximum visual feedback)
   */
  sparkIntensity: number;

  /**
   * Primary spark emission direction (unit vector; [0,0] if no contacts).
   * Derived from average contact normal — sparks scatter along this axis.
   */
  sparkDirection: [number, number];
}

// ─────────────────────────────────────────────────────────────────────────────
// Species → Morphology mapping (semantic, not per-cellId)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Maps a species string to its morphology archetype.
 *
 * The mapping is based on the *role* each species plays in the Transformer
 * architecture analogy, not on the cell's identity:
 *
 *   Sensory/attention species     → jellyfish  (sensing, pulsing, responsive)
 *   Encoding/embedding species    → petal      (soft geometry, latent-space warmth)
 *   Structural/normalisation spp. → coral      (rigid, stabilising, branching)
 *   Routing/skip-connection spp.  → mycelium   (networked, directional, spreading)
 *   Computational/transform spp.  → crystal    (faceted, refractive, energetic)
 *
 * Unknown species fall back to 'coral' (the most neutral archetype).
 */
function speciesToMorphology(species: string): Morphology {
  // Attention / sensory — pulsing, responsive, multi-headed
  if (species === 'cil-eye' || species === 'cil-filter') {
    return 'jellyfish';
  }

  // Embedding / encoding — soft, continuous latent space
  if (species === 'cil-vector') {
    return 'petal';
  }

  // Structural / normalisation — rigid anchors, residual stability
  if (species === 'cil-plus' || species === 'cil-layers') {
    return 'coral';
  }

  // Routing / skip connections / control flow — directional, networked
  if (species === 'cil-arrow-right' || species === 'cil-loop') {
    return 'mycelium';
  }

  // Computation / transform — FFN, output projection, topology
  if (species === 'cil-bolt' || species === 'cil-code' || species === 'cil-graph') {
    return 'crystal';
  }

  // Unknown → neutral
  return 'coral';
}

// ─────────────────────────────────────────────────────────────────────────────
// QoS → visual derivation helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Derive border sharpness from QoS reliability.
 *
 * RELIABLE connections maintain strong, well-defined boundaries —
 * every message arrives, so the cell membrane is confident and crisp.
 *
 * BEST_EFFORT connections have lossy, probabilistic delivery —
 * the cell edge dissipates and blurs, reflecting uncertainty.
 *
 * The intermediate value allows for future QoS profiles that might
 * blend reliability characteristics.
 */
function deriveBorderSharpness(qos: QoSProfile): number {
  if (qos.reliability === 'RELIABLE') {
    // Crisp membrane — high stiffness boundary.
    // Durability adds a small boost: TRANSIENT_LOCAL cells are even crisper
    // because they persist state and maintain structural integrity.
    const durabilityBoost = qos.durability === 'TRANSIENT_LOCAL' ? 0.08 : 0.0;
    return Math.min(1.0, 0.85 + durabilityBoost);
  }

  // BEST_EFFORT — soft, fuzzy edge.
  // Higher mps partially compensates: frequent updates keep the boundary
  // somewhat coherent even without guaranteed delivery.
  const mpsCompensation = Math.min(0.15, qos.mps * 0.001);
  return 0.3 + mpsCompensation;
}

/**
 * Derive internal flow speed from QoS mps (messages per second).
 *
 * Flow speed is a visual proxy for information throughput: cells processing
 * more messages per second have faster-moving internal particle streams,
 * conveying busyness and activity.
 *
 * The mapping uses a sqrt curve so the visual difference between 0→10 mps
 * is more pronounced than 90→100 mps (diminishing visual returns at
 * high throughput avoids overwhelming noise).
 */
function deriveFlowSpeed(qos: QoSProfile): number {
  if (qos.mps <= 0) {
    // Zero mps → gentle ambient drift, not static.
    // Even idle cells have subtle internal motion (biological rest state).
    return 0.1;
  }

  // sqrt mapping: 1 mps → 0.3, 10 mps → 0.95, 100 mps → 3.0
  // Capped at 5.0 to prevent visual chaos at extreme throughput.
  return Math.min(5.0, 0.3 * Math.sqrt(qos.mps));
}

/**
 * Derive internal particle density from QoS history depth and durability.
 *
 * Cells that retain more historical state have richer, denser internal
 * particle populations — visually conveying stored information mass.
 *
 * TRANSIENT_LOCAL durability adds density because the cell actively
 * maintains a local state cache (more "stuff" inside).
 */
function deriveParticleDensity(qos: QoSProfile): number {
  // historyDepth 1 → sparse (0.15), 20 → dense (0.85)
  const historyFactor = Math.min(1.0, qos.historyDepth / 25);

  const durabilityBoost = qos.durability === 'TRANSIENT_LOCAL' ? 0.15 : 0.0;

  return Math.min(1.0, 0.15 + historyFactor * 0.7 + durabilityBoost);
}

// ─────────────────────────────────────────────────────────────────────────────
// Force field → decoration derivation
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Derive decoration particle direction and speed from the cell's force vector.
 *
 * The force field represents the net data-flow direction for each cell in the
 * Transformer diagram.  Decoration particles (orbiting halos, trailing wisps)
 * flow in the direction of the force, and their speed is proportional to the
 * force magnitude.
 */
function deriveDecoration(force: ForceInput): {
  direction: [number, number];
  speed: number;
} {
  const mag = Math.sqrt(force.dx * force.dx + force.dy * force.dy);

  if (mag < 1e-6) {
    // No force → no directional decoration; particles drift randomly.
    return { direction: [0, 0], speed: 0 };
  }

  // Normalise to unit direction
  const direction: [number, number] = [force.dx / mag, force.dy / mag];

  // Speed: log-scale mapping to compress the wide dynamic range of force
  // magnitudes (force_field.json has values from ~1 to ~72).
  // log(1+1) ≈ 0.69 → speed ≈ 0.46
  // log(1+72) ≈ 4.29 → speed ≈ 2.86
  const speed = Math.log(1 + mag) * 0.667;

  return { direction, speed };
}

// ─────────────────────────────────────────────────────────────────────────────
// Contact → spark derivation
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Derive spark/ripple visual intensity from collision contacts.
 *
 * More contacts and higher average impulse → stronger visual feedback.
 * The intensity is clamped to [0, 1] for use as a direct multiplier
 * in the contact-sparks system.
 */
function deriveSparks(contacts: ContactSummary): {
  intensity: number;
  direction: [number, number];
} {
  if (contacts.count === 0) {
    return { intensity: 0, direction: [0, 0] };
  }

  // Count contribution: 1 contact → 0.3, 3+ contacts → saturates toward 1.0
  const countFactor = 1.0 - Math.exp(-contacts.count * 0.5);

  // Impulse contribution: normalised with a soft knee at impulse ~50
  const impulseFactor = Math.min(1.0, contacts.avgImpulse / 50);

  // Combined intensity: geometric mean gives balanced response
  const intensity = Math.min(1.0, Math.sqrt(countFactor * impulseFactor));

  return {
    intensity,
    direction: contacts.avgNormal,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Color palette derivation
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Derive the three-colour palette from species shader config + QoS + force.
 *
 * The base colour comes from the species shader registry (albedo).
 * The accent is the base hue-shifted by QoS flow speed (active cells
 * shift warmer).  The rim colour incorporates the force field direction
 * to subtly encode data-flow orientation in the visual.
 */
function deriveColorPalette(
  species: string,
  qos: QoSProfile,
  force: ForceInput,
): {
  base:   [number, number, number];
  accent: [number, number, number];
  rim:    [number, number, number];
} {
  const cfg = getSpeciesShaderConfig(species);

  // Base: species albedo (from materialParams or fallback neutral grey)
  const albedo = cfg.materialParams.albedo ?? [0.5, 0.5, 0.5];
  const base: [number, number, number] = [albedo[0], albedo[1], albedo[2]];

  // Accent: hue-shift the base toward warm tones proportional to flow speed.
  // High-throughput QoS profiles push the accent warmer (more energetic);
  // low-throughput stays close to the base (calm, ambient).
  const flowSpeed = deriveFlowSpeed(qos);
  const warmShift = Math.min(0.15, flowSpeed * 0.04);
  const accent: [number, number, number] = [
    Math.min(1.0, base[0] + warmShift * 1.2),
    Math.min(1.0, base[1] + warmShift * 0.3),
    Math.max(0.0, base[2] - warmShift * 0.5),
  ];

  // Rim: Fresnel colour from species, modulated by force direction.
  // The force vector's angle maps to a subtle hue rotation on the rim,
  // encoding data-flow direction as a colour cue visible at grazing angles.
  const fresnelColor = cfg.materialParams.fresnelColor ?? [0.6, 0.6, 0.6];
  const forceAngle = Math.atan2(force.dy, force.dx);
  // Small cyclic hue offset: ±0.08 in each channel based on force angle
  const hueOffset = Math.sin(forceAngle) * 0.08;
  const rim: [number, number, number] = [
    clamp01(fresnelColor[0] + hueOffset),
    clamp01(fresnelColor[1] - hueOffset * 0.5),
    clamp01(fresnelColor[2] + hueOffset * 0.3),
  ];

  return { base, accent, rim };
}

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}

// ─────────────────────────────────────────────────────────────────────────────
// CellVisualIdentity — the public API
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Derives a complete visual identity for any cell purely from its physical
 * attributes.  No hardcoded per-cellId tables — the same inputs always
 * produce the same visual output.
 *
 * Usage:
 * ```ts
 * import { CellVisualIdentity } from '$lib/sph/cell-visual-identity';
 *
 * const profile = CellVisualIdentity.fromCell(
 *   'self_attn',
 *   'cil-eye',
 *   QOS_PRESETS.SENSOR_DATA,
 *   { dx: 0, dy: 15.16, dz: 0 },
 *   { count: 2, avgImpulse: 30, avgNormal: [0.7, -0.7] },
 * );
 *
 * // Feed directly to ATSceneCompositor
 * compositor.setCellVisual(cellId, profile);
 * ```
 */
export class CellVisualIdentity {
  /**
   * Derive a complete VisualProfile from a cell's physical attributes.
   *
   * @param cellId    Cell identifier (used only for cache keying, not for
   *                  visual derivation — the visual is physics-determined).
   * @param species   Species string from cell_registry.json (e.g. 'cil-eye').
   * @param qos       QoS profile governing the cell's communication channel.
   * @param forces    Force-field vector from force_field.json.
   * @param contacts  Collision contact summary (live or snapshot).
   * @returns         VisualProfile ready for the rendering pipeline.
   */
  static fromCell(
    _cellId: string,
    species: string,
    qos: QoSProfile,
    forces: ForceInput,
    contacts: ContactSummary,
  ): VisualProfile {
    const cfg = getSpeciesShaderConfig(species);
    const spatial = qosToSpatial(qos);

    // ── Morphology ─────────────────────────────────────────────────────────
    const morphology = speciesToMorphology(species);

    // ── Material & pattern (from species shader registry) ──────────────────
    const materialType  = cfg.materialType;
    const patternShader = cfg.patternShader;
    const sdfShape      = cfg.sdfShape;

    // ── QoS-derived visuals ────────────────────────────────────────────────
    const borderSharpness = deriveBorderSharpness(qos);
    const flowSpeed       = deriveFlowSpeed(qos);
    const particleDensity = deriveParticleDensity(qos);

    // ── Glow intensity ─────────────────────────────────────────────────────
    // Species base bloom modulated by QoS bandwidth and boundary stiffness.
    // High-bandwidth RELIABLE cells glow brightest (lots of energy, well-contained).
    // Low-bandwidth BEST_EFFORT cells glow dimly (sparse, dissipating).
    const bandwidthFactor = qos.mps > 0
      ? Math.min(2.0, 0.6 + Math.log(1 + qos.mps) * 0.3)
      : 0.6;
    const stiffnessFactor = spatial.boundaryStiffness / 50000;  // normalise to [0.16, 1.0]
    const glowIntensity   = cfg.bloomStrength * bandwidthFactor * (0.5 + stiffnessFactor * 0.5);

    // ── Force-field decoration ─────────────────────────────────────────────
    const decoration = deriveDecoration(forces);

    // ── Contact sparks ─────────────────────────────────────────────────────
    const sparks = deriveSparks(contacts);

    // ── Color palette ──────────────────────────────────────────────────────
    const colorPalette = deriveColorPalette(species, qos, forces);

    return {
      morphology,
      sdfShape,
      materialType,
      patternShader,
      particleDensity,
      flowSpeed,
      borderSharpness,
      glowIntensity,
      colorPalette,
      decorationDirection: decoration.direction,
      decorationSpeed:     decoration.speed,
      sparkIntensity:      sparks.intensity,
      sparkDirection:      sparks.direction,
    };
  }

  /**
   * Batch-derive visual profiles for all cells in a registry.
   *
   * Convenience method that takes the raw JSON structures from
   * cell_registry.json and force_field.json and produces a Map of profiles.
   *
   * @param cellRegistry  Parsed cell_registry.json `.cells` object.
   * @param forceField    Parsed force_field.json object.
   * @param qosMap        Per-cell QoS profile assignment.
   *                      Falls back to DEFAULT preset if a cell is missing.
   * @param contactMap    Per-cell contact summaries (live collision state).
   *                      Falls back to zero contacts if a cell is missing.
   * @returns             Map<cellId, VisualProfile>.
   */
  static fromRegistry(
    cellRegistry: Record<string, { species: string; bbox: { min: number[]; max: number[] } }>,
    forceField:   Record<string, ForceInput>,
    qosMap:       Record<string, QoSProfile>,
    contactMap:   Record<string, ContactSummary>,
  ): Map<string, VisualProfile> {
    const DEFAULT_QOS: QoSProfile = {
      reliability: 'RELIABLE',
      mps: 0,
      historyDepth: 10,
      durability: 'VOLATILE',
    };

    const ZERO_CONTACTS: ContactSummary = {
      count: 0,
      avgImpulse: 0,
      avgNormal: [0, 0],
    };

    const ZERO_FORCE: ForceInput = { dx: 0, dy: 0, dz: 0 };

    const result = new Map<string, VisualProfile>();

    for (const [cellId, cellDef] of Object.entries(cellRegistry)) {
      const profile = CellVisualIdentity.fromCell(
        cellId,
        cellDef.species,
        qosMap[cellId]     ?? DEFAULT_QOS,
        forceField[cellId] ?? ZERO_FORCE,
        contactMap[cellId] ?? ZERO_CONTACTS,
      );
      result.set(cellId, profile);
    }

    return result;
  }

  /**
   * Re-derive only the contact-dependent portion of a profile.
   *
   * Useful during the animation loop when only collision state changes
   * between frames — avoids recomputing the stable species/QoS/force
   * portions of the profile.
   *
   * @param existing  Previously computed VisualProfile.
   * @param contacts  Updated contact summary.
   * @returns         New VisualProfile with updated spark fields.
   */
  static updateContacts(
    existing: VisualProfile,
    contacts: ContactSummary,
  ): VisualProfile {
    const sparks = deriveSparks(contacts);
    return {
      ...existing,
      sparkIntensity: sparks.intensity,
      sparkDirection: sparks.direction,
    };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Re-exports for convenience
// ─────────────────────────────────────────────────────────────────────────────

export type {
  QoSProfile,
  MaterialType,
  PatternShader,
  SdfShape,
};
