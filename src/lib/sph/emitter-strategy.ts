


// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------




import { APOLLO_PROFILES, qosToSpatial } from './qos-spatial-bridge';
import type { SpatialPhysics, QoSProfile } from './qos-spatial-bridge';
import type { CellPhysicsConfig } from './cell-body-bridge';

export interface EmitterConfig {
  x: number;
  y: number;
  dirX: number;
  dirY: number;
  rate: number;        // particles per second (base, before temporal modulation)
  species: number;
  label: string;       // for debug

  /** Temporal emission pattern applied per-tick by the SPH stepper. */
  pattern: EmissionPattern;
}

/**
 * Emission pattern controls *when* and *how many* particles fire each tick.
 * The stepper calls `pattern.sample(t)` to get an instantaneous multiplier
 * that scales `EmitterConfig.rate`.
 */
export interface EmissionPattern {
  /** Human-readable name for the debug overlay. */
  readonly name: string;

  /**
   * Returns a rate multiplier in [0, ∞) for the given simulation time `t`
   * (seconds).  1.0 means emit at `rate` as-is; 0 means silence.
   */
  sample(t: number): number;

  /** Approximate mean multiplier (for legend / normalisation). */
  readonly meanMultiplier: number;
}

// ---------------------------------------------------------------------------
// EmissionPattern implementations
// ---------------------------------------------------------------------------

/**
 * CONTINUOUS — constant multiplier of 1.0.
 * Used as a fallback and for DEFAULT / SYSTEM_DEFAULT profiles.
 */
export class ContinuousPattern implements EmissionPattern {
  readonly name = 'continuous';
  readonly meanMultiplier = 1.0;
  sample(_t: number): number { return 1.0; }
}

/**
 * HIGH_FREQ_STREAM — slight sinusoidal ripple around 1.0, fast period.
 * Models SENSOR_DATA: near-continuous high-frequency burst with minor
 * sensor-jitter fluctuations (±20 %).
 *
 * Multiplier: 1.0 + 0.2 * sin(2π t / periodSec)
 */
export class HighFreqStreamPattern implements EmissionPattern {
  readonly name = 'high-freq-stream';
  readonly meanMultiplier = 1.0;

  constructor(
    /** Ripple period in seconds (default 0.1 s → 10 Hz jitter). */
    private readonly periodSec = 0.1,
  ) {}

  sample(t: number): number {
    // Fast sinusoidal variation around 1, bounded to [0.8, 1.2]
    return 1.0 + 0.2 * Math.sin((2 * Math.PI * t) / this.periodSec);
  }
}

/**
 * LOW_FREQ_PULSE — emits in short discrete bursts separated by long silences.
 * Models PARAMETERS: rare parameter updates, each update is a compact batch
 * of messages delivered all-at-once.
 *
 * Within each burst window the multiplier is `burstPeak`; outside it is 0.
 * Burst duty cycle: burstDurationSec / periodSec.
 */
export class LowFreqPulsePattern implements EmissionPattern {
  readonly name = 'low-freq-pulse';
  readonly meanMultiplier: number;

  constructor(
    /** Total period between bursts (seconds).  Default 2 s → 0.5 Hz. */
    private readonly periodSec = 2.0,
    /** Duration of each burst within the period (seconds).  Default 0.15 s. */
    private readonly burstDurationSec = 0.15,
    /** Multiplier during the burst.  High value compensates for silence. */
    private readonly burstPeak = 8.0,
  ) {
    this.meanMultiplier = (burstDurationSec / periodSec) * burstPeak;
  }

  sample(t: number): number {
    // Phase within the current period [0, periodSec)
    const phase = ((t % this.periodSec) + this.periodSec) % this.periodSec;
    return phase < this.burstDurationSec ? this.burstPeak : 0.0;
  }
}

/**
 * CONSTANT_FIELD — rock-steady emission at exactly 1.0, but spawns particles
 * across a spatial spread rather than a single point.
 *
 * The pattern itself is temporally flat (multiplier = 1).  Spatial spread is
 * implemented by `createStaticFieldEmitters` which fans out multiple emitter
 * points; each one uses this pattern so their aggregate forms a uniform field.
 *
 * Models TF_STATIC: the transform exists forever and is always "on".
 */
export class ConstantFieldPattern implements EmissionPattern {
  readonly name = 'constant-field';
  readonly meanMultiplier = 1.0;
  sample(_t: number): number { return 1.0; }
}

/**
 * BURST_WAVE — emits an intense wavefront followed by complete silence,
 * repeating at a configured interval.
 *
 * Models TOPO_CHANGE: a topology event triggers a brief, high-amplitude flood
 * of update messages that must race ahead of all other traffic (priority 3),
 * after which the channel goes dark until the next change.
 *
 * Shape: Gaussian envelope centred at t=0 within each period.
 *        peak multiplier can reach `peakMultiplier`; tails fall below 0.05
 *        within ~±sigmaSec of the burst centre.
 */
export class BurstWavePattern implements EmissionPattern {
  readonly name = 'burst-wave';
  readonly meanMultiplier: number;

  constructor(
    /** Period between topology-change events (seconds).  Default 3 s. */
    private readonly periodSec = 3.0,
    /** Gaussian σ (spread of the burst, seconds).  Default 0.12 s. */
    private readonly sigmaSec = 0.12,
    /** Peak multiplier at the centre of the burst.  Default 15×. */
    private readonly peakMultiplier = 15.0,
  ) {
    // Numerical mean: integrate Gaussian over period (≈ peak * σ * √(2π) / T)
    this.meanMultiplier = (peakMultiplier * sigmaSec * Math.sqrt(2 * Math.PI)) / periodSec;
  }

  sample(t: number): number {
    // Map t to phase within [-periodSec/2, +periodSec/2)
    const halfPeriod = this.periodSec / 2;
    let phase = ((t % this.periodSec) + this.periodSec) % this.periodSec;
    if (phase > halfPeriod) phase -= this.periodSec; // centre the Gaussian at 0

    const exponent = -(phase * phase) / (2 * this.sigmaSec * this.sigmaSec);
    return this.peakMultiplier * Math.exp(exponent);
  }
}

// ---------------------------------------------------------------------------
// Pattern factory — derive the right pattern from an Apollo QoS profile name
// ---------------------------------------------------------------------------

/**
 * Return the canonical `EmissionPattern` for a named Apollo QoS profile.
 *
 * | Profile         | Pattern           | Rationale                          |
 * |-----------------|-------------------|------------------------------------|
 * | SENSOR_DATA     | HighFreqStream    | Lidar/camera are near-continuous   |
 * | PARAMETERS      | LowFreqPulse      | Config updates are rare batches    |
 * | TF_STATIC       | ConstantField     | Transforms are always present      |
 * | TOPO_CHANGE     | BurstWave         | Graph changes are sudden floods    |
 * | DEFAULT / *     | Continuous        | Generic fallback                   |
 */
export function patternForProfile(profileName: string): EmissionPattern {
  switch (profileName) {
    case 'SENSOR_DATA':
      // High-frequency continuous stream: fast jitter period (80 ms ≈ 12.5 Hz)
      return new HighFreqStreamPattern(0.08);

    case 'PARAMETERS':
    case 'PARAM_EVENT':
      // Low-frequency pulse: one burst every 2 s, 150 ms wide, 8× peak
      return new LowFreqPulsePattern(2.0, 0.15, 8.0);

    case 'TF_STATIC':
      // Temporally flat — spatial spread is handled by the emitter layout
      return new ConstantFieldPattern();

    case 'TOPO_CHANGE':
      // Burst wave: 3 s period, narrow Gaussian (120 ms), 15× peak
      return new BurstWavePattern(3.0, 0.12, 15.0);

    case 'SERVICES_DEFAULT':
      // RPC: slightly pulsed (request-response pairs), moderate frequency
      return new LowFreqPulsePattern(0.5, 0.08, 4.0);

    default:
      // DEFAULT, SYSTEM_DEFAULT, and anything unknown → steady stream
      return new ContinuousPattern();
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

/** Map a QoS spatial config into a base particle emission rate [minRate, maxRate]. */
function spatialToRate(
  sp: SpatialPhysics,
  minRate = 2,
  maxRate = 60,
): number {
  // emitterRate is the canonical QoS-derived value (camelCase, from qosToSpatial)
  const base = sp.emitterRate ?? 20;
  // forceMultiplier (1.0–2.5) nudges the rate upward for higher-priority channels
  const factor = clamp((sp.forceMultiplier ?? 1.0) / 2.5, 0.4, 1.0);
  return clamp(base * factor, minRate, maxRate);
}

/** Pick species index from force multiplier (proxy for QoS priority 0-3). */
function spatialToSpecies(sp: SpatialPhysics): number {
  // forceMultiplier ∈ [1.0, 2.5]; map to species 0-3
  const priority = Math.round(((sp.forceMultiplier ?? 1.0) - 1.0) / 0.5);
  return clamp(priority, 0, 3);
}

// ---------------------------------------------------------------------------
// Strategy: edge emitters
// ---------------------------------------------------------------------------

/**
 * Place one emitter per topology edge, at the midpoint, pointing from source
 * toward target.  Rate and pattern are derived from the QoS profile name so
 * that different message types produce visually distinct flows.
 */
export function createEdgeEmitters(
  cells: CellPhysicsConfig[],
  edges: Array<{ source: string; target: string }>,
  qos: SpatialPhysics,
  profileName = 'DEFAULT',
): EmitterConfig[] {
  const cellMap = new Map<string, CellPhysicsConfig>(
    cells.map((c) => [c.id, c]),
  );

  const rate    = spatialToRate(qos);
  const species = spatialToSpecies(qos);
  const pattern = patternForProfile(profileName);
  const emitters: EmitterConfig[] = [];

  for (const edge of edges) {
    const src = cellMap.get(edge.source);
    const tgt = cellMap.get(edge.target);
    if (!src || !tgt) continue;

    const mx = (src.x + tgt.x) / 2;
    const my = (src.y + tgt.y) / 2;
    const dx = tgt.x - src.x;
    const dy = tgt.y - src.y;
    const len = Math.hypot(dx, dy) || 1;

    emitters.push({
      x: mx,
      y: my,
      dirX: dx / len,
      dirY: dy / len,
      rate,
      species,
      label: `edge:${edge.source}->${edge.target}:${profileName}`,
      pattern,
    });
  }

  return emitters;
}

// ---------------------------------------------------------------------------
// Strategy: boundary emitters
// ---------------------------------------------------------------------------

/**
 * Place emitters on the top boundary (ingress) and bottom boundary (egress).
 * Pattern is derived from the profile so world-edge traffic respects the same
 * temporal shape as in-graph traffic.
 */
export function createBoundaryEmitters(
  worldW: number,
  worldH: number,
  qos: SpatialPhysics,
  profileName = 'DEFAULT',
): EmitterConfig[] {
  const rate    = spatialToRate(qos);
  const species = spatialToSpecies(qos);
  const pattern = patternForProfile(profileName);

  return [
    {
      x: worldW / 2,
      y: 0,
      dirX: 0,
      dirY: 1,
      rate,
      species,
      label: `boundary:top-ingress:${profileName}`,
      pattern,
    },
    {
      x: worldW / 2,
      y: worldH,
      dirX: 0,
      dirY: -1,
      rate: rate * 0.3, // ACK/return traffic is lighter
      species: clamp(species + 1, 0, 3),
      label: `boundary:bottom-egress:${profileName}`,
      pattern,
    },
  ];
}

// ---------------------------------------------------------------------------
// Strategy: rain emitter
// ---------------------------------------------------------------------------

/**
 * A "rain" emitter centred at the top of the world, pointing straight down.
 * Useful for ambient background traffic.  Uses Continuous pattern regardless
 * of profile so the background stays visually calm.
 */
export function createRainEmitter(
  worldW: number,
  qos: SpatialPhysics,
  species: number,
): EmitterConfig {
  const rate = clamp(spatialToRate(qos, 5, 30), 5, 30);

  return {
    x: worldW / 2,
    y: 0,
    dirX: 0,
    dirY: 1,
    rate,
    species: clamp(species, 0, 3),
    label: 'rain:uniform-top',
    pattern: new ContinuousPattern(),
  };
}

// ---------------------------------------------------------------------------
// Strategy: static field emitters (TF_STATIC — spread across the world)
// ---------------------------------------------------------------------------

/**
 * Spawn a grid of evenly-spaced emitters pointing downward, each using the
 * ConstantFieldPattern.  Together they produce a uniform "always-on" field
 * that visually matches TF_STATIC semantics: the transform is omnipresent.
 *
 * @param worldW  World width in pixels.
 * @param worldH  World height in pixels.
 * @param columns Number of columns in the emitter grid (default 4).
 * @param qos     Spatial physics derived from TF_STATIC profile.
 */
export function createStaticFieldEmitters(
  worldW: number,
  worldH: number,
  columns = 4,
  qos: SpatialPhysics,
): EmitterConfig[] {
  const rate    = spatialToRate(qos, 3, 20); // lower per-emitter rate; many emitters
  const species = spatialToSpecies(qos);
  const pattern = new ConstantFieldPattern();
  const emitters: EmitterConfig[] = [];

  const step = worldW / (columns + 1);
  for (let col = 1; col <= columns; col++) {
    emitters.push({
      x: step * col,
      y: worldH * 0.05, // near the top
      dirX: 0,
      dirY: 1,
      rate,
      species,
      label: `static-field:col${col}`,
      pattern,
    });
  }

  return emitters;
}

// ---------------------------------------------------------------------------
// Strategy: burst-origin emitters (TOPO_CHANGE — radial burst from a cell)
// ---------------------------------------------------------------------------

/**
 * Place a radial fan of emitters around a specific cell (or the world centre),
 * all using BurstWavePattern.  The fan fires in multiple directions to simulate
 * a topology-change event propagating outward from its origin.
 *
 * @param originX   X-coordinate of the burst origin.
 * @param originY   Y-coordinate of the burst origin.
 * @param rays      Number of radial rays (default 8 → every 45°).
 * @param qos       Spatial physics derived from TOPO_CHANGE profile.
 * @param periodSec Seconds between burst events.
 */
export function createBurstOriginEmitters(
  originX: number,
  originY: number,
  rays = 8,
  qos: SpatialPhysics,
  periodSec = 3.0,
): EmitterConfig[] {
  const rate    = spatialToRate(qos, 10, 60);
  const species = spatialToSpecies(qos);
  const pattern = new BurstWavePattern(periodSec, 0.12, 15.0);
  const emitters: EmitterConfig[] = [];

  for (let i = 0; i < rays; i++) {
    const angle = (2 * Math.PI * i) / rays;
    emitters.push({
      x: originX,
      y: originY,
      dirX: Math.cos(angle),
      dirY: Math.sin(angle),
      rate,
      species,
      label: `burst-origin:ray${i}`,
      pattern,
    });
  }

  return emitters;
}

// ---------------------------------------------------------------------------
// Combined default strategy (profile-aware)
// ---------------------------------------------------------------------------

/**
 * Compose the appropriate set of emitters for a given QoS profile name.
 *
 * | Profile     | Layout strategy                                         |
 * |-------------|---------------------------------------------------------|
 * | SENSOR_DATA | Edge emitters + boundaries (HighFreqStream pattern)     |
 * | PARAMETERS  | Edge emitters + boundaries (LowFreqPulse pattern)       |
 * | TF_STATIC   | Static field grid (ConstantField pattern)               |
 * | TOPO_CHANGE | Burst-origin radial fans at hub cells (BurstWave)       |
 * | *           | Edge + boundary + rain emitters (Continuous pattern)    |
 *
 * Duplicate positions are deduplicated by rounded pixel + species key.
 */
export function defaultEmitterStrategy(
  cells: CellPhysicsConfig[],
  edges: Array<{ source: string; target: string }>,
  worldW: number,
  worldH: number,
  qos: SpatialPhysics,
  profileName = 'DEFAULT',
): EmitterConfig[] {
  let all: EmitterConfig[];

  switch (profileName) {

    // ── SENSOR_DATA: high-frequency continuous stream ─────────────────────
    case 'SENSOR_DATA': {
      // Full edge coverage + both boundary walls; no rain (already high-freq)
      all = [
        ...createEdgeEmitters(cells, edges, qos, profileName),
        ...createBoundaryEmitters(worldW, worldH, qos, profileName),
      ];
      break;
    }

    // ── PARAMETERS / PARAM_EVENT: low-frequency pulsed delivery ──────────
    case 'PARAMETERS':
    case 'PARAM_EVENT': {
      // Edge emitters only — parameter updates travel the graph, no ambient
      all = createEdgeEmitters(cells, edges, qos, profileName);
      break;
    }

    // ── TF_STATIC: spatially uniform, temporally constant field ──────────
    case 'TF_STATIC': {
      // Grid of static-field emitters; no edge emitters (not link-specific)
      all = createStaticFieldEmitters(worldW, worldH, 5, qos);
      break;
    }

    // ── TOPO_CHANGE: burst waves from high-degree hub cells ───────────────
    case 'TOPO_CHANGE': {
      // Identify hub cells (appear in the most edges)
      const degree = new Map<string, number>();
      for (const e of edges) {
        degree.set(e.source, (degree.get(e.source) ?? 0) + 1);
        degree.set(e.target, (degree.get(e.target) ?? 0) + 1);
      }

      // Pick top-2 hubs (fall back to world centre if no cells/edges)
      const cellMap = new Map<string, CellPhysicsConfig>(
        cells.map((c) => [c.id, c]),
      );
      const hubs = [...degree.entries()]
        .sort((a, b) => b[1] - a[1])
        .slice(0, 2)
        .map(([id]) => cellMap.get(id))
        .filter(Boolean) as CellPhysicsConfig[];

      if (hubs.length === 0) {
        // No topology info — single burst from world centre
        all = createBurstOriginEmitters(worldW / 2, worldH / 2, 8, qos);
      } else {
        all = hubs.flatMap((hub, idx) =>
          createBurstOriginEmitters(
            hub.x,
            hub.y,
            6, // 6 rays each
            qos,
            3.0 + idx * 0.5, // stagger periods slightly so waves don't sync
          ),
        );
      }
      break;
    }

    // ── Default / everything else ─────────────────────────────────────────
    default: {
      all = [
        ...createEdgeEmitters(cells, edges, qos, profileName),
        ...createBoundaryEmitters(worldW, worldH, qos, profileName),
        createRainEmitter(worldW, qos, 0),
      ];
    }
  }

  // Deduplicate by rounded position + species (avoids stacked emitters at
  // shared edge midpoints in dense star topologies).
  const seen = new Set<string>();
  return all.filter((e) => {
    const key = `${Math.round(e.x)},${Math.round(e.y)},${e.species},${e.pattern.name}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

// ---------------------------------------------------------------------------
// Convenience: build emitters directly from a named Apollo profile
// ---------------------------------------------------------------------------

/**
 * High-level helper that resolves the Apollo profile by name, derives the
 * spatial physics, and delegates to `defaultEmitterStrategy`.
 *
 * ```ts
 * const emitters = emittersForApolloProfile(
 *   'SENSOR_DATA', cells, edges, 800, 600,
 * );
 * ```
 */
export function emittersForApolloProfile(
  profileName: keyof typeof APOLLO_PROFILES | string,
  cells: CellPhysicsConfig[],
  edges: Array<{ source: string; target: string }>,
  worldW: number,
  worldH: number,
): EmitterConfig[] {
  const profile = APOLLO_PROFILES[profileName] ?? APOLLO_PROFILES['DEFAULT'];
  const spatial = qosToSpatial(profile);
  return defaultEmitterStrategy(cells, edges, worldW, worldH, spatial, profileName);
}
