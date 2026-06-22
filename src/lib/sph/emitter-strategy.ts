import { SpatialPhysics } from './qos-spatial-bridge';
import { CellPhysicsConfig } from './cell-body-bridge';

export interface EmitterConfig {
  x: number;
  y: number;
  dirX: number;
  dirY: number;
  rate: number;      // particles per second (from QoS)
  species: number;
  label: string;     // for debug
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

/** Map a QoS latency + throughput into a particle emission rate.
 *  Higher throughput → more particles; higher latency → slower rate.
 *  Returns particles-per-second in [minRate, maxRate].
 */
function qosToRate(
  qos: SpatialPhysics,
  minRate = 2,
  maxRate = 60
): number {
  // emitter_rate is the canonical QoS-derived value
  const base = qos.emitter_rate ?? 20;

  // Scale by throughput factor (0–1 normalized, saturates at 1 Gbps)
  const throughputMbps = qos.throughput_mbps ?? 100;
  const throughputFactor = clamp(throughputMbps / 1000, 0.05, 1.0);

  // Penalise by latency (low latency = factor near 1)
  const latencyMs = qos.latency_ms ?? 20;
  const latencyFactor = clamp(1 - latencyMs / 200, 0.1, 1.0);

  const raw = base * throughputFactor * latencyFactor;
  return clamp(raw, minRate, maxRate);
}

/** Pick species index from QoS priority tier (0–3). */
function qosToSpecies(qos: SpatialPhysics): number {
  const priority = qos.priority ?? 0;
  return clamp(Math.round(priority), 0, 3);
}

// ---------------------------------------------------------------------------
// Strategy: edge emitters
// ---------------------------------------------------------------------------

/**
 * Place one emitter per topology edge, at the midpoint, pointing from source
 * toward target.  Rate is derived from the shared QoS profile.
 */
export function createEdgeEmitters(
  cells: CellPhysicsConfig[],
  edges: Array<{ source: string; target: string }>,
  qos: SpatialPhysics
): EmitterConfig[] {
  const cellMap = new Map<string, CellPhysicsConfig>(
    cells.map((c) => [c.id, c])
  );

  const rate = qosToRate(qos);
  const species = qosToSpecies(qos);
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
      label: `edge:${edge.source}->${edge.target}`,
    });
  }

  return emitters;
}

// ---------------------------------------------------------------------------
// Strategy: boundary emitters
// ---------------------------------------------------------------------------

/**
 * Place emitters on the top boundary (ingress / input traffic) and bottom
 * boundary (egress / output traffic) of the simulation world.
 */
export function createBoundaryEmitters(
  worldW: number,
  worldH: number,
  qos: SpatialPhysics
): EmitterConfig[] {
  const rate = qosToRate(qos);
  const species = qosToSpecies(qos);

  return [
    // Top-center → pointing downward (ingress)
    {
      x: worldW / 2,
      y: 0,
      dirX: 0,
      dirY: 1,
      rate,
      species,
      label: 'boundary:top-ingress',
    },
    // Bottom-center → pointing upward (egress / ACK flow)
    {
      x: worldW / 2,
      y: worldH,
      dirX: 0,
      dirY: -1,
      rate: rate * 0.3, // ACK traffic is lighter
      species: clamp(species + 1, 0, 3),
      label: 'boundary:bottom-egress',
    },
  ];
}

// ---------------------------------------------------------------------------
// Strategy: rain emitter
// ---------------------------------------------------------------------------

/**
 * A single "rain" emitter centred at the top of the world, fan-angle wide,
 * pointing straight down.  Useful for ambient background traffic.
 */
export function createRainEmitter(
  worldW: number,
  qos: SpatialPhysics,
  species: number
): EmitterConfig {
  const rate = qosToRate(qos, 5, 30);

  return {
    x: worldW / 2,
    y: 0,
    dirX: 0,
    dirY: 1,
    rate,
    species: clamp(species, 0, 3),
    label: 'rain:uniform-top',
  };
}

// ---------------------------------------------------------------------------
// Combined default strategy
// ---------------------------------------------------------------------------

/**
 * Compose all strategies into a single emitter list:
 *   1. One edge emitter per topology connection  (primary data flow)
 *   2. Two boundary emitters                     (world-edge ingress/egress)
 *   3. One rain emitter                          (background ambient traffic)
 *
 * Duplicate positions are deduplicated by rounding to the nearest pixel.
 */
export function defaultEmitterStrategy(
  cells: CellPhysicsConfig[],
  edges: Array<{ source: string; target: string }>,
  worldW: number,
  worldH: number,
  qos: SpatialPhysics
): EmitterConfig[] {
  const all: EmitterConfig[] = [
    ...createEdgeEmitters(cells, edges, qos),
    ...createBoundaryEmitters(worldW, worldH, qos),
    createRainEmitter(worldW, qos, 0),
  ];

  // Deduplicate by rounded position to avoid stacked emitters at shared
  // midpoints (e.g. star topology where many edges share a hub cell).
  const seen = new Set<string>();
  return all.filter((e) => {
    const key = `${Math.round(e.x)},${Math.round(e.y)},${e.species}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}
