# === src/lib/sph/qosSpatial.ts ===
// qosSpatial.ts — QoS preset → physical parameter mapping

import { QoSProfile, SpatialConfig } from './types';

export type QoSProfileName =
  | 'DEFAULT'
  | 'SENSOR_DATA'
  | 'PARAMETERS'
  | 'TF_STATIC'
  | 'TOPO_CHANGE';

export const QOS_PRESETS: Record<QoSProfileName, QoSProfile> = {
  DEFAULT: {
    reliability: 'RELIABLE',
    mps: 0,
    historyDepth: 10,
    durability: 'VOLATILE',
  },
  SENSOR_DATA: {
    reliability: 'BEST_EFFORT',
    mps: 100,
    historyDepth: 5,
    durability: 'VOLATILE',
  },
  PARAMETERS: {
    reliability: 'RELIABLE',
    mps: 10,
    historyDepth: 20,
    durability: 'TRANSIENT_LOCAL',
  },
  TF_STATIC: {
    reliability: 'RELIABLE',
    mps: 0,
    historyDepth: 1,
    durability: 'TRANSIENT_LOCAL',
  },
  TOPO_CHANGE: {
    reliability: 'RELIABLE',
    mps: 0,
    historyDepth: 1,
    durability: 'TRANSIENT_LOCAL',
  },
};

// alias used by SPHWorld.ts (imported as `qosSpatial`)
export const qosSpatial = QOS_PRESETS;

export function qosToSpatial(qos: QoSProfile): SpatialConfig {
  const boundaryStiffness =
    qos.reliability === 'RELIABLE' ? 50000 : 8000;

  const viscosity =
    qos.mps > 0
      ? Math.max(0.001, 0.1 / Math.sqrt(qos.mps))
      : 0.01;

  const persistence = qos.historyDepth * 0.5;

  const restDensity =
    qos.durability === 'TRANSIENT_LOCAL' ? 1200 : 1000;

  const surfaceTension = 0.02;

  return {
    boundaryStiffness,
    viscosity,
    persistence,
    restDensity,
    surfaceTension,
  };
}

export function interpolateConfigs(
  a: SpatialConfig,
  b: SpatialConfig,
  t: number,
): SpatialConfig {
  const lerp = (x: number, y: number) => x + (y - x) * t;
  return {
    boundaryStiffness: lerp(a.boundaryStiffness, b.boundaryStiffness),
    viscosity:         lerp(a.viscosity,         b.viscosity),
    persistence:       lerp(a.persistence,       b.persistence),
    restDensity:       lerp(a.restDensity,        b.restDensity),
    surfaceTension:    lerp(a.surfaceTension,     b.surfaceTension),
  };
}
