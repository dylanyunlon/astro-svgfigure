// src/lib/sph/qos-spatial-bridge.ts
// QoS → Physics mapping for SPH spatial renderer
// Mirrors qos_spatial.py (task 07) — same formulas, TypeScript types

export type Reliability = 'RELIABLE' | 'BEST_EFFORT';
export type Durability  = 'VOLATILE' | 'TRANSIENT_LOCAL';

export interface QoSProfile {
  reliability:  Reliability;
  durability:   Durability;
  historyDepth: number;
  mps:          number;
  priority:     number; // 0-3
}

export interface SpatialPhysics {
  viscosity:        number;
  boundaryFriction: number;
  trailLength:      number;
  emitterRate:      number;
  forceMultiplier:  number;
}

// ---------------------------------------------------------------------------
// Core mapping function — identical formulas to Python qos_to_physics()
// ---------------------------------------------------------------------------
export function qosToSpatial(qos: QoSProfile): SpatialPhysics {
  // Reliability → viscosity
  // RELIABLE packets stay in order → high viscosity (thick, ordered flow)
  // BEST_EFFORT packets can drop   → low viscosity  (thin, turbulent flow)
  const viscosity = qos.reliability === 'RELIABLE' ? 0.02 : 0.001;

  // Durability → boundary friction
  // TRANSIENT_LOCAL retains last messages → sticky walls (high friction)
  // VOLATILE forgets immediately          → slippery walls (low friction)
  const boundaryFriction = qos.durability === 'TRANSIENT_LOCAL' ? 0.95 : 0.30;

  // History depth → trail length (capped at 30)
  const trailLength = Math.min(qos.historyDepth * 3, 30);

  // MPS → emitter rate
  // 0 means unlimited → render as max burst (120 particles/s)
  const emitterRate = qos.mps === 0 ? 120.0 : Math.min(qos.mps * 1.5, 120.0);

  // Priority 0-3 → force multiplier 1.0-2.5 (linear interpolation)
  const forceMultiplier = 1.0 + qos.priority * 0.5;

  return { viscosity, boundaryFriction, trailLength, emitterRate, forceMultiplier };
}

// ---------------------------------------------------------------------------
// Apollo CyberRT QoS profiles
// Source: apollo/cyber/transport/qos/qos_profile_conf.cc
// ---------------------------------------------------------------------------
export const APOLLO_PROFILES: Record<string, QoSProfile> = {
  // Default profile — moderate depth, no rate limit
  DEFAULT: {
    reliability:  'RELIABLE',
    durability:   'VOLATILE',
    historyDepth: 1,
    mps:          0,
    priority:     1,
  },

  // High-frequency sensor streams (lidar, camera, radar)
  SENSOR_DATA: {
    reliability:  'BEST_EFFORT',
    durability:   'VOLATILE',
    historyDepth: 5,
    mps:          0,
    priority:     0,
  },

  // ROS-style parameters — need reliable delivery + late-join history
  PARAMETERS: {
    reliability:  'RELIABLE',
    durability:   'TRANSIENT_LOCAL',
    historyDepth: 1000,
    mps:          0,
    priority:     2,
  },

  // Service call channels (request-reply pattern)
  SERVICES_DEFAULT: {
    reliability:  'RELIABLE',
    durability:   'VOLATILE',
    historyDepth: 10,
    mps:          0,
    priority:     2,
  },

  // Parameter events — subset of parameter changes
  PARAM_EVENT: {
    reliability:  'RELIABLE',
    durability:   'TRANSIENT_LOCAL',
    historyDepth: 1000,
    mps:          0,
    priority:     2,
  },

  // System-level default (matches ROS 2 /rosout style)
  SYSTEM_DEFAULT: {
    reliability:  'RELIABLE',
    durability:   'VOLATILE',
    historyDepth: 1,
    mps:          0,
    priority:     1,
  },

  // Static transforms — must survive late joiners, keep full history
  TF_STATIC: {
    reliability:  'RELIABLE',
    durability:   'TRANSIENT_LOCAL',
    historyDepth: 1,
    mps:          0,
    priority:     1,
  },

  // Topology / graph change events
  TOPO_CHANGE: {
    reliability:  'RELIABLE',
    durability:   'TRANSIENT_LOCAL',
    historyDepth: 10,
    mps:          0,
    priority:     3,
  },
};

// ---------------------------------------------------------------------------
// Human-readable descriptions for UI tooltips / legend panels
// ---------------------------------------------------------------------------
export const PROFILE_DESCRIPTIONS: Record<string, string> = {
  DEFAULT:
    'General-purpose channel. Reliable delivery, volatile history, ' +
    'shallow queue (depth 1). Suitable for infrequent control messages.',
  SENSOR_DATA:
    'High-frequency sensor streams (lidar/camera/radar). Best-effort ' +
    'delivery tolerates occasional drops; volatile history discards stale frames.',
  PARAMETERS:
    'Parameter server channel. Reliable + transient-local so late-joining ' +
    'nodes receive the full parameter history on connect.',
  SERVICES_DEFAULT:
    'RPC-style service calls. Reliable, volatile, shallow queue; ' +
    'each request expects a matched response within the session.',
  PARAM_EVENT:
    'Parameter-change event bus. Same reliability/durability as PARAMETERS ' +
    'so subscribers never miss a configuration update.',
  SYSTEM_DEFAULT:
    'System-level fallback profile. Matches Apollo internal infra channels; ' +
    'reliable, volatile, depth 1.',
  TF_STATIC:
    'Static coordinate-frame transforms. Transient-local depth-1 ensures ' +
    'any node that joins after broadcast still receives the transform.',
  TOPO_CHANGE:
    'Graph topology updates (node/edge add-remove). Highest priority (3), ' +
    'transient-local so late joiners reconstruct the current topology.',
};
