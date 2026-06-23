// === src/lib/sph/types.ts ===
// types.ts --- shared interfaces

export interface GPUBufferSet {
  posX: GPUBuffer;
  posY: GPUBuffer;
  velX: GPUBuffer;
  velY: GPUBuffer;
  density: GPUBuffer;
  pressure: GPUBuffer;
  forceX: GPUBuffer;
  forceY: GPUBuffer;
  species: GPUBuffer;
  count: GPUBuffer;
}

export interface SimParams {
  h: number;
  restDensity: number;
  gasConstant: number;
  viscosity: number;
  gravity: number;
  dt: number;
  domainW: number;
  domainH: number;
}

export interface ParticleData {
  x: Float32Array;
  y: Float32Array;
  vx: Float32Array;
  vy: Float32Array;
  species: Uint32Array;
  count: number;
}

export interface ObstacleData {
  cx: number;
  cy: number;
  r: number;
  stiffness: number;
}

export interface NeighborCSR {
  offsetBuf: GPUBuffer;
  listBuf: GPUBuffer;
  offsetCPU: Int32Array;
  listCPU: Int32Array;
}

export interface QoSProfile {
  reliability: 'RELIABLE' | 'BEST_EFFORT';
  mps: number;
  historyDepth: number;
  durability: 'VOLATILE' | 'TRANSIENT_LOCAL';
}

export interface SpatialConfig {
  boundaryStiffness: number;
  viscosity: number;
  restDensity: number;
  surfaceTension: number;
  persistence: number;
}

export const MAX_PARTICLES = 50000;
export const WORKGROUP_SIZE = 256;
export const MAX_NEIGHBORS = 64;

// ------ Rigid body types (used by collision pipeline) ------------------------------------------------------
export interface RigidBody {
  x: number; y: number;        // position
  vx: number; vy: number;      // velocity
  angle: number;                // rotation angle
  omega: number;                // angular velocity
  invMass: number;              // 1/mass (0 = static/kinematic)
  invInertia: number;           // 1/moment of inertia
  restitution: number;          // bounce coefficient
  friction: number;             // coulomb friction
}

export interface ContactConstraint {
  bodyA: number; bodyB: number;
  normal: { x: number; y: number };
  point: { x: number; y: number };
  depth: number;
  normalImpulse: number;        // accumulated (warm start)
  tangentImpulse: number;       // accumulated (warm start)
}
