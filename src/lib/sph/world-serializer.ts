/**
 * world-serializer.ts
 *
 * Binary serialization / deserialization for World snapshots.
 *
 * Format (little-endian throughout):
 *
 * ┌─────────────────────────────────────────────────────────────────┐
 * │ HEADER  (64 bytes)                                             │
 * │  [0..3]   u32  magic        0x53504857  ("SPHW")               │
 * │  [4..7]   u32  version      1                                  │
 * │  [8..11]  u32  particleCount                                   │
 * │  [12..15] u32  bodyCount                                       │
 * │  [16..19] f32  config.width                                    │
 * │  [20..23] f32  config.height                                   │
 * │  [24..27] f32  config.gravity                                  │
 * │  [28..31] f32  config.particleRadius                           │
 * │  [32..35] f32  config.smoothingRadius                          │
 * │  [36..39] f32  config.restDensity                              │
 * │  [40..43] f32  config.viscosity                                │
 * │  [44..47] f32  config.dt                                       │
 * │  [48..51] u32  config.substeps                                 │
 * │  [52..55] u32  config.maxParticles                             │
 * │  [56..59] f32  config.restitution                              │
 * │  [60..63] u32  frame                                           │
 * ├─────────────────────────────────────────────────────────────────┤
 * │ PARTICLES  (particleCount × 48 bytes each)                     │
 * │  Per particle:                                                 │
 * │   [0..3]   u32  id                                             │
 * │   [4..7]   f32  x                                              │
 * │   [8..11]  f32  y                                              │
 * │   [12..15] f32  vx                                             │
 * │   [16..19] f32  vy                                             │
 * │   [20..23] f32  ax                                             │
 * │   [24..27] f32  ay                                             │
 * │   [28..31] f32  density                                        │
 * │   [32..35] f32  pressure                                       │
 * │   [36..39] f32  mass                                           │
 * │   [40..43] f32  alpha                                          │
 * │   [44..47] u32  speciesLen  (followed by species string        │
 * │                              in the species string table)      │
 * ├─────────────────────────────────────────────────────────────────┤
 * │ BODIES  (bodyCount × 80 bytes each)                            │
 * │  Per body:                                                     │
 * │   [0..3]   u32  id                                             │
 * │   [4..7]   f32  x                                              │
 * │   [8..11]  f32  y                                              │
 * │   [12..15] f32  vx                                             │
 * │   [16..19] f32  vy                                             │
 * │   [20..23] f32  angle                                          │
 * │   [24..27] f32  angVel                                         │
 * │   [28..31] f32  mass                                           │
 * │   [32..35] f32  inertia                                        │
 * │   [36..39] f32  w   (half-width)                               │
 * │   [40..43] f32  h   (half-height)                              │
 * │   [44..47] f32  restitution                                    │
 * │   [48..51] f32  friction                                       │
 * │   [52..55] u32  pinned  (0 or 1)                               │
 * │   [56..59] u32  species                                        │
 * │   [60..63] f32  fx                                             │
 * │   [64..67] f32  fy                                             │
 * │   [68..71] f32  torque                                         │
 * │   [72..75] u32  boundaryCount                                  │
 * │   [76..79] u32  _reserved                                      │
 * ├─────────────────────────────────────────────────────────────────┤
 * │ SPECIES STRING TABLE                                           │
 * │  Concatenated UTF-8 strings, lengths given by per-particle     │
 * │  speciesLen fields, no delimiters.                             │
 * ├─────────────────────────────────────────────────────────────────┤
 * │ BOUNDARY DATA  (per body, bodyCount sections)                  │
 * │  Per body with boundaryCount > 0:                              │
 * │    boundaryX   Float64[boundaryCount]                          │
 * │    boundaryY   Float64[boundaryCount]                          │
 * │    boundaryNx  Float64[boundaryCount]                          │
 * │    boundaryNy  Float64[boundaryCount]                          │
 * └─────────────────────────────────────────────────────────────────┘
 */

import type { Particle, World, WorldConfig, Emitter } from './world-stepper';
import type { RigidBody } from './rigid-body';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAGIC = 0x53504857;          // "SPHW" in little-endian
const VERSION = 1;
const HEADER_BYTES = 64;
const PARTICLE_BYTES = 48;         // fixed fields per particle
const BODY_BYTES = 80;             // fixed fields per body

const encoder = new TextEncoder();
const decoder = new TextDecoder();

// ---------------------------------------------------------------------------
// serializeWorld
// ---------------------------------------------------------------------------

/**
 * Serialize a `World` snapshot into a compact binary `ArrayBuffer`.
 *
 * The encoding captures all information needed to reconstruct the physical
 * state: particle positions/velocities/properties, rigid body state including
 * boundary particle arrays, and the world configuration.
 *
 * Transient / derived fields (spatial hash, solver state, trails, emitters,
 * performance budget, QoS) are intentionally omitted — the caller is expected
 * to re-initialize those after deserialization via `createWorld` + patching.
 */
export function serializeWorld(world: World): ArrayBuffer {
  const { particles, rigidBodies, config, frame } = world;
  const particleCount = particles.length;
  const bodyCount = rigidBodies.length;

  // ── 1. Encode species strings ────────────────────────────────────────
  const speciesBuffers: Uint8Array[] = [];
  let speciesTableBytes = 0;
  for (const p of particles) {
    const buf = encoder.encode(p.species);
    speciesBuffers.push(buf);
    speciesTableBytes += buf.byteLength;
  }

  // ── 2. Compute boundary data size ────────────────────────────────────
  let boundaryDataBytes = 0;
  for (const body of rigidBodies) {
    // 4 arrays × Float64 (8 bytes) × boundaryCount
    boundaryDataBytes += body.boundaryCount * 4 * 8;
  }

  // ── 3. Allocate ──────────────────────────────────────────────────────
  const totalBytes =
    HEADER_BYTES +
    particleCount * PARTICLE_BYTES +
    bodyCount * BODY_BYTES +
    speciesTableBytes +
    boundaryDataBytes;

  const buffer = new ArrayBuffer(totalBytes);
  const view = new DataView(buffer);
  const LE = true; // little-endian

  // ── 4. Write header ──────────────────────────────────────────────────
  let off = 0;
  view.setUint32(off, MAGIC, LE);            off += 4;  // [0]
  view.setUint32(off, VERSION, LE);          off += 4;  // [4]
  view.setUint32(off, particleCount, LE);    off += 4;  // [8]
  view.setUint32(off, bodyCount, LE);        off += 4;  // [12]
  view.setFloat32(off, config.width, LE);    off += 4;  // [16]
  view.setFloat32(off, config.height, LE);   off += 4;  // [20]
  view.setFloat32(off, config.gravity, LE);  off += 4;  // [24]
  view.setFloat32(off, config.particleRadius, LE); off += 4;  // [28]
  view.setFloat32(off, config.smoothingRadius, LE); off += 4; // [32]
  view.setFloat32(off, config.restDensity, LE);     off += 4; // [36]
  view.setFloat32(off, config.viscosity, LE);        off += 4; // [40]
  view.setFloat32(off, config.dt, LE);               off += 4; // [44]
  view.setUint32(off, config.substeps, LE);          off += 4; // [48]
  view.setUint32(off, config.maxParticles, LE);      off += 4; // [52]
  view.setFloat32(off, config.restitution, LE);      off += 4; // [56]
  view.setUint32(off, frame, LE);                    off += 4; // [60]

  // ── 5. Write particles ───────────────────────────────────────────────
  for (let i = 0; i < particleCount; i++) {
    const p = particles[i];
    const specLen = speciesBuffers[i].byteLength;
    view.setUint32(off, p.id, LE);       off += 4;
    view.setFloat32(off, p.x, LE);       off += 4;
    view.setFloat32(off, p.y, LE);       off += 4;
    view.setFloat32(off, p.vx, LE);      off += 4;
    view.setFloat32(off, p.vy, LE);      off += 4;
    view.setFloat32(off, p.ax, LE);      off += 4;
    view.setFloat32(off, p.ay, LE);      off += 4;
    view.setFloat32(off, p.density, LE); off += 4;
    view.setFloat32(off, p.pressure, LE);off += 4;
    view.setFloat32(off, p.mass, LE);    off += 4;
    view.setFloat32(off, p.alpha, LE);   off += 4;
    view.setUint32(off, specLen, LE);    off += 4;
  }

  // ── 6. Write bodies ──────────────────────────────────────────────────
  for (const body of rigidBodies) {
    view.setUint32(off, body.id, LE);             off += 4;
    view.setFloat32(off, body.x, LE);             off += 4;
    view.setFloat32(off, body.y, LE);             off += 4;
    view.setFloat32(off, body.vx, LE);            off += 4;
    view.setFloat32(off, body.vy, LE);            off += 4;
    view.setFloat32(off, body.angle, LE);         off += 4;
    view.setFloat32(off, body.angVel, LE);        off += 4;
    view.setFloat32(off, body.mass, LE);          off += 4;
    view.setFloat32(off, body.inertia, LE);       off += 4;
    view.setFloat32(off, body.w, LE);             off += 4;
    view.setFloat32(off, body.h, LE);             off += 4;
    view.setFloat32(off, body.restitution, LE);   off += 4;
    view.setFloat32(off, body.friction, LE);      off += 4;
    view.setUint32(off, body.pinned ? 1 : 0, LE); off += 4;
    view.setUint32(off, body.species, LE);        off += 4;
    view.setFloat32(off, body.fx, LE);            off += 4;
    view.setFloat32(off, body.fy, LE);            off += 4;
    view.setFloat32(off, body.torque, LE);        off += 4;
    view.setUint32(off, body.boundaryCount, LE);  off += 4;
    view.setUint32(off, 0, LE);                   off += 4; // _reserved
  }

  // ── 7. Write species string table ────────────────────────────────────
  const byteView = new Uint8Array(buffer);
  for (const buf of speciesBuffers) {
    byteView.set(buf, off);
    off += buf.byteLength;
  }

  // ── 8. Write boundary data ───────────────────────────────────────────
  for (const body of rigidBodies) {
    const bc = body.boundaryCount;
    if (bc === 0) continue;
    // Each array: Float64 × boundaryCount
    for (let i = 0; i < bc; i++) { view.setFloat64(off, body.boundaryX[i], LE);  off += 8; }
    for (let i = 0; i < bc; i++) { view.setFloat64(off, body.boundaryY[i], LE);  off += 8; }
    for (let i = 0; i < bc; i++) { view.setFloat64(off, body.boundaryNx[i], LE); off += 8; }
    for (let i = 0; i < bc; i++) { view.setFloat64(off, body.boundaryNy[i], LE); off += 8; }
  }

  return buffer;
}

// ---------------------------------------------------------------------------
// deserializeWorld
// ---------------------------------------------------------------------------

/**
 * Reconstruct a `World` from a binary `ArrayBuffer` produced by
 * `serializeWorld`.
 *
 * The returned `World` has fully populated `particles` and `rigidBodies`
 * arrays and a valid `config`. Transient fields (`_hash`, `_solver`,
 * `_collisionWorld`, `trails`, `emitters`, `qos`, `perfBudget`) are set to
 * sensible zero/empty defaults — the caller should re-initialize the
 * simulation infrastructure (e.g. via the helpers in `world-stepper.ts`)
 * before resuming stepping.
 */
export function deserializeWorld(buf: ArrayBuffer): World {
  const view = new DataView(buf);
  const LE = true;
  let off = 0;

  // ── 1. Read & validate header ────────────────────────────────────────
  const magic = view.getUint32(off, LE); off += 4;
  if (magic !== MAGIC) {
    throw new Error(
      `world-serializer: bad magic 0x${magic.toString(16)} (expected 0x${MAGIC.toString(16)})`,
    );
  }

  const version = view.getUint32(off, LE); off += 4;
  if (version !== VERSION) {
    throw new Error(
      `world-serializer: unsupported version ${version} (expected ${VERSION})`,
    );
  }

  const particleCount = view.getUint32(off, LE); off += 4;
  const bodyCount     = view.getUint32(off, LE); off += 4;

  const config: WorldConfig = {
    width:           view.getFloat32(off, LE),  // [16]
    height:          (off += 4, view.getFloat32(off, LE)),  // [20]
    gravity:         (off += 4, view.getFloat32(off, LE)),  // [24]
    particleRadius:  (off += 4, view.getFloat32(off, LE)),  // [28]
    smoothingRadius: (off += 4, view.getFloat32(off, LE)),  // [32]
    restDensity:     (off += 4, view.getFloat32(off, LE)),  // [36]
    viscosity:       (off += 4, view.getFloat32(off, LE)),  // [40]
    dt:              (off += 4, view.getFloat32(off, LE)),  // [44]
    substeps:        (off += 4, view.getUint32(off, LE)),   // [48]
    maxParticles:    (off += 4, view.getUint32(off, LE)),   // [52]
    restitution:     (off += 4, view.getFloat32(off, LE)),  // [56]
    trailLength:     20,  // default, not serialized
  };
  off += 4;

  const frame = view.getUint32(off, LE); off += 4;  // [60]

  // ── 2. Read particles (fixed fields) ─────────────────────────────────
  const particles: Particle[] = new Array(particleCount);
  const speciesLengths: number[] = new Array(particleCount);

  for (let i = 0; i < particleCount; i++) {
    const id       = view.getUint32(off, LE);  off += 4;
    const x        = view.getFloat32(off, LE); off += 4;
    const y        = view.getFloat32(off, LE); off += 4;
    const vx       = view.getFloat32(off, LE); off += 4;
    const vy       = view.getFloat32(off, LE); off += 4;
    const ax       = view.getFloat32(off, LE); off += 4;
    const ay       = view.getFloat32(off, LE); off += 4;
    const density  = view.getFloat32(off, LE); off += 4;
    const pressure = view.getFloat32(off, LE); off += 4;
    const mass     = view.getFloat32(off, LE); off += 4;
    const alpha    = view.getFloat32(off, LE); off += 4;
    const specLen  = view.getUint32(off, LE);  off += 4;

    speciesLengths[i] = specLen;
    particles[i] = {
      id, x, y, vx, vy, ax, ay,
      density, pressure, mass, alpha,
      species: '',  // filled in after reading the string table
    };
  }

  // ── 3. Read bodies (fixed fields) ────────────────────────────────────
  const rigidBodies: RigidBody[] = new Array(bodyCount);
  const boundaryCounts: number[] = new Array(bodyCount);

  for (let i = 0; i < bodyCount; i++) {
    const id            = view.getUint32(off, LE);  off += 4;
    const x             = view.getFloat32(off, LE); off += 4;
    const y             = view.getFloat32(off, LE); off += 4;
    const vx            = view.getFloat32(off, LE); off += 4;
    const vy            = view.getFloat32(off, LE); off += 4;
    const angle         = view.getFloat32(off, LE); off += 4;
    const angVel        = view.getFloat32(off, LE); off += 4;
    const mass          = view.getFloat32(off, LE); off += 4;
    const inertia       = view.getFloat32(off, LE); off += 4;
    const w             = view.getFloat32(off, LE); off += 4;
    const h             = view.getFloat32(off, LE); off += 4;
    const restitution   = view.getFloat32(off, LE); off += 4;
    const friction      = view.getFloat32(off, LE); off += 4;
    const pinned        = view.getUint32(off, LE) !== 0; off += 4;
    const species       = view.getUint32(off, LE);  off += 4;
    const fx            = view.getFloat32(off, LE); off += 4;
    const fy            = view.getFloat32(off, LE); off += 4;
    const torque        = view.getFloat32(off, LE); off += 4;
    const boundaryCount = view.getUint32(off, LE);  off += 4;
    /* _reserved */                                  off += 4;

    boundaryCounts[i] = boundaryCount;
    rigidBodies[i] = {
      id, x, y, vx, vy, angle, angVel,
      mass, inertia, w, h,
      restitution, friction, pinned, species,
      fx, fy, torque,
      boundaryCount,
      boundaryX:  new Float64Array(boundaryCount),
      boundaryY:  new Float64Array(boundaryCount),
      boundaryNx: new Float64Array(boundaryCount),
      boundaryNy: new Float64Array(boundaryCount),
    };
  }

  // ── 4. Read species string table ─────────────────────────────────────
  const byteView = new Uint8Array(buf);
  for (let i = 0; i < particleCount; i++) {
    const len = speciesLengths[i];
    particles[i].species = decoder.decode(byteView.slice(off, off + len));
    off += len;
  }

  // ── 5. Read boundary data ────────────────────────────────────────────
  for (let i = 0; i < bodyCount; i++) {
    const bc = boundaryCounts[i];
    if (bc === 0) continue;
    const body = rigidBodies[i];
    for (let j = 0; j < bc; j++) { body.boundaryX[j]  = view.getFloat64(off, LE); off += 8; }
    for (let j = 0; j < bc; j++) { body.boundaryY[j]  = view.getFloat64(off, LE); off += 8; }
    for (let j = 0; j < bc; j++) { body.boundaryNx[j] = view.getFloat64(off, LE); off += 8; }
    for (let j = 0; j < bc; j++) { body.boundaryNy[j] = view.getFloat64(off, LE); off += 8; }
  }

  // ── 6. Assemble World ────────────────────────────────────────────────
  // Transient fields are zeroed / emptied. The caller is responsible for
  // re-initializing the solver, collision world, spatial hash, QoS, and
  // performance budget before resuming simulation.
  return {
    particles,
    rigidBodies,
    wallParticles: [],
    config,
    qos: null as unknown as World['qos'],
    perfBudget: null as unknown as World['perfBudget'],
    frame,
    time: frame * config.dt,
    substeps: config.substeps,
    emitters: [],
    trails: new Map(),
    _hash: null,
    _solver: null as unknown as World['_solver'],
    _collisionWorld: null as unknown as World['_collisionWorld'],
    _nextParticleId: particles.length > 0
      ? Math.max(...particles.map((p) => p.id)) + 1
      : 0,
    _nextEmitterId: 0,
  };
}
