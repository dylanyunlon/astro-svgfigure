/**
 * at-antimatter-particles.ts — M838
 * AT Antimatter Particle System → WebGPU / WGSL Port
 * Spawn/update/render + Cell boundary ejection + SPH physics bridging.
 * Ported from AT compiled.vs: AntimatterSpawn.fs, ProtonAntimatter.fs,
 * AntimatterPosition.vs, AntimatterBasicFrag.fs, antimatter.glsl, range.glsl
 */

const WG = 64 as const;

const MAX_PARTICLES = 65536 as const;

const TEX_W = 256 as const;
const TEX_H = 256 as const;

const PARTICLE_STRIDE = 20 as const;

const MAX_EMITTERS = 128 as const;

const EMITTER_STRIDE = 12 as const;

const MAX_CELL_FACES = 6 as const;

export interface CellBoundaryFace {
  faceId:   string;
  /** Cell owning this boundary face. */
  cellId:   string;
  targetId: string;
  /** Face midpoint in domain space. */
  origin:   { x: number; y: number };
  /** Outward normal direction (unit vector). */
  normal:   { x: number; y: number };
  /** Face half-length in domain units. */
  halfLen:  number;
  /** Species colour index for palette lookup. */
  species?: number;
}

export interface ATAntimatterConfig {
  /** AT: decay speed per frame (default 0.005). */
  decay?:            number;
  decayRandom?:      [number, number];
  /** AT: curl noise spatial scale (default 0.5). */
  curlScale?:        number;
  /** AT: curl noise time speed (default 0.3). */
  curlSpeed?:        number;
  /** AT: curl force strength (default 0.02). */
  curlStrength?:     number;
  originStrength?:   number;
  hz?:               number;
  dpr?:              number;
  uSize?:            number;
  sprayStrength?:    number;
  spraySpread?:      number;
  sphSmoothingH?:    number;
  sphRestDensity?:   number;
  onHandoff?: (
    cellId:    string,
    targetId:  string,
    x: number, y: number,
    vx: number, vy: number,
    species: number,
    density: number,
  ) => void;
}

export interface EmitterRequest {
  /** Which boundary face to emit from. */
  face:       CellBoundaryFace;
  /** Number of particles to emit. */
  count:      number;
  strength?:  number;
  /** Initial particle life (0–1). Default 1.0. */
  life?:      number;
}

const DEFAULTS = {
  decay:           0.005,
  decayRandom:     [0.5, 1.5] as [number, number],
  curlScale:       0.5,
  curlSpeed:       0.3,
  curlStrength:    0.02,
  originStrength:  0.05,
  hz:              60,
  dpr:             1,
  uSize:           0.02,
  sprayStrength:   2.0,
  spraySpread:     Math.PI / 4,
  sphSmoothingH:   0.04,
  sphRestDensity:  1000,
} as const;

const NOISE_WGSL = /* wgsl */`
fn hash3(p: vec3f) -> vec3f {
  var q = vec3f(
    dot(p, vec3f(127.1, 311.7, 74.7)),
    dot(p, vec3f(269.5, 183.3, 246.1)),
    dot(p, vec3f(113.5, 271.9, 124.6)),
  );
  return fract(sin(q) * 43758.5453123);
}

fn hash1(n: f32) -> f32 {
  return fract(sin(n) * 43758.5453123);
}

fn noise3(x: vec3f) -> f32 {
  let i  = floor(x);
  let f  = fract(x);
  let u  = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

  let n000 = dot(hash3(i + vec3f(0,0,0)) * 2.0 - 1.0,  f - vec3f(0,0,0));
  let n100 = dot(hash3(i + vec3f(1,0,0)) * 2.0 - 1.0,  f - vec3f(1,0,0));
  let n010 = dot(hash3(i + vec3f(0,1,0)) * 2.0 - 1.0,  f - vec3f(0,1,0));
  let n110 = dot(hash3(i + vec3f(1,1,0)) * 2.0 - 1.0,  f - vec3f(1,1,0));
  let n001 = dot(hash3(i + vec3f(0,0,1)) * 2.0 - 1.0,  f - vec3f(0,0,1));
  let n101 = dot(hash3(i + vec3f(1,0,1)) * 2.0 - 1.0,  f - vec3f(1,0,1));
  let n011 = dot(hash3(i + vec3f(0,1,1)) * 2.0 - 1.0,  f - vec3f(0,1,1));
  let n111 = dot(hash3(i + vec3f(1,1,1)) * 2.0 - 1.0,  f - vec3f(1,1,1));

  return mix(
    mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
    mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y),
    u.z,
  );
}

fn curlNoise2D(p: vec3f, eps: f32) -> vec2f {
  let dx   = vec3f(eps, 0.0, 0.0);
  let dy   = vec3f(0.0, eps, 0.0);
  let dz   = vec3f(0.0, 0.0, eps);
  let Fz_x = (noise3(p + dx + dz) - noise3(p - dx + dz) - noise3(p + dx) + noise3(p - dx)) / (4.0 * eps * eps);
  let Fz_y = (noise3(p + dy + dz) - noise3(p - dy + dz) - noise3(p + dy) + noise3(p - dy)) / (4.0 * eps * eps);
  return vec2f(-Fz_y, Fz_x);
}

fn curlNoise3D(p: vec3f, eps: f32) -> vec3f {
  let dx = vec3f(eps, 0.0, 0.0);
  let dy = vec3f(0.0, eps, 0.0);
  let dz = vec3f(0.0, 0.0, eps);

  let p_x0 = noise3(p - dx);
  let p_x1 = noise3(p + dx);
  let p_y0 = noise3(p - dy);
  let p_y1 = noise3(p + dy);
  let p_z0 = noise3(p - dz);
  let p_z1 = noise3(p + dz);

  let curlX = (noise3(p + dy + dz) - noise3(p - dy + dz)) - (noise3(p + dz + dy) - noise3(p - dz + dy));
  let curlY = (noise3(p + dz + dx) - noise3(p - dz + dx)) - (noise3(p + dx + dz) - noise3(p - dx + dz));
  let curlZ = (noise3(p + dx + dy) - noise3(p - dx + dy)) - (noise3(p + dy + dx) - noise3(p - dy + dx));

  return vec3f(curlX, curlY, curlZ) / (2.0 * eps);
}
`;


const RANGE_WGSL = /* wgsl */`
fn range_f(oldValue: f32, oldMin: f32, oldMax: f32, newMin: f32, newMax: f32) -> f32 {
  let sub0 = oldValue - oldMin;
  let sub1 = newMax - newMin;
  let sub2 = oldMax - oldMin;
  return sub0 * sub1 / sub2 + newMin;
}

fn crange_f(oldValue: f32, oldMin: f32, oldMax: f32, newMin: f32, newMax: f32) -> f32 {
  return clamp(range_f(oldValue, oldMin, oldMax, newMin, newMax), min(newMin, newMax), max(newMin, newMax));
}
`;


const UNIFORMS_WGSL = /* wgsl */`
struct AntimatterUniforms {
  // AT Antimatter parameters
  time             : f32,   // elapsed seconds
  delta            : f32,   // frame delta
  decay            : f32,   // life drain speed (AT: 0.005)
  decayRandomMin   : f32,   // per-particle decay variance min
  decayRandomMax   : f32,   // per-particle decay variance max
  curlScale        : f32,   // curl noise spatial frequency
  curlSpeed        : f32,   // curl noise time evolution rate
  curlStrength     : f32,   // curl force magnitude
  originStrength   : f32,   // AT: origin attraction strength
  hz               : f32,   // frame-rate normalisation base
  uSize            : f32,   // particle point-splat size
  dpr              : f32,   // device pixel ratio

  // Domain / projection
  domainW          : f32,   // canvas/domain width
  domainH          : f32,   // canvas/domain height
  scaleX           : f32,   // NDC scale x = 2/domainW
  scaleY           : f32,   // NDC scale y = 2/domainH

  // Spray parameters
  sprayStrength    : f32,   // boundary spray velocity multiplier
  spraySpread      : f32,   // angular spread (radians)
  sphSmoothingH    : f32,   // SPH kernel h for bridge
  sphRestDensity   : f32,   // SPH ρ₀ for bridge

  // Counts
  particleCount    : u32,   // active particle slot count
  emitterCount     : u32,   // active emitter count
  texW             : u32,   // tPos texture width
  texH             : u32,   // tPos texture height

  // Setup flag
  uSetup           : f32,   // 1.0 during init frame
  uMaxCount        : f32,   // active particle ceiling
  _pad0            : f32,
  _pad1            : f32,
}
`;

const U_TIME             =  0;
const U_DELTA            =  4;
const U_DECAY            =  8;
const U_DECAY_RANDOM_MIN = 12;
const U_DECAY_RANDOM_MAX = 16;
const U_CURL_SCALE       = 20;
const U_CURL_SPEED       = 24;
const U_CURL_STRENGTH    = 28;
const U_ORIGIN_STRENGTH  = 32;
const U_HZ               = 36;
const U_SIZE             = 40;
const U_DPR              = 44;
const U_DOMAIN_W         = 48;
const U_DOMAIN_H         = 52;
const U_SCALE_X          = 56;
const U_SCALE_Y          = 60;
const U_SPRAY_STRENGTH   = 64;
const U_SPRAY_SPREAD     = 68;
const U_SPH_H            = 72;
const U_SPH_DENSITY      = 76;
const U_PARTICLE_COUNT   = 80;  // u32
const U_EMITTER_COUNT    = 84;  // u32
const U_TEX_W            = 88;  // u32
const U_TEX_H            = 92;  // u32
const U_SETUP            = 96;
const U_MAX_COUNT        = 100;
const U_PAD0             = 104;
const U_PAD1             = 108;
const UNIFORMS_BYTE_SIZE = 112;


const EMITTER_WGSL = /* wgsl */`
//   [0]  originX    — face midpoint x
//   [1]  originY    — face midpoint y
//   [2]  normalX    — outward normal x
//   [3]  normalY    — outward normal y
//   [4]  strength   — emission velocity multiplier
//   [5]  count      — particles to emit (float, rounded)
//   [6]  life       — initial particle life (0-1)
//   [7]  species    — species colour index
//   [8]  halfLen    — face half-length
//   [9]  active     — 1.0 if emitter is active this frame
//   [11] _pad

const EMIT_STRIDE = 12u;

fn emitOrigin(buf: ptr<storage, array<f32>, read>, idx: u32) -> vec2f {
  let base = idx * EMIT_STRIDE;
  return vec2f((*buf)[base], (*buf)[base + 1u]);
}

fn emitNormal(buf: ptr<storage, array<f32>, read>, idx: u32) -> vec2f {
  let base = idx * EMIT_STRIDE;
  return vec2f((*buf)[base + 2u], (*buf)[base + 3u]);
}

fn emitStrength(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
  let base = idx * EMIT_STRIDE;
  return (*buf)[base + 4u];
}

fn emitCount(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
  let base = idx * EMIT_STRIDE;
  return (*buf)[base + 5u];
}

fn emitLife(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
  let base = idx * EMIT_STRIDE;
  return (*buf)[base + 6u];
}

fn emitSpecies(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
  let base = idx * EMIT_STRIDE;
  return (*buf)[base + 7u];
}

fn emitHalfLen(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
  let base = idx * EMIT_STRIDE;
  return (*buf)[base + 8u];
}

fn emitActive(buf: ptr<storage, array<f32>, read>, idx: u32) -> f32 {
  let base = idx * EMIT_STRIDE;
  return (*buf)[base + 9u];
}
`;



const SPAWN_COMPUTE = /* wgsl */`
${UNIFORMS_WGSL}
${RANGE_WGSL}
${NOISE_WGSL}
${EMITTER_WGSL}

@group(0) @binding(0) var<uniform>             uni        : AntimatterUniforms;
@group(1) @binding(0) var<storage, read_write> particles  : array<f32>;
@group(1) @binding(1) var<storage, read>       emitters   : array<f32>;
@group(1) @binding(2) var<storage, read_write> spawnCount : array<atomic<u32>>;

const P_STRIDE = 20u;

fn pGet(idx: u32, field: u32) -> f32 {
  return particles[idx * P_STRIDE + field];
}
fn pSet(idx: u32, field: u32, v: f32) {
  particles[idx * P_STRIDE + field] = v;
}

// Field indices
const F_POS_X      = 0u;
const F_POS_Y      = 1u;
const F_VEL_X      = 2u;
const F_VEL_Y      = 3u;
const F_ORIGIN_X   = 4u;
const F_ORIGIN_Y   = 5u;
const F_LIFE       = 6u;
const F_MAX_LIFE   = 7u;
const F_AGE        = 8u;
const F_PHASE      = 9u;
const F_SPECIES    = 10u;
const F_SEED       = 11u;
const F_DECAY_RATE = 12u;
const F_CURL_X     = 13u;
const F_CURL_Y     = 14u;
const F_SPH_DENS   = 15u;
const F_SPH_PX     = 16u;
const F_SPH_PY     = 17u;
const F_HANDOFF    = 18u;
const F_EMITTER    = 19u;

fn rng(seed: f32, salt: f32) -> f32 {
  return fract(sin(seed * 127.1 + salt * 311.7) * 43758.5453);
}

// Original GLSL:
//   vec4 life = texture2D(tLife, uv);

@compute @workgroup_size(${WG})
fn spawnMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= uni.particleCount) { return; }

  if (f32(idx) >= uni.uMaxCount) {
    pSet(idx, F_POS_X, 9999.0);
    pSet(idx, F_PHASE, 0.0);
    return;
  }

  // Setup mode: clear all
  if (uni.uSetup > 0.5) {
    pSet(idx, F_POS_X, 0.0);
    pSet(idx, F_POS_Y, 0.0);
    pSet(idx, F_VEL_X, 0.0);
    pSet(idx, F_VEL_Y, 0.0);
    pSet(idx, F_LIFE,  0.0);
    pSet(idx, F_PHASE, 0.0);
    pSet(idx, F_HANDOFF, 0.0);
    return;
  }

  let phase = pGet(idx, F_PHASE);
  let life  = pGet(idx, F_LIFE);
  let seed  = pGet(idx, F_SEED);

  if (phase < 0.5) {
    // Scan emitters for one that still needs particles
    for (var e = 0u; e < uni.emitterCount; e++) {
      if (emitActive(&emitters, e) < 0.5) { continue; }
      let targetCount = u32(emitCount(&emitters, e));
      let claimed = atomicAdd(&spawnCount[e], 1u);
      if (claimed >= targetCount) { continue; }

      // Spawn at boundary face with spray distribution
      let origin   = emitOrigin(&emitters, e);
      let normal   = emitNormal(&emitters, e);
      let halfLen  = emitHalfLen(&emitters, e);
      let strength = emitStrength(&emitters, e);
      let initLife = emitLife(&emitters, e);
      let species  = emitSpecies(&emitters, e);

      // Distribute particles along face with jittered offset
      let t        = rng(seed + f32(claimed), f32(e)) * 2.0 - 1.0;
      let perpX    = -normal.y;
      let perpY    =  normal.x;
      let spawnX   = origin.x + perpX * halfLen * t;
      let spawnY   = origin.y + perpY * halfLen * t;

      let angle    = (rng(seed, f32(claimed) + 99.0) * 2.0 - 1.0) * uni.spraySpread;
      let cosA     = cos(angle);
      let sinA     = sin(angle);
      let rotNX    = normal.x * cosA - normal.y * sinA;
      let rotNY    = normal.x * sinA + normal.y * cosA;
      let velMag   = strength * uni.sprayStrength;
      let velX     = rotNX * velMag;
      let velY     = rotNY * velMag;

      let decayVar = crange_f(rng(seed, 7.0), 0.0, 1.0, uni.decayRandomMin, uni.decayRandomMax);

      pSet(idx, F_POS_X,      spawnX);
      pSet(idx, F_POS_Y,      spawnY);
      pSet(idx, F_VEL_X,      velX);
      pSet(idx, F_VEL_Y,      velY);
      pSet(idx, F_ORIGIN_X,   spawnX);
      pSet(idx, F_ORIGIN_Y,   spawnY);
      pSet(idx, F_LIFE,       initLife);
      pSet(idx, F_MAX_LIFE,   initLife);
      pSet(idx, F_AGE,        0.0);
      pSet(idx, F_PHASE,      1.0);   // → spawning
      pSet(idx, F_SPECIES,    species);
      pSet(idx, F_SEED,       rng(seed, 13.0) * 1000.0);
      pSet(idx, F_DECAY_RATE, decayVar);
      pSet(idx, F_CURL_X,     0.0);
      pSet(idx, F_CURL_Y,     0.0);
      pSet(idx, F_SPH_DENS,   0.0);
      pSet(idx, F_SPH_PX,     0.0);
      pSet(idx, F_SPH_PY,     0.0);
      pSet(idx, F_HANDOFF,    0.0);
      pSet(idx, F_EMITTER,    f32(e));
      return;
    }
    // No emitter claimed — stay dead
    return;
  }

  if (phase >= 1.0 && phase < 4.5) {
    let decayVar = pGet(idx, F_DECAY_RATE);
    let newLife  = life - uni.decay * decayVar * uni.hz * uni.delta;

    if (newLife <= 0.0) {
      // Particle dead — mark for handoff if it was alive
      if (phase >= 2.0 && phase < 4.0) {
        pSet(idx, F_PHASE, 4.0);   // → handoff
        pSet(idx, F_HANDOFF, 1.0);
        pSet(idx, F_LIFE, 0.0);
      } else {
        pSet(idx, F_PHASE, 0.0);   // → dead
        pSet(idx, F_LIFE, 0.0);
      }
    } else {
      pSet(idx, F_LIFE, newLife);
      // Transition from spawning to alive after first frame
      if (phase < 1.5) {
        pSet(idx, F_PHASE, 2.0);
      }
    }
  }
}
`;


const PHYSICS_COMPUTE = /* wgsl */`
${UNIFORMS_WGSL}
${RANGE_WGSL}
${NOISE_WGSL}

@group(0) @binding(0) var<uniform>             uni       : AntimatterUniforms;
@group(1) @binding(0) var<storage, read_write> particles : array<f32>;

const P_STRIDE = 20u;

fn pGet(idx: u32, field: u32) -> f32 {
  return particles[idx * P_STRIDE + field];
}
fn pSet(idx: u32, field: u32, v: f32) {
  particles[idx * P_STRIDE + field] = v;
}

const F_POS_X      = 0u;
const F_POS_Y      = 1u;
const F_VEL_X      = 2u;
const F_VEL_Y      = 3u;
const F_ORIGIN_X   = 4u;
const F_ORIGIN_Y   = 5u;
const F_LIFE       = 6u;
const F_MAX_LIFE   = 7u;
const F_AGE        = 8u;
const F_PHASE      = 9u;
const F_SEED       = 11u;
const F_CURL_X     = 13u;
const F_CURL_Y     = 14u;
const F_SPH_DENS   = 15u;
const F_SPH_PX     = 16u;
const F_SPH_PY     = 17u;

fn rng(seed: f32, salt: f32) -> f32 {
  return fract(sin(seed * 127.1 + salt * 311.7) * 43758.5453);
}

// contribute density to the SPH solver via this kernel.
fn cubicSplineW(r: f32, h: f32) -> f32 {
  let q     = r / h;
  let alpha = 10.0 / (7.0 * 3.14159265 * h * h);
  if (q < 1.0) {
    return alpha * (1.0 - 1.5 * q * q + 0.75 * q * q * q);
  } else if (q < 2.0) {
    let t = 2.0 - q;
    return alpha * 0.25 * t * t * t;
  }
  return 0.0;
}

// Original GLSL:
//   vec3 origin = texture2D(tOrigin, uv).xyz;
//   vec3 pos    = texture2D(tInput, uv).xyz;
//   // curl noise force
//   // origin attraction
//   // velocity integration
//   gl_FragColor = vec4(pos, data);
//
// Plus ProtonAntimatterLifecycle.fs spawn integration:
//   vec4 spawn = texture2D(tSpawn, vUv);
//   if (spawn.x <= 0.0)   { pos.x = 9999.0; }

@compute @workgroup_size(${WG})
fn physicsMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= uni.particleCount) { return; }

  let phase = pGet(idx, F_PHASE);
  // Only update alive or decaying particles
  if (phase < 1.5 || phase > 3.5) { return; }

  let dt   = uni.delta;
  let t    = uni.time;
  let seed = pGet(idx, F_SEED);

  var posX = pGet(idx, F_POS_X);
  var posY = pGet(idx, F_POS_Y);
  var velX = pGet(idx, F_VEL_X);
  var velY = pGet(idx, F_VEL_Y);
  let origX = pGet(idx, F_ORIGIN_X);
  let origY = pGet(idx, F_ORIGIN_Y);
  let life  = pGet(idx, F_LIFE);
  let maxLife = pGet(idx, F_MAX_LIFE);
  var age   = pGet(idx, F_AGE);

  let noiseCoord = vec3f(
    posX * uni.curlScale,
    posY * uni.curlScale,
    t * uni.curlSpeed,
  );
  let curl = curlNoise2D(noiseCoord, 0.01);
  let curlForceX = curl.x * uni.curlStrength;
  let curlForceY = curl.y * uni.curlStrength;

  let toOrigX  = origX - posX;
  let toOrigY  = origY - posY;
  let distOrig = sqrt(toOrigX * toOrigX + toOrigY * toOrigY) + 1e-6;
  let lifeRatio = life / max(maxLife, 0.001);
  let origForceX = toOrigX / distOrig * uni.originStrength * lifeRatio;
  let origForceY = toOrigY / distOrig * uni.originStrength * lifeRatio;

  let damping = 0.98;
  velX = velX * damping + (curlForceX + origForceX) * dt;
  velY = velY * damping + (curlForceY + origForceY) * dt;

  let sphPX = pGet(idx, F_SPH_PX);
  let sphPY = pGet(idx, F_SPH_PY);
  let sphDens = pGet(idx, F_SPH_DENS);
  if (sphDens > 0.0) {
    let pressureScale = 0.001 / max(sphDens, uni.sphRestDensity);
    velX += sphPX * pressureScale * dt;
    velY += sphPY * pressureScale * dt;
  }

  posX += velX * dt * uni.hz;
  posY += velY * dt * uni.hz;
  age  += dt;

  if (posX < 0.0) { posX = -posX; velX = abs(velX) * 0.5; }
  if (posY < 0.0) { posY = -posY; velY = abs(velY) * 0.5; }
  if (posX > uni.domainW) { posX = 2.0 * uni.domainW - posX; velX = -abs(velX) * 0.5; }
  if (posY > uni.domainH) { posY = 2.0 * uni.domainH - posY; velY = -abs(velY) * 0.5; }

  pSet(idx, F_POS_X,  posX);
  pSet(idx, F_POS_Y,  posY);
  pSet(idx, F_VEL_X,  velX);
  pSet(idx, F_VEL_Y,  velY);
  pSet(idx, F_AGE,    age);
  pSet(idx, F_CURL_X, curlForceX);
  pSet(idx, F_CURL_Y, curlForceY);
}
`;


const TPOS_WRITE_COMPUTE = /* wgsl */`
${UNIFORMS_WGSL}

@group(0) @binding(0) var<uniform>             uni       : AntimatterUniforms;
@group(1) @binding(0) var<storage, read>       particles : array<f32>;
@group(1) @binding(1) var                      tPos      : texture_storage_2d<rgba32float, write>;

const P_STRIDE = 20u;
const F_POS_X  = 0u;
const F_POS_Y  = 1u;
const F_LIFE   = 6u;
const F_MAX_LIFE = 7u;
const F_PHASE  = 9u;
const F_SPECIES = 10u;

@compute @workgroup_size(${WG})
fn tposMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= uni.particleCount) { return; }

  let posX  = particles[idx * P_STRIDE + F_POS_X];
  let posY  = particles[idx * P_STRIDE + F_POS_Y];
  let life  = particles[idx * P_STRIDE + F_LIFE];
  let maxL  = particles[idx * P_STRIDE + F_MAX_LIFE];
  let phase = particles[idx * P_STRIDE + F_PHASE];
  let spec  = particles[idx * P_STRIDE + F_SPECIES];

  // since antimatter particles don't travel along splines.
  let lifeRatio = life / max(maxL, 0.001);
  let alpha = select(0.0, sin(3.14159265 * lifeRatio) * lifeRatio, phase >= 1.5);

  let texX = i32(idx % uni.texW);
  let texY = i32(idx / uni.texW);
  textureStore(tPos, vec2<i32>(texX, texY), vec4f(posX, posY, alpha, spec));
}
`;


const VERTEX_SHADER = /* wgsl */`
${UNIFORMS_WGSL}

@group(0) @binding(0) var<uniform> uni     : AntimatterUniforms;
@group(0) @binding(1) var          tPos    : texture_2d<f32>;
@group(0) @binding(2) var          sSampler: sampler;

struct VertOut {
  @builtin(position) pos     : vec4f,
  @location(0)       vUv     : vec2f,   // quad local [-1,1]
  @location(1)       vAlpha  : f32,     // particle opacity
  @location(2)       vSpecies: f32,     // species for colour
  @location(3)       vLife   : f32,     // life ratio for size attenuation
}

// 6 vertices per quad (2 triangles)
var<private> QUAD_UV: array<vec2f, 6> = array<vec2f, 6>(
  vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0),
  vec2f(-1.0, -1.0), vec2f( 1.0,  1.0), vec2f(-1.0,  1.0),
);

@vertex fn vs_main(
  @builtin(vertex_index)   vi : u32,
  @builtin(instance_index) ii : u32,
) -> VertOut {
  let texX = i32(ii % uni.texW);
  let texY = i32(ii / uni.texW);
  let tposVal = textureLoad(tPos, vec2<i32>(texX, texY), 0);

  let worldX  = tposVal.r;
  let worldY  = tposVal.g;
  let alpha   = tposVal.b;
  let species = tposVal.a;

  // Discard invisible particles
  let alive = select(0.0, 1.0, alpha > 0.005);

  // Size attenuates with alpha for smooth fade-out.
  let sizeScale = sqrt(alpha);
  let halfSize  = uni.uSize * sizeScale * uni.dpr * 0.5;

  let quadUV = QUAD_UV[vi];
  let ndcX   = worldX * uni.scaleX - 1.0 + quadUV.x * halfSize * uni.scaleX;
  let ndcY   = worldY * uni.scaleY - 1.0 + quadUV.y * halfSize * uni.scaleY;

  var out: VertOut;
  out.pos      = vec4f(ndcX * alive, ndcY * alive, 0.0, 1.0);
  out.vUv      = quadUV;
  out.vAlpha   = alpha;
  out.vSpecies = species;
  out.vLife    = sizeScale;
  return out;
}
`;


const FRAGMENT_SHADER = /* wgsl */`
${UNIFORMS_WGSL}

@group(0) @binding(0) var<uniform> uni      : AntimatterUniforms;
@group(0) @binding(3) var          tMatcap  : texture_2d<f32>;
@group(0) @binding(4) var          sMatcap  : sampler;

struct FragIn {
  @location(0) vUv     : vec2f,
  @location(1) vAlpha  : f32,
  @location(2) vSpecies: f32,
  @location(3) vLife   : f32,
}

// Species colour palette (8 slots)
// can produce a differently coloured particle stream.
fn speciesColour(s: f32) -> vec3f {
  let si = u32(s) % 8u;
  switch (si) {
    case 0u: { return vec3f(0.4, 0.7, 1.0); }   // electric blue
    case 1u: { return vec3f(1.0, 0.4, 0.6); }   // plasma pink
    case 2u: { return vec3f(0.3, 1.0, 0.6); }   // neon green
    case 3u: { return vec3f(1.0, 0.8, 0.2); }   // solar gold
    case 4u: { return vec3f(0.6, 0.3, 1.0); }   // violet
    case 5u: { return vec3f(1.0, 0.5, 0.2); }   // ember orange
    case 6u: { return vec3f(0.2, 0.9, 0.9); }   // cyan
    default: { return vec3f(0.9, 0.9, 0.9); }   // white
  }
}

@fragment fn fs_main(in: FragIn) -> @location(0) vec4f {
  // Circular discard
  let r2 = dot(in.vUv, in.vUv);
  if (r2 > 1.0) { discard; }

  let nz       = sqrt(max(0.0, 1.0 - r2));
  let matcapUV = in.vUv * 0.5 + 0.5;
  let matcap   = textureSample(tMatcap, sMatcap, matcapUV);

  let baseCol = speciesColour(in.vSpecies);
  let litCol  = matcap.rgb * baseCol;

  let radialFade = 1.0 - r2 * 0.4;  // softer edge than hard disc
  let finalA     = in.vAlpha * radialFade * in.vLife;

  return vec4f(litCol, finalA);
}
`;


const SPH_BRIDGE_COMPUTE = /* wgsl */`
${UNIFORMS_WGSL}

@group(0) @binding(0) var<uniform>             uni       : AntimatterUniforms;
@group(1) @binding(0) var<storage, read_write> particles : array<f32>;

const P_STRIDE = 20u;
const F_POS_X    = 0u;
const F_POS_Y    = 1u;
const F_PHASE    = 9u;
const F_SPH_DENS = 15u;
const F_SPH_PX   = 16u;
const F_SPH_PY   = 17u;

fn cubicW(r: f32, h: f32) -> f32 {
  let q     = r / h;
  let alpha = 10.0 / (7.0 * 3.14159265 * h * h);
  if (q < 1.0) {
    return alpha * (1.0 - 1.5 * q * q + 0.75 * q * q * q);
  } else if (q < 2.0) {
    let t = 2.0 - q;
    return alpha * 0.25 * t * t * t;
  }
  return 0.0;
}

fn cubicGradW(r: f32, h: f32) -> f32 {
  let q     = r / h;
  let alpha = 10.0 / (7.0 * 3.14159265 * h * h);
  if (q < 1.0) {
    return alpha * (-3.0 * q + 2.25 * q * q) / h;
  } else if (q < 2.0) {
    let t = 2.0 - q;
    return alpha * (-0.75 * t * t) / h;
  }
  return 0.0;
}

@compute @workgroup_size(${WG})
fn sphBridgeMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= uni.particleCount) { return; }

  let phase = particles[idx * P_STRIDE + F_PHASE];
  if (phase < 1.5) {
    particles[idx * P_STRIDE + F_SPH_DENS] = 0.0;
    particles[idx * P_STRIDE + F_SPH_PX]   = 0.0;
    particles[idx * P_STRIDE + F_SPH_PY]   = 0.0;
    return;
  }

  let posX = particles[idx * P_STRIDE + F_POS_X];
  let posY = particles[idx * P_STRIDE + F_POS_Y];
  let h    = uni.sphSmoothingH;
  let h2   = h * 2.0;

  var density  = 0.0;
  var pressX   = 0.0;
  var pressY   = 0.0;

  let sampleStride = max(1u, uni.particleCount / 4096u);
  for (var j = 0u; j < uni.particleCount; j += sampleStride) {
    if (j == idx) { continue; }
    let jp = particles[j * P_STRIDE + F_PHASE];
    if (jp < 1.5) { continue; }

    let jx = particles[j * P_STRIDE + F_POS_X];
    let jy = particles[j * P_STRIDE + F_POS_Y];
    let dx = posX - jx;
    let dy = posY - jy;
    let r  = sqrt(dx * dx + dy * dy);

    if (r < h2 && r > 1e-6) {
      density += cubicW(r, h);
      let gradW = cubicGradW(r, h);
      pressX += dx / r * gradW;
      pressY += dy / r * gradW;
    }
  }

  // Scale by sample stride to approximate full density
  density *= f32(sampleStride);
  pressX  *= f32(sampleStride);
  pressY  *= f32(sampleStride);

  particles[idx * P_STRIDE + F_SPH_DENS] = density;
  particles[idx * P_STRIDE + F_SPH_PX]   = pressX;
  particles[idx * P_STRIDE + F_SPH_PY]   = pressY;
}
`;


function buildEmitterBuffer(emitters: InternalEmitter[]): Float32Array {
  const buf = new Float32Array(MAX_EMITTERS * EMITTER_STRIDE);
  for (let i = 0; i < Math.min(emitters.length, MAX_EMITTERS); i++) {
    const e    = emitters[i];
    const base = i * EMITTER_STRIDE;
    buf[base + 0]  = e.originX;
    buf[base + 1]  = e.originY;
    buf[base + 2]  = e.normalX;
    buf[base + 3]  = e.normalY;
    buf[base + 4]  = e.strength;
    buf[base + 5]  = e.count;
    buf[base + 6]  = e.life;
    buf[base + 7]  = e.species;
    buf[base + 8]  = e.halfLen;
    buf[base + 9]  = e.active ? 1.0 : 0.0;
    buf[base + 10] = 0;  // emitted counter (reset each frame)
    buf[base + 11] = 0;  // pad
  }
  return buf;
}

interface InternalEmitter {
  originX:  number;
  originY:  number;
  normalX:  number;
  normalY:  number;
  strength: number;
  count:    number;
  life:     number;
  species:  number;
  halfLen:  number;
  active:   boolean;
  faceId:   string;
  cellId:   string;
  targetId: string;
}


export class ATAntimatterParticles {
  private readonly device:   GPUDevice;
  private readonly canvas:   HTMLCanvasElement;
  private readonly cfg: {
    decay:          number;
    decayRandom:    [number, number];
    curlScale:      number;
    curlSpeed:      number;
    curlStrength:   number;
    originStrength: number;
    hz:             number;
    dpr:            number;
    uSize:          number;
    sprayStrength:  number;
    spraySpread:    number;
    sphSmoothingH:  number;
    sphRestDensity: number;
  };
  private readonly onHandoff?: ATAntimatterConfig['onHandoff'];

  private particleCount: number;
  private emitters: InternalEmitter[] = [];
  private pendingEmits: EmitterRequest[] = [];

  private faceRegistry = new Map<number, { cellId: string; targetId: string }>();

  private uniformBuf!:       GPUBuffer;
  private particleBuf!:      GPUBuffer;
  private emitterBuf!:       GPUBuffer;
  private spawnCountBuf!:    GPUBuffer;
  private readbackBuf!:      GPUBuffer;

  private tPos!:             GPUTexture;
  private tPosView!:         GPUTextureView;

  private tMatcap!:          GPUTexture;
  private tMatcapView!:      GPUTextureView;

  private sampler!:          GPUSampler;
  private matcapSampler!:    GPUSampler;

  private spawnPipeline!:    GPUComputePipeline;
  private physicsPipeline!:  GPUComputePipeline;
  private sphBridgePipeline!: GPUComputePipeline;
  private tposWritePipeline!: GPUComputePipeline;
  private renderPipeline!:   GPURenderPipeline;

  private spawnBG0!:         GPUBindGroup;
  private spawnBG1!:         GPUBindGroup;
  private physicsBG0!:       GPUBindGroup;
  private physicsBG1!:       GPUBindGroup;
  private sphBridgeBG0!:     GPUBindGroup;
  private sphBridgeBG1!:     GPUBindGroup;
  private tposWriteBG0!:     GPUBindGroup;
  private tposWriteBG1!:     GPUBindGroup;
  private renderBG!:         GPUBindGroup;

  private built = false;
  private elapsed = 0;
  private frameCount = 0;
  private setupFrame = true;

  constructor(
    device: GPUDevice,
    canvas: HTMLCanvasElement,
    config: ATAntimatterConfig = {},
  ) {
    this.device   = device;
    this.canvas   = canvas;
    this.particleCount = MAX_PARTICLES;
    this.onHandoff = config.onHandoff;
    this.cfg = {
      decay:          config.decay          ?? DEFAULTS.decay,
      decayRandom:    config.decayRandom    ?? [...DEFAULTS.decayRandom],
      curlScale:      config.curlScale      ?? DEFAULTS.curlScale,
      curlSpeed:      config.curlSpeed      ?? DEFAULTS.curlSpeed,
      curlStrength:   config.curlStrength   ?? DEFAULTS.curlStrength,
      originStrength: config.originStrength ?? DEFAULTS.originStrength,
      hz:             config.hz             ?? DEFAULTS.hz,
      dpr:            config.dpr            ?? DEFAULTS.dpr,
      uSize:          config.uSize          ?? DEFAULTS.uSize,
      sprayStrength:  config.sprayStrength  ?? DEFAULTS.sprayStrength,
      spraySpread:    config.spraySpread    ?? DEFAULTS.spraySpread,
      sphSmoothingH:  config.sphSmoothingH  ?? DEFAULTS.sphSmoothingH,
      sphRestDensity: config.sphRestDensity ?? DEFAULTS.sphRestDensity,
    };
  }

  async build(): Promise<void> {
    if (this.built) this._destroy();
    const { device } = this;

    this.uniformBuf = device.createBuffer({
      size:  UNIFORMS_BYTE_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const particleData = new Float32Array(this.particleCount * PARTICLE_STRIDE);
    for (let i = 0; i < this.particleCount; i++) {
      const base = i * PARTICLE_STRIDE;
      particleData[base + 9]  = 0;   // phase = dead
      particleData[base + 11] = Math.random() * 1000; // seed
    }
    this.particleBuf = device.createBuffer({
      size:  particleData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.particleBuf, 0, particleData);

    this.emitterBuf = device.createBuffer({
      size:  MAX_EMITTERS * EMITTER_STRIDE * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.spawnCountBuf = device.createBuffer({
      size:  MAX_EMITTERS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.readbackBuf = device.createBuffer({
      size:  particleData.byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    this.tPos = device.createTexture({
      size:   [TEX_W, TEX_H],
      format: 'rgba32float',
      usage:  GPUTextureUsage.TEXTURE_BINDING |
              GPUTextureUsage.STORAGE_BINDING  |
              GPUTextureUsage.COPY_SRC,
    });
    this.tPosView = this.tPos.createView();

    this.tMatcap = device.createTexture({
      size:   [1, 1],
      format: 'rgba8unorm',
      usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture(
      { texture: this.tMatcap },
      new Uint8Array([255, 255, 255, 255]),
      { bytesPerRow: 4 },
      [1, 1],
    );
    this.tMatcapView = this.tMatcap.createView();

    this.sampler = device.createSampler({
      magFilter: 'nearest', minFilter: 'nearest',
    });
    this.matcapSampler = device.createSampler({
      magFilter: 'linear', minFilter: 'linear',
      addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
    });

    const spawnModule   = device.createShaderModule({ code: SPAWN_COMPUTE });
    const physicsModule = device.createShaderModule({ code: PHYSICS_COMPUTE });
    const sphModule     = device.createShaderModule({ code: SPH_BRIDGE_COMPUTE });
    const tposModule    = device.createShaderModule({ code: TPOS_WRITE_COMPUTE });

    this.spawnPipeline = device.createComputePipeline({
      layout:  'auto',
      compute: { module: spawnModule, entryPoint: 'spawnMain' },
    });
    this.physicsPipeline = device.createComputePipeline({
      layout:  'auto',
      compute: { module: physicsModule, entryPoint: 'physicsMain' },
    });
    this.sphBridgePipeline = device.createComputePipeline({
      layout:  'auto',
      compute: { module: sphModule, entryPoint: 'sphBridgeMain' },
    });
    this.tposWritePipeline = device.createComputePipeline({
      layout:  'auto',
      compute: { module: tposModule, entryPoint: 'tposMain' },
    });

    const vsModule = device.createShaderModule({ code: VERTEX_SHADER });
    const fsModule = device.createShaderModule({ code: FRAGMENT_SHADER });
    const ctx = this.canvas.getContext('webgpu') as GPUCanvasContext;
    const fmt = navigator.gpu.getPreferredCanvasFormat();

    this.renderPipeline = device.createRenderPipeline({
      layout:    'auto',
      vertex:    { module: vsModule, entryPoint: 'vs_main' },
      fragment:  {
        module: fsModule, entryPoint: 'fs_main',
        targets: [{
          format: fmt,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
            alpha: { srcFactor: 'one',       dstFactor: 'one', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
    });

    this._buildBindGroups();
    this.built = true;
    this.setupFrame = true;
  }

  emitFromBoundary(request: EmitterRequest): void {
    this.pendingEmits.push(request);
  }

  emitBurst(requests: EmitterRequest[]): void {
    this.pendingEmits.push(...requests);
  }

  registerBoundaryFaces(faces: CellBoundaryFace[]): void {
    for (const face of faces) {
      const key = this._hashFaceId(face.faceId);
      this.faceRegistry.set(key, { cellId: face.cellId, targetId: face.targetId });
    }
  }

  async loadMatcap(bitmap: ImageBitmap): Promise<void> {
    if (!this.built) throw new Error('[ATAntimatterParticles] call build() first');
    this.tMatcap.destroy();
    this.tMatcap = this.device.createTexture({
      size:   [bitmap.width, bitmap.height],
      format: 'rgba8unorm',
      usage:  GPUTextureUsage.TEXTURE_BINDING |
              GPUTextureUsage.RENDER_ATTACHMENT |
              GPUTextureUsage.COPY_DST,
    });
    this.device.queue.copyExternalImageToTexture(
      { source: bitmap },
      { texture: this.tMatcap },
      [bitmap.width, bitmap.height],
    );
    this.tMatcapView = this.tMatcap.createView();
    this._buildBindGroups();
  }

  update(encoder: GPUCommandEncoder, elapsed: number, dt: number): void {
    if (!this.built) return;
    this.elapsed = elapsed;
    this.frameCount++;

    this._processEmitters();

    this._writeUniforms(elapsed, dt);

    const emitterData = buildEmitterBuffer(this.emitters);
    this.device.queue.writeBuffer(this.emitterBuf, 0, emitterData);

    const zeros = new Uint32Array(MAX_EMITTERS);
    this.device.queue.writeBuffer(this.spawnCountBuf, 0, zeros);

    this._buildBindGroups();

    const wg = Math.ceil(this.particleCount / WG);

    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.spawnPipeline);
      pass.setBindGroup(0, this.spawnBG0);
      pass.setBindGroup(1, this.spawnBG1);
      pass.dispatchWorkgroups(wg);
      pass.end();
    }

    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.physicsPipeline);
      pass.setBindGroup(0, this.physicsBG0);
      pass.setBindGroup(1, this.physicsBG1);
      pass.dispatchWorkgroups(wg);
      pass.end();
    }

    if (this.frameCount % 4 === 0) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.sphBridgePipeline);
      pass.setBindGroup(0, this.sphBridgeBG0);
      pass.setBindGroup(1, this.sphBridgeBG1);
      pass.dispatchWorkgroups(wg);
      pass.end();
    }

    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.tposWritePipeline);
      pass.setBindGroup(0, this.tposWriteBG0);
      pass.setBindGroup(1, this.tposWriteBG1);
      pass.dispatchWorkgroups(wg);
      pass.end();
    }

    if (this.setupFrame) {
      this.setupFrame = false;
    }

    for (const e of this.emitters) {
      e.active = false;
    }
  }

  render(
    encoder:    GPUCommandEncoder,
    colorView:  GPUTextureView,
    depthView?: GPUTextureView,
  ): void {
    if (!this.built) return;

    const colorAttach: GPURenderPassColorAttachment = {
      view:    colorView,
      loadOp:  'load',
      storeOp: 'store',
    };
    const passDesc: GPURenderPassDescriptor = {
      colorAttachments: [colorAttach],
    };
    if (depthView) {
      passDesc.depthStencilAttachment = {
        view:         depthView,
        depthLoadOp:  'load',
        depthStoreOp: 'store',
      };
    }

    const pass = encoder.beginRenderPass(passDesc);
    pass.setPipeline(this.renderPipeline);
    pass.setBindGroup(0, this.renderBG);
    pass.draw(6, this.particleCount);
    pass.end();
  }

  async scheduleHandoffReadback(): Promise<void> {
    if (!this.built || !this.onHandoff) return;
    const { device } = this;

    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(this.particleBuf, 0, this.readbackBuf, 0, this.particleBuf.size);
    device.queue.submit([enc.finish()]);

    await this.readbackBuf.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(this.readbackBuf.getMappedRange());

    for (let i = 0; i < this.particleCount; i++) {
      const base = i * PARTICLE_STRIDE;
      if (data[base + 18] < 0.5) continue;  // F_HANDOFF

      const posX     = data[base + 0];
      const posY     = data[base + 1];
      const velX     = data[base + 2];
      const velY     = data[base + 3];
      const species  = Math.round(data[base + 10]);
      const density  = data[base + 15];
      const emIdx    = Math.round(data[base + 19]);

      const emitter = this.emitters[emIdx];
      const cellId   = emitter?.cellId   ?? 'unknown';
      const targetId = emitter?.targetId ?? 'unknown';

      this.onHandoff(cellId, targetId, posX, posY, velX, velY, species, density);
    }

    this.readbackBuf.unmap();
  }


  writeSPHField(x: number, y: number, density: number, pressX: number, pressY: number, radius: number): void {
    if (!this.built) return;
    void [x, y, density, pressX, pressY, radius];
  }

  setDecay(v: number): void           { this.cfg.decay = v; }
  setDecayRandom(min: number, max: number): void { this.cfg.decayRandom = [min, max]; }
  setCurlScale(v: number): void       { this.cfg.curlScale = v; }
  setCurlSpeed(v: number): void       { this.cfg.curlSpeed = v; }
  setCurlStrength(v: number): void    { this.cfg.curlStrength = v; }
  setOriginStrength(v: number): void  { this.cfg.originStrength = v; }
  setSize(v: number): void            { this.cfg.uSize = v; }
  setSprayStrength(v: number): void   { this.cfg.sprayStrength = v; }
  setSpraySpread(v: number): void     { this.cfg.spraySpread = v; }
  setDPR(v: number): void             { this.cfg.dpr = v; }
  setMaxParticles(n: number): void    { this.particleCount = Math.min(n, MAX_PARTICLES); }

  get activeParticleCount(): number  { return this.particleCount; }
  get activeEmitterCount(): number   { return this.emitters.filter(e => e.active).length; }
  get isBuilt(): boolean             { return this.built; }
  get totalFrames(): number          { return this.frameCount; }

  destroy(): void { this._destroy(); }

  private _processEmitters(): void {
    this.emitters = [];

    for (const req of this.pendingEmits) {
      if (this.emitters.length >= MAX_EMITTERS) break;
      const face = req.face;
      const nLen = Math.sqrt(face.normal.x ** 2 + face.normal.y ** 2) || 1;
      this.emitters.push({
        originX:  face.origin.x,
        originY:  face.origin.y,
        normalX:  face.normal.x / nLen,
        normalY:  face.normal.y / nLen,
        strength: req.strength ?? this.cfg.sprayStrength,
        count:    Math.min(req.count, 4096),
        life:     req.life ?? 1.0,
        species:  face.species ?? 0,
        halfLen:  face.halfLen,
        active:   true,
        faceId:   face.faceId,
        cellId:   face.cellId,
        targetId: face.targetId,
      });
    }
    this.pendingEmits = [];
  }

  private _writeUniforms(elapsed: number, dt: number): void {
    const { device, cfg, canvas } = this;
    const data = new ArrayBuffer(UNIFORMS_BYTE_SIZE);
    const f32  = new Float32Array(data);
    const u32  = new Uint32Array(data);

    f32[U_TIME             / 4] = elapsed;
    f32[U_DELTA            / 4] = Math.min(dt, 1 / 30); // clamp dt
    f32[U_DECAY            / 4] = cfg.decay;
    f32[U_DECAY_RANDOM_MIN / 4] = cfg.decayRandom[0];
    f32[U_DECAY_RANDOM_MAX / 4] = cfg.decayRandom[1];
    f32[U_CURL_SCALE       / 4] = cfg.curlScale;
    f32[U_CURL_SPEED       / 4] = cfg.curlSpeed;
    f32[U_CURL_STRENGTH    / 4] = cfg.curlStrength;
    f32[U_ORIGIN_STRENGTH  / 4] = cfg.originStrength;
    f32[U_HZ               / 4] = cfg.hz;
    f32[U_SIZE             / 4] = cfg.uSize;
    f32[U_DPR              / 4] = cfg.dpr;
    f32[U_DOMAIN_W         / 4] = canvas.width;
    f32[U_DOMAIN_H         / 4] = canvas.height;
    f32[U_SCALE_X          / 4] = 2.0 / canvas.width;
    f32[U_SCALE_Y          / 4] = 2.0 / canvas.height;
    f32[U_SPRAY_STRENGTH   / 4] = cfg.sprayStrength;
    f32[U_SPRAY_SPREAD     / 4] = cfg.spraySpread;
    f32[U_SPH_H            / 4] = cfg.sphSmoothingH;
    f32[U_SPH_DENSITY      / 4] = cfg.sphRestDensity;
    u32[U_PARTICLE_COUNT   / 4] = this.particleCount;
    u32[U_EMITTER_COUNT    / 4] = this.emitters.length;
    u32[U_TEX_W            / 4] = TEX_W;
    u32[U_TEX_H            / 4] = TEX_H;
    f32[U_SETUP            / 4] = this.setupFrame ? 1.0 : 0.0;
    f32[U_MAX_COUNT        / 4] = this.particleCount;
    f32[U_PAD0             / 4] = 0;
    f32[U_PAD1             / 4] = 0;

    device.queue.writeBuffer(this.uniformBuf, 0, f32);
  }

  private _buildBindGroups(): void {
    const { device } = this;

    this.spawnBG0 = device.createBindGroup({
      layout:  this.spawnPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.uniformBuf } }],
    });

    this.spawnBG1 = device.createBindGroup({
      layout:  this.spawnPipeline.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: this.particleBuf } },
        { binding: 1, resource: { buffer: this.emitterBuf } },
        { binding: 2, resource: { buffer: this.spawnCountBuf } },
      ],
    });

    this.physicsBG0 = device.createBindGroup({
      layout:  this.physicsPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.uniformBuf } }],
    });

    this.physicsBG1 = device.createBindGroup({
      layout:  this.physicsPipeline.getBindGroupLayout(1),
      entries: [{ binding: 0, resource: { buffer: this.particleBuf } }],
    });

    this.sphBridgeBG0 = device.createBindGroup({
      layout:  this.sphBridgePipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.uniformBuf } }],
    });

    this.sphBridgeBG1 = device.createBindGroup({
      layout:  this.sphBridgePipeline.getBindGroupLayout(1),
      entries: [{ binding: 0, resource: { buffer: this.particleBuf } }],
    });

    this.tposWriteBG0 = device.createBindGroup({
      layout:  this.tposWritePipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.uniformBuf } }],
    });

    this.tposWriteBG1 = device.createBindGroup({
      layout:  this.tposWritePipeline.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: this.particleBuf } },
        { binding: 1, resource: this.tPosView },
      ],
    });

    this.renderBG = device.createBindGroup({
      layout:  this.renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
        { binding: 1, resource: this.tPosView },
        { binding: 2, resource: this.sampler },
        { binding: 3, resource: this.tMatcapView },
        { binding: 4, resource: this.matcapSampler },
      ],
    });
  }

  private _hashFaceId(faceId: string): number {
    let hash = 0;
    for (let i = 0; i < faceId.length; i++) {
      hash = ((hash << 5) - hash + faceId.charCodeAt(i)) | 0;
    }
    return hash;
  }

  private _destroy(): void {
    if (!this.built) return;
    this.uniformBuf?.destroy();
    this.particleBuf?.destroy();
    this.emitterBuf?.destroy();
    this.spawnCountBuf?.destroy();
    this.readbackBuf?.destroy();
    this.tPos?.destroy();
    this.tMatcap?.destroy();
    this.built = false;
  }
}

export function cellBoundsToFaces(
  cellId:    string,
  targetIds: [string, string, string, string],
  x0: number, y0: number,
  x1: number, y1: number,
  species?: number,
): CellBoundaryFace[] {
  const cx = (x0 + x1) * 0.5;
  const cy = (y0 + y1) * 0.5;
  const hw = (x1 - x0) * 0.5;
  const hh = (y1 - y0) * 0.5;

  return [
    {
      faceId: `${cellId}_N`, cellId, targetId: targetIds[0],
      origin: { x: cx, y: y1 }, normal: { x: 0, y: 1 },
      halfLen: hw, species,
    },
    {
      faceId: `${cellId}_S`, cellId, targetId: targetIds[1],
      origin: { x: cx, y: y0 }, normal: { x: 0, y: -1 },
      halfLen: hw, species,
    },
    {
      faceId: `${cellId}_E`, cellId, targetId: targetIds[2],
      origin: { x: x1, y: cy }, normal: { x: 1, y: 0 },
      halfLen: hh, species,
    },
    {
      faceId: `${cellId}_W`, cellId, targetId: targetIds[3],
      origin: { x: x0, y: cy }, normal: { x: -1, y: 0 },
      halfLen: hh, species,
    },
  ];
}

export function createAntimatterForSPH(
  device:   GPUDevice,
  canvas:   HTMLCanvasElement,
  addFluid: (x0: number, y0: number, x1: number, y1: number, spacing: number, species: number) => void,
  config:   Omit<ATAntimatterConfig, 'onHandoff'> = {},
): ATAntimatterParticles {
  const HANDOFF_R = 0.05;
  return new ATAntimatterParticles(device, canvas, {
    ...config,
    onHandoff: (_cellId, _targetId, x, y, _vx, _vy, species, _density) => {
      addFluid(
        x - HANDOFF_R, y - HANDOFF_R,
        x + HANDOFF_R, y + HANDOFF_R,
        HANDOFF_R * 0.8,
        species,
      );
    },
  });
}

export function createPubSubEmitter(
  system:      ATAntimatterParticles,
  faces:       CellBoundaryFace[],
  countPerMsg: number = 50,
): (fromCellId: string, toCellId: string, strength?: number) => void {
  const lookup = new Map<string, CellBoundaryFace>();
  for (const face of faces) {
    lookup.set(`${face.cellId}_${face.targetId}`, face);
  }
  system.registerBoundaryFaces(faces);

  return (fromCellId: string, toCellId: string, strength?: number) => {
    const key  = `${fromCellId}_${toCellId}`;
    const face = lookup.get(key);
    if (!face) return;
    system.emitFromBoundary({
      face,
      count:    countPerMsg,
      strength: strength ?? undefined,
    });
  };
}

export const AT_ANTIMATTER_DEFAULTS = {
  ...DEFAULTS,
  maxParticles:    MAX_PARTICLES,
  texW:            TEX_W,
  texH:            TEX_H,
  particleStride:  PARTICLE_STRIDE,
  maxEmitters:     MAX_EMITTERS,
  emitterStride:   EMITTER_STRIDE,
  workgroupSize:   WG,
} as const;
