/**
 * particle-effect-system.ts — M783: Unified Particle Effect System
 * ─────────────────────────────────────────────────────────────────────────────
 * Single entry point for all transient particle effects in the SPH world.
 *
 * Problem
 * ───────
 * The codebase has accumulated many independent particle subsystems:
 *   - contact-sparks.ts       → collision spark particles (Canvas 2D)
 *   - collision-fx-system.ts  → flower petal bursts (Canvas 2D)
 *   - collision-shockwave.ts  → shockwave ring particles (GPU)
 *   - transition-system.ts    → dissolve scatter particles
 *   - environment-fx.ts       → ambient atmospheric particles
 *   - trails.ts               → flow trail ring-buffers
 *
 * Each manages its own allocation, lifetime, update loop, and draw call,
 * leading to duplicated random utilities, separate GC pressure, and no
 * unified tick/render path.
 *
 * Solution
 * ────────
 * ParticleEffectSystem provides:
 *   1. A typed `emit(type, position, params)` API for six effect categories
 *   2. A single object pool (pre-allocated, no GC) shared across all types
 *   3. Per-type behaviour configs with species-aware shader references
 *      (read from species-shader-registry.ts)
 *   4. A unified `tick(dt)` that advances all active particles
 *   5. A unified `render(encoder)` that batches all particles into one
 *      GPU instanced draw call per effect type
 *
 * Effect types
 * ────────────
 *   collision_spark  — impulse-driven bright sparks at rigid-body contacts
 *   flow_trail       — velocity-aligned ribbon particles along fluid paths
 *   ambient_dust     — slow-drifting atmospheric motes (always-on background)
 *   qos_transition   — burst particles when a cell changes QoS profile
 *   cell_birth       — inward-coalescing particles when a cell appears
 *   cell_death       — outward-scattering particles when a cell is removed
 *
 * Object pooling
 * ──────────────
 * All Particle objects are pre-allocated in a flat typed-array-backed pool.
 * `emit()` acquires particles from the free list; when a particle's life
 * reaches zero it is returned to the free list.  No allocations occur
 * during gameplay — only during initial construction.
 *
 * GPU instanced rendering
 * ───────────────────────
 * Active particles are packed into an interleaved Float32Array each frame
 * (position, velocity, life, size, colour, type) and uploaded to a single
 * GPUBuffer.  A WGSL vertex shader expands screen-aligned quads; the
 * fragment shader selects per-type appearance from the effect config.
 *
 * Integration with species-shader-registry
 * ─────────────────────────────────────────
 * Each effect type can optionally reference a species id.  When provided,
 * the system reads SpeciesShaderConfig from the registry to derive particle
 * colour palette, bloom strength, and material hints.  This keeps the
 * visual language consistent: a cil-eye cell death scatters iridescent
 * fragments, while a cil-bolt cell death throws lightning-coloured sparks.
 *
 * Upstream references
 * ───────────────────
 *   src/lib/sph/contact-sparks.ts         — spark physics model
 *   src/lib/sph/collision-fx-system.ts    — flower burst model
 *   src/lib/sph/transition-system.ts      — dissolve scatter model
 *   src/lib/sph/environment-fx.ts         — ambient atmosphere
 *   src/lib/sph/species-shader-registry.ts — per-species shader configs
 *   src/lib/sph/particle-instancing.ts    — GPU instanced quad pattern
 *   src/lib/sph/types.ts                  — MAX_PARTICLES, ParticleData
 *
 * Usage
 * ─────
 *   const fx = new ParticleEffectSystem({ poolSize: 4096 });
 *   fx.initGPU(device, format);
 *
 *   // On collision:
 *   fx.emit('collision_spark', { x: 100, y: 200 }, {
 *     impulse: 0.8,
 *     normal: { x: 0, y: -1 },
 *     species: 'cil-bolt',
 *   });
 *
 *   // On cell death:
 *   fx.emit('cell_death', { x: 300, y: 150 }, {
 *     species: 'cil-eye',
 *     radius: 24,
 *   });
 *
 *   // Each frame:
 *   fx.tick(dt);
 *   fx.render(passEncoder);
 *
 *   // Cleanup:
 *   fx.dispose();
 *
 * [ASTRO-PARTICLE-FX] debug prefix.
 */




// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────


import {

<<<<<<< HEAD
// [orphan-precise] /** Default pool capacity (pre-allocated particle slots) */
=======
/** Default pool capacity (pre-allocated particle slots) */




  getSpeciesShaderConfig,
  type SpeciesShaderConfig,
} from './species-shader-registry';

>>>>>>> ecb00e743307774715a4cdccaff74dfb0983baea
const DEFAULT_POOL_SIZE = 4096;

/** Floats per particle in the GPU instance buffer */
const GPU_STRIDE = 12;
// layout: posX, posY, velX, velY, life, maxLife, size, r, g, b, type, seed

/** Maximum concurrent effect types (used for type index in shader) */
const MAX_EFFECT_TYPES = 6;

/** Minimum alpha before a particle is considered dead */
const ALPHA_EPSILON = 0.001;

// ─────────────────────────────────────────────────────────────────────────────
// Lygia random port (sin-less hash — shared convention with contact-sparks.ts)
// ─────────────────────────────────────────────────────────────────────────────

const SCALE_X = 0.1031;
const SCALE_Y = 0.1030;
const SCALE_Z = 0.0973;

function fract(x: number): number {
  return x - Math.floor(x);
}

/** Scalar → scalar hash, range [0, 1). */
function hashScalar(p: number): number {
  let x = fract(p * SCALE_X);
  x *= x + 33.33;
  x *= x + x;
  return fract(x);
}

/** vec2 → scalar hash, range [0, 1). */
function hashVec2(sx: number, sy: number): number {
  let p3x = fract(sx * SCALE_X);
  let p3y = fract(sy * SCALE_Y);
  let p3z = fract(sx * SCALE_Z);
  const d = p3x * (p3y + 33.33) + p3y * (p3z + 33.33) + p3z * (p3x + 33.33);
  p3x += d; p3y += d; p3z += d;
  return fract((p3x + p3y) * p3z);
}

/** vec2 → vec2 hash, each component in [0, 1). */
function hashVec2To2(sx: number, sy: number): [number, number] {
  let p3x = fract(sx * SCALE_X);
  let p3y = fract(sy * SCALE_Y);
  let p3z = fract(sx * SCALE_Z);
  const d = p3x * (p3y + 19.19) + p3y * (p3z + 19.19) + p3z * (p3x + 19.19);
  p3x += d; p3y += d; p3z += d;
  return [fract((p3x + p3x) * (p3y + p3z)), fract((p3x + p3y) * (p3y + p3z))];
}

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/** The six unified effect categories */
export type ParticleEffectType =
  | 'collision_spark'
  | 'flow_trail'
  | 'ambient_dust'
  | 'qos_transition'
  | 'cell_birth'
  | 'cell_death';

/** Effect type → integer index for GPU type uniform */
const EFFECT_TYPE_INDEX: Record<ParticleEffectType, number> = {
  collision_spark:  0,
  flow_trail:       1,
  ambient_dust:     2,
  qos_transition:   3,
  cell_birth:       4,
  cell_death:       5,
};

/** 2D position */
export interface Vec2 {
  x: number;
  y: number;
}

/** Parameters for collision_spark emission */
export interface CollisionSparkParams {
  /** Normalised impulse magnitude [0, 1].  Drives count and brightness. */
  impulse: number;
  /** Surface normal at the contact point (unit vector). */
  normal: Vec2;
  /** Optional species id for colour derivation. */
  species?: string;
}

/** Parameters for flow_trail emission */
export interface FlowTrailParams {
  /** Flow velocity at the emission point (world units/s). */
  velocity: Vec2;
  /** Optional species id for colour derivation. */
  species?: string;
  /** Trail ribbon width in world units. Default 1.5. */
  width?: number;
}

/** Parameters for ambient_dust emission */
export interface AmbientDustParams {
  /** Spawn radius around the emission point. Default 100. */
  radius?: number;
  /** Number of dust motes to emit. Default 1. */
  count?: number;
}

/** Parameters for qos_transition emission */
export interface QosTransitionParams {
  /** Species id of the transitioning cell. */
  species?: string;
  /** Cell radius for the burst envelope. Default 16. */
  radius?: number;
  /** Whether transitioning to a higher QoS tier (affects colour). */
  upgrade?: boolean;
}

/** Parameters for cell_birth emission */
export interface CellBirthParams {
  /** Species id for colour / bloom derivation. */
  species?: string;
  /** Cell radius for the coalescence envelope. Default 16. */
  radius?: number;
}

/** Parameters for cell_death emission */
export interface CellDeathParams {
  /** Species id for colour / bloom derivation. */
  species?: string;
  /** Cell radius for the scatter envelope. Default 16. */
  radius?: number;
}

/** Union of all emission parameter types */
export type EmitParams =
  | CollisionSparkParams
  | FlowTrailParams
  | AmbientDustParams
  | QosTransitionParams
  | CellBirthParams
  | CellDeathParams;

/** System-wide configuration */
export interface ParticleEffectSystemConfig {
  /** Pre-allocated pool capacity. Default 4096. */
  poolSize?: number;
  /** Global time scale for all particle lifetimes. Default 1.0. */
  timeScale?: number;
  /** Global opacity multiplier. Default 1.0. */
  globalOpacity?: number;
  /** Domain width for ambient dust spawning bounds. Default 800. */
  domainWidth?: number;
  /** Domain height for ambient dust spawning bounds. Default 600. */
  domainHeight?: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal particle representation (pool-friendly, no nested objects)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Flat particle structure.  All fields are value types so the pool array
 * can be pre-allocated once and mutated in place without GC pressure.
 */
interface PooledParticle {
  /** Whether this slot is currently in use */
  active: boolean;

  // ── Spatial ────────────────────────────────────────────────────────────────
  x: number;
  y: number;
  vx: number;
  vy: number;

  // ── Lifetime ───────────────────────────────────────────────────────────────
  life: number;      // remaining lifetime (seconds)
  maxLife: number;   // total lifetime (for normalised ramp)

  // ── Visual ─────────────────────────────────────────────────────────────────
  size: number;      // current radius in world units
  baseSize: number;  // initial size (for attenuation curves)
  r: number;         // colour red   [0, 1] linear
  g: number;         // colour green [0, 1] linear
  b: number;         // colour blue  [0, 1] linear
  alpha: number;     // opacity [0, 1]

  // ── Physics ────────────────────────────────────────────────────────────────
  drag: number;      // velocity damping per second (0 = none, 1 = full stop)
  gravityY: number;  // downward acceleration (world units/s²)

  // ── Identity ───────────────────────────────────────────────────────────────
  type: ParticleEffectType;
  typeIndex: number; // integer index for GPU
  seed: number;      // per-particle random seed (for shader noise)
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-effect-type default behaviour
// ─────────────────────────────────────────────────────────────────────────────

interface EffectDefaults {
  /** Base particle count per emit call */
  count: number;
  /** Base lifetime in seconds */
  lifetime: number;
  /** Base particle radius */
  size: number;
  /** Velocity damping per second */
  drag: number;
  /** Downward gravity (world units/s²) */
  gravity: number;
  /** Base emission speed (world units/s) */
  speed: number;
  /** Half-angle of the emission cone (radians) */
  coneAngle: number;
  /** Default RGB colour */
  color: [number, number, number];
}

const EFFECT_DEFAULTS: Record<ParticleEffectType, EffectDefaults> = {
  collision_spark: {
    count: 24,
    lifetime: 0.6,
    size: 1.5,
    drag: 0.7,
    gravity: 280,
    speed: 200,
    coneAngle: Math.PI / 3,
    color: [1.0, 0.85, 0.4],    // warm amber
  },
  flow_trail: {
    count: 3,
    lifetime: 0.8,
    size: 1.0,
    drag: 0.3,
    gravity: 0,
    speed: 20,
    coneAngle: Math.PI / 8,
    color: [0.4, 0.7, 1.0],    // cool blue
  },
  ambient_dust: {
    count: 1,
    lifetime: 4.0,
    size: 0.8,
    drag: 0.1,
    gravity: 5,
    speed: 8,
    coneAngle: Math.PI,
    color: [0.6, 0.65, 0.7],   // neutral grey-blue
  },
  qos_transition: {
    count: 32,
    lifetime: 0.7,
    size: 2.0,
    drag: 0.5,
    gravity: 0,
    speed: 80,
    coneAngle: Math.PI,
    color: [0.3, 1.0, 0.5],    // upgrade green
  },
  cell_birth: {
    count: 28,
    lifetime: 0.55,
    size: 2.5,
    drag: 0.6,
    gravity: 0,
    speed: 60,
    coneAngle: Math.PI,
    color: [0.9, 0.95, 1.0],   // soft white
  },
  cell_death: {
    count: 36,
    lifetime: 0.9,
    size: 2.0,
    drag: 0.4,
    gravity: 40,
    speed: 100,
    coneAngle: Math.PI,
    color: [0.8, 0.3, 0.2],    // fading red
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Species → colour derivation
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Derive a representative particle colour from a species shader config.
 * Uses the material albedo as the base, boosted by bloom strength for
 * emissive effects (sparks, death scatter).
 */
function speciesColor(
  cfg: SpeciesShaderConfig,
  emissiveBoost: number = 0.0,
): [number, number, number] {
  const albedo = cfg.materialParams.albedo ?? [0.5, 0.5, 0.5];
  const fresnel = cfg.materialParams.fresnelColor ?? albedo;
  const bloom = cfg.bloomStrength;

  // Blend albedo toward fresnel rim colour, then boost by bloom
  const t = Math.min(emissiveBoost, 1.0);
  const boost = 1.0 + bloom * emissiveBoost * 0.3;
  return [
    Math.min((albedo[0] * (1 - t) + fresnel[0] * t) * boost, 1.0),
    Math.min((albedo[1] * (1 - t) + fresnel[1] * t) * boost, 1.0),
    Math.min((albedo[2] * (1 - t) + fresnel[2] * t) * boost, 1.0),
  ];
}

// ─────────────────────────────────────────────────────────────────────────────
// WGSL shader source — instanced quad rendering
// ─────────────────────────────────────────────────────────────────────────────

const PARTICLE_FX_WGSL = /* wgsl */ `
struct Uniforms {
  scale_x    : f32,
  scale_y    : f32,
  offset_x   : f32,
  offset_y   : f32,
  time       : f32,
  opacity    : f32,
  _pad0      : f32,
  _pad1      : f32,
}

struct ParticleInstance {
  pos_x    : f32,
  pos_y    : f32,
  vel_x    : f32,
  vel_y    : f32,
  life     : f32,
  max_life : f32,
  size     : f32,
  r        : f32,
  g        : f32,
  b        : f32,
  fx_type  : f32,
  seed     : f32,
}

@group(0) @binding(0) var<uniform> uni : Uniforms;
@group(0) @binding(1) var<storage, read> instances : array<ParticleInstance>;

struct VertexOut {
  @builtin(position) pos   : vec4f,
  @location(0)       uv    : vec2f,
  @location(1)       color : vec3f,
  @location(2)       alpha : f32,
  @location(3)       fxType: f32,
  @location(4)       seed  : f32,
}

@vertex fn vs_main(
  @builtin(vertex_index)   vi : u32,
  @builtin(instance_index) ii : u32,
) -> VertexOut {
  // Screen-aligned quad (2 triangles)
  var quadUV = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0),
    vec2f(-1.0, -1.0), vec2f( 1.0,  1.0), vec2f(-1.0,  1.0),
  );
  let uv = quadUV[vi];
  let p  = instances[ii];

  // Life ratio: 1 = just born, 0 = about to die
  let lifeRatio = clamp(p.life / max(p.max_life, 0.001), 0.0, 1.0);

  // Size attenuation: shrink near end of life
  let sizeScale = p.size * (0.3 + 0.7 * lifeRatio);

  // Speed-based elongation for sparks (type 0) and trails (type 1)
  let speed = length(vec2f(p.vel_x, p.vel_y));
  var stretch = vec2f(1.0, 1.0);
  if (p.fx_type < 1.5) {
    let elongation = min(speed * 0.003, 2.0);
    if (speed > 0.1) {
      let dir = normalize(vec2f(p.vel_x, p.vel_y));
      // Elongate along velocity direction
      let cosA = dir.x;
      let sinA = dir.y;
      let rotUV = vec2f(
        uv.x * cosA - uv.y * sinA,
        uv.x * sinA + uv.y * cosA,
      );
      // Apply stretch along local X (velocity direction)
      stretch = vec2f(1.0 + elongation, 1.0);
    }
  }

  let halfSize = sizeScale * 0.5;
  let quadPos = uv * stretch * halfSize;

  let ndcX = p.pos_x * uni.scale_x + uni.offset_x + quadPos.x * uni.scale_x;
  let ndcY = p.pos_y * uni.scale_y + uni.offset_y + quadPos.y * uni.scale_y;

  // Alpha fade: ease out near death, flash at birth
  let birthFlash = 1.0 - pow(1.0 - min(lifeRatio * 4.0, 1.0), 2.0);
  let deathFade  = smoothstep(0.0, 0.15, lifeRatio);
  let alpha      = birthFlash * deathFade * uni.opacity;

  var out : VertexOut;
  out.pos    = vec4f(ndcX, ndcY, 0.0, 1.0);
  out.uv     = uv;
  out.color  = vec3f(p.r, p.g, p.b);
  out.alpha  = alpha;
  out.fxType = p.fx_type;
  out.seed   = p.seed;
  return out;
}

@fragment fn fs_main(in: VertexOut) -> @location(0) vec4f {
  let r2 = dot(in.uv, in.uv);

  // Discard outside unit circle
  if (r2 > 1.0) { discard; }

  let fxType = u32(in.fxType + 0.5);

  // Base radial falloff
  var falloff : f32;

  switch (fxType) {
    // collision_spark: hard bright core, sharp falloff
    case 0u: {
      falloff = pow(1.0 - sqrt(r2), 3.0);
    }
    // flow_trail: soft elongated glow
    case 1u: {
      falloff = exp(-r2 * 2.5);
    }
    // ambient_dust: very soft, nearly uniform
    case 2u: {
      falloff = 1.0 - r2 * r2;
    }
    // qos_transition: ring-shaped burst
    case 3u: {
      let ring = abs(sqrt(r2) - 0.6);
      falloff = exp(-ring * ring * 20.0) + (1.0 - r2) * 0.3;
    }
    // cell_birth: soft inward glow
    case 4u: {
      falloff = pow(1.0 - r2, 2.0);
    }
    // cell_death: fragmenting, noisy edge
    case 5u: {
      let noise = fract(sin(in.seed * 12.9898 + r2 * 78.233) * 43758.5453);
      let edge  = smoothstep(0.7 + noise * 0.3, 1.0, sqrt(r2));
      falloff   = (1.0 - edge) * pow(1.0 - r2, 1.5);
    }
    default: {
      falloff = 1.0 - r2;
    }
  }

  let finalAlpha = in.alpha * falloff;
  if (finalAlpha < 0.002) { discard; }

  return vec4f(in.color * falloff, finalAlpha);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// ParticleEffectSystem
// ─────────────────────────────────────────────────────────────────────────────

export class ParticleEffectSystem {
  // ── Configuration ──────────────────────────────────────────────────────────
  private readonly poolSize: number;
  private timeScale: number;
  private globalOpacity: number;
  private domainWidth: number;
  private domainHeight: number;

  // ── Object pool ────────────────────────────────────────────────────────────
  private readonly pool: PooledParticle[];
  private readonly freeIndices: number[];
  private activeCount = 0;

  // ── GPU resources (initialised lazily via initGPU) ─────────────────────────
  private device: GPUDevice | null = null;
  private pipeline: GPURenderPipeline | null = null;
  private uniformBuffer: GPUBuffer | null = null;
  private instanceBuffer: GPUBuffer | null = null;
  private bindGroup: GPUBindGroup | null = null;
  private instanceData: Float32Array;
  private gpuReady = false;

  // ── Timing ─────────────────────────────────────────────────────────────────
  private elapsedTime = 0;

  // ── Seed counter for unique per-particle seeds ─────────────────────────────
  private seedCounter = 0;

  constructor(config: ParticleEffectSystemConfig = {}) {
    this.poolSize      = config.poolSize      ?? DEFAULT_POOL_SIZE;
    this.timeScale     = config.timeScale     ?? 1.0;
    this.globalOpacity = config.globalOpacity ?? 1.0;
    this.domainWidth   = config.domainWidth   ?? 800;
    this.domainHeight  = config.domainHeight  ?? 600;

    // Pre-allocate the entire pool
    this.pool = new Array<PooledParticle>(this.poolSize);
    this.freeIndices = new Array<number>(this.poolSize);

    for (let i = 0; i < this.poolSize; i++) {
      this.pool[i] = this._createBlankParticle();
      this.freeIndices[i] = this.poolSize - 1 - i; // stack: top = last index
    }

    // Pre-allocate the GPU upload buffer
    this.instanceData = new Float32Array(this.poolSize * GPU_STRIDE);
  }

  // ─── GPU initialisation ──────────────────────────────────────────────────

  /**
   * Initialise WebGPU resources.  Must be called once before `render()`.
   * Safe to call multiple times (no-ops after the first).
   */
  initGPU(device: GPUDevice, format: GPUTextureFormat): void {
    if (this.gpuReady) return;
    this.device = device;

    // Shader module
    const shaderModule = device.createShaderModule({
      label: 'particle-effect-system shader',
      code: PARTICLE_FX_WGSL,
    });

    // Uniform buffer (8 floats = 32 bytes, padded to 16-byte alignment)
    this.uniformBuffer = device.createBuffer({
      label: 'particle-fx uniforms',
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Instance storage buffer
    const instanceByteSize = this.poolSize * GPU_STRIDE * 4;
    this.instanceBuffer = device.createBuffer({
      label: 'particle-fx instances',
      size: instanceByteSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Bind group layout
    const bindGroupLayout = device.createBindGroupLayout({
      label: 'particle-fx bind group layout',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: 'read-only-storage' },
        },
      ],
    });

    this.bindGroup = device.createBindGroup({
      label: 'particle-fx bind group',
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.instanceBuffer } },
      ],
    });

    // Pipeline layout
    const pipelineLayout = device.createPipelineLayout({
      label: 'particle-fx pipeline layout',
      bindGroupLayouts: [bindGroupLayout],
    });

    // Render pipeline — additive blending for all particle effects
    this.pipeline = device.createRenderPipeline({
      label: 'particle-fx pipeline',
      layout: pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_main',
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_main',
        targets: [
          {
            format,
            blend: {
              color: {
                srcFactor: 'src-alpha',
                dstFactor: 'one',          // additive
                operation: 'add',
              },
              alpha: {
                srcFactor: 'one',
                dstFactor: 'one',
                operation: 'add',
              },
            },
            writeMask: GPUColorWrite.ALL,
          },
        ],
      },
      primitive: {
        topology: 'triangle-list',
      },
    });

    this.gpuReady = true;
  }

  // ─── Emission API ────────────────────────────────────────────────────────

  /**
   * Emit particles of the given effect type at the specified position.
   *
   * @param type     One of the six particle effect categories.
   * @param position World-space emission origin.
   * @param params   Type-specific parameters (impulse, species, radius, etc).
   * @returns        Number of particles actually emitted (may be less than
   *                 requested if the pool is full).
   */
  emit(type: ParticleEffectType, position: Vec2, params: EmitParams): number {
    switch (type) {
      case 'collision_spark':
        return this._emitCollisionSpark(position, params as CollisionSparkParams);
      case 'flow_trail':
        return this._emitFlowTrail(position, params as FlowTrailParams);
      case 'ambient_dust':
        return this._emitAmbientDust(position, params as AmbientDustParams);
      case 'qos_transition':
        return this._emitQosTransition(position, params as QosTransitionParams);
      case 'cell_birth':
        return this._emitCellBirth(position, params as CellBirthParams);
      case 'cell_death':
        return this._emitCellDeath(position, params as CellDeathParams);
      default:
        return 0;
    }
  }

  // ─── Tick (CPU physics update) ─────────────────────────────────────────────

  /**
   * Advance all active particles by `dt` seconds.
   * Dead particles are returned to the free list.
   */
  tick(dt: number): void {
    const scaledDt = dt * this.timeScale;
    this.elapsedTime += scaledDt;

    for (let i = 0; i < this.poolSize; i++) {
      const p = this.pool[i];
      if (!p.active) continue;

      // Integrate velocity
      p.vx *= (1.0 - p.drag * scaledDt);
      p.vy *= (1.0 - p.drag * scaledDt);
      p.vy += p.gravityY * scaledDt;

      p.x += p.vx * scaledDt;
      p.y += p.vy * scaledDt;

      // Decrement life
      p.life -= scaledDt;

      // Size attenuation
      const lifeRatio = Math.max(p.life / p.maxLife, 0);
      p.size = p.baseSize * (0.3 + 0.7 * lifeRatio);

      // Alpha ramp: fade out near death
      p.alpha = Math.min(lifeRatio * 4.0, 1.0) * smoothstep(0, 0.15, lifeRatio);

      // Reclaim dead particles
      if (p.life <= 0 || p.alpha < ALPHA_EPSILON) {
        p.active = false;
        this.freeIndices.push(i);
        this.activeCount--;
      }
    }
  }

  // ─── GPU Render ────────────────────────────────────────────────────────────

  /**
   * Pack active particles into the GPU buffer and issue a single instanced
   * draw call.  The render pass must already be begun on the encoder.
   *
   * @param passEncoder An active GPURenderPassEncoder.
   * @param scaleX      NDC scale for X axis (e.g. 2/domainWidth).
   * @param scaleY      NDC scale for Y axis (e.g. -2/domainHeight, Y-down).
   * @param offsetX     NDC offset X. Default -1.
   * @param offsetY     NDC offset Y. Default  1.
   */
  render(
    passEncoder: GPURenderPassEncoder,
    scaleX: number = 2.0 / this.domainWidth,
    scaleY: number = -2.0 / this.domainHeight,
    offsetX: number = -1.0,
    offsetY: number = 1.0,
  ): void {
    if (!this.gpuReady || this.activeCount === 0) return;

    const device = this.device!;

    // Pack active particles into the instance array
    let writeIdx = 0;
    for (let i = 0; i < this.poolSize; i++) {
      const p = this.pool[i];
      if (!p.active) continue;

      const offset = writeIdx * GPU_STRIDE;
      this.instanceData[offset + 0]  = p.x;
      this.instanceData[offset + 1]  = p.y;
      this.instanceData[offset + 2]  = p.vx;
      this.instanceData[offset + 3]  = p.vy;
      this.instanceData[offset + 4]  = p.life;
      this.instanceData[offset + 5]  = p.maxLife;
      this.instanceData[offset + 6]  = p.size;
      this.instanceData[offset + 7]  = p.r;
      this.instanceData[offset + 8]  = p.g;
      this.instanceData[offset + 9]  = p.b;
      this.instanceData[offset + 10] = p.typeIndex;
      this.instanceData[offset + 11] = p.seed;
      writeIdx++;
    }

    // Upload uniform data
    const uniformData = new Float32Array([
      scaleX, scaleY, offsetX, offsetY,
      this.elapsedTime, this.globalOpacity, 0, 0,
    ]);
    device.queue.writeBuffer(this.uniformBuffer!, 0, uniformData);

    // Upload instance data (only the active portion)
    const activeByteLength = writeIdx * GPU_STRIDE * 4;
    device.queue.writeBuffer(
      this.instanceBuffer!, 0,
      this.instanceData.buffer, 0, activeByteLength,
    );

    // Draw
    passEncoder.setPipeline(this.pipeline!);
    passEncoder.setBindGroup(0, this.bindGroup!);
    passEncoder.draw(6, writeIdx, 0, 0); // 6 verts per quad, writeIdx instances
  }

  // ─── Canvas 2D fallback renderer ──────────────────────────────────────────

  /**
   * Render all active particles to a Canvas 2D context.
   * Use this when WebGPU is not available.
   */
  renderCanvas(ctx: CanvasRenderingContext2D): void {
    for (let i = 0; i < this.poolSize; i++) {
      const p = this.pool[i];
      if (!p.active) continue;

      const lifeRatio = Math.max(p.life / p.maxLife, 0);
      const alpha = Math.min(lifeRatio * 4.0, 1.0)
                    * smoothstep(0, 0.15, lifeRatio)
                    * this.globalOpacity;
      if (alpha < ALPHA_EPSILON) continue;

      const r = Math.round(p.r * 255);
      const g = Math.round(p.g * 255);
      const b = Math.round(p.b * 255);

      ctx.globalAlpha = alpha;

      // Speed-based elongation for sparks and trails
      if (p.typeIndex <= 1) {
        const speed = Math.sqrt(p.vx * p.vx + p.vy * p.vy);
        if (speed > 1) {
          const dx = p.vx / speed;
          const dy = p.vy / speed;
          const tailLen = Math.min(speed * 0.06, p.size * 3);

          ctx.strokeStyle = `rgba(${r},${g},${b},${alpha})`;
          ctx.lineWidth = p.size * 0.6;
          ctx.lineCap = 'round';
          ctx.beginPath();
          ctx.moveTo(p.x - dx * tailLen, p.y - dy * tailLen);
          ctx.lineTo(p.x, p.y);
          ctx.stroke();
        }
      }

      // Draw the particle disc
      ctx.fillStyle = `rgba(${r},${g},${b},${alpha})`;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.globalAlpha = 1;
  }

  // ─── Queries ───────────────────────────────────────────────────────────────

  /** Number of currently active particles across all effect types. */
  getActiveCount(): number {
    return this.activeCount;
  }

  /** Number of free slots remaining in the pool. */
  getFreeCount(): number {
    return this.freeIndices.length;
  }

  /** Total pool capacity. */
  getPoolSize(): number {
    return this.poolSize;
  }

  /** Per-type active particle counts for diagnostics. */
  getActiveCountByType(): Record<ParticleEffectType, number> {
    const counts: Record<ParticleEffectType, number> = {
      collision_spark: 0,
      flow_trail: 0,
      ambient_dust: 0,
      qos_transition: 0,
      cell_birth: 0,
      cell_death: 0,
    };
    for (let i = 0; i < this.poolSize; i++) {
      const p = this.pool[i];
      if (p.active) {
        counts[p.type]++;
      }
    }
    return counts;
  }

  // ─── Configuration setters ─────────────────────────────────────────────────

  setTimeScale(scale: number): void {
    this.timeScale = Math.max(0, scale);
  }

  setGlobalOpacity(opacity: number): void {
    this.globalOpacity = Math.max(0, Math.min(1, opacity));
  }

  setDomainSize(width: number, height: number): void {
    this.domainWidth = width;
    this.domainHeight = height;
  }

  // ─── Cleanup ───────────────────────────────────────────────────────────────

  /** Release all GPU resources. The system cannot render after this call. */
  dispose(): void {
    this.uniformBuffer?.destroy();
    this.instanceBuffer?.destroy();
    this.uniformBuffer = null;
    this.instanceBuffer = null;
    this.pipeline = null;
    this.bindGroup = null;
    this.device = null;
    this.gpuReady = false;
  }

  /** Kill all active particles immediately (return to pool). */
  clear(): void {
    for (let i = 0; i < this.poolSize; i++) {
      if (this.pool[i].active) {
        this.pool[i].active = false;
        this.freeIndices.push(i);
      }
    }
    this.activeCount = 0;
  }

  // ─── Private: pool management ──────────────────────────────────────────────

  private _acquire(): PooledParticle | null {
    if (this.freeIndices.length === 0) return null;
    const idx = this.freeIndices.pop()!;
    const p = this.pool[idx];
    p.active = true;
    this.activeCount++;
    return p;
  }

  private _createBlankParticle(): PooledParticle {
    return {
      active: false,
      x: 0, y: 0, vx: 0, vy: 0,
      life: 0, maxLife: 0,
      size: 0, baseSize: 0,
      r: 0, g: 0, b: 0, alpha: 0,
      drag: 0, gravityY: 0,
      type: 'collision_spark',
      typeIndex: 0,
      seed: 0,
    };
  }

  private _nextSeed(): number {
    return hashScalar(++this.seedCounter * 1.618033988749895);
  }

  // ─── Private: per-type emission logic ──────────────────────────────────────

  private _emitCollisionSpark(pos: Vec2, params: CollisionSparkParams): number {
    const defaults = EFFECT_DEFAULTS.collision_spark;
    const impulse = Math.max(0, Math.min(1, params.impulse));
    const count = Math.ceil(defaults.count * Math.sqrt(impulse));

    // Derive colour from species if provided
    let color = defaults.color;
    if (params.species) {
      const cfg = getSpeciesShaderConfig(params.species);
      color = speciesColor(cfg, impulse);
    }

    // Normal direction → base emission angle
    const nx = params.normal.x;
    const ny = params.normal.y;
    const baseAngle = Math.atan2(ny, nx);

    let emitted = 0;
    for (let i = 0; i < count; i++) {
      const p = this._acquire();
      if (!p) break;

      const seed = this._nextSeed();
      const [rndAngle, rndSpeed] = hashVec2To2(seed, i * 0.37);

      // Cone scatter around normal
      const angle = baseAngle + (rndAngle - 0.5) * 2 * defaults.coneAngle;
      const speed = defaults.speed * (0.4 + impulse * 0.6) * (0.5 + rndSpeed);

      p.x = pos.x;
      p.y = pos.y;
      p.vx = Math.cos(angle) * speed;
      p.vy = Math.sin(angle) * speed;
      p.life = defaults.lifetime * (0.5 + impulse * 0.5) * (0.7 + rndSpeed * 0.3);
      p.maxLife = p.life;
      p.size = defaults.size * (0.5 + impulse * 0.5);
      p.baseSize = p.size;
      p.r = Math.min(color[0] + (1 - color[0]) * impulse * 0.4, 1.0);
      p.g = Math.min(color[1] + (1 - color[1]) * impulse * 0.2, 1.0);
      p.b = color[2];
      p.alpha = 1.0;
      p.drag = defaults.drag;
      p.gravityY = defaults.gravity;
      p.type = 'collision_spark';
      p.typeIndex = EFFECT_TYPE_INDEX.collision_spark;
      p.seed = seed;
      emitted++;
    }
    return emitted;
  }

  private _emitFlowTrail(pos: Vec2, params: FlowTrailParams): number {
    const defaults = EFFECT_DEFAULTS.flow_trail;

    let color = defaults.color;
    if (params.species) {
      const cfg = getSpeciesShaderConfig(params.species);
      color = speciesColor(cfg, 0.2);
    }

    const vx = params.velocity.x;
    const vy = params.velocity.y;
    const speed = Math.sqrt(vx * vx + vy * vy);
    const width = params.width ?? 1.5;

    let emitted = 0;
    for (let i = 0; i < defaults.count; i++) {
      const p = this._acquire();
      if (!p) break;

      const seed = this._nextSeed();
      const rnd = hashScalar(seed + i * 0.53);

      // Emit along the velocity vector with slight perpendicular scatter
      const perpX = speed > 0.1 ? -vy / speed : 0;
      const perpY = speed > 0.1 ?  vx / speed : 0;
      const scatter = (rnd - 0.5) * width;

      p.x = pos.x + perpX * scatter;
      p.y = pos.y + perpY * scatter;
      p.vx = vx * 0.8 + (rnd - 0.5) * defaults.speed;
      p.vy = vy * 0.8 + (hashScalar(seed + 7.7) - 0.5) * defaults.speed;
      p.life = defaults.lifetime * (0.7 + rnd * 0.3);
      p.maxLife = p.life;
      p.size = defaults.size * width;
      p.baseSize = p.size;
      p.r = color[0];
      p.g = color[1];
      p.b = color[2];
      p.alpha = 1.0;
      p.drag = defaults.drag;
      p.gravityY = defaults.gravity;
      p.type = 'flow_trail';
      p.typeIndex = EFFECT_TYPE_INDEX.flow_trail;
      p.seed = seed;
      emitted++;
    }
    return emitted;
  }

  private _emitAmbientDust(pos: Vec2, params: AmbientDustParams): number {
    const defaults = EFFECT_DEFAULTS.ambient_dust;
    const radius = params.radius ?? 100;
    const count = params.count ?? 1;

    let emitted = 0;
    for (let i = 0; i < count; i++) {
      const p = this._acquire();
      if (!p) break;

      const seed = this._nextSeed();
      const [rndX, rndY] = hashVec2To2(seed, i * 1.17);
      const rndSize = hashScalar(seed + 3.3);
      const rndAngle = hashScalar(seed + 5.5) * Math.PI * 2;

      // Scatter within the spawn radius
      p.x = pos.x + (rndX - 0.5) * 2 * radius;
      p.y = pos.y + (rndY - 0.5) * 2 * radius;
      p.vx = Math.cos(rndAngle) * defaults.speed * (0.3 + rndSize * 0.7);
      p.vy = Math.sin(rndAngle) * defaults.speed * (0.3 + rndSize * 0.7);
      p.life = defaults.lifetime * (0.6 + rndSize * 0.4);
      p.maxLife = p.life;
      p.size = defaults.size * (0.4 + rndSize * 0.6);
      p.baseSize = p.size;
      p.r = defaults.color[0] + (rndX - 0.5) * 0.1;
      p.g = defaults.color[1] + (rndY - 0.5) * 0.1;
      p.b = defaults.color[2];
      p.alpha = 0.3 + rndSize * 0.4; // dust is always somewhat transparent
      p.drag = defaults.drag;
      p.gravityY = defaults.gravity;
      p.type = 'ambient_dust';
      p.typeIndex = EFFECT_TYPE_INDEX.ambient_dust;
      p.seed = seed;
      emitted++;
    }
    return emitted;
  }

  private _emitQosTransition(pos: Vec2, params: QosTransitionParams): number {
    const defaults = EFFECT_DEFAULTS.qos_transition;
    const radius = params.radius ?? 16;

    // Derive colour: green for upgrade, orange for downgrade, or from species
    let color = defaults.color;
    if (params.species) {
      const cfg = getSpeciesShaderConfig(params.species);
      color = speciesColor(cfg, 0.5);
    }
    if (params.upgrade === false) {
      color = [1.0, 0.5, 0.15]; // downgrade orange
    }

    let emitted = 0;
    for (let i = 0; i < defaults.count; i++) {
      const p = this._acquire();
      if (!p) break;

      const seed = this._nextSeed();
      const angle = (i / defaults.count) * Math.PI * 2 + hashScalar(seed) * 0.3;
      const rndSpeed = 0.6 + hashScalar(seed + 1.1) * 0.4;

      // Radial burst from the cell boundary
      p.x = pos.x + Math.cos(angle) * radius * 0.8;
      p.y = pos.y + Math.sin(angle) * radius * 0.8;
      p.vx = Math.cos(angle) * defaults.speed * rndSpeed;
      p.vy = Math.sin(angle) * defaults.speed * rndSpeed;
      p.life = defaults.lifetime * (0.7 + hashScalar(seed + 2.2) * 0.3);
      p.maxLife = p.life;
      p.size = defaults.size * (0.6 + hashScalar(seed + 3.3) * 0.4);
      p.baseSize = p.size;
      p.r = color[0];
      p.g = color[1];
      p.b = color[2];
      p.alpha = 1.0;
      p.drag = defaults.drag;
      p.gravityY = defaults.gravity;
      p.type = 'qos_transition';
      p.typeIndex = EFFECT_TYPE_INDEX.qos_transition;
      p.seed = seed;
      emitted++;
    }
    return emitted;
  }

  private _emitCellBirth(pos: Vec2, params: CellBirthParams): number {
    const defaults = EFFECT_DEFAULTS.cell_birth;
    const radius = params.radius ?? 16;

    let color = defaults.color;
    if (params.species) {
      const cfg = getSpeciesShaderConfig(params.species);
      color = speciesColor(cfg, 0.3);
    }

    let emitted = 0;
    for (let i = 0; i < defaults.count; i++) {
      const p = this._acquire();
      if (!p) break;

      const seed = this._nextSeed();
      const angle = (i / defaults.count) * Math.PI * 2 + hashScalar(seed) * 0.4;
      const spawnDist = radius * (1.5 + hashScalar(seed + 0.7) * 1.0);

      // Birth: particles start far out and converge inward
      p.x = pos.x + Math.cos(angle) * spawnDist;
      p.y = pos.y + Math.sin(angle) * spawnDist;
      // Velocity points inward (toward pos)
      p.vx = -Math.cos(angle) * defaults.speed * (0.6 + hashScalar(seed + 1.1) * 0.4);
      p.vy = -Math.sin(angle) * defaults.speed * (0.6 + hashScalar(seed + 2.2) * 0.4);
      p.life = defaults.lifetime * (0.8 + hashScalar(seed + 3.3) * 0.2);
      p.maxLife = p.life;
      p.size = defaults.size * (0.5 + hashScalar(seed + 4.4) * 0.5);
      p.baseSize = p.size;
      p.r = color[0];
      p.g = color[1];
      p.b = color[2];
      p.alpha = 1.0;
      p.drag = defaults.drag;
      p.gravityY = defaults.gravity;
      p.type = 'cell_birth';
      p.typeIndex = EFFECT_TYPE_INDEX.cell_birth;
      p.seed = seed;
      emitted++;
    }
    return emitted;
  }

  private _emitCellDeath(pos: Vec2, params: CellDeathParams): number {
    const defaults = EFFECT_DEFAULTS.cell_death;
    const radius = params.radius ?? 16;

    let color = defaults.color;
    if (params.species) {
      const cfg = getSpeciesShaderConfig(params.species);
      // Death uses a desaturated, dimmer variant of the species colour
      const base = speciesColor(cfg, 0.1);
      color = [
        base[0] * 0.7 + 0.3 * defaults.color[0],
        base[1] * 0.7 + 0.3 * defaults.color[1],
        base[2] * 0.7 + 0.3 * defaults.color[2],
      ];
    }

    let emitted = 0;
    for (let i = 0; i < defaults.count; i++) {
      const p = this._acquire();
      if (!p) break;

      const seed = this._nextSeed();
      const angle = (i / defaults.count) * Math.PI * 2 + hashScalar(seed) * 0.5;
      const rndSpeed = 0.5 + hashScalar(seed + 1.1) * 0.5;

      // Death: particles start near the cell centre and scatter outward
      const startDist = radius * (0.2 + hashScalar(seed + 0.3) * 0.6);
      p.x = pos.x + Math.cos(angle) * startDist;
      p.y = pos.y + Math.sin(angle) * startDist;
      p.vx = Math.cos(angle) * defaults.speed * rndSpeed;
      p.vy = Math.sin(angle) * defaults.speed * rndSpeed;
      p.life = defaults.lifetime * (0.6 + hashScalar(seed + 2.2) * 0.4);
      p.maxLife = p.life;
      p.size = defaults.size * (0.4 + hashScalar(seed + 3.3) * 0.6);
      p.baseSize = p.size;
      // Per-particle colour variation for organic feel
      const colorJitter = hashScalar(seed + 5.5) * 0.15;
      p.r = Math.min(color[0] + colorJitter, 1.0);
      p.g = Math.max(color[1] - colorJitter * 0.5, 0.0);
      p.b = Math.max(color[2] - colorJitter * 0.3, 0.0);
      p.alpha = 1.0;
      p.drag = defaults.drag;
      p.gravityY = defaults.gravity;
      p.type = 'cell_death';
      p.typeIndex = EFFECT_TYPE_INDEX.cell_death;
      p.seed = seed;
      emitted++;
    }
    return emitted;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility
// ─────────────────────────────────────────────────────────────────────────────

/** GLSL-style smoothstep — clamped Hermite interpolation */
function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

// ─────────────────────────────────────────────────────────────────────────────
// Re-exports for barrel
// ─────────────────────────────────────────────────────────────────────────────

export { EFFECT_DEFAULTS, EFFECT_TYPE_INDEX, GPU_STRIDE, PARTICLE_FX_WGSL };
