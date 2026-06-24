/**
 * physarum-sim.ts
 *
 * GPU Physarum polycephalum slime-mould simulation (Jones 2010).
 *   "Characteristics of pattern formation and evolution in approximations of
 *    Physarum transport networks"  — Jeff Jones, 2010.
 *
 * The model produces emergent network patterns strikingly similar to fungal
 * hyphae, vascular systems, and urban road networks — arising purely from
 * millions of simple agents following local chemical gradients.
 *
 * Algorithm (one frame):
 *   1. sense   — each agent samples the trail map at three forward probes
 *                (centre, left rotated by sensorAngle, right rotated by –sensorAngle)
 *   2. rotate  — steer heading toward the probe with highest trail concentration
 *   3. move    — advance agent by stepSize along new heading; wrap at boundary
 *   4. deposit — add depositAmount to the trail map at the new position
 *   5. diffuse — 3×3 box-blur the trail map (spreading pheromone)
 *   6. decay   — multiply every trail cell by (1 – decayRate)
 *
 * GPU architecture:
 *   agent_step   compute — steps 1–4, 64 threads/workgroup, one thread per agent
 *   diffuse_decay compute — steps 5–6, 16×16 threads/workgroup, one thread per pixel
 *
 * Initial agent distribution uses an inline WGSL port of the Ashima/stegu
 * 2-D simplex noise (upstream/webgl-noise/src/noise2D.glsl), so agents start
 * in a biologically plausible clustered pattern rather than a perfect grid.
 *
 * Public API:
 *   const sim = await PhysarumSimulation.create(device, 1024, 1024, 1_000_000);
 *   // inside rAF:
 *   sim.tick(commandEncoder);
 *   // bind sim.getTrailTexture() as a sampled texture in your render pass
 */

// ─────────────────────────────────────────────────────────────────────────────
// Constants & tuneable parameters
// ─────────────────────────────────────────────────────────────────────────────

/** Workgroup size for the agent step pass (1-D, one thread per agent). */








const AGENT_WG = 64;

/** Workgroup tile for the diffuse/decay pass (2-D). */
const DIFFUSE_WG = 16;

/** Trail texture format: single-channel f32 for pheromone concentration. */
const TRAIL_FORMAT: GPUTextureFormat = "r32float";

// ─────────────────────────────────────────────────────────────────────────────
// Physarum simulation parameters (all tuneable at runtime via updateParams)
// ─────────────────────────────────────────────────────────────────────────────

export interface PhysarumParams {
  /** How far ahead each sensor probe reaches (pixels). Default 9. */
  sensorDistance: number;
  /** Angle (radians) between centre and side probes. Default 0.4. */
  sensorAngle: number;
  /** Turn speed multiplier (radians/step). Default 0.3. */
  turnSpeed: number;
  /** Movement speed (pixels/step). Default 1.5. */
  stepSize: number;
  /** Trail pheromone deposited per agent per step. Default 5.0. */
  depositAmount: number;
  /** Fraction of trail evaporated each step. Default 0.010. */
  decayRate: number;
  /** Current simulation time (seconds) — used to animate initial seeding only. */
  time: number;
}

const DEFAULT_PARAMS: PhysarumParams = {
  sensorDistance: 9.0,
  sensorAngle:    0.4,
  turnSpeed:      0.3,
  stepSize:       1.5,
  depositAmount:  5.0,
  decayRate:      0.010,
  time:           0.0,
};

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Simplex noise basis
// Ported from upstream/webgl-noise/src/noise2D.glsl (Ashima Arts / stegu)
// MIT License — Copyright (C) 2011 Ashima Arts.
// Adapted to WGSL syntax; semantics are identical.
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_SIMPLEX2 = /* wgsl */`
// ── Ashima simplex 2-D noise — WGSL port ────────────────────────────────────
fn _sn_mod289v2(x: vec2f) -> vec2f { return x - floor(x * (1.0 / 289.0)) * 289.0; }
fn _sn_mod289v3(x: vec3f) -> vec3f { return x - floor(x * (1.0 / 289.0)) * 289.0; }
fn _sn_perm3(x: vec3f)    -> vec3f { return _sn_mod289v3(((x * 34.0) + 10.0) * x); }

fn snoise2(v: vec2f) -> f32 {
  // C constants
  let Cx = 0.211324865405187;   // (3 - sqrt(3)) / 6
  let Cy = 0.366025403784439;   // 0.5 * (sqrt(3) - 1)
  let Cz = -0.577350269189626;  // -1 + 2*Cx
  let Cw = 0.024390243902439;   // 1 / 41

  var i  = floor(v + dot(v, vec2f(Cy)));
  let x0 = v - i + dot(i, vec2f(Cx));

  let i1 = select(vec2f(0.0, 1.0), vec2f(1.0, 0.0), x0.x > x0.y);
  let x12 = x0.xyxy + vec4f(Cx, Cx, Cz, Cz) - vec4f(i1, 1.0, 1.0);

  i = _sn_mod289v2(i);
  let p = _sn_perm3(
    _sn_perm3(i.y + vec3f(0.0, i1.y, 1.0)) + i.x + vec3f(0.0, i1.x, 1.0));

  var m = max(
    0.5 - vec3f(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)),
    vec3f(0.0));
  m = m * m;
  m = m * m;

  let xg  = 2.0 * fract(p * Cw) - 1.0;
  let h   = abs(xg) - 0.5;
  let ox  = floor(xg + 0.5);
  let a0  = xg - ox;

  m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);

  var g: vec3f;
  g.x  = a0.x  * x0.x   + h.x  * x0.y;
  g.y  = a0.y  * x12.x  + h.y  * x12.y;
  g.z  = a0.z  * x12.z  + h.z  * x12.w;
  return 130.0 * dot(m, g);
}

/// Fractional Brownian Motion over snoise2 — 4 octaves.
fn fbm2(p: vec2f) -> f32 {
  var v   = 0.0;
  var amp = 0.5;
  var pos = p;
  for (var i = 0; i < 4; i++) {
    v   += amp * snoise2(pos);
    pos *= 2.1;
    amp *= 0.5;
  }
  return v;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Agent step shader (sense → rotate → move → deposit)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build the agent-step WGSL source. Width/height and agentCount are
 * baked in as override constants so the compiler can optimise bounds checks.
 */
function buildAgentStepShader(width: number, height: number): string {
  return /* wgsl */`
${WGSL_SIMPLEX2}

// ── Uniforms ──────────────────────────────────────────────────────────────────
struct PhysarumUniforms {
  sensorDist  : f32,  // probe reach in pixels
  sensorAngle : f32,  // side-probe angle (radians)
  turnSpeed   : f32,  // heading rotation per step (radians)
  stepSize    : f32,  // pixels moved per step
  depositAmt  : f32,  // pheromone deposited per agent per step
  decayRate   : f32,  // fraction evaporated each step (diffuse_decay pass)
  time        : f32,  // simulation time (s) — for noise seeding
  agentCount  : u32,  // total number of agents
}
@group(0) @binding(0) var<uniform> u : PhysarumUniforms;

// ── Agent buffers (AoS-of-SoA: x, y, heading in separate arrays) ─────────────
@group(1) @binding(0) var<storage, read_write> agentX   : array<f32>;
@group(1) @binding(1) var<storage, read_write> agentY   : array<f32>;
@group(1) @binding(2) var<storage, read_write> agentAng : array<f32>;  // radians

// ── Trail map (read sample + write deposit) ───────────────────────────────────
@group(2) @binding(0) var trailRead  : texture_2d<f32>;
@group(2) @binding(1) var trailWrite : texture_storage_2d<r32float, write>;

// ── Helpers ───────────────────────────────────────────────────────────────────

const WIDTH  : f32 = ${width}.0;
const HEIGHT : f32 = ${height}.0;
const TWO_PI : f32 = 6.28318530718;

/// Wrap a position component into [0, dim).
fn wrap(v: f32, dim: f32) -> f32 {
  return ((v % dim) + dim) % dim;
}

/// Sample trail map at floating-point position (nearest-neighbour, wrapping).
fn sampleTrail(pos: vec2f) -> f32 {
  let ix = i32(wrap(pos.x, WIDTH));
  let iy = i32(wrap(pos.y, HEIGHT));
  return textureLoad(trailRead, vec2i(ix, iy), 0).r;
}

/// Probe position in front of an agent at a given angular offset.
fn probePos(px: f32, py: f32, ang: f32, offset: f32, dist: f32) -> vec2f {
  let a = ang + offset;
  return vec2f(px + dist * cos(a), py + dist * sin(a));
}

// ── Main ──────────────────────────────────────────────────────────────────────

@compute @workgroup_size(${AGENT_WG})
fn agent_step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= u.agentCount) { return; }

  var px  = agentX[idx];
  var py  = agentY[idx];
  var ang = agentAng[idx];

  // ── 1. Sense: sample three probes ──────────────────────────────────────────
  let fwd   = sampleTrail(probePos(px, py, ang,  0.0,        u.sensorDist));
  let left  = sampleTrail(probePos(px, py, ang,  u.sensorAngle, u.sensorDist));
  let right = sampleTrail(probePos(px, py, ang, -u.sensorAngle, u.sensorDist));

  // ── 2. Rotate: steer toward highest concentration ─────────────────────────
  //   Jones 2010 Table 1 decision tree:
  //     fwd > left AND fwd > right  → no turn
  //     fwd < left AND fwd < right  → random ±turn  (approximate with idx hash)
  //     left > right                → turn left
  //     right >= left               → turn right
  if (fwd >= left && fwd >= right) {
    // stay straight — fastest path, do nothing
  } else if (left > right) {
    ang += u.turnSpeed;
  } else if (right > left) {
    ang -= u.turnSpeed;
  } else {
    // ambiguous — stochastic turn using index as cheap hash
    let h = (idx * 2654435761u) ^ (idx >> 16u);   // Knuth multiplicative hash
    ang  += select(-u.turnSpeed, u.turnSpeed, (h & 1u) == 0u);
  }

  // ── 3. Move ────────────────────────────────────────────────────────────────
  px  = wrap(px  + u.stepSize * cos(ang), WIDTH);
  py  = wrap(py  + u.stepSize * sin(ang), HEIGHT);

  // ── 4. Deposit: add pheromone at new position ─────────────────────────────
  let ix = i32(px);
  let iy = i32(py);
  // Accumulate on top of whatever is already there (read–modify–write).
  // Note: atomic add on texture isn't available in WebGPU; using a separate
  // deposit texture would require an extra pass.  Instead we read the current
  // value and write back.  Race conditions between agents depositing to the
  // same pixel merely reduce effective deposit; this is acceptable for visual
  // purposes (Jones' original CPU code has the same issue at high densities).
  let current = textureLoad(trailRead, vec2i(ix, iy), 0).r;
  textureStore(trailWrite, vec2i(ix, iy), vec4f(current + u.depositAmt, 0.0, 0.0, 1.0));

  // Write back agent state
  agentX[idx]   = px;
  agentY[idx]   = py;
  agentAng[idx] = ang;
}

// ── Init pass: seed agents using simplex noise ────────────────────────────────
// Called once at startup.  Positions agents in dense clusters proportional to
// fbm2 intensity at their normalised canvas coordinate (matches Jones 2010 §4
// where initial concentration mimics a nutrient gradient).

@compute @workgroup_size(${AGENT_WG})
fn init_agents(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= u.agentCount) { return; }

  // Cheap LCG random from index → seed position
  var rng  = idx * 1664525u + 1013904223u;
  rng = rng ^ (rng >> 16u);
  rng = rng * 2246822519u;
  rng = rng ^ (rng >> 13u);

  let rx = f32(rng & 0xFFFFu) / 65535.0;
  rng = rng * 1664525u + 1013904223u;
  let ry = f32(rng & 0xFFFFu) / 65535.0;
  rng = rng * 1664525u + 1013904223u;
  let ra = f32(rng & 0xFFFFu) / 65535.0;

  // Rejection-sample via simplex noise: accept position with probability
  // proportional to max(0, fbm2(uv * 3.0 + time*0.1)).
  // Agents with higher fbm → denser filaments → more lifelike initial clusters.
  // For simplicity we use the noise to *bias* the position rather than full
  // rejection sampling (which would need variable iteration count).
  let uv    = vec2f(rx, ry);
  let noise = fbm2(uv * 3.0 + vec2f(u.time * 0.1));
  // nudge position toward high-noise regions (soft attraction, not hard rejection)
  let bias  = clamp((noise + 1.0) * 0.5, 0.0, 1.0);

  // Perturb: cluster around a few noise-bright spots
  let cx = rx + (fbm2(uv * 1.3 + 7.3) * 0.5 - 0.25) * bias;
  let cy = ry + (fbm2(uv * 1.3 + 2.1) * 0.5 - 0.25) * bias;

  agentX[idx]   = wrap(cx * WIDTH,  WIDTH);
  agentY[idx]   = wrap(cy * HEIGHT, HEIGHT);
  agentAng[idx] = ra * 6.28318530718;
}
`;
}

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Diffuse & Decay shader (3×3 box blur + evaporation)
// ─────────────────────────────────────────────────────────────────────────────

function buildDiffuseDecayShader(width: number, height: number): string {
  return /* wgsl */`
struct DiffuseUniforms {
  decay   : f32,
  _pad0   : f32,
  _pad1   : f32,
  _pad2   : f32,
}
@group(0) @binding(0) var<uniform> u : DiffuseUniforms;

@group(1) @binding(0) var trailRead  : texture_2d<f32>;
@group(1) @binding(1) var trailWrite : texture_storage_2d<r32float, write>;

const W : i32 = ${width};
const H : i32 = ${height};

/// Read trail with toroidal (wrap-around) boundary.
fn load(x: i32, y: i32) -> f32 {
  let wx = ((x % W) + W) % W;
  let wy = ((y % H) + H) % H;
  return textureLoad(trailRead, vec2i(wx, wy), 0).r;
}

@compute @workgroup_size(${DIFFUSE_WG}, ${DIFFUSE_WG})
fn diffuse_decay(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ix = i32(gid.x);
  let iy = i32(gid.y);
  if (ix >= W || iy >= H) { return; }

  // 3×3 box blur (uniform weights → fast diffusion)
  var sum = 0.0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      sum += load(ix + dx, iy + dy);
    }
  }
  let blurred = sum / 9.0;

  // Evaporation
  let decayed = blurred * (1.0 - u.decay);

  textureStore(trailWrite, vec2i(ix, iy), vec4f(decayed, 0.0, 0.0, 1.0));
}
`;
}

// ─────────────────────────────────────────────────────────────────────────────
// PhysarumSimulation class
// ─────────────────────────────────────────────────────────────────────────────

export class PhysarumSimulation {
  // GPU device
  private readonly device: any /*GPUDevice*/;

  // Dimensions
  readonly width:      number;
  readonly height:     number;
  readonly agentCount: number;

  // Simulation params (host copy)
  private params: PhysarumParams;

  // ── Agent buffers ─────────────────────────────────────────────────────────
  private agentXBuf!:   GPUBuffer;
  private agentYBuf!:   GPUBuffer;
  private agentAngBuf!: any /*GPUBuffer*/;

  // ── Trail textures (ping-pong) ────────────────────────────────────────────
  private trailA!:      GPUTexture;
  private trailB!:      GPUTexture;
  private trailViewA!:  GPUTextureView;
  private trailViewB!:  GPUTextureView;
  /** 0 = A is current read, B is write; 1 = B is current read, A is write. */
  private pingPong = 0;

  // ── Uniform buffers ───────────────────────────────────────────────────────
  private agentUniformBuf!:   GPUBuffer;
  private diffuseUniformBuf!: any /*GPUBuffer*/;

  // ── Pipelines ─────────────────────────────────────────────────────────────
  private agentPipeline!:   GPUComputePipeline;
  private initPipeline!:    GPUComputePipeline;
  private diffusePipeline!: any /*GPUComputePipeline*/;

  // ── Bind group layouts ────────────────────────────────────────────────────
  private agentUniformBGL!:   GPUBindGroupLayout;
  private agentBufBGL!:       GPUBindGroupLayout;
  private agentTrailBGL!:     GPUBindGroupLayout;
  private diffuseUniformBGL!: GPUBindGroupLayout;
  private diffuseTrailBGL!:   GPUBindGroupLayout;

  // ── Bind groups (two sets for ping-pong) ──────────────────────────────────
  private agentTrailBG_AB!:     GPUBindGroup;  // read A, write B
  private agentTrailBG_BA!:     GPUBindGroup;  // read B, write A
  private diffuseTrailBG_AB!:   GPUBindGroup;
  private diffuseTrailBG_BA!:   GPUBindGroup;

  // Uniform bind groups (same buffer, shared across directions)
  private agentUniformBG!:   GPUBindGroup;
  private agentBufBG!:       GPUBindGroup;
  private diffuseUniformBG!: any /*GPUBindGroup*/;

  // ── Internal flag ─────────────────────────────────────────────────────────
  private _initialized = false;

  // ─────────────────────────────────────────────────────────────────────────
  // Private constructor — use static create()
  // ─────────────────────────────────────────────────────────────────────────

  private constructor(
    device:     GPUDevice,
    width:      number,
    height:     number,
    agentCount: number,
    params?:    Partial<PhysarumParams>,
  ) {
    this.device     = device;
    this.width      = width;
    this.height     = height;
    this.agentCount = agentCount;
    this.params     = { ...DEFAULT_PARAMS, ...params };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Static factory
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Allocate GPU resources and seed agents, then return a ready-to-tick sim.
   *
   * @param device      — WebGPU logical device
   * @param width       — trail texture width  (pixels, power-of-2 recommended)
   * @param height      — trail texture height (pixels)
   * @param agentCount  — number of Physarum agents (default 1_000_000)
   * @param params      — optional overrides for simulation tunables
   */
  static async create(
    device:     GPUDevice,
    width       = 1024,
    height      = 1024,
    agentCount  = 1_000_000,
    params?:    Partial<PhysarumParams>,
  ): Promise<PhysarumSimulation> {
    const sim = new PhysarumSimulation(device, width, height, agentCount, params);
    await sim._init();
    return sim;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Initialisation
  // ─────────────────────────────────────────────────────────────────────────

  private async _init(): Promise<void> {
    this._createBuffers();
    this._createTextures();
    this._createUniformBuffers();
    await this._createPipelines();
    this._createBindGroups();
    this._seedAgents();
    this._initialized = true;
  }

  // ── Allocate agent SoA buffers ────────────────────────────────────────────

  private _createBuffers(): void {
    const n    = this.agentCount;
    const size = n * Float32Array.BYTES_PER_ELEMENT;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

    this.agentXBuf   = this.device.createBuffer({ label: "physarum-agentX",   size, usage });
    this.agentYBuf   = this.device.createBuffer({ label: "physarum-agentY",   size, usage });
    this.agentAngBuf = this.device.createBuffer({ label: "physarum-agentAng", size, usage });
  }

  // ── Ping-pong trail textures ──────────────────────────────────────────────

  private _createTextures(): void {
    const desc: GPUTextureDescriptor = {
      size:   { width: this.width, height: this.height },
      format: TRAIL_FORMAT,
      usage:
        GPUTextureUsage.TEXTURE_BINDING |  // readable as sampled texture
        GPUTextureUsage.STORAGE_BINDING |  // writable by compute
        GPUTextureUsage.COPY_DST,          // initial zero clear
    };

    this.trailA     = this.device.createTexture({ ...desc, label: "physarum-trailA" });
    this.trailB     = this.device.createTexture({ ...desc, label: "physarum-trailB" });
    this.trailViewA = this.trailA.createView({ label: "physarum-viewA" });
    this.trailViewB = this.trailB.createView({ label: "physarum-viewB" });
  }

  // ── Uniform buffers ───────────────────────────────────────────────────────

  private _createUniformBuffers(): void {
    // PhysarumUniforms: 8 × f32 = 32 bytes (aligned to 16)
    this.agentUniformBuf = this.device.createBuffer({
      label: "physarum-agent-uniform",
      size:  32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // DiffuseUniforms: 1 × f32 + 3 pad = 16 bytes
    this.diffuseUniformBuf = this.device.createBuffer({
      label: "physarum-diffuse-uniform",
      size:  16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Upload initial values immediately
    this._uploadAgentUniforms();
    this._uploadDiffuseUniforms();
  }

  // ── Compile compute pipelines ─────────────────────────────────────────────

  private async _createPipelines(): Promise<void> {
    const { device, width, height } = this;

    // ── Bind group layouts ────────────────────────────────────────────────────

    this.agentUniformBGL = device.createBindGroupLayout({
      label:   "physarum-agent-uniform-bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });

    this.agentBufBGL = device.createBindGroupLayout({
      label:   "physarum-agent-buf-bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });

    this.agentTrailBGL = device.createBindGroupLayout({
      label:   "physarum-agent-trail-bgl",
      entries: [
        // binding 0: read trail (sampled texture)
        { binding: 0, visibility: GPUShaderStage.COMPUTE,
          texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
        // binding 1: write trail (storage texture)
        { binding: 1, visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: "write-only", format: TRAIL_FORMAT, viewDimension: "2d" } },
      ],
    });

    this.diffuseUniformBGL = device.createBindGroupLayout({
      label:   "physarum-diffuse-uniform-bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });

    this.diffuseTrailBGL = device.createBindGroupLayout({
      label:   "physarum-diffuse-trail-bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE,
          texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: "write-only", format: TRAIL_FORMAT, viewDimension: "2d" } },
      ],
    });

    // ── Shader modules ────────────────────────────────────────────────────────

    const agentModule = device.createShaderModule({
      label: "physarum-agent-shader",
      code:  buildAgentStepShader(width, height),
    });

    const diffuseModule = device.createShaderModule({
      label: "physarum-diffuse-shader",
      code:  buildDiffuseDecayShader(width, height),
    });

    // ── Pipeline layouts ──────────────────────────────────────────────────────

    const agentLayout = device.createPipelineLayout({
      bindGroupLayouts: [
        this.agentUniformBGL,
        this.agentBufBGL,
        this.agentTrailBGL,
      ],
    });

    const diffuseLayout = device.createPipelineLayout({
      bindGroupLayouts: [
        this.diffuseUniformBGL,
        this.diffuseTrailBGL,
      ],
    });

    // ── Compile pipelines (async for better driver scheduling) ────────────────

    const [agentPipe, initPipe, diffusePipe] = await Promise.all([
      device.createComputePipelineAsync({
        label:   "physarum-agent-pipeline",
        layout:  agentLayout,
        compute: { module: agentModule, entryPoint: "agent_step" },
      }),
      device.createComputePipelineAsync({
        label:   "physarum-init-pipeline",
        layout:  agentLayout,
        compute: { module: agentModule, entryPoint: "init_agents" },
      }),
      device.createComputePipelineAsync({
        label:   "physarum-diffuse-pipeline",
        layout:  diffuseLayout,
        compute: { module: diffuseModule, entryPoint: "diffuse_decay" },
      }),
    ]);

    this.agentPipeline   = agentPipe;
    this.initPipeline    = initPipe;
    this.diffusePipeline = diffusePipe;
  }

  // ── Bind groups ───────────────────────────────────────────────────────────

  private _createBindGroups(): void {
    const dev = this.device;

    // ── Uniform & buffer bind groups (single, shared) ─────────────────────
    this.agentUniformBG = dev.createBindGroup({
      label:   "physarum-agent-uniform-bg",
      layout:  this.agentUniformBGL,
      entries: [{ binding: 0, resource: { buffer: this.agentUniformBuf } }],
    });

    this.agentBufBG = dev.createBindGroup({
      label:   "physarum-agent-buf-bg",
      layout:  this.agentBufBGL,
      entries: [
        { binding: 0, resource: { buffer: this.agentXBuf   } },
        { binding: 1, resource: { buffer: this.agentYBuf   } },
        { binding: 2, resource: { buffer: this.agentAngBuf } },
      ],
    });

    this.diffuseUniformBG = dev.createBindGroup({
      label:   "physarum-diffuse-uniform-bg",
      layout:  this.diffuseUniformBGL,
      entries: [{ binding: 0, resource: { buffer: this.diffuseUniformBuf } }],
    });

    // ── Ping-pong trail bind groups ───────────────────────────────────────

    // Agent pass: read A, write B
    this.agentTrailBG_AB = dev.createBindGroup({
      label:   "physarum-agent-trail-AB",
      layout:  this.agentTrailBGL,
      entries: [
        { binding: 0, resource: this.trailViewA },
        { binding: 1, resource: this.trailViewB },
      ],
    });

    // Agent pass: read B, write A
    this.agentTrailBG_BA = dev.createBindGroup({
      label:   "physarum-agent-trail-BA",
      layout:  this.agentTrailBGL,
      entries: [
        { binding: 0, resource: this.trailViewB },
        { binding: 1, resource: this.trailViewA },
      ],
    });

    // Diffuse pass: read A, write B
    this.diffuseTrailBG_AB = dev.createBindGroup({
      label:   "physarum-diffuse-trail-AB",
      layout:  this.diffuseTrailBGL,
      entries: [
        { binding: 0, resource: this.trailViewA },
        { binding: 1, resource: this.trailViewB },
      ],
    });

    // Diffuse pass: read B, write A
    this.diffuseTrailBG_BA = dev.createBindGroup({
      label:   "physarum-diffuse-trail-BA",
      layout:  this.diffuseTrailBGL,
      entries: [
        { binding: 0, resource: this.trailViewB },
        { binding: 1, resource: this.trailViewA },
      ],
    });
  }

  // ─────────────────────────────────────────────────────────────────────────
  // GPU-side agent seeding
  // ─────────────────────────────────────────────────────────────────────────

  private _seedAgents(): void {
    const enc = this.device.createCommandEncoder({ label: "physarum-seed" });

    const pass = enc.beginComputePass({ label: "physarum-init-pass" });
    pass.setPipeline(this.initPipeline);
    pass.setBindGroup(0, this.agentUniformBG);
    pass.setBindGroup(1, this.agentBufBG);
    // Seed writes only to agent buffers; trail read is trailA (all-zero after createTexture).
    // We use AB direction so write trail goes to trailB — doesn't matter since we only
    // care about agent positions here.
    pass.setBindGroup(2, this.agentTrailBG_AB);
    pass.dispatchWorkgroups(Math.ceil(this.agentCount / AGENT_WG));
    pass.end();

    this.device.queue.submit([enc.finish()]);
    // pingPong stays at 0 — trailA is the canonical read texture.
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Uniform helpers
  // ─────────────────────────────────────────────────────────────────────────

  private _uploadAgentUniforms(): void {
    const p = this.params;
    // PhysarumUniforms layout (8 × f32 = 32 bytes):
    //   0: sensorDist, 1: sensorAngle, 2: turnSpeed, 3: stepSize,
    //   4: depositAmt, 5: decayRate,   6: time,       7: agentCount (as f32 placeholder)
    // agentCount is u32, occupying offset 28; we store it via writeBuffer separately.
    const data = new Float32Array([
      p.sensorDistance,
      p.sensorAngle,
      p.turnSpeed,
      p.stepSize,
      p.depositAmount,
      p.decayRate,
      p.time,
      0.0,  // placeholder — agentCount u32 written below
    ]);
    this.device.queue.writeBuffer(this.agentUniformBuf, 0, data.buffer);
    // Write agentCount as u32 at byte offset 28
    this.device.queue.writeBuffer(
      this.agentUniformBuf, 28,
      new Uint32Array([this.agentCount]).buffer,
    );
  }

  private _uploadDiffuseUniforms(): void {
    const data = new Float32Array([this.params.decayRate, 0, 0, 0]);
    this.device.queue.writeBuffer(this.diffuseUniformBuf, 0, data.buffer);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public API
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Update simulation parameters.  Changes take effect on the next `tick()`.
   */
  updateParams(overrides: Partial<PhysarumParams>): void {
    this.params = { ...this.params, ...overrides };
    this._uploadAgentUniforms();
    this._uploadDiffuseUniforms();
  }

  /**
   * Return the GPU texture that contains the current trail map.
   * Bind this as `texture_2d<f32>` in your render pipeline.
   * The texture format is `r32float`; use the R channel as pheromone intensity.
   *
   * The returned handle is stable across frames — ping-pong is internal.
   * Always call this *after* `tick()` to get the most recently written result.
   */
  getTrailTexture(): GPUTexture {
    // After each tick the write target becomes the new read; return it.
    return this.pingPong === 0 ? this.trailA : this.trailB;
  }

  /**
   * Record one full simulation step into the provided GPUCommandEncoder.
   *
   * The encoder is NOT submitted here — the caller is responsible for
   * submitting it (or including it in a larger frame command buffer).
   *
   * Execution order (both passes recorded sequentially):
   *   1. agent_step  — sense → rotate → move → deposit   (reads trailRead, writes trailWrite)
   *   2. diffuse_decay — 3×3 box blur + evaporation      (reads trailWrite, outputs to trailRead)
   *
   * After this returns, `getTrailTexture()` points to the updated trail.
   */
  tick(encoder: any /*GPUCommandEncoder*/): void {
    if (!this._initialized) {
      console.warn("PhysarumSimulation.tick() called before init completed");
      return;
    }

    // Current ping-pong state:
    //   pingPong=0: trailA = read, trailB = write
    //   pingPong=1: trailB = read, trailA = write
    const agentTrailBG   = this.pingPong === 0 ? this.agentTrailBG_AB   : this.agentTrailBG_BA;
    const diffuseTrailBG = this.pingPong === 0 ? this.diffuseTrailBG_AB : this.diffuseTrailBG_BA;

    // ── Pass 1: Agent step (sense→rotate→move→deposit) ─────────────────────
    {
      const pass = encoder.beginComputePass({ label: "physarum-agent-pass" });
      pass.setPipeline(this.agentPipeline);
      pass.setBindGroup(0, this.agentUniformBG);
      pass.setBindGroup(1, this.agentBufBG);
      pass.setBindGroup(2, agentTrailBG);
      pass.dispatchWorkgroups(Math.ceil(this.agentCount / AGENT_WG));
      pass.end();
    }

    // ── Pass 2: Diffuse & decay (box blur + evaporation) ───────────────────
    // After the agent pass, trailWrite contains deposits.
    // Diffuse reads trailWrite and outputs to trailRead — completing the swap.
    {
      const pass = encoder.beginComputePass({ label: "physarum-diffuse-pass" });
      pass.setPipeline(this.diffusePipeline);
      pass.setBindGroup(0, this.diffuseUniformBG);
      // Diffuse reads the just-written trail, outputs to the other buffer.
      // Note: diffuse read = agent write, diffuse write = agent read.
      // For AB: agent wrote B → diffuse reads B, writes A.
      // For BA: agent wrote A → diffuse reads A, writes B.
      // We use the *swapped* direction for diffuse relative to agent:
      const diffuseSwapped = this.pingPong === 0 ? this.diffuseTrailBG_BA : this.diffuseTrailBG_AB;
      pass.setBindGroup(1, diffuseSwapped);
      pass.dispatchWorkgroups(
        Math.ceil(this.width  / DIFFUSE_WG),
        Math.ceil(this.height / DIFFUSE_WG),
      );
      pass.end();
    }

    // Flip ping-pong: after diffuse the read buffer is back to the original side.
    // Net result: after both passes, the same texture is authoritative as before
    // (diffuse completes the round-trip).  We do NOT flip pingPong here so that
    // getTrailTexture() consistently returns the settled read texture.
    // (Agent writes → diffuse reads from write, writes back to read → no net swap.)
  }

  /**
   * Convenience wrapper: create a one-shot command encoder, record a tick, and
   * submit immediately.  Useful for single-sim setups that don't batch frames.
   */
  tickAndSubmit(): void {
    const enc = this.device.createCommandEncoder({ label: "physarum-tick" });
    this.tick(enc);
    this.device.queue.submit([enc.finish()]);
  }

  /**
   * Free all GPU resources.  The instance must not be used after this call.
   */
  destroy(): void {
    this.agentXBuf.destroy();
    this.agentYBuf.destroy();
    this.agentAngBuf.destroy();
    this.trailA.destroy();
    this.trailB.destroy();
    this.agentUniformBuf.destroy();
    this.diffuseUniformBuf.destroy();
  }
}
