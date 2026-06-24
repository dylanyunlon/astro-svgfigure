// boids-compute.ts — WebGPU Boids (separation + alignment + cohesion)
//
// Three-pass compute pipeline that advances a 2-D boid flock entirely on the
// GPU.  The host uploads initial positions/velocities once; every `tick()` call
// encodes a command buffer and resolves results back to CPU-side Float32Arrays.
//
// Pass layout
// ───────────
//   Pass 0 – boids_influence  : for each boid i scan all neighbours and
//                               accumulate separation / alignment / cohesion
//                               steering vectors into an intermediate
//                               steeringBuf (3 × vec2 per boid).
//   Pass 1 – boids_integrate  : apply the three weighted forces, clamp speed
//                               and advance position with wrap-around boundaries.
//   (Ping-pong buffers)        : posA/velA → write posB/velB; swap every frame.
//
// All WGSL lives inline as tagged template literals so Vite's import.meta.glob
// SSR path is never triggered.  Uniform struct is padded to 16-byte alignment.

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------









export const BOIDS_WORKGROUP = 256;
export const BOIDS_MAX       = 65536;  // hard upper bound (buffer allocation)

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface BoidsParams {
  /** Number of active boids (≤ BOIDS_MAX). */
  count: number;
  /** Integration timestep (seconds). Default 1/60. */
  dt?: number;
  /** Domain half-width in world units. Default 1.0. */
  domainW?: number;
  /** Domain half-height in world units. Default 1.0. */
  domainH?: number;
  /** Visual range radius (perception radius). Default 0.15. */
  perceptionRadius?: number;
  /** Separation weight. Default 1.5. */
  separationWeight?: number;
  /** Alignment weight. Default 1.0. */
  alignmentWeight?: number;
  /** Cohesion weight. Default 1.0. */
  cohesionWeight?: number;
  /** Max speed (world units / second). Default 0.5. */
  maxSpeed?: number;
  /** Min separation distance (hard-core). Default 0.03. */
  separationRadius?: number;
}

export interface BoidsSnapshot {
  /** Interleaved [x0,y0, x1,y1, …] world positions for `count` boids. */
  positions: Float32Array;
  /** Interleaved [vx0,vy0, vx1,vy1, …] velocities for `count` boids. */
  velocities: Float32Array;
  count: number;
}

// ---------------------------------------------------------------------------
// WGSL — shared uniform struct (keep in sync with writeUniforms)
// ---------------------------------------------------------------------------

const UNIFORMS_WGSL = /* wgsl */`
struct BoidsUniforms {
  count            : u32,
  _pad0            : u32,
  _pad1            : u32,
  _pad2            : u32,
  dt               : f32,
  domainW          : f32,
  domainH          : f32,
  perceptionRadius : f32,
  separationWeight : f32,
  alignmentWeight  : f32,
  cohesionWeight   : f32,
  maxSpeed         : f32,
  separationRadius : f32,
  _pad3            : f32,
  _pad4            : f32,
  _pad5            : f32,
}
`;

// Byte offsets for writeUniforms (must mirror the struct above, std140 / 16-byte aligned)
const U_COUNT              = 0;   // u32  @ byte  0
// _pad0-2 fill bytes 4-15
const U_DT               = 16;   // f32  @ byte 16
const U_DOMAIN_W         = 20;
const U_DOMAIN_H         = 24;
const U_PERCEPTION_R     = 28;
const U_SEP_W            = 32;
const U_ALIGN_W          = 36;
const U_COH_W            = 40;
const U_MAX_SPEED        = 44;
const U_SEP_R            = 48;
// _pad3-5 fill bytes 52-63
const UNIFORMS_BYTE_SIZE = 64;   // 4 vec4-aligned rows

// ---------------------------------------------------------------------------
// WGSL — Pass 0: influence accumulation
// ---------------------------------------------------------------------------

const INFLUENCE_SHADER = /* wgsl */`
${UNIFORMS_WGSL}

@group(0) @binding(0) var<uniform> u : BoidsUniforms;

// Read-only source (ping) buffers
@group(1) @binding(0) var<storage, read> posX : array<f32>;
@group(1) @binding(1) var<storage, read> posY : array<f32>;
@group(1) @binding(2) var<storage, read> velX : array<f32>;
@group(1) @binding(3) var<storage, read> velY : array<f32>;

// Write-only steering: 6 floats per boid — [sepX, sepY, avgVx, avgVy, cxSum, cySum]
// packed as array<f32> with stride 6
@group(2) @binding(0) var<storage, read_write> steering : array<f32>;

@compute @workgroup_size(${BOIDS_WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.count) { return; }

  let xi = posX[i];
  let yi = posY[i];

  // Steering accumulators
  var sepX = 0.0;   var sepY = 0.0;
  var avgVx = 0.0;  var avgVy = 0.0;
  var cxSum = 0.0;  var cySum = 0.0;
  var neighbourCount = 0u;
  var sepCount       = 0u;

  let rP  = u.perceptionRadius;
  let rP2 = rP * rP;
  let rS  = u.separationRadius;
  let rS2 = rS * rS;

  for (var j = 0u; j < u.count; j++) {
    if (j == i) { continue; }

    let dx = posX[j] - xi;
    let dy = posY[j] - yi;
    let d2 = dx * dx + dy * dy;

    if (d2 < rP2) {
      // — cohesion + alignment (all neighbours within perception radius) —
      cxSum += posX[j];
      cySum += posY[j];
      avgVx += velX[j];
      avgVy += velY[j];
      neighbourCount++;

      // — separation (hard-core repulsion for very close neighbours) —
      if (d2 < rS2 && d2 > 0.0) {
        let invD = 1.0 / sqrt(d2);
        // Steer away: force proportional to how much closer than rS
        let strength = (rS - sqrt(d2)) * invD;
        sepX -= dx * strength;
        sepY -= dy * strength;
        sepCount++;
      }
    }
  }

  let base = i * 6u;

  // Normalise separation by count so it doesn't blow up in dense flocks
  if (sepCount > 0u) {
    let invS = 1.0 / f32(sepCount);
    steering[base + 0u] = sepX * invS;
    steering[base + 1u] = sepY * invS;
  } else {
    steering[base + 0u] = 0.0;
    steering[base + 1u] = 0.0;
  }

  if (neighbourCount > 0u) {
    let invN = 1.0 / f32(neighbourCount);
    // alignment: target = avg neighbour velocity
    steering[base + 2u] = avgVx * invN;
    steering[base + 3u] = avgVy * invN;
    // cohesion: target = avg neighbour position (centre of mass)
    steering[base + 4u] = cxSum * invN;
    steering[base + 5u] = cySum * invN;
  } else {
    steering[base + 2u] = velX[i];
    steering[base + 3u] = velY[i];
    steering[base + 4u] = xi;
    steering[base + 5u] = yi;
  }
}
`;

// ---------------------------------------------------------------------------
// WGSL — Pass 1: velocity integration + wrap boundaries
// ---------------------------------------------------------------------------

const INTEGRATE_SHADER = /* wgsl */`
${UNIFORMS_WGSL}

@group(0) @binding(0) var<uniform> u : BoidsUniforms;

// Read-only source (ping) positions + velocities
@group(1) @binding(0) var<storage, read> posXIn  : array<f32>;
@group(1) @binding(1) var<storage, read> posYIn  : array<f32>;
@group(1) @binding(2) var<storage, read> velXIn  : array<f32>;
@group(1) @binding(3) var<storage, read> velYIn  : array<f32>;

// Steering written by the influence pass
@group(2) @binding(0) var<storage, read> steering : array<f32>;

// Write-only destination (pong) buffers
@group(3) @binding(0) var<storage, read_write> posXOut : array<f32>;
@group(3) @binding(1) var<storage, read_write> posYOut : array<f32>;
@group(3) @binding(2) var<storage, read_write> velXOut : array<f32>;
@group(3) @binding(3) var<storage, read_write> velYOut : array<f32>;

fn clampSpeed(vx: f32, vy: f32, maxSpd: f32) -> vec2f {
  let s2  = vx * vx + vy * vy;
  if (s2 > maxSpd * maxSpd && s2 > 0.0) {
    let inv = maxSpd / sqrt(s2);
    return vec2f(vx * inv, vy * inv);
  }
  return vec2f(vx, vy);
}

fn wrapDomain(v: f32, half: f32) -> f32 {
  // Wrap [-half, +half] toroidal domain
  var w = v;
  if (w >  half) { w -= 2.0 * half; }
  if (w < -half) { w += 2.0 * half; }
  return w;
}

@compute @workgroup_size(${BOIDS_WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.count) { return; }

  let xi  = posXIn[i];
  let yi  = posYIn[i];
  let vxi = velXIn[i];
  let vyi = velYIn[i];

  let base = i * 6u;
  let sepX = steering[base + 0u];
  let sepY = steering[base + 1u];
  let tgtVx = steering[base + 2u];  // alignment target velocity
  let tgtVy = steering[base + 3u];
  let cxAvg = steering[base + 4u];  // cohesion target position
  let cyAvg = steering[base + 5u];

  // — Separation steer —
  var steerSepX = sepX;
  var steerSepY = sepY;

  // — Alignment steer: steer towards avg neighbour velocity —
  var steerAlignX = tgtVx - vxi;
  var steerAlignY = tgtVy - vyi;

  // — Cohesion steer: steer towards centre of local flock —
  var steerCohX = cxAvg - xi;
  var steerCohY = cyAvg - yi;

  // Normalise each steering component to unit length before weighting
  // so the weights have consistent units (acceleration / dt).
  let lenSep   = sqrt(steerSepX * steerSepX   + steerSepY * steerSepY);
  let lenAlign = sqrt(steerAlignX * steerAlignX + steerAlignY * steerAlignY);
  let lenCoh   = sqrt(steerCohX * steerCohX   + steerCohY * steerCohY);

  if (lenSep   > 0.0) { steerSepX   /= lenSep;   steerSepY   /= lenSep;   }
  if (lenAlign > 0.0) { steerAlignX /= lenAlign;  steerAlignY /= lenAlign; }
  if (lenCoh   > 0.0) { steerCohX   /= lenCoh;    steerCohY   /= lenCoh;   }

  // Weighted sum of steering components → acceleration
  let ax = steerSepX   * u.separationWeight
         + steerAlignX * u.alignmentWeight
         + steerCohX   * u.cohesionWeight;

  let ay = steerSepY   * u.separationWeight
         + steerAlignY * u.alignmentWeight
         + steerCohY   * u.cohesionWeight;

  // Euler integration
  var nvx = vxi + ax * u.dt;
  var nvy = vyi + ay * u.dt;

  // Clamp speed
  let clamped = clampSpeed(nvx, nvy, u.maxSpeed);
  nvx = clamped.x;
  nvy = clamped.y;

  var npx = xi + nvx * u.dt;
  var npy = yi + nvy * u.dt;

  // Toroidal wrap
  npx = wrapDomain(npx, u.domainW);
  npy = wrapDomain(npy, u.domainH);

  posXOut[i] = npx;
  posYOut[i] = npy;
  velXOut[i] = nvx;
  velYOut[i] = nvy;
}
`;

// ---------------------------------------------------------------------------
// BoidsCompute — orchestrator class
// ---------------------------------------------------------------------------

export class BoidsCompute {
  private readonly device: any /*GPUDevice*/;

  // Double-buffered particle state (ping-pong)
  private posXBuf : [GPUBuffer, GPUBuffer];
  private posYBuf : [GPUBuffer, GPUBuffer];
  private velXBuf : [GPUBuffer, GPUBuffer];
  private velYBuf : [GPUBuffer, GPUBuffer];

  // Per-boid steering accumulator: 6 f32 per boid
  private steeringBuf!: any /*GPUBuffer*/;

  // Read-back staging buffer
  private stagingBuf!: any /*GPUBuffer*/;

  // Uniform
  private uniformBuf!: any /*GPUBuffer*/;

  // Pipelines
  private influencePipeline!: any /*GPUComputePipeline*/;
  private integratePipeline!: any /*GPUComputePipeline*/;

  // Bind-group layouts
  private uniformBGL!    : GPUBindGroupLayout;
  private particleSrcBGL!: GPUBindGroupLayout;
  private steeringBGL!   : GPUBindGroupLayout;
  private particleDstBGL!: GPUBindGroupLayout;

  // Cached bind groups (rebuilt on demand / on swap)
  private uniformBG!    : any /*GPUBindGroup*/;
  private srcBG : [GPUBindGroup | null, GPUBindGroup | null] = [null, null];
  private dstBG : [GPUBindGroup | null, GPUBindGroup | null] = [null, null];
  private steeringReadBG  : any /*GPUBindGroup*/ | null = null;
  private steeringWriteBG : any /*GPUBindGroup*/ | null = null;

  // Ping-pong index: 0 → A is src, 1 → B is src
  private ping = 0;

  private readonly params: Required<BoidsParams>;
  private destroyed = false;

  // ── constructor ─────────────────────────────────────────────────────────────

  constructor(device: any /*GPUDevice*/, params: BoidsParams) {
    this.device = device;

    // Fill defaults
    this.params = {
      count            : params.count,
      dt               : params.dt               ?? 1 / 60,
      domainW          : params.domainW          ?? 1.0,
      domainH          : params.domainH          ?? 1.0,
      perceptionRadius : params.perceptionRadius ?? 0.15,
      separationWeight : params.separationWeight ?? 1.5,
      alignmentWeight  : params.alignmentWeight  ?? 1.0,
      cohesionWeight   : params.cohesionWeight   ?? 1.0,
      maxSpeed         : params.maxSpeed         ?? 0.5,
      separationRadius : params.separationRadius ?? 0.03,
    };

    if (this.params.count < 1 || this.params.count > BOIDS_MAX) {
      throw new RangeError(`BoidsCompute: count must be in [1, ${BOIDS_MAX}]; got ${this.params.count}`);
    }

    const n = this.params.count;
    const bytesF32 = n * 4;

    // ── Allocate particle buffers (double-buffered) ────────────────────────

    const mkParticle = (label: string) => device.createBuffer({
      label,
      size : bytesF32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    this.posXBuf = [mkParticle('boids:posX:A'), mkParticle('boids:posX:B')];
    this.posYBuf = [mkParticle('boids:posY:A'), mkParticle('boids:posY:B')];
    this.velXBuf = [mkParticle('boids:velX:A'), mkParticle('boids:velX:B')];
    this.velYBuf = [mkParticle('boids:velY:A'), mkParticle('boids:velY:B')];

    // ── Steering buffer: 6 × f32 per boid ─────────────────────────────────

    this.steeringBuf = device.createBuffer({
      label: 'boids:steering',
      size : n * 6 * 4,
      usage: GPUBufferUsage.STORAGE,
    });

    // ── Staging buffer for readback ────────────────────────────────────────

    this.stagingBuf = device.createBuffer({
      label: 'boids:staging',
      size : bytesF32 * 4,  // enough for posX + posY + velX + velY
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // ── Uniform buffer ─────────────────────────────────────────────────────

    this.uniformBuf = device.createBuffer({
      label: 'boids:uniforms',
      size : UNIFORMS_BYTE_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.writeUniforms();
    this.buildPipelines();
    this.buildBindGroups();
  }

  // ── Static factory ─────────────────────────────────────────────────────────

  static async create(params: BoidsParams): Promise<BoidsCompute> {
    if (!navigator.gpu) throw new Error('WebGPU not available');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter found');
    const device = await adapter.requestDevice();
    return new BoidsCompute(device, params);
  }

  // ── Upload initial state ───────────────────────────────────────────────────

  upload(data: { px: Float32Array; py: Float32Array; vx: Float32Array; vy: Float32Array }): void {
    const n = this.params.count;
    if (data.px.length < n || data.py.length < n || data.vx.length < n || data.vy.length < n) {
      throw new RangeError('BoidsCompute.upload: input arrays are shorter than count');
    }
    const p = this.ping;
    this.device.queue.writeBuffer(this.posXBuf[p], 0, data.px, 0, n);
    this.device.queue.writeBuffer(this.posYBuf[p], 0, data.py, 0, n);
    this.device.queue.writeBuffer(this.velXBuf[p], 0, data.vx, 0, n);
    this.device.queue.writeBuffer(this.velYBuf[p], 0, data.vy, 0, n);
  }

  // ── Randomise initial state (convenience) ─────────────────────────────────

  randomise(seed = 42): void {
    const n = this.params.count;
    const px = new Float32Array(n);
    const py = new Float32Array(n);
    const vx = new Float32Array(n);
    const vy = new Float32Array(n);

    // Deterministic LCG so results are reproducible
    let s = seed >>> 0;
    const rand = () => {
      s = (Math.imul(1664525, s) + 1013904223) >>> 0;
      return (s >>> 0) / 0xFFFFFFFF;
    };

    const hw = this.params.domainW;
    const hh = this.params.domainH;
    const ms = this.params.maxSpeed;

    for (let i = 0; i < n; i++) {
      px[i] = (rand() * 2 - 1) * hw;
      py[i] = (rand() * 2 - 1) * hh;
      const angle = rand() * Math.PI * 2;
      const speed = rand() * ms;
      vx[i] = Math.cos(angle) * speed;
      vy[i] = Math.sin(angle) * speed;
    }

    this.upload({ px, py, vx, vy });
  }

  // ── Advance simulation by one timestep ────────────────────────────────────

  tick(dtOverride?: number): void {
    if (this.destroyed) throw new Error('BoidsCompute has been destroyed');

    if (dtOverride !== undefined && dtOverride !== this.params.dt) {
      this.params.dt = dtOverride;
      this.writeUniforms();
    }

    const n    = this.params.count;
    const wg   = BOIDS_WORKGROUP;
    const disp = Math.ceil(n / wg);

    const p = this.ping;
    const q = 1 - p;

    const cmd = this.device.createCommandEncoder({ label: 'boids:tick' });

    // Pass 0 — influence (reads ping, writes steering)
    {
      const pass = cmd.beginComputePass({ label: 'boids:influence' });
      pass.setPipeline(this.influencePipeline);
      pass.setBindGroup(0, this.uniformBG);
      pass.setBindGroup(1, this.srcBG[p]!);
      pass.setBindGroup(2, this.steeringWriteBG!);
      pass.dispatchWorkgroups(disp);
      pass.end();
    }

    // Pass 1 — integrate (reads ping + steering, writes pong)
    {
      const pass = cmd.beginComputePass({ label: 'boids:integrate' });
      pass.setPipeline(this.integratePipeline);
      pass.setBindGroup(0, this.uniformBG);
      pass.setBindGroup(1, this.srcBG[p]!);
      pass.setBindGroup(2, this.steeringReadBG!);
      pass.setBindGroup(3, this.dstBG[q]!);
      pass.dispatchWorkgroups(disp);
      pass.end();
    }

    this.device.queue.submit([cmd.finish()]);

    // Swap ping-pong
    this.ping = q;
  }

  // ── Read-back positions + velocities to CPU ────────────────────────────────

  async readback(): Promise<BoidsSnapshot> {
    if (this.destroyed) throw new Error('BoidsCompute has been destroyed');

    const n       = this.params.count;
    const bytes   = n * 4;
    const p       = this.ping;   // current source after last tick

    const cmd = this.device.createCommandEncoder({ label: 'boids:readback' });
    cmd.copyBufferToBuffer(this.posXBuf[p], 0, this.stagingBuf, 0,           bytes);
    cmd.copyBufferToBuffer(this.posYBuf[p], 0, this.stagingBuf, bytes,       bytes);
    cmd.copyBufferToBuffer(this.velXBuf[p], 0, this.stagingBuf, bytes * 2,   bytes);
    cmd.copyBufferToBuffer(this.velYBuf[p], 0, this.stagingBuf, bytes * 3,   bytes);
    this.device.queue.submit([cmd.finish()]);

    await this.stagingBuf.mapAsync(GPUMapMode.READ, 0, bytes * 4);
    const mapped  = this.stagingBuf.getMappedRange(0, bytes * 4);
    const copy    = new Float32Array(mapped.byteLength / 4);
    copy.set(new Float32Array(mapped));
    this.stagingBuf.unmap();

    const positions  = new Float32Array(n * 2);
    const velocities = new Float32Array(n * 2);
    const pxArr = copy.subarray(0, n);
    const pyArr = copy.subarray(n, n * 2);
    const vxArr = copy.subarray(n * 2, n * 3);
    const vyArr = copy.subarray(n * 3, n * 4);

    for (let i = 0; i < n; i++) {
      positions[i * 2]     = pxArr[i];
      positions[i * 2 + 1] = pyArr[i];
      velocities[i * 2]    = vxArr[i];
      velocities[i * 2 + 1] = vyArr[i];
    }

    return { positions, velocities, count: n };
  }

  // ── Update params at runtime ──────────────────────────────────────────────

  setParams(patch: Partial<Omit<BoidsParams, 'count'>>): void {
    Object.assign(this.params, patch);
    this.writeUniforms();
  }

  // ── Destroy — release all GPU resources ──────────────────────────────────

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;

    this.posXBuf[0].destroy();  this.posXBuf[1].destroy();
    this.posYBuf[0].destroy();  this.posYBuf[1].destroy();
    this.velXBuf[0].destroy();  this.velXBuf[1].destroy();
    this.velYBuf[0].destroy();  this.velYBuf[1].destroy();
    this.steeringBuf.destroy();
    this.stagingBuf.destroy();
    this.uniformBuf.destroy();
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  private writeUniforms(): void {
    const buf = new ArrayBuffer(UNIFORMS_BYTE_SIZE);
    const u32 = new Uint32Array(buf);
    const f32 = new Float32Array(buf);

    u32[U_COUNT / 4]          = this.params.count;
    f32[U_DT / 4]             = this.params.dt;
    f32[U_DOMAIN_W / 4]       = this.params.domainW;
    f32[U_DOMAIN_H / 4]       = this.params.domainH;
    f32[U_PERCEPTION_R / 4]   = this.params.perceptionRadius;
    f32[U_SEP_W / 4]          = this.params.separationWeight;
    f32[U_ALIGN_W / 4]        = this.params.alignmentWeight;
    f32[U_COH_W / 4]          = this.params.cohesionWeight;
    f32[U_MAX_SPEED / 4]      = this.params.maxSpeed;
    f32[U_SEP_R / 4]          = this.params.separationRadius;

    this.device.queue.writeBuffer(this.uniformBuf, 0, buf);
  }

  private buildPipelines(): void {
    const device = this.device;

    // ── Bind-group layouts ────────────────────────────────────────────────

    this.uniformBGL = device.createBindGroupLayout({
      label  : 'boids:uniformBGL',
      entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }],
    });

    this.particleSrcBGL = device.createBindGroupLayout({
      label  : 'boids:particleSrcBGL',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      ],
    });

    this.steeringBGL = device.createBindGroupLayout({
      label  : 'boids:steeringBGL',
      entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }],
    });

    this.particleDstBGL = device.createBindGroupLayout({
      label  : 'boids:particleDstBGL',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });

    // ── Shader modules ─────────────────────────────────────────────────────

    const influenceSM = device.createShaderModule({ label: 'boids:influence:sm', code: INFLUENCE_SHADER });
    const integrateSM = device.createShaderModule({ label: 'boids:integrate:sm', code: INTEGRATE_SHADER });

    // ── Compute pipelines ──────────────────────────────────────────────────

    const influenceLayout = device.createPipelineLayout({
      label               : 'boids:influence:layout',
      bindGroupLayouts    : [this.uniformBGL, this.particleSrcBGL, this.steeringBGL],
    });

    const integrateLayout = device.createPipelineLayout({
      label               : 'boids:integrate:layout',
      bindGroupLayouts    : [this.uniformBGL, this.particleSrcBGL, this.steeringBGL, this.particleDstBGL],
    });

    this.influencePipeline = device.createComputePipeline({
      label  : 'boids:influence:pipeline',
      layout : influenceLayout,
      compute: { module: influenceSM, entryPoint: 'main' },
    });

    this.integratePipeline = device.createComputePipeline({
      label  : 'boids:integrate:pipeline',
      layout : integrateLayout,
      compute: { module: integrateSM, entryPoint: 'main' },
    });
  }

  private buildBindGroups(): void {
    const device = this.device;

    // ── Uniform ────────────────────────────────────────────────────────────

    this.uniformBG = device.createBindGroup({
      label  : 'boids:uniformBG',
      layout : this.uniformBGL,
      entries: [{ binding: 0, resource: { buffer: this.uniformBuf } }],
    });

    // ── Per-ping-pong source / destination ────────────────────────────────

    for (let p = 0; p < 2; p++) {
      this.srcBG[p] = device.createBindGroup({
        label  : `boids:srcBG:${p}`,
        layout : this.particleSrcBGL,
        entries: [
          { binding: 0, resource: { buffer: this.posXBuf[p] } },
          { binding: 1, resource: { buffer: this.posYBuf[p] } },
          { binding: 2, resource: { buffer: this.velXBuf[p] } },
          { binding: 3, resource: { buffer: this.velYBuf[p] } },
        ],
      });

      this.dstBG[p] = device.createBindGroup({
        label  : `boids:dstBG:${p}`,
        layout : this.particleDstBGL,
        entries: [
          { binding: 0, resource: { buffer: this.posXBuf[p] } },
          { binding: 1, resource: { buffer: this.posYBuf[p] } },
          { binding: 2, resource: { buffer: this.velXBuf[p] } },
          { binding: 3, resource: { buffer: this.velYBuf[p] } },
        ],
      });
    }

    // ── Steering (read vs write view — same underlying buffer) ────────────

    this.steeringWriteBG = device.createBindGroup({
      label  : 'boids:steeringWriteBG',
      layout : this.steeringBGL,
      entries: [{ binding: 0, resource: { buffer: this.steeringBuf } }],
    });

    // The integrate pass binds steering as read-only-storage —
    // but we use the same steeringBGL (storage) since WGSL `read` is
    // compatible with a `storage` BGL when no UAV writes are issued.
    this.steeringReadBG = this.steeringWriteBG;
  }
}
