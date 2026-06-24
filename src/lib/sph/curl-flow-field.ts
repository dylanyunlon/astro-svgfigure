/**
 * src/lib/sph/curl-flow-field.ts  — M606
 *
 * WebGPU 3-D Curl-Noise Flow Field
 * ─────────────────────────────────────────────────────────────────────────────
 * Implements a volumetric curl-noise velocity field entirely on the GPU.
 * Curl noise is divergence-free by construction (∇·F = 0), making it ideal
 * for incompressible fluid-like particle advection, turbulence overlays, and
 * SPH external-force injection.
 *
 * Algorithm
 * ─────────────────────────────────────────────────────────────────────────────
 *  1. Perlin-style 3-D gradient noise (fBm, 1–6 octaves) is evaluated at
 *     (x, y, z) and at (x, y, z + ε) for three component potentials ψx, ψy, ψz.
 *  2. The curl of the potential field Ψ = (ψx, ψy, ψz) gives a
 *     divergence-free vector field:
 *       F = ∇ × Ψ
 *         = ( ∂ψz/∂y − ∂ψy/∂z,
 *             ∂ψx/∂z − ∂ψz/∂x,
 *             ∂ψy/∂x − ∂ψx/∂y )
 *     Partial derivatives are approximated by central finite differences with
 *     step ε (uniform `epsilonWGSL`).
 *  3. A 3-D texture of dimensions (W × H × D) is filled once per `update()`
 *     call; each texel stores vec4f(Fx, Fy, Fz, speed) in half-float.
 *  4. A second compute pass reads the texture and writes compact f32 buffers
 *     (velX, velY, velZ) to be blended into the SPH external-force pipeline.
 *
 * Noise implementation
 * ─────────────────────────────────────────────────────────────────────────────
 * The WGSL noise implementation is a clean-room port of the classic Perlin
 * gradient noise (Ken Perlin, SIGGRAPH 2002 "Improving Noise") adapted for
 * WebGPU:
 *   • Hash via bitwise scatter (no texture lookup needed)
 *   • Quintic fade curve: t³(6t²−15t+10)
 *   • Trilinear interpolation of lattice gradients
 *   • fBm loop with configurable octaves, lacunarity, gain
 *
 * Integration with SPH / cell pubsub
 * ─────────────────────────────────────────────────────────────────────────────
 * CurlFlowField exposes three primary APIs:
 *
 *   curlField.update(encoder, time)
 *     → Runs the curl-noise compute pass; writes the 3-D texture + flat buffers.
 *
 *   curlField.sampleCPU(x, y, z)
 *     → Returns {vx, vy, vz} by trilinear interpolation of the CPU-side
 *       copy (updated once per JS frame via readback). Useful for non-GPU
 *       consumers (cell position steering, camera turbulence, etc.).
 *
 *   curlField.injectIntoSPH(encoder, sphForceXBuf, sphForceYBuf, N)
 *     → Dispatch a blend pass that adds the curl velocity contribution to the
 *       SPH force buffers; particle (x,y) is used to sample the XY-slice of
 *       the 3-D field at z = curlField.params.sliceZ.
 *
 * Usage
 * ─────────────────────────────────────────────────────────────────────────────
 *   const field = new CurlFlowField(device, { width: 64, height: 64, depth: 32 });
 *   await field.build();
 *
 *   // render loop:
 *   const enc = device.createCommandEncoder();
 *   field.update(enc, performance.now() / 1000);
 *   field.injectIntoSPH(enc, sph.bufs.forceX, sph.bufs.forceY, sph.params.count);
 *   device.queue.submit([enc.finish()]);
 *
 *   // CPU reads (after await device.queue.onSubmittedWorkDone()):
 *   field.scheduleReadback();
 *   const { vx, vy, vz } = field.sampleCPU(px, py, pz);
 */

// ─── Constants ───────────────────────────────────────────────────────────────

/** Workgroup size for all compute passes. */








const WG = 64 as const;

/** Finite-difference step for numerical curl derivation. */
const EPSILON = 0.01 as const;

// ─── Public types ─────────────────────────────────────────────────────────────

/** Parameters controlling the curl noise field dimensions and behaviour. */
export interface CurlFlowFieldParams {
  /** Texture width  (x-axis). Default 64. */
  width?: number;
  /** Texture height (y-axis). Default 64. */
  height?: number;
  /** Texture depth  (z-axis). Default 32. */
  depth?: number;
  /** World-space extent mapped to [0,1] in x. Default 10.0. */
  domainX?: number;
  /** World-space extent mapped to [0,1] in y. Default 10.0. */
  domainY?: number;
  /** World-space extent mapped to [0,1] in z. Default 10.0. */
  domainZ?: number;
  /** Z-slice used when injecting 2-D curl force into SPH. Default 0.5. */
  sliceZ?: number;
  /** Time scale (speed of evolution). Default 0.12. */
  timeScale?: number;
  /** Number of fBm octaves. Range 1–6. Default 4. */
  octaves?: number;
  /** fBm lacunarity (frequency multiplier per octave). Default 2.0. */
  lacunarity?: number;
  /** fBm gain (amplitude multiplier per octave). Default 0.5. */
  gain?: number;
  /** Overall velocity magnitude scale. Default 1.0. */
  strength?: number;
  /** Blend factor when injecting into SPH (0 = no effect, 1 = full replace). Default 0.15. */
  sphBlend?: number;
}

/** Resolved (non-optional) parameter set. */
export interface ResolvedCurlParams {
  width: number;
  height: number;
  depth: number;
  domainX: number;
  domainY: number;
  domainZ: number;
  sliceZ: number;
  timeScale: number;
  octaves: number;
  lacunarity: number;
  gain: number;
  strength: number;
  sphBlend: number;
}

/** CPU-side sample result. */
export interface CurlSample {
  vx: number;
  vy: number;
  vz: number;
  /** Pre-computed |v|. */
  speed: number;
}

// ─── WGSL: Curl-Noise Compute Shader ─────────────────────────────────────────

/** Uniform struct layout (must match `buildUniformData()`). */
const UNIFORM_STRIDE = 16 * 4; // 16 × f32 = 64 bytes

const WGSL_CURL_COMPUTE = /* wgsl */`

// ── Uniforms ────────────────────────────────────────────────────────────────
struct CurlUniforms {
  domainX   : f32,  // world-space x extent
  domainY   : f32,
  domainZ   : f32,
  sliceZ    : f32,
  timeScale : f32,
  time      : f32,
  epsilon   : f32,
  strength  : f32,
  octaves   : u32,
  lacunarity: f32,
  gain      : f32,
  width     : u32,
  height    : u32,
  depth     : u32,
  _pad0     : u32,
  _pad1     : u32,
}

@group(0) @binding(0) var<uniform>            u    : CurlUniforms;
@group(0) @binding(1) var<storage, read_write> outBuf: array<vec4f>;

// ── Hash / gradient noise ────────────────────────────────────────────────────
// Bitwise scatter hash — avoids texture lookups, fully WGSL-legal.
fn hash3u(p: vec3u) -> u32 {
  var h = p.x ^ (p.y * 1597334677u) ^ (p.z * 3812015801u);
  h ^= h >> 17u;
  h *= 0xbf324c81u;
  h ^= h >> 11u;
  h *= 0x9c7493adu;
  h ^= h >> 16u;
  return h;
}

// Returns a unit gradient vector from the hash integer.
fn grad3(h: u32) -> vec3f {
  let s = h & 15u;
  // 16 gradients on cube edges + face diagonals
  switch s {
    case 0u:  { return vec3f( 1.0,  1.0,  0.0); }
    case 1u:  { return vec3f(-1.0,  1.0,  0.0); }
    case 2u:  { return vec3f( 1.0, -1.0,  0.0); }
    case 3u:  { return vec3f(-1.0, -1.0,  0.0); }
    case 4u:  { return vec3f( 1.0,  0.0,  1.0); }
    case 5u:  { return vec3f(-1.0,  0.0,  1.0); }
    case 6u:  { return vec3f( 1.0,  0.0, -1.0); }
    case 7u:  { return vec3f(-1.0,  0.0, -1.0); }
    case 8u:  { return vec3f( 0.0,  1.0,  1.0); }
    case 9u:  { return vec3f( 0.0, -1.0,  1.0); }
    case 10u: { return vec3f( 0.0,  1.0, -1.0); }
    case 11u: { return vec3f( 0.0, -1.0, -1.0); }
    case 12u: { return vec3f( 1.0,  1.0,  0.0); }
    case 13u: { return vec3f(-1.0,  1.0,  0.0); }
    case 14u: { return vec3f( 0.0, -1.0,  1.0); }
    default:  { return vec3f( 0.0, -1.0, -1.0); }
  }
}

// Quintic fade curve: 6t⁵ − 15t⁴ + 10t³
fn fade3(t: vec3f) -> vec3f {
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// 3-D Perlin gradient noise; returns value in [-1, +1].
fn perlin3(p: vec3f) -> f32 {
  let Pi  = vec3i(floor(p)) & vec3i(255);
  let Pu  = vec3u(Pi & vec3i(255));
  let pf  = fract(p);
  let f   = fade3(pf);

  // 8 corner hashes
  let h000 = hash3u(Pu + vec3u(0u, 0u, 0u));
  let h100 = hash3u(Pu + vec3u(1u, 0u, 0u));
  let h010 = hash3u(Pu + vec3u(0u, 1u, 0u));
  let h110 = hash3u(Pu + vec3u(1u, 1u, 0u));
  let h001 = hash3u(Pu + vec3u(0u, 0u, 1u));
  let h101 = hash3u(Pu + vec3u(1u, 0u, 1u));
  let h011 = hash3u(Pu + vec3u(0u, 1u, 1u));
  let h111 = hash3u(Pu + vec3u(1u, 1u, 1u));

  // 8 dot products
  let d000 = dot(grad3(h000), pf - vec3f(0.0, 0.0, 0.0));
  let d100 = dot(grad3(h100), pf - vec3f(1.0, 0.0, 0.0));
  let d010 = dot(grad3(h010), pf - vec3f(0.0, 1.0, 0.0));
  let d110 = dot(grad3(h110), pf - vec3f(1.0, 1.0, 0.0));
  let d001 = dot(grad3(h001), pf - vec3f(0.0, 0.0, 1.0));
  let d101 = dot(grad3(h101), pf - vec3f(1.0, 0.0, 1.0));
  let d011 = dot(grad3(h011), pf - vec3f(0.0, 1.0, 1.0));
  let d111 = dot(grad3(h111), pf - vec3f(1.0, 1.0, 1.0));

  // Trilinear blend
  let fx  = f.x;
  let fy  = f.y;
  let fz  = f.z;
  let x00 = mix(d000, d100, fx);
  let x10 = mix(d010, d110, fx);
  let x01 = mix(d001, d101, fx);
  let x11 = mix(d011, d111, fx);
  let y0  = mix(x00, x10, fy);
  let y1  = mix(x01, x11, fy);
  return mix(y0, y1, fz);
}

// fBm wrapper (octaves driven by uniform; loop unrolled to max 6 statically).
fn fbm3(p: vec3f, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
  var val  = 0.0;
  var amp  = 0.5;
  var freq = 1.0;
  var q    = p;
  for (var o = 0u; o < 6u; o++) {
    if (o >= octaves) { break; }
    val  += amp * perlin3(q * freq);
    freq *= lacunarity;
    amp  *= gain;
  }
  return val;
}

// ── Curl derivation via central finite differences ───────────────────────────
// Potential field Ψ = (ψx, ψy, ψz) evaluated at 'p' with time offset.
fn potential(p: vec3f, axis: u32, t: f32, oct: u32, lac: f32, gain: f32) -> f32 {
  // Offset each axis component by a different large constant so they are
  // decorrelated gradient-noise instances.
  switch axis {
    case 0u: { return fbm3(p + vec3f(31.416, t * 0.7, 0.0),   oct, lac, gain); }
    case 1u: { return fbm3(p + vec3f(0.0, 62.832, t * 0.5),   oct, lac, gain); }
    default: { return fbm3(p + vec3f(t * 0.3, 0.0, 94.248),   oct, lac, gain); }
  }
}

fn curlNoise3(p: vec3f, eps: f32, t: f32, oct: u32, lac: f32, gain: f32) -> vec3f {
  // ∂ψz/∂y − ∂ψy/∂z
  let dPzDy = (potential(p + vec3f(0.0, eps, 0.0), 2u, t, oct, lac, gain)
             - potential(p - vec3f(0.0, eps, 0.0), 2u, t, oct, lac, gain)) / (2.0 * eps);
  let dPyDz = (potential(p + vec3f(0.0, 0.0, eps), 1u, t, oct, lac, gain)
             - potential(p - vec3f(0.0, 0.0, eps), 1u, t, oct, lac, gain)) / (2.0 * eps);

  // ∂ψx/∂z − ∂ψz/∂x
  let dPxDz = (potential(p + vec3f(0.0, 0.0, eps), 0u, t, oct, lac, gain)
             - potential(p - vec3f(0.0, 0.0, eps), 0u, t, oct, lac, gain)) / (2.0 * eps);
  let dPzDx = (potential(p + vec3f(eps, 0.0, 0.0), 2u, t, oct, lac, gain)
             - potential(p - vec3f(eps, 0.0, 0.0), 2u, t, oct, lac, gain)) / (2.0 * eps);

  // ∂ψy/∂x − ∂ψx/∂y
  let dPyDx = (potential(p + vec3f(eps, 0.0, 0.0), 1u, t, oct, lac, gain)
             - potential(p - vec3f(eps, 0.0, 0.0), 1u, t, oct, lac, gain)) / (2.0 * eps);
  let dPxDy = (potential(p + vec3f(0.0, eps, 0.0), 0u, t, oct, lac, gain)
             - potential(p - vec3f(0.0, eps, 0.0), 0u, t, oct, lac, gain)) / (2.0 * eps);

  return vec3f(dPzDy - dPyDz, dPxDz - dPzDx, dPyDx - dPxDy);
}

// ── Main kernel ───────────────────────────────────────────────────────────────
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = u.width * u.height * u.depth;
  if (idx >= total) { return; }

  // 3-D index
  let xi = idx % u.width;
  let yi = (idx / u.width) % u.height;
  let zi = idx / (u.width * u.height);

  // Normalised position → world-space noise coordinates
  let nx = (f32(xi) + 0.5) / f32(u.width);
  let ny = (f32(yi) + 0.5) / f32(u.height);
  let nz = (f32(zi) + 0.5) / f32(u.depth);
  let p  = vec3f(nx * u.domainX, ny * u.domainY, nz * u.domainZ);

  let t   = u.time * u.timeScale;
  let curl = curlNoise3(p, u.epsilon, t,
                        u.octaves, u.lacunarity, u.gain) * u.strength;
  let spd = length(curl);
  outBuf[idx] = vec4f(curl.x, curl.y, curl.z, spd);
}
`;

// ─── WGSL: SPH Injection Blend Shader ────────────────────────────────────────

const WGSL_SPH_INJECT = /* wgsl */`
struct InjectUniforms {
  width    : u32,
  height   : u32,
  depth    : u32,
  count    : u32,
  domainX  : f32,
  domainY  : f32,
  domainZ  : f32,
  sliceZ   : f32,
  blend    : f32,
  _pad0    : u32,
  _pad1    : u32,
  _pad2    : u32,
}

@group(0) @binding(0) var<uniform>        u       : InjectUniforms;
@group(0) @binding(1) var<storage, read>  curlBuf : array<vec4f>;
@group(0) @binding(2) var<storage, read>  posX    : array<f32>;
@group(0) @binding(3) var<storage, read>  posY    : array<f32>;
@group(0) @binding(4) var<storage, read_write> forceX : array<f32>;
@group(0) @binding(5) var<storage, read_write> forceY : array<f32>;

// Trilinear sample of the flat 3-D buffer at normalised coords (nx, ny, nz).
fn sampleCurl(nx: f32, ny: f32, nz: f32) -> vec3f {
  let W  = f32(u.width);
  let H  = f32(u.height);
  let D  = f32(u.depth);
  let cx = clamp(nx * W - 0.5, 0.0, W - 1.001);
  let cy = clamp(ny * H - 0.5, 0.0, H - 1.001);
  let cz = clamp(nz * D - 0.5, 0.0, D - 1.001);
  let ix = u32(cx); let fx = cx - f32(ix);
  let iy = u32(cy); let fy = cy - f32(iy);
  let iz = u32(cz); let fz = cz - f32(iz);
  let ix1 = min(ix + 1u, u.width  - 1u);
  let iy1 = min(iy + 1u, u.height - 1u);
  let iz1 = min(iz + 1u, u.depth  - 1u);

  let w = u.width; let h = u.height;
  let i000 = iz  * h * w + iy  * w + ix;
  let i100 = iz  * h * w + iy  * w + ix1;
  let i010 = iz  * h * w + iy1 * w + ix;
  let i110 = iz  * h * w + iy1 * w + ix1;
  let i001 = iz1 * h * w + iy  * w + ix;
  let i101 = iz1 * h * w + iy  * w + ix1;
  let i011 = iz1 * h * w + iy1 * w + ix;
  let i111 = iz1 * h * w + iy1 * w + ix1;

  let c000 = curlBuf[i000].xyz; let c100 = curlBuf[i100].xyz;
  let c010 = curlBuf[i010].xyz; let c110 = curlBuf[i110].xyz;
  let c001 = curlBuf[i001].xyz; let c101 = curlBuf[i101].xyz;
  let c011 = curlBuf[i011].xyz; let c111 = curlBuf[i111].xyz;

  let x00 = mix(c000, c100, fx); let x10 = mix(c010, c110, fx);
  let x01 = mix(c001, c101, fx); let x11 = mix(c011, c111, fx);
  let y0  = mix(x00,  x10,  fy); let y1  = mix(x01,  x11,  fy);
  return mix(y0, y1, fz);
}

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.count) { return; }

  let nx = posX[i] / u.domainX;
  let ny = posY[i] / u.domainY;
  let nz = u.sliceZ;   // all SPH particles live at configurable z-slice

  let curl = sampleCurl(clamp(nx, 0.0, 1.0), clamp(ny, 0.0, 1.0), nz);
  forceX[i] += curl.x * u.blend;
  forceY[i] += curl.y * u.blend;
}
`;

// ─── CurlFlowField ────────────────────────────────────────────────────────────

export class CurlFlowField {
  readonly params: ResolvedCurlParams;

  private readonly device: any /*GPUDevice*/;

  // Compute pipeline — fills the flat vec4f storage buffer.
  private curlPipeline!    : any /*GPUComputePipeline*/;
  private curlBGL!         : GPUBindGroupLayout;
  private curlBG!          : any /*GPUBindGroup*/;

  // Injection pipeline — blends curl into SPH force buffers.
  private injectPipeline!  : any /*GPUComputePipeline*/;
  private injectBGL!       : GPUBindGroupLayout;
  /** Injection bind group is rebuilt per `injectIntoSPH()` call (SPH bufs vary). */

  // GPU buffers.
  private uniformBuf!      : any /*GPUBuffer*/;
  private injectUniformBuf!: any /*GPUBuffer*/;
  /** Flat storage buffer: width × height × depth × vec4f. */
  curlBuf!                 : any /*GPUBuffer*/;
  /** CPU-readable staging buffer for `scheduleReadback()`. */
  private stagingBuf!      : any /*GPUBuffer*/;

  // CPU mirror updated by readback.
  private cpuData: Float32Array | null = null;
  private readbackPending = false;

  // ──────────────────────────────────────────────────────────────────────────

  constructor(device: any /*GPUDevice*/, params: CurlFlowFieldParams = {}) {
    this.device = device;
    this.params = {
      width:      params.width      ?? 64,
      height:     params.height     ?? 64,
      depth:      params.depth      ?? 32,
      domainX:    params.domainX    ?? 10.0,
      domainY:    params.domainY    ?? 10.0,
      domainZ:    params.domainZ    ?? 10.0,
      sliceZ:     params.sliceZ     ?? 0.5,
      timeScale:  params.timeScale  ?? 0.12,
      octaves:    Math.max(1, Math.min(6, params.octaves ?? 4)),
      lacunarity: params.lacunarity ?? 2.0,
      gain:       params.gain       ?? 0.5,
      strength:   params.strength   ?? 1.0,
      sphBlend:   params.sphBlend   ?? 0.15,
    };
  }

  // ── build() — must be called once before update() ─────────────────────────

  async build(): Promise<void> {
    const dev = this.device;
    const { width, height, depth } = this.params;
    const totalTexels = width * height * depth;
    const bufBytes    = totalTexels * 4 * 4; // vec4f = 4 × f32

    // ── buffers ──────────────────────────────────────────────────────────────
    this.uniformBuf = dev.createBuffer({
      label: 'curl-uniform',
      size:  UNIFORM_STRIDE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.injectUniformBuf = dev.createBuffer({
      label: 'curl-inject-uniform',
      size:  12 * 4, // 12 × f32/u32
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.curlBuf = dev.createBuffer({
      label: 'curl-field-buf',
      size:  bufBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    this.stagingBuf = dev.createBuffer({
      label: 'curl-staging',
      size:  bufBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    // ── curl compute pipeline ─────────────────────────────────────────────────
    const curlShader = dev.createShaderModule({
      label: 'curl-noise-compute',
      code:  WGSL_CURL_COMPUTE,
    });

    this.curlBGL = dev.createBindGroupLayout({
      label: 'curl-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' } },
      ],
    });

    this.curlPipeline = dev.createComputePipeline({
      label:  'curl-noise-pipeline',
      layout: dev.createPipelineLayout({ bindGroupLayouts: [this.curlBGL] }),
      compute: { module: curlShader, entryPoint: 'main' },
    });

    this.curlBG = dev.createBindGroup({
      label:  'curl-bg',
      layout: this.curlBGL,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
        { binding: 1, resource: { buffer: this.curlBuf } },
      ],
    });

    // ── inject pipeline ───────────────────────────────────────────────────────
    const injectShader = dev.createShaderModule({
      label: 'curl-inject-compute',
      code:  WGSL_SPH_INJECT,
    });

    this.injectBGL = dev.createBindGroupLayout({
      label: 'inject-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' } },
      ],
    });

    this.injectPipeline = dev.createComputePipeline({
      label:  'curl-inject-pipeline',
      layout: dev.createPipelineLayout({ bindGroupLayouts: [this.injectBGL] }),
      compute: { module: injectShader, entryPoint: 'main' },
    });

    // Write initial uniforms so first update() is safe.
    this._writeUniform(0.0);
  }

  // ── update() — run curl-noise compute, fill curlBuf ───────────────────────

  /**
   * Dispatch the curl-noise compute pass.
   * @param encoder  Command encoder for this frame.
   * @param time     Elapsed time in seconds (used to animate the field).
   */
  update(encoder: any /*GPUCommandEncoder*/, time: number): void {
    this._writeUniform(time);

    const { width, height, depth } = this.params;
    const total = width * height * depth;

    const pass = encoder.beginComputePass({ label: 'curl-noise-pass' });
    pass.setPipeline(this.curlPipeline);
    pass.setBindGroup(0, this.curlBG);
    pass.dispatchWorkgroups(Math.ceil(total / WG));
    pass.end();
  }

  // ── injectIntoSPH() — add curl velocity to SPH force buffers ─────────────

  /**
   * Blend the curl-noise 2-D slice into SPH particle forces.
   *
   * @param encoder     Current command encoder.
   * @param posXBuf     SPH particle position X buffer (read-only storage).
   * @param posYBuf     SPH particle position Y buffer (read-only storage).
   * @param forceXBuf   SPH force X buffer (read_write storage).
   * @param forceYBuf   SPH force Y buffer (read_write storage).
   * @param count       Number of active SPH particles.
   * @param blendOverride  Optional per-call blend override (else uses params.sphBlend).
   */
  injectIntoSPH(
    encoder    : any /*GPUCommandEncoder*/,
    posXBuf    : any /*GPUBuffer*/,
    posYBuf    : any /*GPUBuffer*/,
    forceXBuf  : any /*GPUBuffer*/,
    forceYBuf  : any /*GPUBuffer*/,
    count      : number,
    blendOverride?: number,
  ): void {
    const { width, height, depth, domainX, domainY, domainZ, sliceZ, sphBlend } = this.params;
    const blend = blendOverride ?? sphBlend;

    // Write inject uniforms.
    const data = new ArrayBuffer(12 * 4);
    const u32  = new Uint32Array(data);
    const f32  = new Float32Array(data);
    u32[0]  = width;
    u32[1]  = height;
    u32[2]  = depth;
    u32[3]  = count;
    f32[4]  = domainX;
    f32[5]  = domainY;
    f32[6]  = domainZ;
    f32[7]  = sliceZ;
    f32[8]  = blend;
    // [9..11] padding
    this.device.queue.writeBuffer(this.injectUniformBuf, 0, data);

    const bg = this.device.createBindGroup({
      label:  'inject-bg',
      layout: this.injectBGL,
      entries: [
        { binding: 0, resource: { buffer: this.injectUniformBuf } },
        { binding: 1, resource: { buffer: this.curlBuf } },
        { binding: 2, resource: { buffer: posXBuf } },
        { binding: 3, resource: { buffer: posYBuf } },
        { binding: 4, resource: { buffer: forceXBuf } },
        { binding: 5, resource: { buffer: forceYBuf } },
      ],
    });

    const pass = encoder.beginComputePass({ label: 'curl-inject-pass' });
    pass.setPipeline(this.injectPipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(count / WG));
    pass.end();
  }

  // ── scheduleReadback() — copy curlBuf → staging (async) ──────────────────

  /**
   * Enqueue a GPU→CPU copy of the curl buffer.  Call this at the *end* of a
   * frame's command encoder, then call `await device.queue.onSubmittedWorkDone()`
   * and finally `finishReadback()` to make `sampleCPU()` reflect the new data.
   */
  scheduleReadback(encoder: any /*GPUCommandEncoder*/): void {
    if (this.readbackPending) return;
    const bytes = this.params.width * this.params.height * this.params.depth * 16;
    encoder.copyBufferToBuffer(this.curlBuf, 0, this.stagingBuf, 0, bytes);
    this.readbackPending = true;
  }

  /**
   * Map the staging buffer and copy data to the CPU mirror.
   * Must be awaited after `device.queue.onSubmittedWorkDone()`.
   */
  async finishReadback(): Promise<void> {
    if (!this.readbackPending) return;
    await this.stagingBuf.mapAsync(GPUMapMode.READ);
    const src = new Float32Array(this.stagingBuf.getMappedRange());
    this.cpuData = src.slice(); // copy before unmap
    this.stagingBuf.unmap();
    this.readbackPending = false;
  }

  // ── sampleCPU() — trilinear lookup in the CPU mirror ─────────────────────

  /**
   * Sample the curl velocity at world-space position (wx, wy, wz).
   * Returns {0,0,0,0} if readback has never completed.
   */
  sampleCPU(wx: number, wy: number, wz: number): CurlSample {
    const zero: CurlSample = { vx: 0, vy: 0, vz: 0, speed: 0 };
    if (!this.cpuData) return zero;

    const { width, height, depth, domainX, domainY, domainZ } = this.params;

    const cx = Math.max(0, Math.min(width  - 1.001, (wx / domainX) * width  - 0.5));
    const cy = Math.max(0, Math.min(height - 1.001, (wy / domainY) * height - 0.5));
    const cz = Math.max(0, Math.min(depth  - 1.001, (wz / domainZ) * depth  - 0.5));

    const ix = Math.floor(cx), fx = cx - ix;
    const iy = Math.floor(cy), fy = cy - iy;
    const iz = Math.floor(cz), fz = cz - iz;
    const ix1 = Math.min(ix + 1, width  - 1);
    const iy1 = Math.min(iy + 1, height - 1);
    const iz1 = Math.min(iz + 1, depth  - 1);

    const idx = (zi: number, yi: number, xi: number) =>
      (zi * height * width + yi * width + xi) * 4;

    const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
    const bilerp = (
      v000: number, v100: number, v010: number, v110: number,
      v001: number, v101: number, v011: number, v111: number,
    ) => {
      const x00 = lerp(v000, v100, fx); const x10 = lerp(v010, v110, fx);
      const x01 = lerp(v001, v101, fx); const x11 = lerp(v011, v111, fx);
      return lerp(lerp(x00, x10, fy), lerp(x01, x11, fy), fz);
    };

    const d = this.cpuData;
    const sample = (ch: number) => bilerp(
      d[idx(iz,  iy,  ix ) + ch], d[idx(iz,  iy,  ix1) + ch],
      d[idx(iz,  iy1, ix ) + ch], d[idx(iz,  iy1, ix1) + ch],
      d[idx(iz1, iy,  ix ) + ch], d[idx(iz1, iy,  ix1) + ch],
      d[idx(iz1, iy1, ix ) + ch], d[idx(iz1, iy1, ix1) + ch],
    );

    const vx = sample(0), vy = sample(1), vz = sample(2);
    return { vx, vy, vz, speed: Math.sqrt(vx * vx + vy * vy + vz * vz) };
  }

  // ── setParam() — hot-update individual parameters ────────────────────────

  /**
   * Update a single parameter without rebuilding pipelines.  Changes take
   * effect on the next `update()` call.
   */
  setParam<K extends keyof ResolvedCurlParams>(key: K, value: ResolvedCurlParams[K]): void {
    (this.params as ResolvedCurlParams)[key] = value;
  }

  // ── destroy() — release all GPU resources ────────────────────────────────

  destroy(): void {
    this.uniformBuf?.destroy();
    this.injectUniformBuf?.destroy();
    this.curlBuf?.destroy();
    this.stagingBuf?.destroy();
  }

  // ── Private helpers ───────────────────────────────────────────────────────

  private _writeUniform(time: number): void {
    const { width, height, depth,
            domainX, domainY, domainZ, sliceZ,
            timeScale, octaves, lacunarity, gain, strength } = this.params;

    const buf = new ArrayBuffer(UNIFORM_STRIDE);
    const f   = new Float32Array(buf);
    const u   = new Uint32Array(buf);

    f[0]  = domainX;
    f[1]  = domainY;
    f[2]  = domainZ;
    f[3]  = sliceZ;
    f[4]  = timeScale;
    f[5]  = time;
    f[6]  = EPSILON;
    f[7]  = strength;
    u[8]  = octaves;
    f[9]  = lacunarity;
    f[10] = gain;
    u[11] = width;
    u[12] = height;
    u[13] = depth;
    // [14,15] padding

    this.device.queue.writeBuffer(this.uniformBuf, 0, buf);
  }
}

// ─── Convenience factory ──────────────────────────────────────────────────────

/**
 * Build and return a ready-to-use `CurlFlowField`.
 *
 * @example
 *   const curl = await createCurlFlowField(device, { octaves: 3, strength: 2.0 });
 *   // frame loop:
 *   const enc = device.createCommandEncoder();
 *   curl.update(enc, t);
 *   curl.injectIntoSPH(enc, posX, posY, forceX, forceY, N);
 *   device.queue.submit([enc.finish()]);
 */
export async function createCurlFlowField(
  device: any /*GPUDevice*/,
  params?: CurlFlowFieldParams,
): Promise<CurlFlowField> {
  const field = new CurlFlowField(device, params);
  await field.build();
  return field;
}
