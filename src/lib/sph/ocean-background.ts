// ocean-background.ts — Gerstner wave ocean background for the Cell canvas
//
// Visual layers:
//   1. OCEAN MESH      — subdivided grid whose vertices are displaced by 4
//                        summed Gerstner waves (lygia gerstnerWave2 port).
//                        Simplex noise (lygia snoise2 port) adds detail ripples.
//   2. CELL FLOAT      — each cell's Y position is biased upward by the wave
//                        height sampled at (cellX, time), so they bob on the
//                        sea surface rather than sinking into it.
//   3. SPLASH PARTICLES — a small GPU particle system that emits "water drops"
//                        when a cell crosses a wave crest; particles follow a
//                        parabolic arc and fade out.
//
// Lygia references:
//   upstream/lygia/generative/gerstnerWave.wgsl  (Patricio Gonzalez Vivo)
//   upstream/lygia/generative/snoise.wgsl        (Stefan Gustavson / Ian McEwan)
//
// Both are self-contained inline WGSL; no lygia preprocessor is needed at
// runtime.  The code follows the same "inline snippet + TypeScript driver"
// pattern used by noise-flow-field.ts and natural-patterns.ts.




// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────




import { WORKGROUP_SIZE } from './types';

export interface OceanConfig {
  /** Width of the domain in simulation units (matches SimParams.domainW). */
  domainW: number;
  /** Height of the domain in simulation units (matches SimParams.domainH). */
  domainH: number;
  /** How many subdivisions along X for the ocean mesh (default 128). */
  gridX?: number;
  /** How many subdivisions along Y for the ocean mesh (default 48). */
  gridY?: number;
  /** Overall wave amplitude multiplier (default 1). */
  amplitude?: number;
  /** Wind direction angle in radians (default 0 = rightward). */
  windAngle?: number;
  /** Deep-water colour as [r,g,b,a] in linear sRGB (default dark navy). */
  deepColor?: [number, number, number, number];
  /** Shallow/crest colour (default cyan-white). */
  crestColor?: [number, number, number, number];
  /** Maximum number of simultaneous splash particles (default 1024). */
  maxSplashParticles?: number;
}

export interface OceanUniforms {
  time: number;
  /** NDC scale / offset — same mapping used by ParticleRenderer. */
  scaleX: number;
  scaleY: number;
  offsetX: number;
  offsetY: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// WGSL: Lygia-derived simplex noise helpers (snoise2, snoise3)
// Ported from upstream/lygia/generative/snoise.wgsl
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_SNOISE = /* wgsl */`
// ── lygia snoise2 helpers ────────────────────────────────────────────────────
fn oc_mod289_2(x: vec2f) -> vec2f { return x - floor(x * (1.0/289.0)) * 289.0; }
fn oc_mod289_3(x: vec3f) -> vec3f { return x - floor(x * (1.0/289.0)) * 289.0; }
fn oc_perm3(x: vec3f) -> vec3f {
  return oc_mod289_3(((x * 34.0) + 1.0) * x);
}

/// 2-D simplex noise in [-1, 1].
/// Direct port of lygia snoise2 (upstream/lygia/generative/snoise.wgsl).
fn snoise2(v: vec2f) -> f32 {
  let C = vec4f(
     0.211324865405187,   // (3 - sqrt(3)) / 6
     0.366025403784439,   // 0.5 * (sqrt(3) - 1)
    -0.577350269189626,   // -1 + 2*C.x
     0.024390243902439);  // 1/41
  var i  = floor(v + dot(v, C.yy));
  let x0 = v - i + dot(i, C.xx);
  let i1  = select(vec2f(0.0, 1.0), vec2f(1.0, 0.0), x0.x > x0.y);
  let x12 = x0.xyxy + C.xxzz - vec4f(i1, 0.0, 0.0);
  i = oc_mod289_2(i);
  let p = oc_perm3(oc_perm3(i.y + vec3f(0.0, i1.y, 1.0)) + i.x + vec3f(0.0, i1.x, 1.0));
  var m = max(0.5 - vec3f(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), vec3f(0.0));
  m = m * m;
  m = m * m;
  let x  = 2.0 * fract(p * C.www) - 1.0;
  let h  = abs(x) - 0.5;
  let ox = floor(x + 0.5);
  let a0 = x - ox;
  m *= 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);
  let gx  = a0.x  * x0.x  + h.x  * x0.y;
  let gyz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, vec3f(gx, gyz));
}

/// Fractional Brownian Motion: 4 octaves of snoise2.
fn fbm2(p: vec2f, t: f32) -> f32 {
  var v  = 0.0;
  var a  = 0.5;
  var pp = p;
  for (var i = 0; i < 4; i++) {
    v  += a * snoise2(pp + vec2f(t * 0.07, t * 0.05));
    pp *= 2.1;
    a  *= 0.5;
  }
  return v;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL: Lygia-derived Gerstner wave
// Ported from upstream/lygia/generative/gerstnerWave.wgsl
// (Patricio Gonzalez Vivo — Prosperity / Patron license)
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_GERSTNER = /* wgsl */`
const OC_PI: f32 = 3.14159265358979323846;

/// Single Gerstner wave contribution.
/// Returns XYZ vertex displacement.  _tangent / _binormal are accumulated
/// in-place (passed by value here; caller accumulates manually).
///
/// Direct port of lygia gerstnerWave2
/// (upstream/lygia/generative/gerstnerWave.wgsl).
fn gerstnerWave(
  uv        : vec2f,
  dir       : vec2f,
  steepness : f32,
  wavelength: f32,
  t         : f32,
) -> vec3f {
  let k  = 2.0 * OC_PI / wavelength;
  let c  = sqrt(9.8 / k);
  let d  = normalize(dir);
  let f  = k * (dot(d, uv) - c * t);
  let a  = steepness / k;
  return vec3f(
    d.x * (a * cos(f)),
    a   *       sin(f),
    d.y * (a * cos(f)),
  );
}

/// Sample the combined ocean surface height at world-XZ position p.
/// Sums 4 Gerstner waves with different directions + frequencies.
fn oceanHeight(p: vec2f, t: f32, ampScale: f32) -> f32 {
  var h = 0.0;

  // Wave 1 — primary swell (wind direction)
  h += gerstnerWave(p, vec2f( 1.0,  0.3), 0.35 * ampScale, 8.0,  t).y;
  // Wave 2 — secondary cross-swell
  h += gerstnerWave(p, vec2f( 0.6, -0.8), 0.20 * ampScale, 5.0,  t).y;
  // Wave 3 — short choppy wave
  h += gerstnerWave(p, vec2f(-0.4,  0.9), 0.12 * ampScale, 2.5,  t).y;
  // Wave 4 — high-frequency detail
  h += gerstnerWave(p, vec2f( 0.9,  0.4), 0.08 * ampScale, 1.3,  t).y;

  // Simplex noise detail ripple
  h += fbm2(p * 0.5, t) * 0.06 * ampScale;

  return h;
}

/// Full XZ+Y displacement for mesh vertex.
fn oceanDisplace(uv: vec2f, t: f32, ampScale: f32) -> vec3f {
  var d = vec3f(0.0);
  d += gerstnerWave(uv, vec2f( 1.0,  0.3), 0.35 * ampScale, 8.0,  t);
  d += gerstnerWave(uv, vec2f( 0.6, -0.8), 0.20 * ampScale, 5.0,  t);
  d += gerstnerWave(uv, vec2f(-0.4,  0.9), 0.12 * ampScale, 2.5,  t);
  d += gerstnerWave(uv, vec2f( 0.9,  0.4), 0.08 * ampScale, 1.3,  t);

  // Noise detail
  let noiseXZ = snoise2(uv * 0.8 + vec2f(t * 0.11, 0.0)) * 0.04 * ampScale;
  d.x += noiseXZ;
  d.y += fbm2(uv * 0.5, t) * 0.06 * ampScale;
  return d;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL: Ocean mesh vertex + fragment shaders
// ─────────────────────────────────────────────────────────────────────────────

const OCEAN_MESH_SHADER = /* wgsl */`
${WGSL_SNOISE}
${WGSL_GERSTNER}

struct OceanUniforms {
  time      : f32,
  scaleX    : f32,
  scaleY    : f32,
  offsetX   : f32,
  offsetY   : f32,
  domainW   : f32,
  domainH   : f32,
  ampScale  : f32,
  deepR     : f32,  deepG  : f32,  deepB  : f32,  deepA  : f32,
  crestR    : f32,  crestG : f32,  crestB : f32,  crestA : f32,
  gridX     : u32,
  gridY     : u32,
}

@group(0) @binding(0) var<uniform> u: OceanUniforms;

struct MeshVert {
  @builtin(position) pos   : vec4f,
  @location(0)       wave  : f32,   // normalised crest height [0, 1]
  @location(1)       foam  : f32,   // edge-foam factor
}

@vertex
fn vs_ocean(@builtin(vertex_index) vi: u32) -> MeshVert {
  // Reconstruct grid (ix, iy) from flat vertex index.
  // Each quad uses 6 vertices (2 triangles).
  let quad     = vi / 6u;
  let quadVert = vi % 6u;
  let ix_base  = quad % u.gridX;
  let iy_base  = quad / u.gridX;

  // Per-vertex offset within quad (CCW winding)
  let offsets  = array<vec2u, 6>(
    vec2u(0u, 0u), vec2u(1u, 0u), vec2u(1u, 1u),
    vec2u(0u, 0u), vec2u(1u, 1u), vec2u(0u, 1u),
  );
  let off = offsets[quadVert];
  let ix  = ix_base + off.x;
  let iy  = iy_base + off.y;

  // World-space XZ on the seabed plane
  let fx   = (f32(ix) / f32(u.gridX)) * u.domainW;
  let fz   = (f32(iy) / f32(u.gridY)) * u.domainH;
  let uv2  = vec2f(fx, fz);

  // Gerstner + noise displacement
  let disp = oceanDisplace(uv2, u.time, u.ampScale);
  let worldX = fx + disp.x;
  let worldY = disp.y;   // height above seabed
  let worldZ = fz + disp.z;

  // Map to NDC (we treat the domain as 2-D; worldZ drives depth for painter sort)
  let ndcX = worldX * u.scaleX + u.offsetX;
  let ndcY = worldY * u.scaleY + u.offsetY;

  // Crest factor: normalise wave height to [0,1]
  let waveNorm = clamp(disp.y / (0.45 * u.ampScale) * 0.5 + 0.5, 0.0, 1.0);

  // Foam mask: near zero-crossing on x-gradient of snoise2 → spray edge
  let foamN  = abs(snoise2(uv2 * 1.2 + vec2f(u.time * 0.2, 0.0)));
  let foam   = smoothstep(0.35, 0.55, foamN) * waveNorm;

  var out: MeshVert;
  out.pos  = vec4f(ndcX, ndcY, 0.5 - worldZ * 0.001, 1.0);
  out.wave = waveNorm;
  out.foam = foam;
  return out;
}

@fragment
fn fs_ocean(in: MeshVert) -> @location(0) vec4f {
  let deep  = vec4f(u.deepR,  u.deepG,  u.deepB,  u.deepA);
  let crest = vec4f(u.crestR, u.crestG, u.crestB, u.crestA);

  // Mix deep → crest colour by wave height
  var col = mix(deep, crest, pow(in.wave, 2.2));

  // Foam: blend toward white at edges
  col = mix(col, vec4f(1.0, 1.0, 1.0, col.a), in.foam * 0.75);

  // Subtle depth-based transparency — deep water is more opaque
  col.a = mix(0.55, 0.90, in.wave);

  return col;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL: Splash particle compute (emit + integrate)
// ─────────────────────────────────────────────────────────────────────────────

const SPLASH_COMPUTE_SHADER = /* wgsl */`
${WGSL_SNOISE}
${WGSL_GERSTNER}

struct SplashParticle {
  x    : f32,  y    : f32,
  vx   : f32,  vy   : f32,
  life : f32,  // remaining life in [0, 1]; 0 = dead
  size : f32,
}

struct SplashUniforms {
  time     : f32,
  dt       : f32,
  ampScale : f32,
  domainW  : f32,
  domainH  : f32,
  seed     : f32,
}

@group(0) @binding(0) var<storage, read_write> particles: array<SplashParticle>;
@group(0) @binding(1) var<storage, read>       posX:      array<f32>;
@group(0) @binding(2) var<storage, read>       posY:      array<f32>;
@group(0) @binding(3) var<storage, read>       cellCount: array<u32>;
@group(0) @binding(4) var<uniform>             su:        SplashUniforms;

const GRAVITY: f32 = -9.8;

fn rand(seed: vec2f) -> f32 {
  return fract(sin(dot(seed, vec2f(12.9898, 78.233))) * 43758.5453);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn cs_splash(@builtin(global_invocation_id) gid: vec3u) {
  let pid = gid.x;
  let maxP = arrayLength(&particles);
  if (pid >= maxP) { return; }

  var p = particles[pid];

  // ── Integrate live particles ──────────────────────────────────────────────
  if (p.life > 0.0) {
    p.vy   += GRAVITY * su.dt * 0.15;
    p.x    += p.vx * su.dt;
    p.y    += p.vy * su.dt;
    p.life -= su.dt * 0.8;
    if (p.life < 0.0) { p.life = 0.0; }
    particles[pid] = p;
    return;
  }

  // ── Respawn: one dead particle per live cell that crests a wave ───────────
  let n = cellCount[0];
  if (n == 0u) { return; }
  // Distribute cell responsibility across invocations
  let cid = pid % n;
  let cx  = posX[cid];
  let cy  = posY[cid];
  let waveH = oceanHeight(vec2f(cx, cy), su.time, su.ampScale);

  // Only spray when cell is near the crest (wave height > 60 % amplitude)
  let threshold = 0.6 * su.ampScale * 0.45;
  if (waveH < threshold) { return; }

  // Probabilistic emission — scale by dt so frame-rate-independent
  let emitProb = 0.4 * su.dt;
  let r0 = rand(vec2f(cx + su.seed, su.time * 13.7 + f32(pid)));
  if (r0 > emitProb) { return; }

  // Randomise splash velocity (fan outward + upward)
  let r1  = rand(vec2f(f32(pid) * 0.37, su.time + 1.0));
  let r2  = rand(vec2f(f32(pid) * 0.53, su.time + 2.0));
  let ang = (r1 - 0.5) * 2.4 + 1.5708; // near-vertical fan ±70°
  let spd = 1.5 + r2 * 2.0;

  p.x    = cx;
  p.y    = cy + waveH;
  p.vx   = cos(ang) * spd;
  p.vy   = sin(ang) * spd;
  p.life = 0.6 + r1 * 0.4;
  p.size = 0.015 + r2 * 0.025;

  particles[pid] = p;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL: Splash particle render (instanced quads)
// ─────────────────────────────────────────────────────────────────────────────

const SPLASH_RENDER_SHADER = /* wgsl */`
struct SplashParticle {
  x    : f32,  y    : f32,
  vx   : f32,  vy   : f32,
  life : f32,
  size : f32,
}

struct SplashCamera {
  scaleX  : f32,  scaleY  : f32,
  offsetX : f32,  offsetY : f32,
}

@group(0) @binding(0) var<storage, read> particles: array<SplashParticle>;
@group(0) @binding(1) var<uniform>       cam:       SplashCamera;

struct SplashVert {
  @builtin(position) pos  : vec4f,
  @location(0)       uv   : vec2f,
  @location(1)       life : f32,
}

@vertex
fn vs_splash(
  @builtin(vertex_index)   vi: u32,
  @builtin(instance_index) ii: u32,
) -> SplashVert {
  var quad = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0),
    vec2f(-1.0, -1.0), vec2f( 1.0,  1.0), vec2f(-1.0,  1.0),
  );
  let p  = particles[ii];
  let uv = quad[vi];

  let ndcX = (p.x + uv.x * p.size) * cam.scaleX + cam.offsetX;
  let ndcY = (p.y + uv.y * p.size) * cam.scaleY + cam.offsetY;

  var out: SplashVert;
  out.pos  = vec4f(ndcX, ndcY, 0.2, 1.0);
  out.uv   = uv;
  out.life = p.life;
  return out;
}

@fragment
fn fs_splash(in: SplashVert) -> @location(0) vec4f {
  let r2    = dot(in.uv, in.uv);
  if (r2 > 1.0) { discard; }
  let alpha = smoothstep(1.0, 0.0, r2) * in.life * 0.85;
  // White-blue spray
  let col   = mix(vec3f(0.75, 0.92, 1.0), vec3f(1.0, 1.0, 1.0), in.life);
  return vec4f(col, alpha);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// TypeScript driver
// ─────────────────────────────────────────────────────────────────────────────

/** Uniform buffer layout (must match OceanUniforms struct, 64 bytes). */
const OCEAN_UNI_FLOATS = 20; // 5 + 2 domain + 1 amp + 8 colour + 2 grid (u32)
const OCEAN_UNI_BYTES  = OCEAN_UNI_FLOATS * 4;

/** Scratch uniform buffer layout for splash compute (SplashUniforms). */
const SPLASH_UNI_FLOATS = 6;
const SPLASH_UNI_BYTES  = SPLASH_UNI_FLOATS * 4;

/** Camera uniform for splash render (SplashCamera). */
const SPLASH_CAM_FLOATS = 4;
const SPLASH_CAM_BYTES  = SPLASH_CAM_FLOATS * 4;

/** Bytes per SplashParticle struct (6 × f32). */
const PARTICLE_STRIDE = 6 * 4;

export class OceanBackground {
  private device: GPUDevice;
  private cfg: Required<OceanConfig>;

  // Ocean mesh
  private meshPipeline!: GPURenderPipeline;
  private oceanUniBuf!:  GPUBuffer;
  private oceanBindGroup!: GPUBindGroup;
  private meshVertexCount!: number;

  // Splash compute
  private splashComputePipeline!: GPUComputePipeline;
  private splashParticleBuf!:     GPUBuffer;
  private splashUniBuf!:          GPUBuffer;
  private splashComputeBindGroup!: GPUBindGroup;

  // Splash render
  private splashRenderPipeline!: GPURenderPipeline;
  private splashCamBuf!:         GPUBuffer;
  private splashRenderBindGroup!: GPUBindGroup;

  // External cell buffers (set before each frame)
  private cellPosXBuf: GPUBuffer | null = null;
  private cellPosYBuf: GPUBuffer | null = null;
  private cellCountBuf: GPUBuffer | null = null;

  constructor(device: GPUDevice, config: OceanConfig) {
    this.device = device;
    this.cfg = {
      domainW:            config.domainW,
      domainH:            config.domainH,
      gridX:              config.gridX              ?? 128,
      gridY:              config.gridY              ?? 48,
      amplitude:          config.amplitude          ?? 1.0,
      windAngle:          config.windAngle          ?? 0.0,
      deepColor:          config.deepColor          ?? [0.05, 0.12, 0.28, 1.0],
      crestColor:         config.crestColor         ?? [0.62, 0.88, 1.00, 1.0],
      maxSplashParticles: config.maxSplashParticles ?? 1024,
    };
  }

  // ── Async init ─────────────────────────────────────────────────────────────

  async init(presentationFormat: GPUTextureFormat): Promise<void> {
    await this._buildOceanMeshPipeline(presentationFormat);
    await this._buildSplashPipelines(presentationFormat);
  }

  // ── Per-frame encode ───────────────────────────────────────────────────────

  /**
   * Call this once per frame to:
   *   1. Update ocean uniform buffer with current time + camera mapping.
   *   2. Run splash particle compute pass.
   *   3. Encode ocean mesh + splash render passes into `passEncoder`.
   *
   * @param passEncoder  An active GPURenderPassEncoder targeting the swap-chain.
   * @param uniforms     Current time and NDC mapping.
   * @param cellPosXBuf  GPUBuffer holding cell X positions (f32[]).
   * @param cellPosYBuf  GPUBuffer holding cell Y positions (f32[]).
   * @param cellCountBuf GPUBuffer holding [cellCount] as u32[1].
   * @param commandEncoder For the compute dispatch (must be same submit).
   */
  encode(
    passEncoder:    GPURenderPassEncoder,
    commandEncoder: GPUCommandEncoder,
    uniforms:       OceanUniforms,
    cellPosXBuf:    GPUBuffer,
    cellPosYBuf:    GPUBuffer,
    cellCountBuf:   GPUBuffer,
  ): void {
    // Rebuild splash compute bind-group if cell buffers changed
    if (
      this.cellPosXBuf !== cellPosXBuf ||
      this.cellPosYBuf !== cellPosYBuf ||
      this.cellCountBuf !== cellCountBuf
    ) {
      this.cellPosXBuf  = cellPosXBuf;
      this.cellPosYBuf  = cellPosYBuf;
      this.cellCountBuf = cellCountBuf;
      this._rebuildSplashComputeBindGroup();
    }

    // ── Upload ocean uniforms ──────────────────────────────────────────────
    const ou = new Float32Array(OCEAN_UNI_FLOATS);
    ou[0] = uniforms.time;
    ou[1] = uniforms.scaleX;
    ou[2] = uniforms.scaleY;
    ou[3] = uniforms.offsetX;
    ou[4] = uniforms.offsetY;
    ou[5] = this.cfg.domainW;
    ou[6] = this.cfg.domainH;
    ou[7] = this.cfg.amplitude;
    // deep colour
    ou[8]  = this.cfg.deepColor[0];
    ou[9]  = this.cfg.deepColor[1];
    ou[10] = this.cfg.deepColor[2];
    ou[11] = this.cfg.deepColor[3];
    // crest colour
    ou[12] = this.cfg.crestColor[0];
    ou[13] = this.cfg.crestColor[1];
    ou[14] = this.cfg.crestColor[2];
    ou[15] = this.cfg.crestColor[3];
    // grid dims packed as floats (shader reads as u32 — same bit pattern for small ints)
    ou[16] = this.cfg.gridX as unknown as number;
    ou[17] = this.cfg.gridY as unknown as number;
    this.device.queue.writeBuffer(this.oceanUniBuf, 0, ou.buffer);

    // ── Upload splash compute uniforms ────────────────────────────────────
    const su = new Float32Array(SPLASH_UNI_FLOATS);
    su[0] = uniforms.time;
    su[1] = 0.016; // fixed dt ~60 fps
    su[2] = this.cfg.amplitude;
    su[3] = this.cfg.domainW;
    su[4] = this.cfg.domainH;
    su[5] = Math.random() * 999; // random seed per frame
    this.device.queue.writeBuffer(this.splashUniBuf, 0, su.buffer);

    // ── Upload splash camera uniforms ─────────────────────────────────────
    const sc = new Float32Array(SPLASH_CAM_FLOATS);
    sc[0] = uniforms.scaleX;
    sc[1] = uniforms.scaleY;
    sc[2] = uniforms.offsetX;
    sc[3] = uniforms.offsetY;
    this.device.queue.writeBuffer(this.splashCamBuf, 0, sc.buffer);

    // ── Compute pass: integrate + emit splash particles ───────────────────
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(this.splashComputePipeline);
    computePass.setBindGroup(0, this.splashComputeBindGroup);
    computePass.dispatchWorkgroups(
      Math.ceil(this.cfg.maxSplashParticles / WORKGROUP_SIZE),
    );
    computePass.end();

    // ── Render pass: ocean mesh ───────────────────────────────────────────
    passEncoder.setPipeline(this.meshPipeline);
    passEncoder.setBindGroup(0, this.oceanBindGroup);
    passEncoder.draw(this.meshVertexCount);

    // ── Render pass: splash particles ─────────────────────────────────────
    passEncoder.setPipeline(this.splashRenderPipeline);
    passEncoder.setBindGroup(0, this.splashRenderBindGroup);
    passEncoder.draw(6, this.cfg.maxSplashParticles);
  }

  /**
   * Compute the vertical surface offset for a cell at position (cx, cy)
   * so the caller can elevate the cell to float on the wave.
   *
   * This runs on CPU for simplicity; for large cell counts a compute shader
   * variant can be added.  The formula mirrors oceanHeight() in the WGSL.
   */
  sampleSurfaceHeight(cx: number, cy: number, time: number): number {
    const amp = this.cfg.amplitude;
    const PI  = Math.PI;

    const gerstner = (ux: number, uy: number, dx: number, dy: number,
                      steep: number, wl: number): number => {
      const k   = (2 * PI) / wl;
      const c   = Math.sqrt(9.8 / k);
      const len = Math.hypot(dx, dy);
      const nx  = dx / len; const ny = dy / len;
      const f   = k * (nx * ux + ny * uy - c * time);
      return (steep / k) * Math.sin(f);
    };

    let h = 0;
    h += gerstner(cx, cy,  1.0,  0.3, 0.35 * amp, 8.0);
    h += gerstner(cx, cy,  0.6, -0.8, 0.20 * amp, 5.0);
    h += gerstner(cx, cy, -0.4,  0.9, 0.12 * amp, 2.5);
    h += gerstner(cx, cy,  0.9,  0.4, 0.08 * amp, 1.3);
    return h;
  }

  // ── Cleanup ────────────────────────────────────────────────────────────────

  destroy(): void {
    this.oceanUniBuf?.destroy();
    this.splashParticleBuf?.destroy();
    this.splashUniBuf?.destroy();
    this.splashCamBuf?.destroy();
  }

  // ── Private build helpers ─────────────────────────────────────────────────

  private async _buildOceanMeshPipeline(fmt: GPUTextureFormat): Promise<void> {
    const device = this.device;
    const { gridX, gridY } = this.cfg;

    // We draw gridX * gridY quads, each as 2 triangles (6 verts)
    this.meshVertexCount = gridX * gridY * 6;

    // Uniform buffer (OceanUniforms)
    this.oceanUniBuf = device.createBuffer({
      size:  OCEAN_UNI_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const bgl = device.createBindGroupLayout({
      entries: [{
        binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' },
      }],
    });

    this.oceanBindGroup = device.createBindGroup({
      layout: bgl,
      entries: [{ binding: 0, resource: { buffer: this.oceanUniBuf } }],
    });

    const shaderModule = device.createShaderModule({ code: OCEAN_MESH_SHADER });

    this.meshPipeline = await device.createRenderPipelineAsync({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      vertex:   { module: shaderModule, entryPoint: 'vs_ocean' },
      fragment: {
        module: shaderModule, entryPoint: 'fs_ocean',
        targets: [{
          format: fmt,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one',       dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
    });
  }

  private async _buildSplashPipelines(fmt: GPUTextureFormat): Promise<void> {
    const device = this.device;
    const maxP   = this.cfg.maxSplashParticles;

    // Particle storage buffer
    this.splashParticleBuf = device.createBuffer({
      size:  maxP * PARTICLE_STRIDE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Compute uniforms
    this.splashUniBuf = device.createBuffer({
      size:  SPLASH_UNI_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Splash camera uniforms (render)
    this.splashCamBuf = device.createBuffer({
      size:  SPLASH_CAM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // ── Compute pipeline ───────────────────────────────────────────────────
    // We need real cell buffers before building the bind group; the bind group
    // is rebuilt lazily in _rebuildSplashComputeBindGroup.
    const computeShaderModule = device.createShaderModule({ code: SPLASH_COMPUTE_SHADER });
    this.splashComputePipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module: computeShaderModule, entryPoint: 'cs_splash' },
    });

    // ── Render pipeline ────────────────────────────────────────────────────
    const renderShaderModule = device.createShaderModule({ code: SPLASH_RENDER_SHADER });

    const renderBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      ],
    });

    this.splashRenderBindGroup = device.createBindGroup({
      layout: renderBGL,
      entries: [
        { binding: 0, resource: { buffer: this.splashParticleBuf } },
        { binding: 1, resource: { buffer: this.splashCamBuf } },
      ],
    });

    this.splashRenderPipeline = await device.createRenderPipelineAsync({
      layout: device.createPipelineLayout({ bindGroupLayouts: [renderBGL] }),
      vertex:   { module: renderShaderModule, entryPoint: 'vs_splash' },
      fragment: {
        module: renderShaderModule, entryPoint: 'fs_splash',
        targets: [{
          format: fmt,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one',       operation: 'add' },
            alpha: { srcFactor: 'zero',      dstFactor: 'one',       operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
    });
  }

  /** Called whenever cell position buffers are swapped in by the orchestrator. */
  private _rebuildSplashComputeBindGroup(): void {
    if (!this.cellPosXBuf || !this.cellPosYBuf || !this.cellCountBuf) return;

    this.splashComputeBindGroup = this.device.createBindGroup({
      layout: this.splashComputePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.splashParticleBuf } },
        { binding: 1, resource: { buffer: this.cellPosXBuf } },
        { binding: 2, resource: { buffer: this.cellPosYBuf } },
        { binding: 3, resource: { buffer: this.cellCountBuf } },
        { binding: 4, resource: { buffer: this.splashUniBuf } },
      ],
    });
  }
}
