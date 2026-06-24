// === SPHGPUOrchestrator.ts ===
// SPHGPUOrchestrator.ts --- WebGPU compute pipeline for SPH




// ---------------------------------------------------------------------------
// WGSL Shaders
// ---------------------------------------------------------------------------




import type { GPUBufferSet, SimParams } from './types';
import { WORKGROUP_SIZE } from './types';

const DENSITY_SHADER = /* wgsl */`
struct SimUniforms {
  h          : f32,
  restDensity: f32,
  gasConstant: f32,
  viscosity  : f32,
  gravity    : f32,
  dt         : f32,
  domainW    : f32,
  domainH    : f32,
  count      : u32,
  boundaryN  : u32,
  _pad0      : u32,
  _pad1      : u32,
}

// group 0 --- uniforms
@group(0) @binding(0) var<uniform> params : SimUniforms;

// group 1 --- particle buffers (read-only inputs + rw outputs)
@group(1) @binding(0) var<storage, read>       posX    : array<f32>;
@group(1) @binding(1) var<storage, read>       posY    : array<f32>;
@group(1) @binding(2) var<storage, read_write> density : array<f32>;
@group(1) @binding(3) var<storage, read_write> pressure: array<f32>;

// group 2 --- neighbor CSR
@group(2) @binding(0) var<storage, read> neighborData: array<i32>;
@group(2) @binding(1) var<storage, read> rowPtr      : array<i32>;

// group 3 --- boundary particles
@group(3) @binding(0) var<storage, read> boundaryBuf: array<vec4f>;

fn W_cubic(r: f32, h: f32) -> f32 {
  let q     = r / h;
  let sigma = 10.0 / (7.0 * 3.14159265358979 * h * h);
  if (q <= 1.0) { return sigma * (1.0 - 1.5 * q * q * (1.0 - 0.5 * q)); }
  if (q <= 2.0) { let t = 2.0 - q; return sigma * 0.25 * t * t * t; }
  return 0.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }

  let xi  = posX[i];
  let yi  = posY[i];
  var rho = 0.0;

  let start = rowPtr[i];
  let end   = rowPtr[i + 1u];
  for (var k = start; k < end; k++) {
    let j  = u32(neighborData[k]);
    let dx = posX[j] - xi;
    let dy = posY[j] - yi;
    let r  = sqrt(dx * dx + dy * dy);
    rho   += W_cubic(r, params.h);
  }

  for (var b = 0u; b < params.boundaryN; b++) {
    let bp = boundaryBuf[b];
    let dx = bp.x - xi;
    let dy = bp.y - yi;
    let r  = sqrt(dx * dx + dy * dy);
    rho   += params.restDensity * bp.z * W_cubic(r, params.h);
  }

  density[i] = rho;

  let ratio     = rho / params.restDensity;
  let r2        = ratio  * ratio;
  let r4        = r2     * r2;
  let r7        = r4     * r2 * ratio;
  pressure[i]   = params.gasConstant * (r7 - 1.0);
}
`;

// ---------------------------------------------------------------------------

const FORCE_SHADER = /* wgsl */`
struct SimUniforms {
  h          : f32,
  restDensity: f32,
  gasConstant: f32,
  viscosity  : f32,
  gravity    : f32,
  dt         : f32,
  domainW    : f32,
  domainH    : f32,
  count      : u32,
  boundaryN  : u32,
  _pad0      : u32,
  _pad1      : u32,
}

// group 0 --- uniforms
@group(0) @binding(0) var<uniform> params : SimUniforms;

// group 1 --- particle buffers
@group(1) @binding(0) var<storage, read>       posX    : array<f32>;
@group(1) @binding(1) var<storage, read>       posY    : array<f32>;
@group(1) @binding(2) var<storage, read>       velX    : array<f32>;
@group(1) @binding(3) var<storage, read>       velY    : array<f32>;
@group(1) @binding(4) var<storage, read>       density : array<f32>;
@group(1) @binding(5) var<storage, read>       pressure: array<f32>;
@group(1) @binding(6) var<storage, read_write> forceX  : array<f32>;
@group(1) @binding(7) var<storage, read_write> forceY  : array<f32>;

// group 2 --- neighbor CSR
@group(2) @binding(0) var<storage, read> neighborData: array<i32>;
@group(2) @binding(1) var<storage, read> rowPtr      : array<i32>;

// group 3 --- boundary particles
@group(3) @binding(0) var<storage, read> boundaryBuf: array<vec4f>;

fn W_cubic(r: f32, h: f32) -> f32 {
  let q     = r / h;
  let sigma = 10.0 / (7.0 * 3.14159265358979 * h * h);
  if (q <= 1.0) { return sigma * (1.0 - 1.5 * q * q * (1.0 - 0.5 * q)); }
  if (q <= 2.0) { let t = 2.0 - q; return sigma * 0.25 * t * t * t; }
  return 0.0;
}

fn gradW_cubic(dx: f32, dy: f32, r: f32, h: f32) -> vec2f {
  if (r < 1e-6) { return vec2f(0.0, 0.0); }
  let q     = r / h;
  let sigma = 10.0 / (7.0 * 3.14159265358979 * h * h);
  var dW    = 0.0;
  if (q <= 1.0) {
    dW = sigma * (-3.0 * q + 2.25 * q * q) / h;
  } else if (q <= 2.0) {
    let t = 2.0 - q;
    dW    = sigma * (-0.75 * t * t) / h;
  }
  return vec2f(dx / r * dW, dy / r * dW);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }

  let xi   = posX[i];
  let yi   = posY[i];
  let vxi  = velX[i];
  let vyi  = velY[i];
  let rhoI = max(density[i],  0.001);
  let pI   = pressure[i];

  var fx = 0.0;
  var fy = 0.0;

  let start = rowPtr[i];
  let end   = rowPtr[i + 1u];

  for (var k = start; k < end; k++) {
    let j = u32(neighborData[k]);
    if (j == i) { continue; }

    let dx   = posX[j] - xi;
    let dy   = posY[j] - yi;
    let r    = sqrt(dx * dx + dy * dy);
    let rhoJ = max(density[j], 0.001);
    let pJ   = pressure[j];

    // pressure gradient (symmetric)
    let grad  = gradW_cubic(dx, dy, r, params.h);
    let pTerm = -(pI / (rhoI * rhoI) + pJ / (rhoJ * rhoJ));
    fx += pTerm * grad.x;
    fy += pTerm * grad.y;

    // viscosity (Monaghan 1992 artificial viscosity)
    let dvx  = velX[j] - vxi;
    let dvy  = velY[j] - vyi;
    let vDotR = dvx * dx + dvy * dy;
    if (vDotR < 0.0) {
      let mu     = params.h * vDotR / (r * r + 0.01 * params.h * params.h);
      let avgRho = 0.5 * (rhoI + rhoJ);
      let visc   = params.viscosity * mu / avgRho;
      fx += visc * grad.x;
      fy += visc * grad.y;
    }
  }

  // boundary repulsion (Lennard-Jones style)
  for (var b = 0u; b < params.boundaryN; b++) {
    let bp   = boundaryBuf[b];
    let dx   = xi - bp.x;
    let dy   = yi - bp.y;
    let r    = sqrt(dx * dx + dy * dy);
    let q    = r / params.h;
    if (q < 1.0 && r > 1e-6) {
      let repulse = params.gasConstant * (1.0 / (q * q) - 1.0) / (rhoI * r);
      fx += repulse * dx;
      fy += repulse * dy;
    }
  }

  // gravity
  fy += params.gravity;

  forceX[i] = fx;
  forceY[i] = fy;
}
`;

// ---------------------------------------------------------------------------

const INTEGRATE_SHADER = /* wgsl */`
struct SimUniforms {
  h          : f32,
  restDensity: f32,
  gasConstant: f32,
  viscosity  : f32,
  gravity    : f32,
  dt         : f32,
  domainW    : f32,
  domainH    : f32,
  count      : u32,
  boundaryN  : u32,
  _pad0      : u32,
  _pad1      : u32,
}

// group 0 --- uniforms
@group(0) @binding(0) var<uniform> params : SimUniforms;

// group 1 --- particle buffers (all read-write)
@group(1) @binding(0) var<storage, read_write> posX  : array<f32>;
@group(1) @binding(1) var<storage, read_write> posY  : array<f32>;
@group(1) @binding(2) var<storage, read_write> velX  : array<f32>;
@group(1) @binding(3) var<storage, read_write> velY  : array<f32>;
@group(1) @binding(4) var<storage, read>       forceX: array<f32>;
@group(1) @binding(5) var<storage, read>       forceY: array<f32>;
@group(1) @binding(6) var<storage, read>       density: array<f32>;

const MAX_VEL: f32 = 50.0;
const DAMPING: f32 = 0.5;   // velocity damping on wall hit

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }

  let rho  = max(density[i], 0.001);
  let ax   = forceX[i] / rho;
  let ay   = forceY[i] / rho;
  let dt   = params.dt;

  // symplectic Euler
  var vx = velX[i] + ax * dt;
  var vy = velY[i] + ay * dt;

  // clamp velocity magnitude
  let speed = sqrt(vx * vx + vy * vy);
  if (speed > MAX_VEL) {
    vx *= MAX_VEL / speed;
    vy *= MAX_VEL / speed;
  }

  var px = posX[i] + vx * dt;
  var py = posY[i] + vy * dt;

  let margin = params.h * 0.5;

  // X boundary
  if (px < margin) {
    px = margin;
    vx = abs(vx) * DAMPING;
  } else if (px > params.domainW - margin) {
    px = params.domainW - margin;
    vx = -abs(vx) * DAMPING;
  }

  // Y boundary
  if (py < margin) {
    py = margin;
    vy = abs(vy) * DAMPING;
  } else if (py > params.domainH - margin) {
    py = params.domainH - margin;
    vy = -abs(vy) * DAMPING;
  }

  posX[i] = px;
  posY[i] = py;
  velX[i] = vx;
  velY[i] = vy;
}
`;

// ---------------------------------------------------------------------------
// Hash-Count shader  (spatial hash pass 1 --- Teschner et al. 2003)
//
// For each fluid particle i, compute its cell key via the Teschner hash:
//   key = (ix * p1 XOR iy * p2) mod tableSize
// where ix/iy are the integer grid cell indices, and p1/p2 are large primes.
// Then atomically increment cellCount[key] so that a subsequent prefix-scan
// can convert the counts into start offsets for pass 2 (scatter).
// ---------------------------------------------------------------------------

const HASH_COUNT_SHADER = /* wgsl */`
struct HashUniforms {
  h         : f32,   // smoothing length (= cell size)
  domainW   : f32,   // simulation domain width
  domainH   : f32,   // simulation domain height
  count     : u32,   // number of fluid particles
  tableSize : u32,   // hash-table length (must be power-of-two or prime)
  _pad0     : u32,
  _pad1     : u32,
  _pad2     : u32,
}

// group 0 --- hash uniforms
@group(0) @binding(0) var<uniform> uParams : HashUniforms;

// group 1 --- particle positions (read-only)
@group(1) @binding(0) var<storage, read> posX : array<f32>;
@group(1) @binding(1) var<storage, read> posY : array<f32>;

// group 2 --- cell count table (read-write atomics)
//   cellCount[key] accumulates the number of particles mapping to key
@group(2) @binding(0) var<storage, read_write> cellCount : array<atomic<u32>>;

// Teschner hash primes (Teschner et al. 2003, "Optimized Spatial Hashing")
const P1 : u32 = 73856093u;
const P2 : u32 = 19349663u;

/// Map a particle position to a hash-table bucket index.
fn teschnerHash(px: f32, py: f32, h: f32, tableSize: u32) -> u32 {
  // Integer grid coordinates of the cell containing (px, py)
  let ix = u32(max(floor(px / h), 0.0));
  let iy = u32(max(floor(py / h), 0.0));

  // Teschner hash: XOR of coordinate-scaled primes, then modulo table size.
  // The XOR spreads keys uniformly even for spatially coherent input.
  let raw = (ix * P1) ^ (iy * P2);
  return raw % tableSize;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= uParams.count) { return; }

  let key = teschnerHash(posX[i], posY[i], uParams.h, uParams.tableSize);

  // Atomic increment: safe when multiple threads hash to the same cell.
  atomicAdd(&cellCount[key], 1u);
}
`;

// ---------------------------------------------------------------------------
// Prefix-Sum shader  (Blelloch exclusive scan --- spatial-hash pass 1.5)
//
// Converts the raw per-cell particle counts produced by HASH_COUNT_SHADER
// into exclusive prefix-sum offsets so that pass 2 (scatter) knows where
// in the sorted-particle array each cell starts.
//
// Algorithm: Blelloch work-efficient parallel scan (CUDA "scan" chapter,
// Harris et al. 2007).  The shader runs a two-phase up-sweep / down-sweep
// entirely in shared memory for arrays up to SCAN_BLOCK -- 2 elements per
// workgroup invocation.  For larger tables the host chains multiple passes
// (see `dispatchPrefixSum`).
//
// Shared-memory layout (SCAN_BLOCK = 256 --- 512 u32 = 2 KiB):
//   temp[0 .. 2*SCAN_BLOCK-1]  --- ping-pong scratch
// ---------------------------------------------------------------------------

const PREFIX_SUM_SHADER = /* wgsl */`
// One workgroup processes 2 -- SCAN_BLOCK elements of the cellCount array.
// SCAN_BLOCK must equal the workgroup_size declared below (256).
const SCAN_BLOCK : u32 = 256u;

struct ScanUniforms {
  n         : u32,   // total number of elements in the array
  blockOffset: u32,  // element offset for this dispatch (multi-pass support)
  _pad0     : u32,
  _pad1     : u32,
}

@group(0) @binding(0) var<uniform>            uScan    : ScanUniforms;
@group(1) @binding(0) var<storage, read_write> data     : array<u32>;
// blockSum[i] receives the total sum of workgroup i (used in multi-pass)
@group(2) @binding(0) var<storage, read_write> blockSum : array<u32>;

var<workgroup> temp: array<u32, 512>;   // 2 -- SCAN_BLOCK

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid : vec3<u32>,
  @builtin(local_invocation_id)  lid : vec3<u32>,
  @builtin(workgroup_id)         wid : vec3<u32>,
) {
  let thid  = lid.x;                          // thread index within workgroup [0, 255]
  let base  = uScan.blockOffset + wid.x * (SCAN_BLOCK * 2u);   // global element base

  // ------ Load two elements per thread into shared memory ------------------------------------------------------------------
  let ai = base + thid;
  let bi = base + thid + SCAN_BLOCK;

  temp[thid]             = select(0u, data[ai], ai < uScan.n);
  temp[thid + SCAN_BLOCK] = select(0u, data[bi], bi < uScan.n);
  workgroupBarrier();

  // ------ Up-sweep (reduce) phase ---------------------------------------------------------------------------------------------------------------------------------------------
  // Build a partial-sum tree in-place.  After k iterations temp[2^(k+1)-1]
  // holds the partial sum of the first 2^(k+1) elements.
  var offset = 1u;
  var d      = SCAN_BLOCK;   // d starts at N/2
  loop {
    if (d == 0u) { break; }
    workgroupBarrier();
    if (thid < d) {
      let ai2 = offset * (2u * thid + 1u) - 1u;
      let bi2 = offset * (2u * thid + 2u) - 1u;
      temp[bi2] += temp[ai2];
    }
    offset *= 2u;
    d      /= 2u;
  }

  // ------ Store block total, then clear the last element (exclusive scan) ---------------------
  if (thid == 0u) {
    blockSum[wid.x] = temp[SCAN_BLOCK * 2u - 1u];
    temp[SCAN_BLOCK * 2u - 1u] = 0u;
  }
  workgroupBarrier();

  // ------ Down-sweep phase ------------------------------------------------------------------------------------------------------------------------------------------------------------------
  // Propagate the zero seed back down the tree to produce the exclusive scan.
  d      = 1u;
  offset = SCAN_BLOCK;
  loop {
    if (d > SCAN_BLOCK) { break; }
    offset /= 2u;
    workgroupBarrier();
    if (thid < d) {
      let ai2 = offset * (2u * thid + 1u) - 1u;
      let bi2 = offset * (2u * thid + 2u) - 1u;
      let t    = temp[ai2];
      temp[ai2] = temp[bi2];
      temp[bi2] += t;
    }
    d *= 2u;
  }
  workgroupBarrier();

  // ------ Write results back to global memory ------------------------------------------------------------------------------------------------------
  if (ai < uScan.n) { data[ai] = temp[thid]; }
  if (bi < uScan.n) { data[bi] = temp[thid + SCAN_BLOCK]; }
}
`;

// Shader that adds the per-block totals accumulated in the first pass back
// into each block's elements so that the final array is a globally-correct
// exclusive prefix sum (second pass of a two-pass Blelloch scan).
const PREFIX_SUM_ADD_SHADER = /* wgsl */`
const SCAN_BLOCK : u32 = 256u;

struct ScanUniforms {
  n          : u32,
  blockOffset: u32,
  _pad0      : u32,
  _pad1      : u32,
}

@group(0) @binding(0) var<uniform>            uScan    : ScanUniforms;
@group(1) @binding(0) var<storage, read_write> data     : array<u32>;
@group(2) @binding(0) var<storage, read>       blockSum : array<u32>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid : vec3<u32>,
  @builtin(local_invocation_id)  lid : vec3<u32>,
  @builtin(workgroup_id)         wid : vec3<u32>,
) {
  // Skip the very first block --- its offset is already 0.
  if (wid.x == 0u) { return; }

  let base = uScan.blockOffset + wid.x * (SCAN_BLOCK * 2u);
  let add  = blockSum[wid.x];   // exclusive prefix of block totals

  let ai = base + lid.x;
  let bi = base + lid.x + SCAN_BLOCK;

  if (ai < uScan.n) { data[ai] += add; }
  if (bi < uScan.n) { data[bi] += add; }
}
`;

// ---------------------------------------------------------------------------
// Types (local aliases)
// ---------------------------------------------------------------------------

interface NeighborCSR {
  neighborBuf: any /*GPUBuffer*/; // array<i32>
  rowPtrBuf  : any /*GPUBuffer*/; // array<i32>, length = N+1
}

// ---------------------------------------------------------------------------
// Perf-log entry produced by GPU timestamp profiling
// ---------------------------------------------------------------------------

export interface PerfEntry {
  /** Monotonic frame counter (increments every `tick` call). */
  frame     : number;
  /** Density + pressure pass GPU time (ms). */
  densityMs : number;
  /** Force computation pass GPU time (ms). */
  forceMs   : number;
  /** Integration pass GPU time (ms). */
  integrateMs: number;
  /** Sum of the three pass durations (ms). */
  totalMs   : number;
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

export class SPHGPUOrchestrator {
  private readonly device  : any /*GPUDevice*/;
  private readonly bufs    : GPUBufferSet;

  private uniformBuf!      : any /*GPUBuffer*/;

  // pipelines
  private densityPipeline   !: any /*GPUComputePipeline*/;
  private forcePipeline     !: any /*GPUComputePipeline*/;
  private integratePipeline !: any /*GPUComputePipeline*/;
  private hashCountPipeline !: any /*GPUComputePipeline*/;   // spatial-hash pass 1
  private prefixSumPipeline !: any /*GPUComputePipeline*/;   // Blelloch scan  (pass 1.5 --- local)
  private prefixSumAddPipeline !: any /*GPUComputePipeline*/; // Blelloch add   (pass 1.5 --- global fixup)

  // bind-group layouts (group 0 is uniform, shared)
  private uniformBGL           !: GPUBindGroupLayout;
  private densityParticleBGL   !: GPUBindGroupLayout;
  private forceParticleBGL     !: GPUBindGroupLayout;
  private integrateParticleBGL !: GPUBindGroupLayout;
  private neighborBGL          !: GPUBindGroupLayout;
  private boundaryBGL          !: GPUBindGroupLayout;
  private hashPosBGL           !: GPUBindGroupLayout; // hash-count: posX/posY
  private hashCountBGL         !: GPUBindGroupLayout; // hash-count: cellCount (atomic)

  // prefix-sum BGLs (group 0 = scan uniforms, group 1 = data, group 2 = blockSum)
  private scanUniformBGL !: GPUBindGroupLayout;
  private scanDataBGL    !: GPUBindGroupLayout;
  private scanBlockBGL   !: GPUBindGroupLayout;

  // dedicated uniform buffer for the hash-count pass (32 bytes)
  private hashUniformBuf       !: any /*GPUBuffer*/;
  // dedicated uniform buffer for prefix-sum passes (16 bytes)
  private scanUniformBuf       !: any /*GPUBuffer*/;

  // cached bind groups
  private uniformBG           !: any /*GPUBindGroup*/;
  private densityParticleBG   !: any /*GPUBindGroup*/;
  private forceParticleBG     !: any /*GPUBindGroup*/;
  private integrateParticleBG !: any /*GPUBindGroup*/;
  private neighborBG          : any /*GPUBindGroup*/ | null = null;
  private boundaryBG          : any /*GPUBindGroup*/ | null = null;

  // hash-count bind groups (recreated when the cell-count buffer changes)
  private hashUniformBG       !: any /*GPUBindGroup*/;
  private hashPosBG           !: any /*GPUBindGroup*/;
  private hashCountBG         : any /*GPUBindGroup*/ | null = null;
  private lastCellCountBuf    : any /*GPUBuffer*/    | null = null;

  // prefix-sum bind groups (recreated when the data/blockSum buffers change)
  private scanUniformBG       !: any /*GPUBindGroup*/;
  private scanDataBG          : any /*GPUBindGroup*/ | null = null;
  private scanBlockSumBG      : any /*GPUBindGroup*/ | null = null;
  private lastScanDataBuf     : any /*GPUBuffer*/    | null = null;
  // internal blockSum scratch buffer (resized on demand)
  private scanBlockSumBuf     : any /*GPUBuffer*/    | null = null;
  private scanBlockSumCapacity: number              = 0;

  // last CSR / boundary refs for dirty-checking
  private lastNeighborBuf  : any /*GPUBuffer*/ | null = null;
  private lastRowPtrBuf    : any /*GPUBuffer*/ | null = null;
  private lastBoundaryBuf  : any /*GPUBuffer*/ | null = null;

  // ------ GPU timestamp profiling ------------------------------------------------------------------------------------------------------------------------------------------
  // Enabled only when the device was requested with `timestamp-query` feature.
  private readonly tsEnabled : boolean = false;

  // QuerySet holds 2 timestamps per named pass (write + end = 2 timestamps).
  // We profile 3 core passes: density(0-1), force(2-3), integrate(4-5).
  private tsQuerySet    : GPUQuerySet   | null = null;
  // Resolve buffer receives the raw u64 ns values from the GPU.
  private tsResolveBuf  : any /*GPUBuffer*/     | null = null;
  // Map buffer is COPY_DST + MAP_READ so the CPU can read timestamps back.
  private tsMapBuf      : any /*GPUBuffer*/     | null = null;

  // Number of timestamp slots we allocate (2 per pass -- 3 passes).
  private static readonly TS_SLOTS = 6;

  /** Rolling performance log: ring buffer of `perfLog` entries (last 60). */
  readonly perfLog: PerfEntry[] = [];
  private static readonly PERF_LOG_MAX = 60;

  /** Monotonic counter incremented every `tick()` call. */
  private frameCounter = 0;

  constructor(device: any /*GPUDevice*/, bufs: GPUBufferSet) {
    this.device = device;
    this.bufs   = bufs;

    // Enable GPU timestamp profiling if the device supports it.
    (this as { tsEnabled: boolean }).tsEnabled =
      device.features.has("timestamp-query");

    if (this.tsEnabled) {
      const slots = SPHGPUOrchestrator.TS_SLOTS;
      this.tsQuerySet = device.createQuerySet({
        label: "sph-ts-queryset",
        type : "timestamp",
        count: slots,
      });
      // resolve buffer: slots x 8 bytes (u64 nanoseconds each)
      this.tsResolveBuf = device.createBuffer({
        label: "sph-ts-resolve",
        size : slots * 8,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      this.tsMapBuf = device.createBuffer({
        label: "sph-ts-map",
        size : slots * 8,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    }

    this.init();
  }

  // -------------------------------------------------------------------------
  private init(): void {
    const dev = this.device;

    // ---------- uniform buffer (48 bytes --- 12 -- f32/u32) ----------
    this.uniformBuf = dev.createBuffer({
      size : 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // ---------- hash-count uniform buffer (32 bytes --- 8 -- f32/u32) ----------
    this.hashUniformBuf = dev.createBuffer({
      size : 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // ---------- prefix-sum uniform buffer (16 bytes --- 4 -- u32) ----------
    this.scanUniformBuf = dev.createBuffer({
      size : 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // ---------- bind-group layouts ----------
    this.uniformBGL = dev.createBindGroupLayout({
      label  : "uniform-bgl",
      entries: [{
        binding   : 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer    : { type: "uniform" },
      }],
    });

    // density pass: posX, posY, density(rw), pressure(rw)
    this.densityParticleBGL = dev.createBindGroupLayout({
      label  : "density-particle-bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });

    // force pass: posX, posY, velX, velY, density(r), pressure(r), forceX(rw), forceY(rw)
    this.forceParticleBGL = dev.createBindGroupLayout({
      label  : "force-particle-bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });

    // integrate pass: posX(rw), posY(rw), velX(rw), velY(rw), forceX(r), forceY(r), density(r)
    this.integrateParticleBGL = dev.createBindGroupLayout({
      label  : "integrate-particle-bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ],
    });

    // neighbor CSR: neighborData(r), rowPtr(r)
    this.neighborBGL = dev.createBindGroupLayout({
      label  : "neighbor-bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ],
    });

    // boundary: boundaryBuf(r)
    this.boundaryBGL = dev.createBindGroupLayout({
      label  : "boundary-bgl",
      entries: [{
        binding   : 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer    : { type: "read-only-storage" },
      }],
    });

    // ---------- pipelines ----------
    this.densityPipeline   = this.makePipeline("density",
      DENSITY_SHADER,
      [this.uniformBGL, this.densityParticleBGL, this.neighborBGL, this.boundaryBGL]);

    this.forcePipeline     = this.makePipeline("force",
      FORCE_SHADER,
      [this.uniformBGL, this.forceParticleBGL, this.neighborBGL, this.boundaryBGL]);

    this.integratePipeline = this.makePipeline("integrate",
      INTEGRATE_SHADER,
      [this.uniformBGL, this.integrateParticleBGL]);

    this.createHashCountPipeline();
    this.createPrefixSumPipeline();

    // ---------- static bind groups (particle buffers, uniform) ----------
    this.uniformBG = dev.createBindGroup({
      label  : "uniform-bg",
      layout : this.uniformBGL,
      entries: [{ binding: 0, resource: { buffer: this.uniformBuf } }],
    });

    this.densityParticleBG = dev.createBindGroup({
      label  : "density-particle-bg",
      layout : this.densityParticleBGL,
      entries: [
        { binding: 0, resource: { buffer: bufs.posX     } },
        { binding: 1, resource: { buffer: bufs.posY     } },
        { binding: 2, resource: { buffer: bufs.density  } },
        { binding: 3, resource: { buffer: bufs.pressure } },
      ],
    });

    this.forceParticleBG = dev.createBindGroup({
      label  : "force-particle-bg",
      layout : this.forceParticleBGL,
      entries: [
        { binding: 0, resource: { buffer: bufs.posX     } },
        { binding: 1, resource: { buffer: bufs.posY     } },
        { binding: 2, resource: { buffer: bufs.velX     } },
        { binding: 3, resource: { buffer: bufs.velY     } },
        { binding: 4, resource: { buffer: bufs.density  } },
        { binding: 5, resource: { buffer: bufs.pressure } },
        { binding: 6, resource: { buffer: bufs.forceX   } },
        { binding: 7, resource: { buffer: bufs.forceY   } },
      ],
    });

    this.integrateParticleBG = dev.createBindGroup({
      label  : "integrate-particle-bg",
      layout : this.integrateParticleBGL,
      entries: [
        { binding: 0, resource: { buffer: bufs.posX     } },
        { binding: 1, resource: { buffer: bufs.posY     } },
        { binding: 2, resource: { buffer: bufs.velX     } },
        { binding: 3, resource: { buffer: bufs.velY     } },
        { binding: 4, resource: { buffer: bufs.forceX   } },
        { binding: 5, resource: { buffer: bufs.forceY   } },
        { binding: 6, resource: { buffer: bufs.density  } },
      ],
    });

    // hash-count particle-position bind group (created after createHashCountPipeline
    // has set up this.hashPosBGL)
    this.hashPosBG = dev.createBindGroup({
      label  : "hash-pos-bg",
      layout : this.hashPosBGL,
      entries: [
        { binding: 0, resource: { buffer: bufs.posX } },
        { binding: 1, resource: { buffer: bufs.posY } },
      ],
    });
  }

  // -------------------------------------------------------------------------
  private makePipeline(
    label  : string,
    wgsl   : string,
    layouts: GPUBindGroupLayout[],
  ): any /*GPUComputePipeline*/ {
    const module = this.device.createShaderModule({ label: `${label}-shader`, code: wgsl });
    const layout = this.device.createPipelineLayout({
      label              : `${label}-layout`,
      bindGroupLayouts   : layouts,
    });
    return this.device.createComputePipeline({
      label : `${label}-pipeline`,
      layout,
      compute: { module, entryPoint: "main" },
    });
  }

  // -------------------------------------------------------------------------
  private updateNeighborBG(csr: NeighborCSR): void {
    if (
      this.neighborBG !== null &&
      csr.neighborBuf === this.lastNeighborBuf &&
      csr.rowPtrBuf   === this.lastRowPtrBuf
    ) { return; }

    this.neighborBG = this.device.createBindGroup({
      label  : "neighbor-bg",
      layout : this.neighborBGL,
      entries: [
        { binding: 0, resource: { buffer: csr.neighborBuf } },
        { binding: 1, resource: { buffer: csr.rowPtrBuf   } },
      ],
    });
    this.lastNeighborBuf = csr.neighborBuf;
    this.lastRowPtrBuf   = csr.rowPtrBuf;
  }

  private updateBoundaryBG(boundaryBuf: any /*GPUBuffer*/): void {
    if (this.boundaryBG !== null && boundaryBuf === this.lastBoundaryBuf) { return; }

    this.boundaryBG = this.device.createBindGroup({
      label  : "boundary-bg",
      layout : this.boundaryBGL,
      entries: [{ binding: 0, resource: { buffer: boundaryBuf } }],
    });
    this.lastBoundaryBuf = boundaryBuf;
  }

  // -------------------------------------------------------------------------
  /** Write the uniform buffer from a SimParams object. */
  private uploadUniforms(p: SimParams): void {
    // Layout (48 bytes):
    //   f32 h, f32 restDensity, f32 gasConstant, f32 viscosity
    //   f32 gravity, f32 dt, f32 domainW, f32 domainH
    //   u32 count, u32 boundaryN, u32 _pad0, u32 _pad1
    const data = new ArrayBuffer(48);
    const f    = new Float32Array(data);
    const u    = new Uint32Array(data);
    f[0]  = p.h;
    f[1]  = p.restDensity;
    f[2]  = p.gasConstant;
    f[3]  = p.viscosity;
    f[4]  = p.gravity;
    f[5]  = p.dt;
    f[6]  = p.domainW;
    f[7]  = p.domainH;
    u[8]  = p.count;
    u[9]  = p.boundaryN;
    u[10] = 0;
    u[11] = 0;
    this.device.queue.writeBuffer(this.uniformBuf, 0, data);
  }

  // -------------------------------------------------------------------------
  /**
   * Execute one simulation frame on the GPU.
   *
   * @param simParams  - scalar simulation parameters
   * @param neighborCSR - current-frame neighbor lists (CSR format)
   * @param boundaryBuf - vec4f boundary-particle buffer (x, y, volume, 0)
   */
  tick(
    simParams  : SimParams,
    neighborCSR: NeighborCSR,
    boundaryBuf: any /*GPUBuffer*/,
  ): void {
    const dev   = this.device;
    const n     = simParams.count;
    const wg    = Math.ceil(n / WORKGROUP_SIZE);
    const frame = this.frameCounter++;
    const ts    = this.tsEnabled;

    // 1. Upload uniforms
    this.uploadUniforms(simParams);

    // 2. Refresh dynamic bind groups if buffers changed
    this.updateNeighborBG(neighborCSR);
    this.updateBoundaryBG(boundaryBuf);

    const neighborBG  = this.neighborBG!;
    const boundaryBG  = this.boundaryBG!;

    const encoder = dev.createCommandEncoder({ label: "sph-frame" });

    // ------------------------------------------------------------------
    // Pass 1 --- density + pressure
    // Timestamp slots: begin=0, end=1
    // ------------------------------------------------------------------
    {
      const passDesc: GPUComputePassDescriptor = { label: "density-pass" };
      if (ts) {
        passDesc.timestampWrites = {
          querySet                 : this.tsQuerySet!,
          beginningOfPassWriteIndex: 0,
          endOfPassWriteIndex      : 1,
        };
      }
      const pass = encoder.beginComputePass(passDesc);
      pass.setPipeline(this.densityPipeline);
      pass.setBindGroup(0, this.uniformBG);
      pass.setBindGroup(1, this.densityParticleBG);
      pass.setBindGroup(2, neighborBG);
      pass.setBindGroup(3, boundaryBG);
      pass.dispatchWorkgroups(wg);
      pass.end();
    }

    // ------------------------------------------------------------------
    // Pass 2 --- pressure gradient + viscosity --- forceX / forceY
    // Timestamp slots: begin=2, end=3
    // ------------------------------------------------------------------
    {
      const passDesc: GPUComputePassDescriptor = { label: "force-pass" };
      if (ts) {
        passDesc.timestampWrites = {
          querySet                 : this.tsQuerySet!,
          beginningOfPassWriteIndex: 2,
          endOfPassWriteIndex      : 3,
        };
      }
      const pass = encoder.beginComputePass(passDesc);
      pass.setPipeline(this.forcePipeline);
      pass.setBindGroup(0, this.uniformBG);
      pass.setBindGroup(1, this.forceParticleBG);
      pass.setBindGroup(2, neighborBG);
      pass.setBindGroup(3, boundaryBG);
      pass.dispatchWorkgroups(wg);
      pass.end();
    }

    // ------------------------------------------------------------------
    // Pass 3 --- symplectic Euler integration + boundary clamp
    // Timestamp slots: begin=4, end=5
    // ------------------------------------------------------------------
    {
      const passDesc: GPUComputePassDescriptor = { label: "integrate-pass" };
      if (ts) {
        passDesc.timestampWrites = {
          querySet                 : this.tsQuerySet!,
          beginningOfPassWriteIndex: 4,
          endOfPassWriteIndex      : 5,
        };
      }
      const pass = encoder.beginComputePass(passDesc);
      pass.setPipeline(this.integratePipeline);
      pass.setBindGroup(0, this.uniformBG);
      pass.setBindGroup(1, this.integrateParticleBG);
      pass.dispatchWorkgroups(wg);
      pass.end();
    }

    // ------------------------------------------------------------------
    // Timestamp resolve: copy raw u64 ns values from QuerySet --- GPU buffer
    // ------------------------------------------------------------------
    if (ts) {
      encoder.resolveQuerySet(
        this.tsQuerySet!,
        0,
        SPHGPUOrchestrator.TS_SLOTS,
        this.tsResolveBuf!,
        0,
      );
      // Copy to MAP_READ staging buffer so the CPU can read asynchronously.
      encoder.copyBufferToBuffer(
        this.tsResolveBuf!, 0,
        this.tsMapBuf!,    0,
        SPHGPUOrchestrator.TS_SLOTS * 8,
      );
    }

    dev.queue.submit([encoder.finish()]);

    // ------------------------------------------------------------------
    // Async readback: map the staging buffer and append a PerfEntry.
    // Fire-and-forget so tick() stays synchronous.
    // ------------------------------------------------------------------
    if (ts) {
      this._readbackTimestamps(frame).catch(() => { /* profiling is best-effort */ });
    }
  }

  // -------------------------------------------------------------------------
  /**
   * Map the timestamp staging buffer, convert raw u64 ns --- ms, push a
   * PerfEntry into `perfLog`, then unmap.
   *
   * Called asynchronously after each tick when timestamp-query is available.
   */
  private async _readbackTimestamps(frame: number): Promise<void> {
    const mapBuf = this.tsMapBuf!;

    // Wait for the GPU to finish writing the resolve output.
    await mapBuf.mapAsync(GPUMapMode.READ);

    try {
      const raw = new BigUint64Array(mapBuf.getMappedRange());
      // Convert nanoseconds (BigInt) --- milliseconds (float).
      const densityMs    = Number(raw[1] - raw[0]) / 1_000_000;
      const forceMs      = Number(raw[3] - raw[2]) / 1_000_000;
      const integrateMs  = Number(raw[5] - raw[4]) / 1_000_000;
      const totalMs      = densityMs + forceMs + integrateMs;

      const entry: PerfEntry = { frame, densityMs, forceMs, integrateMs, totalMs };

      // Maintain the ring buffer.
      if (this.perfLog.length >= SPHGPUOrchestrator.PERF_LOG_MAX) {
        this.perfLog.shift();
      }
      this.perfLog.push(entry);
    } finally {
      mapBuf.unmap();
    }
  }

  // -------------------------------------------------------------------------
  /**
   * Return a human-readable summary of the last N entries in `perfLog`.
   * Useful for on-screen overlays or console debugging.
   *
   * @param n - number of recent frames to average (default: all available)
   */
  perfSummary(n?: number): string {
    const log = n !== undefined ? this.perfLog.slice(-n) : this.perfLog;
    if (log.length === 0) {
      return this.tsEnabled
        ? "GPU profiling enabled --- no data yet"
        : "GPU profiling unavailable (timestamp-query feature not present)";
    }
    const avg = (fn: (e: PerfEntry) => number): string =>
      (log.reduce((s, e) => s + fn(e), 0) / log.length).toFixed(3);

    return [
      `[SPH GPU perf --- avg over ${log.length} frame(s)]`,
      `  density   : ${avg(e => e.densityMs)} ms`,
      `  force     : ${avg(e => e.forceMs)} ms`,
      `  integrate : ${avg(e => e.integrateMs)} ms`,
      `  total     : ${avg(e => e.totalMs)} ms`,
    ].join("\n");
  }

  // -------------------------------------------------------------------------
  /**
   * Build the BGLs and pipeline for the spatial-hash pass-1 (count) shader.
   * Called once during init(); separated for clarity.
   */
  private createHashCountPipeline(): void {
    const dev = this.device;

    // group 0 --- HashUniforms (uniform buffer)
    const hashUniformBGL = dev.createBindGroupLayout({
      label  : "hash-uniform-bgl",
      entries: [{
        binding   : 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer    : { type: "uniform" },
      }],
    });

    // group 1 --- posX (r), posY (r)
    this.hashPosBGL = dev.createBindGroupLayout({
      label  : "hash-pos-bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ],
    });

    // group 2 --- cellCount (rw, atomic<u32>)
    this.hashCountBGL = dev.createBindGroupLayout({
      label  : "hash-count-bgl",
      entries: [{
        binding   : 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer    : { type: "storage" },   // read_write for atomics
      }],
    });

    this.hashCountPipeline = this.makePipeline(
      "hash-count",
      HASH_COUNT_SHADER,
      [hashUniformBGL, this.hashPosBGL, this.hashCountBGL],
    );

    // The hashUniformBG uses the dedicated hashUniformBuf
    // (hashPosBG and hashCountBG are created in init() / updateHashCountBG)
    this.hashUniformBG = dev.createBindGroup({
      label  : "hash-uniform-bg",
      layout : hashUniformBGL,
      entries: [{ binding: 0, resource: { buffer: this.hashUniformBuf } }],
    });
  }

  // -------------------------------------------------------------------------
  /**
   * Dirty-check the cellCount buffer and rebuild the bind group if needed.
   */
  private updateHashCountBG(cellCountBuf: any /*GPUBuffer*/): void {
    if (this.hashCountBG !== null && cellCountBuf === this.lastCellCountBuf) { return; }

    this.hashCountBG = this.device.createBindGroup({
      label  : "hash-count-bg",
      layout : this.hashCountBGL,
      entries: [{ binding: 0, resource: { buffer: cellCountBuf } }],
    });
    this.lastCellCountBuf = cellCountBuf;
  }

  // -------------------------------------------------------------------------
  /**
   * Encode and submit the spatial-hash pass 1 (count) compute pass.
   *
   * Each particle atomically increments `cellCountBuf[hash(particle)]`.
   * The caller is responsible for:
   *   - zeroing `cellCountBuf` before calling this (e.g. with copyBufferToBuffer
   *     from a zeroed source, or a clearBuffer if the device supports it).
   *   - running a prefix-sum / exclusive scan on `cellCountBuf` afterwards
   *     to produce start offsets for the scatter pass (pass 2).
   *
   * @param count         - number of fluid particles
   * @param tableSize     - hash-table size (e.g. next power-of-two --- 2--count)
   * @param h             - smoothing length / cell size
   * @param domainW       - simulation domain width
   * @param domainH       - simulation domain height
   * @param cellCountBuf  - GPU buffer of `tableSize` -- u32 (atomic, zeroed)
   */
  dispatchHashCount(
    count       : number,
    tableSize   : number,
    h           : number,
    domainW     : number,
    domainH     : number,
    cellCountBuf: any /*GPUBuffer*/,
  ): void {
    const dev = this.device;

    // Upload HashUniforms (32 bytes)
    // Layout: f32 h, f32 domainW, f32 domainH, u32 count, u32 tableSize, u32--3 pad
    const data = new ArrayBuffer(32);
    const f    = new Float32Array(data);
    const u    = new Uint32Array(data);
    f[0] = h;
    f[1] = domainW;
    f[2] = domainH;
    u[3] = count;
    u[4] = tableSize;
    u[5] = 0; u[6] = 0; u[7] = 0;
    dev.queue.writeBuffer(this.hashUniformBuf, 0, data);

    // Refresh cellCount bind group if the buffer changed
    this.updateHashCountBG(cellCountBuf);

    const wg      = Math.ceil(count / WORKGROUP_SIZE);
    const encoder = dev.createCommandEncoder({ label: "hash-count-frame" });

    {
      const pass = encoder.beginComputePass({ label: "hash-count-pass" });
      pass.setPipeline(this.hashCountPipeline);
      pass.setBindGroup(0, this.hashUniformBG);
      pass.setBindGroup(1, this.hashPosBG);
      pass.setBindGroup(2, this.hashCountBG!);
      pass.dispatchWorkgroups(wg);
      pass.end();
    }

    dev.queue.submit([encoder.finish()]);
  }

  // -------------------------------------------------------------------------
  /**
   * Build the BGLs and both pipelines needed for the two-phase Blelloch
   * exclusive prefix-sum (scan) over the cellCount array.
   *
   * Layout of bind groups used by both scan shaders:
   *   group 0 --- ScanUniforms  { n, blockOffset, _pad0, _pad1 }  (uniform)
   *   group 1 --- data[]        (read_write u32 --- the cellCount array)
   *   group 2 --- blockSum[]    (read_write u32 for pass-1, read for pass-2)
   */
  private createPrefixSumPipeline(): void {
    const dev = this.device;

    // group 0 --- scan uniforms
    this.scanUniformBGL = dev.createBindGroupLayout({
      label  : "scan-uniform-bgl",
      entries: [{
        binding   : 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer    : { type: "uniform" },
      }],
    });

    // group 1 --- data buffer (read_write)
    this.scanDataBGL = dev.createBindGroupLayout({
      label  : "scan-data-bgl",
      entries: [{
        binding   : 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer    : { type: "storage" },
      }],
    });

    // group 2 --- blockSum buffer (read_write for local scan, read for add pass)
    this.scanBlockBGL = dev.createBindGroupLayout({
      label  : "scan-block-bgl",
      entries: [{
        binding   : 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer    : { type: "storage" },
      }],
    });

    const bgls = [this.scanUniformBGL, this.scanDataBGL, this.scanBlockBGL];
    this.prefixSumPipeline    = this.makePipeline("prefix-sum",     PREFIX_SUM_SHADER,     bgls);
    this.prefixSumAddPipeline = this.makePipeline("prefix-sum-add", PREFIX_SUM_ADD_SHADER, bgls);

    // Static bind group for the scan uniform buffer
    this.scanUniformBG = dev.createBindGroup({
      label  : "scan-uniform-bg",
      layout : this.scanUniformBGL,
      entries: [{ binding: 0, resource: { buffer: this.scanUniformBuf } }],
    });
  }

  // -------------------------------------------------------------------------
  /**
   * Encode a full Blelloch exclusive prefix-sum over `cellCountBuf` in-place.
   *
   * This is the GPU-side "pass 1.5" that converts per-cell particle counts
   * (produced by `dispatchHashCount`) into cell-start offsets ready for the
   * scatter pass (pass 2).
   *
   * The algorithm runs a standard two-phase Blelloch scan:
   *   Phase A --- local scan  : each workgroup independently scans 512 elements
   *                           and writes its total to `blockSum[wg]`.
   *   Phase B --- block scan  : single-workgroup scan over `blockSum[]` itself.
   *   Phase C --- add pass    : every workgroup adds its block-offset back into
   *                           its 512-element window of `data[]`.
   *
   * The entire sequence is encoded into a **single** GPUCommandEncoder for
   * maximum efficiency; the caller submits the encoder's result.
   *
   * @param n             - number of elements in cellCountBuf (--- 1)
   * @param cellCountBuf  - GPU buffer of `n` u32 values to scan in-place
   * @param encoder       - command encoder to append compute passes into;
   *                        if omitted a new one is created and submitted
   */
  dispatchPrefixSum(
    n            : number,
    cellCountBuf : any /*GPUBuffer*/,
    encoder?     : any /*GPUCommandEncoder*/,
  ): void {
    const dev          = this.device;
    const BLOCK_ELEMS  = 512;                              // 2 -- SCAN_BLOCK (workgroup)
    const numBlocks    = Math.ceil(n / BLOCK_ELEMS);       // workgroups for phase A
    const ownEncoder   = encoder === undefined;
    const enc          = ownEncoder
      ? dev.createCommandEncoder({ label: "prefix-sum-frame" })
      : encoder;

    // ------ Lazily resize the blockSum scratch buffer ---------------------------------------------------------------------------
    const neededBytes = Math.max(numBlocks, 1) * 4;       // u32 per block
    if (this.scanBlockSumBuf === null || this.scanBlockSumCapacity < neededBytes) {
      this.scanBlockSumBuf?.destroy();
      this.scanBlockSumBuf = dev.createBuffer({
        label : "scan-block-sum",
        size  : neededBytes,
        usage : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.scanBlockSumCapacity = neededBytes;
      this.scanBlockSumBG       = null;   // force BG rebuild below
      this.scanDataBG           = null;
    }

    // ------ Rebuild data bind group if cellCountBuf changed ---------------------------------------------------------
    if (this.scanDataBG === null || cellCountBuf !== this.lastScanDataBuf) {
      this.scanDataBG = dev.createBindGroup({
        label  : "scan-data-bg",
        layout : this.scanDataBGL,
        entries: [{ binding: 0, resource: { buffer: cellCountBuf } }],
      });
      this.lastScanDataBuf = cellCountBuf;
    }

    // ------ Rebuild blockSum bind group if the scratch buffer was reallocated ------
    if (this.scanBlockSumBG === null) {
      this.scanBlockSumBG = dev.createBindGroup({
        label  : "scan-block-sum-bg",
        layout : this.scanBlockBGL,
        entries: [{ binding: 0, resource: { buffer: this.scanBlockSumBuf! } }],
      });
    }

    // ------ Upload ScanUniforms ---------------------------------------------------------------------------------------------------------------------------------------------
    const uData = new Uint32Array([n, 0, 0, 0]);
    dev.queue.writeBuffer(this.scanUniformBuf, 0, uData);

    const dataBG      = this.scanDataBG!;
    const blockSumBG  = this.scanBlockSumBG!;

    // ------ Phase A: local Blelloch scan per workgroup ------------------------------------------------------------------------
    {
      const pass = enc.beginComputePass({ label: "prefix-sum-local" });
      pass.setPipeline(this.prefixSumPipeline);
      pass.setBindGroup(0, this.scanUniformBG);
      pass.setBindGroup(1, dataBG);
      pass.setBindGroup(2, blockSumBG);
      pass.dispatchWorkgroups(numBlocks);
      pass.end();
    }

    // ------ Phase B: scan the blockSum array (single workgroup, recursive) ------------
    if (numBlocks > 1) {
      // Re-use the same pipeline but now scanning blockSum[] into itself.
      // We need a temporary BG that points group-1 at blockSum and group-2 at
      // a dummy 1-element buffer (the single-block blockSum of blockSum).
      const blockBlockSumBuf = dev.createBuffer({
        label : "scan-block-block-sum",
        size  : 4,
        usage : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      const blockBlockSumBG = dev.createBindGroup({
        label  : "scan-block-block-sum-bg",
        layout : this.scanBlockBGL,
        entries: [{ binding: 0, resource: { buffer: blockBlockSumBuf } }],
      });

      // Re-upload uniforms with n = numBlocks
      const uData2 = new Uint32Array([numBlocks, 0, 0, 0]);
      dev.queue.writeBuffer(this.scanUniformBuf, 0, uData2);

      {
        const pass = enc.beginComputePass({ label: "prefix-sum-block-scan" });
        pass.setPipeline(this.prefixSumPipeline);
        pass.setBindGroup(0, this.scanUniformBG);
        pass.setBindGroup(1, blockSumBG);          // blockSum[] as the data
        pass.setBindGroup(2, blockBlockSumBG);      // ignored (single block)
        pass.dispatchWorkgroups(1);
        pass.end();
      }

      // Restore uniform n for the add pass
      dev.queue.writeBuffer(this.scanUniformBuf, 0, uData);

      // ------ Phase C: add block offsets back into each window ---------------------------------------------------
      {
        const pass = enc.beginComputePass({ label: "prefix-sum-add" });
        pass.setPipeline(this.prefixSumAddPipeline);
        pass.setBindGroup(0, this.scanUniformBG);
        pass.setBindGroup(1, dataBG);
        pass.setBindGroup(2, blockSumBG);
        pass.dispatchWorkgroups(numBlocks);
        pass.end();
      }

      // The temp buffer will be GC'd; schedule a microTask destroy to be safe.
      // (WebGPU buffers are GC'd automatically, but an explicit destroy is good practice.)
      Promise.resolve().then(() => blockBlockSumBuf.destroy());
    }

    if (ownEncoder) {
      dev.queue.submit([enc.finish()]);
    }
  }

  // =========================================================================
  // Public encode-style API used by SPHWorld and world-stepper DFSPH loop
  // =========================================================================

  /**
   * Async init shim --- the constructor already performs synchronous
   * initialisation; this exists so callers that `await orchestrator.init()`
   * continue to work after refactors.
   */
  async init(): Promise<void> { /* GPU pipelines already built in constructor */ }

  // ------ Neighbor / boundary upload ------------------------------------------------------------------------------------------------------------------------------

  /**
   * Upload neighbour CSR lists built on the CPU into GPU storage buffers and
   * refresh the dynamic neighbor bind group.
   *
   * @param neighborLists - per-particle array-of-arrays of neighbour indices
   * @param n             - number of fluid particles
   */
  uploadNeighborLists(
    neighborLists: number[][],
    n: number,
  ): void {
    const dev = this.device;

    // Build flat CSR arrays
    const rowPtr: number[]  = [0];
    const neighborData: number[] = [];
    for (let i = 0; i < n; i++) {
      for (const j of neighborLists[i]) {
        neighborData.push(j);
      }
      rowPtr.push(neighborData.length);
    }

    const rowPtrArr      = new Int32Array(rowPtr);
    const neighborArr    = neighborData.length > 0
      ? new Int32Array(neighborData)
      : new Int32Array(1);

    const neighborBuf = dev.createBuffer({
      label: "neighbor-data-upload",
      size : Math.max(neighborArr.byteLength, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const rowPtrBuf = dev.createBuffer({
      label: "row-ptr-upload",
      size : Math.max(rowPtrArr.byteLength, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    dev.queue.writeBuffer(neighborBuf, 0, neighborArr);
    dev.queue.writeBuffer(rowPtrBuf,   0, rowPtrArr);

    this.updateNeighborBG({ neighborBuf, rowPtrBuf });

    // Destroy the old staging buffers on the next microtask (GPU queue is async)
    Promise.resolve().then(() => { neighborBuf.destroy(); rowPtrBuf.destroy(); });
  }

  // ------ Per-pass encode helpers (share an existing GPUCommandEncoder) ------------------------

  /**
   * Encode the density + pressure compute pass into `encoder`.
   * Precondition: `uploadNeighborLists` must have been called this frame.
   *
   * @param encoder - command encoder to append the pass into
   * @param n       - number of fluid particles
   */
  encodeDensityPressure(encoder: any /*GPUCommandEncoder*/, n: number): void {
    const wg = Math.ceil(n / WORKGROUP_SIZE);

    const pass = encoder.beginComputePass({ label: "dfsph-density-pressure" });
    pass.setPipeline(this.densityPipeline);
    pass.setBindGroup(0, this.uniformBG);
    pass.setBindGroup(1, this.densityParticleBG);
    pass.setBindGroup(2, this.neighborBG!);
    pass.setBindGroup(3, this._getOrCreateEmptyBoundaryBG());
    pass.dispatchWorkgroups(wg);
    pass.end();
  }

  /**
   * Encode the pressure-gradient + viscosity force compute pass into `encoder`.
   * Precondition: density pass must have been dispatched this frame.
   *
   * @param encoder - command encoder to append the pass into
   * @param n       - number of fluid particles
   */
  encodeForces(encoder: any /*GPUCommandEncoder*/, n: number): void {
    const wg = Math.ceil(n / WORKGROUP_SIZE);

    const pass = encoder.beginComputePass({ label: "dfsph-forces" });
    pass.setPipeline(this.forcePipeline);
    pass.setBindGroup(0, this.uniformBG);
    pass.setBindGroup(1, this.forceParticleBG);
    pass.setBindGroup(2, this.neighborBG!);
    pass.setBindGroup(3, this._getOrCreateEmptyBoundaryBG());
    pass.dispatchWorkgroups(wg);
    pass.end();
  }

  /**
   * Encode the symplectic Euler integration + boundary clamp pass into `encoder`.
   * Also writes the per-frame `dt` into the uniform buffer.
   *
   * @param encoder - command encoder to append the pass into
   * @param n       - number of fluid particles
   * @param dt      - current frame --t (seconds)
   */
  encodeIntegrate(encoder: any /*GPUCommandEncoder*/, n: number, dt: number): void {
    // Patch `dt` in the uniform buffer (offset 5 -- 4 = byte 20)
    const dtBuf = new Float32Array([dt]);
    this.device.queue.writeBuffer(this.uniformBuf, 20, dtBuf);

    const wg = Math.ceil(n / WORKGROUP_SIZE);

    const pass = encoder.beginComputePass({ label: "dfsph-integrate" });
    pass.setPipeline(this.integratePipeline);
    pass.setBindGroup(0, this.uniformBG);
    pass.setBindGroup(1, this.integrateParticleBG);
    pass.dispatchWorkgroups(wg);
    pass.end();
  }

  // ------ Internal helpers ---------------------------------------------------------------------------------------------------------------------------------------------------------------

  /**
   * Returns a lazily-created empty boundary bind group (zero particles) so
   * the density and force shaders can still bind group 3 without a real
   * boundary buffer.
   */
  private _emptyBoundaryBG: any /*GPUBindGroup*/ | null = null;
  private _emptyBoundaryBuf: any /*GPUBuffer*/ | null    = null;

  private _getOrCreateEmptyBoundaryBG(): any /*GPUBindGroup*/ {
    if (this._emptyBoundaryBG) return this._emptyBoundaryBG;

    this._emptyBoundaryBuf = this.device.createBuffer({
      label: "boundary-empty",
      size : 16,          // one vec4f placeholder
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this._emptyBoundaryBG = this.device.createBindGroup({
      label  : "boundary-empty-bg",
      layout : this.boundaryBGL,
      entries: [{ binding: 0, resource: { buffer: this._emptyBoundaryBuf } }],
    });
    return this._emptyBoundaryBG;
  }

  // =========================================================================

  // -------------------------------------------------------------------------
  /** Release GPU resources owned by this orchestrator. */
  destroy(): void {
    this.uniformBuf.destroy();
    this.hashUniformBuf.destroy();
    this.scanUniformBuf.destroy();
    this.scanBlockSumBuf?.destroy();
    this._emptyBoundaryBuf?.destroy();
    this._emptyBoundaryBuf  = null;
    this._emptyBoundaryBG   = null;
    // GPUComputePipelines and GPUBindGroupLayouts are GC'd by the device;
    // no explicit destroy() method exists for them in the WebGPU spec.
    this.neighborBG       = null;
    this.boundaryBG       = null;
    this.hashCountBG      = null;
    this.scanDataBG       = null;
    this.scanBlockSumBG   = null;
    this.scanBlockSumBuf  = null;
    this.lastNeighborBuf  = null;
    this.lastRowPtrBuf    = null;
    this.lastBoundaryBuf  = null;
    this.lastCellCountBuf = null;
    this.lastScanDataBuf  = null;
    // Timestamp profiling resources
    this.tsQuerySet?.destroy();
    this.tsResolveBuf?.destroy();
    this.tsMapBuf?.destroy();
    this.tsQuerySet   = null;
    this.tsResolveBuf = null;
    this.tsMapBuf     = null;
  }
}
