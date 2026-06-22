/**
 * webgpu-sph-compute.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * WebGPU compute shader accelerated DFSPH solver.
 *
 * Mirrors the CPU-side dfsph-solver.ts algorithm (Bender & Koschier 2017)
 * but executes entirely on the GPU via three compute-shader stages:
 *
 *   1. spatial_hash_build — parallel Teschner hash + prefix-sum for grid
 *   2. density_pressure   — compute density + DFSPH α factor
 *   3. force_integrate    — non-pressure forces + Euler integration
 *
 * The iterative pressure / divergence-free correction loops dispatch
 * density_pressure repeatedly on the GPU, reading back only the scalar
 * average-error to decide convergence on the CPU.
 *
 * References:
 *   - SPHGPUOrchestrator.ts  (existing WebGPU infrastructure in this repo)
 *   - dfsph-solver.ts        (CPU reference implementation)
 *   - sph-kernels.ts         (kernel math — reproduced in WGSL)
 * ─────────────────────────────────────────────────────────────────────────────
 */

import type { SPHConfig } from "./sph-kernels";

// ─── Constants ───────────────────────────────────────────────────────────────

const WG_SIZE = 256;

/**
 * Per-particle stride in the upload/readback Float32Array.
 *
 * Layout per particle (14 floats):
 *   [0]  x          [1]  y
 *   [2]  vx         [3]  vy
 *   [4]  ax         [5]  ay
 *   [6]  density    [7]  pressure
 *   [8]  factor     [9]  densityAdv
 *   [10] kappa      [11] kappaV
 *   [12] species    [13] _pad
 */
const PARTICLE_STRIDE = 14;

// Hash-table size is next power-of-two ≥ 2 × maxParticles.
function hashTableSize(maxP: number): number {
  let s = 1;
  while (s < maxP * 2) s <<= 1;
  return s;
}

// ─── WGSL Shaders ────────────────────────────────────────────────────────────

// Shared struct preamble injected into every shader.
const UNIFORM_STRUCT = /* wgsl */ `
struct DFSPHUniforms {
  h          : f32,    // smoothing radius
  rho0       : f32,    // rest density
  mass       : f32,    // particle mass
  dt         : f32,    // time step
  gravityX   : f32,    // gravity vector X
  gravityY   : f32,    // gravity vector Y
  count      : u32,    // number of active particles
  tableSize  : u32,    // spatial hash table size
  domainW    : f32,    // domain width  (for boundary clamp)
  domainH    : f32,    // domain height (for boundary clamp)
  _pad0      : u32,
  _pad1      : u32,
}
`;

const KERNEL_FUNCTIONS = /* wgsl */ `
// ── Cubic spline kernel (2-D, Monaghan 1992) ─────────────────────────
// Normalisation: α₂D = 40 / (7 π h²)
// Support: q ∈ [0, 1]  where q = r / h.

fn cubicAlpha(h: f32) -> f32 {
  return 40.0 / (7.0 * 3.14159265358979 * h * h);
}

fn W_cubic(r: f32, h: f32) -> f32 {
  let q = r / h;
  if (q > 1.0) { return 0.0; }
  let alpha = cubicAlpha(h);
  if (q <= 0.5) {
    let q2 = q * q;
    let q3 = q2 * q;
    return alpha * (1.0 - 6.0 * q2 + 6.0 * q3);
  } else {
    let t = 1.0 - q;
    return alpha * 2.0 * t * t * t;
  }
}

fn gradW_cubic(dx: f32, dy: f32, h: f32) -> vec2f {
  let r = sqrt(dx * dx + dy * dy);
  if (r < 1e-12) { return vec2f(0.0, 0.0); }
  let q = r / h;
  if (q > 1.0) { return vec2f(0.0, 0.0); }

  let alpha = cubicAlpha(h);
  let invR  = 1.0 / r;
  let invH  = 1.0 / h;

  var dWdr: f32;
  if (q <= 0.5) {
    dWdr = alpha * invH * (-12.0 * q + 18.0 * q * q);
  } else {
    let t = 1.0 - q;
    dWdr = alpha * invH * (-6.0 * t * t);
  }
  let scale = dWdr * invR;
  return vec2f(scale * dx, scale * dy);
}
`;

// Teschner hash function, shared between hash-build and neighbor-query shaders.
const HASH_FUNCTION = /* wgsl */ `
const P1 : u32 = 73856093u;
const P2 : u32 = 19349663u;

fn teschnerHash(px: f32, py: f32, h: f32, tableSize: u32) -> u32 {
  let ix = u32(max(floor(px / h), 0.0));
  let iy = u32(max(floor(py / h), 0.0));
  let raw = (ix * P1) ^ (iy * P2);
  return raw % tableSize;
}

fn cellKeyFromGrid(ix: i32, iy: i32, tableSize: u32) -> u32 {
  let ux = u32(max(ix, 0));
  let uy = u32(max(iy, 0));
  let raw = (ux * P1) ^ (uy * P2);
  return raw % tableSize;
}
`;

// ─── Shader 1: Spatial Hash Build ────────────────────────────────────────────
//
// Two-pass approach:
//   Pass 1a (hash_count): Each particle atomically increments cellCount[key].
//   Pass 1b (prefix_sum): Blelloch scan converts counts → start offsets.
//   Pass 1c (hash_scatter): Each particle writes itself into the sorted array
//            at the position given by atomicAdd(&cellCount[key], 1).
//
// After these three sub-passes, we have:
//   cellStart[key] = exclusive prefix-sum start index for cell `key`
//   sortedIdx[cellStart[key] .. cellStart[key+1]] = particle indices in cell

const HASH_COUNT_SHADER = /* wgsl */ `
${UNIFORM_STRUCT}

@group(0) @binding(0) var<uniform> params : DFSPHUniforms;

// Particle positions (SoA)
@group(1) @binding(0) var<storage, read> posX : array<f32>;
@group(1) @binding(1) var<storage, read> posY : array<f32>;

// Atomic cell-count table (size = tableSize)
@group(1) @binding(2) var<storage, read_write> cellCount : array<atomic<u32>>;

${HASH_FUNCTION}

@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }
  let key = teschnerHash(posX[i], posY[i], params.h, params.tableSize);
  atomicAdd(&cellCount[key], 1u);
}
`;

const HASH_SCATTER_SHADER = /* wgsl */ `
${UNIFORM_STRUCT}

@group(0) @binding(0) var<uniform> params : DFSPHUniforms;

// Particle positions
@group(1) @binding(0) var<storage, read> posX : array<f32>;
@group(1) @binding(1) var<storage, read> posY : array<f32>;

// cellStart (after prefix-sum — we atomicAdd to get per-particle slot)
@group(1) @binding(2) var<storage, read_write> cellStart : array<atomic<u32>>;

// Output: sorted particle indices
@group(1) @binding(3) var<storage, read_write> sortedIdx : array<u32>;

${HASH_FUNCTION}

@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }
  let key = teschnerHash(posX[i], posY[i], params.h, params.tableSize);
  let slot = atomicAdd(&cellStart[key], 1u);
  sortedIdx[slot] = i;
}
`;

// ─── Prefix-sum shaders (reuse Blelloch from SPHGPUOrchestrator pattern) ─────

const PREFIX_SUM_SHADER = /* wgsl */ `
const SCAN_BLOCK : u32 = 256u;

struct ScanUniforms {
  n          : u32,
  blockOffset: u32,
  _pad0      : u32,
  _pad1      : u32,
}

@group(0) @binding(0) var<uniform>             uScan    : ScanUniforms;
@group(1) @binding(0) var<storage, read_write> data     : array<u32>;
@group(2) @binding(0) var<storage, read_write> blockSum : array<u32>;

var<workgroup> temp: array<u32, 512>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid : vec3<u32>,
  @builtin(local_invocation_id)  lid : vec3<u32>,
  @builtin(workgroup_id)         wid : vec3<u32>,
) {
  let thid = lid.x;
  let base = uScan.blockOffset + wid.x * (SCAN_BLOCK * 2u);

  let ai = base + thid;
  let bi = base + thid + SCAN_BLOCK;

  temp[thid]              = select(0u, data[ai], ai < uScan.n);
  temp[thid + SCAN_BLOCK] = select(0u, data[bi], bi < uScan.n);
  workgroupBarrier();

  // Up-sweep
  var offset = 1u;
  var d = SCAN_BLOCK;
  loop {
    if (d == 0u) { break; }
    workgroupBarrier();
    if (thid < d) {
      let ai2 = offset * (2u * thid + 1u) - 1u;
      let bi2 = offset * (2u * thid + 2u) - 1u;
      temp[bi2] += temp[ai2];
    }
    offset *= 2u;
    d /= 2u;
  }

  if (thid == 0u) {
    blockSum[wid.x] = temp[SCAN_BLOCK * 2u - 1u];
    temp[SCAN_BLOCK * 2u - 1u] = 0u;
  }
  workgroupBarrier();

  // Down-sweep
  d = 1u;
  offset = SCAN_BLOCK;
  loop {
    if (d > SCAN_BLOCK) { break; }
    offset /= 2u;
    workgroupBarrier();
    if (thid < d) {
      let ai2 = offset * (2u * thid + 1u) - 1u;
      let bi2 = offset * (2u * thid + 2u) - 1u;
      let t = temp[ai2];
      temp[ai2] = temp[bi2];
      temp[bi2] += t;
    }
    d *= 2u;
  }
  workgroupBarrier();

  if (ai < uScan.n) { data[ai] = temp[thid]; }
  if (bi < uScan.n) { data[bi] = temp[thid + SCAN_BLOCK]; }
}
`;

const PREFIX_SUM_ADD_SHADER = /* wgsl */ `
const SCAN_BLOCK : u32 = 256u;

struct ScanUniforms {
  n          : u32,
  blockOffset: u32,
  _pad0      : u32,
  _pad1      : u32,
}

@group(0) @binding(0) var<uniform>             uScan    : ScanUniforms;
@group(1) @binding(0) var<storage, read_write> data     : array<u32>;
@group(2) @binding(0) var<storage, read>       blockSum : array<u32>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid : vec3<u32>,
  @builtin(local_invocation_id)  lid : vec3<u32>,
  @builtin(workgroup_id)         wid : vec3<u32>,
) {
  if (wid.x == 0u) { return; }
  let base = uScan.blockOffset + wid.x * (SCAN_BLOCK * 2u);
  let add  = blockSum[wid.x];
  let ai = base + lid.x;
  let bi = base + lid.x + SCAN_BLOCK;
  if (ai < uScan.n) { data[ai] += add; }
  if (bi < uScan.n) { data[bi] += add; }
}
`;

// ─── Shader 2: Density + DFSPH Factor ───────────────────────────────────────
//
// For each particle i:
//   1. Walk the 3×3 neighbor cells via the spatial hash
//   2. Sum density ρ_i = Σ_j m W(r_ij, h)
//   3. Compute DFSPH factor α_i = ρ_i / ( |Σ_j m ∇W_ij|² + Σ_j |m ∇W_ij|² )

const DENSITY_FACTOR_SHADER = /* wgsl */ `
${UNIFORM_STRUCT}

@group(0) @binding(0) var<uniform> params : DFSPHUniforms;

// Particle SoA buffers
@group(1) @binding(0) var<storage, read>       posX      : array<f32>;
@group(1) @binding(1) var<storage, read>       posY      : array<f32>;
@group(1) @binding(2) var<storage, read_write> density   : array<f32>;
@group(1) @binding(3) var<storage, read_write> factor    : array<f32>;

// Spatial hash (read-only after build)
@group(2) @binding(0) var<storage, read> cellStart  : array<u32>;
@group(2) @binding(1) var<storage, read> cellEnd    : array<u32>;
@group(2) @binding(2) var<storage, read> sortedIdx  : array<u32>;

${KERNEL_FUNCTIONS}
${HASH_FUNCTION}

@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }

  let xi = posX[i];
  let yi = posY[i];
  let h  = params.h;
  let m  = params.mass;
  let ts = params.tableSize;

  // Self-contribution
  var rho = m * W_cubic(0.0, h);

  // Gradient accumulation for DFSPH factor
  var sumGradX : f32 = 0.0;
  var sumGradY : f32 = 0.0;
  var sumGradSq: f32 = 0.0;

  // Grid cell of particle i
  let cix = i32(floor(xi / h));
  let ciy = i32(floor(yi / h));

  // Walk 3×3 neighborhood
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let key = cellKeyFromGrid(cix + dx, ciy + dy, ts);
      let cStart = cellStart[key];
      let cEnd   = cellEnd[key];

      for (var k = cStart; k < cEnd; k++) {
        let j = sortedIdx[k];
        if (j == i) { continue; }

        let ddx = xi - posX[j];
        let ddy = yi - posY[j];
        let r   = sqrt(ddx * ddx + ddy * ddy);

        // Density contribution
        rho += m * W_cubic(r, h);

        // Gradient contribution for DFSPH factor
        let g = gradW_cubic(ddx, ddy, h);
        let mgx = m * g.x;
        let mgy = m * g.y;
        sumGradX  += mgx;
        sumGradY  += mgy;
        sumGradSq += mgx * mgx + mgy * mgy;
      }
    }
  }

  density[i] = rho;

  // DFSPH factor: α_i = ρ_i / ( |Σ m∇W|² + Σ |m∇W|² )
  let denom = sumGradX * sumGradX + sumGradY * sumGradY + sumGradSq;
  if (denom > 1e-6) {
    factor[i] = rho / denom;
  } else {
    factor[i] = 0.0;
  }
}
`;

// ─── Shader 2b: Predicted density / divergence computation ──────────────────
//
// Computes density advection rate or velocity divergence for the pressure /
// divergence-free correction iterations.  A uniform `mode` flag selects:
//   mode == 0 → density error:  ρ*_i = ρ_i + dt * Σ_j m (v_i - v_j)·∇W_ij
//   mode == 1 → divergence:     divV_i = Σ_j m (v_i - v_j)·∇W_ij

const PREDICT_DENSITY_SHADER = /* wgsl */ `
${UNIFORM_STRUCT}

struct CorrectUniforms {
  mode : u32,   // 0 = pressure (density error), 1 = divergence
  _p0  : u32,
  _p1  : u32,
  _p2  : u32,
}

@group(0) @binding(0) var<uniform> params  : DFSPHUniforms;
@group(0) @binding(1) var<uniform> cparams : CorrectUniforms;

@group(1) @binding(0) var<storage, read>       posX       : array<f32>;
@group(1) @binding(1) var<storage, read>       posY       : array<f32>;
@group(1) @binding(2) var<storage, read>       velX       : array<f32>;
@group(1) @binding(3) var<storage, read>       velY       : array<f32>;
@group(1) @binding(4) var<storage, read>       density    : array<f32>;
@group(1) @binding(5) var<storage, read>       factorBuf  : array<f32>;
@group(1) @binding(6) var<storage, read_write> densityAdv : array<f32>;
@group(1) @binding(7) var<storage, read_write> errorBuf   : array<f32>;

@group(2) @binding(0) var<storage, read> cellStart : array<u32>;
@group(2) @binding(1) var<storage, read> cellEnd   : array<u32>;
@group(2) @binding(2) var<storage, read> sortedIdx : array<u32>;

${KERNEL_FUNCTIONS}
${HASH_FUNCTION}

@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }

  let xi  = posX[i];
  let yi  = posY[i];
  let vxi = velX[i];
  let vyi = velY[i];
  let h   = params.h;
  let m   = params.mass;
  let ts  = params.tableSize;

  let cix = i32(floor(xi / h));
  let ciy = i32(floor(yi / h));

  var drho: f32 = 0.0;

  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let key    = cellKeyFromGrid(cix + dx, ciy + dy, ts);
      let cStart = cellStart[key];
      let cEnd   = cellEnd[key];

      for (var k = cStart; k < cEnd; k++) {
        let j = sortedIdx[k];
        if (j == i) { continue; }

        let ddx = xi - posX[j];
        let ddy = yi - posY[j];
        let g   = gradW_cubic(ddx, ddy, h);
        let dvx = vxi - velX[j];
        let dvy = vyi - velY[j];
        drho += m * (dvx * g.x + dvy * g.y);
      }
    }
  }

  if (cparams.mode == 0u) {
    // Pressure mode: predicted density ρ*_i
    let rhoAdv = density[i] + params.dt * drho;
    densityAdv[i] = rhoAdv;
    let err = max(rhoAdv - params.rho0, 0.0);
    errorBuf[i] = err;
  } else {
    // Divergence mode: velocity divergence
    densityAdv[i] = drho;
    errorBuf[i] = abs(drho);
  }
}
`;

// ─── Shader 2c: Pressure / divergence-free velocity correction ──────────────
//
// Applies the DFSPH pressure acceleration to correct velocities.
// mode == 0 → density correction (κ_i = err / (dt² · α_i))
// mode == 1 → divergence correction (κV_i = divV / (dt · α_i))

const CORRECT_VELOCITY_SHADER = /* wgsl */ `
${UNIFORM_STRUCT}

struct CorrectUniforms {
  mode : u32,
  _p0  : u32,
  _p1  : u32,
  _p2  : u32,
}

@group(0) @binding(0) var<uniform> params  : DFSPHUniforms;
@group(0) @binding(1) var<uniform> cparams : CorrectUniforms;

@group(1) @binding(0) var<storage, read>       posX       : array<f32>;
@group(1) @binding(1) var<storage, read>       posY       : array<f32>;
@group(1) @binding(2) var<storage, read_write> velX       : array<f32>;
@group(1) @binding(3) var<storage, read_write> velY       : array<f32>;
@group(1) @binding(4) var<storage, read>       density    : array<f32>;
@group(1) @binding(5) var<storage, read>       factorBuf  : array<f32>;
@group(1) @binding(6) var<storage, read>       densityAdv : array<f32>;

@group(2) @binding(0) var<storage, read> cellStart : array<u32>;
@group(2) @binding(1) var<storage, read> cellEnd   : array<u32>;
@group(2) @binding(2) var<storage, read> sortedIdx : array<u32>;

${KERNEL_FUNCTIONS}
${HASH_FUNCTION}

@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }

  let xi   = posX[i];
  let yi   = posY[i];
  let h    = params.h;
  let m    = params.mass;
  let dt   = params.dt;
  let rho0 = params.rho0;
  let ts   = params.tableSize;

  let rhoI = density[i];
  let alpI = factorBuf[i];

  // Compute stiffness κ_i
  var ki: f32;
  if (cparams.mode == 0u) {
    // Pressure: κ_i = max(ρ*_i - ρ0, 0) / (dt² · α_i)
    let err = max(densityAdv[i] - rho0, 0.0);
    ki = (err / (dt * dt)) * alpI;
  } else {
    // Divergence: κV_i = divV_i / (dt · α_i)
    ki = (densityAdv[i] / dt) * alpI;
  }

  let ki_rho2 = ki / (rhoI * rhoI + 1e-12);

  let cix = i32(floor(xi / h));
  let ciy = i32(floor(yi / h));

  var pax: f32 = 0.0;
  var pay: f32 = 0.0;

  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let key    = cellKeyFromGrid(cix + dx, ciy + dy, ts);
      let cStart = cellStart[key];
      let cEnd   = cellEnd[key];

      for (var k = cStart; k < cEnd; k++) {
        let j = sortedIdx[k];
        if (j == i) { continue; }

        let ddx = xi - posX[j];
        let ddy = yi - posY[j];
        let g   = gradW_cubic(ddx, ddy, h);

        let rhoJ = density[j];
        let alpJ = factorBuf[j];

        var kj: f32;
        if (cparams.mode == 0u) {
          let errJ = max(densityAdv[j] - rho0, 0.0);
          kj = (errJ / (dt * dt)) * alpJ;
        } else {
          kj = (densityAdv[j] / dt) * alpJ;
        }
        let kj_rho2 = kj / (rhoJ * rhoJ + 1e-12);

        let coeff = -m * (ki_rho2 + kj_rho2);
        pax += coeff * g.x;
        pay += coeff * g.y;
      }
    }
  }

  // Update velocity
  velX[i] += dt * pax;
  velY[i] += dt * pay;
}
`;

// ─── Shader 3: Non-pressure Forces + Euler Integration ──────────────────────
//
// Applies gravity (+ any externally accumulated ax/ay) and integrates
// position via symplectic Euler.  Also performs boundary clamping.

const FORCE_INTEGRATE_SHADER = /* wgsl */ `
${UNIFORM_STRUCT}

@group(0) @binding(0) var<uniform> params : DFSPHUniforms;

@group(1) @binding(0) var<storage, read_write> posX : array<f32>;
@group(1) @binding(1) var<storage, read_write> posY : array<f32>;
@group(1) @binding(2) var<storage, read_write> velX : array<f32>;
@group(1) @binding(3) var<storage, read_write> velY : array<f32>;
@group(1) @binding(4) var<storage, read_write> accX : array<f32>;
@group(1) @binding(5) var<storage, read_write> accY : array<f32>;

const MAX_VEL : f32 = 100.0;
const DAMPING : f32 = 0.5;

@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }

  let dt = params.dt;

  // Apply gravity + accumulated non-pressure acceleration
  let ax = accX[i] + params.gravityX;
  let ay = accY[i] + params.gravityY;

  var vx = velX[i] + dt * ax;
  var vy = velY[i] + dt * ay;

  // Reset accelerations
  accX[i] = 0.0;
  accY[i] = 0.0;

  // Clamp velocity magnitude
  let speed = sqrt(vx * vx + vy * vy);
  if (speed > MAX_VEL) {
    vx *= MAX_VEL / speed;
    vy *= MAX_VEL / speed;
  }

  // Update position (symplectic Euler)
  var px = posX[i] + vx * dt;
  var py = posY[i] + vy * dt;

  // Boundary clamp (box: [margin, domain - margin])
  let margin = params.h * 0.5;

  if (px < margin) {
    px = margin;
    vx = abs(vx) * DAMPING;
  } else if (px > params.domainW - margin) {
    px = params.domainW - margin;
    vx = -abs(vx) * DAMPING;
  }

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

// ─── Shader: Apply gravity to velocity (pre-solve step) ─────────────────────
// Separate from the integration pass so the DFSPH correction loop operates
// on the gravity-predicted velocity before position update.

const APPLY_GRAVITY_SHADER = /* wgsl */ `
${UNIFORM_STRUCT}

@group(0) @binding(0) var<uniform> params : DFSPHUniforms;

@group(1) @binding(0) var<storage, read_write> velX : array<f32>;
@group(1) @binding(1) var<storage, read_write> velY : array<f32>;
@group(1) @binding(2) var<storage, read_write> accX : array<f32>;
@group(1) @binding(3) var<storage, read_write> accY : array<f32>;

@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }

  let dt = params.dt;
  velX[i] += dt * (accX[i] + params.gravityX);
  velY[i] += dt * (accY[i] + params.gravityY);

  // Reset accelerations
  accX[i] = 0.0;
  accY[i] = 0.0;
}
`;

// ─── Shader: Position update (post-solve) ───────────────────────────────────

const UPDATE_POSITION_SHADER = /* wgsl */ `
${UNIFORM_STRUCT}

@group(0) @binding(0) var<uniform> params : DFSPHUniforms;

@group(1) @binding(0) var<storage, read_write> posX : array<f32>;
@group(1) @binding(1) var<storage, read_write> posY : array<f32>;
@group(1) @binding(2) var<storage, read>       velX : array<f32>;
@group(1) @binding(3) var<storage, read>       velY : array<f32>;

const MAX_VEL : f32 = 100.0;
const DAMPING : f32 = 0.5;

@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }

  let dt = params.dt;
  var vx = velX[i];
  var vy = velY[i];

  var px = posX[i] + vx * dt;
  var py = posY[i] + vy * dt;

  // Boundary clamp
  let margin = params.h * 0.5;
  if (px < margin)                     { px = margin; }
  else if (px > params.domainW - margin) { px = params.domainW - margin; }
  if (py < margin)                     { py = margin; }
  else if (py > params.domainH - margin) { py = params.domainH - margin; }

  posX[i] = px;
  posY[i] = py;
}
`;

// ─── Shader: Error reduction (parallel sum → single value) ──────────────────
// Sums errorBuf[0..count-1] into reductionOut[0] for convergence check.

const REDUCE_ERROR_SHADER = /* wgsl */ `
struct ReduceUniforms {
  count : u32,
  _p0   : u32,
  _p1   : u32,
  _p2   : u32,
}

@group(0) @binding(0) var<uniform>             rParams    : ReduceUniforms;
@group(1) @binding(0) var<storage, read>       errorBuf   : array<f32>;
@group(1) @binding(1) var<storage, read_write> reduceOut  : array<atomic<u32>>;

// Bit-cast f32 → u32 for atomicAdd approximation.
// We use integer atomicAdd on the bit-reinterpreted float; the caller
// reinterprets the final u32 sum back to f32.  This works because all error
// values are non-negative and we accumulate the sum in integer space on the
// bit-pattern of their IEEE-754 representation — wait, that's not correct.
//
// Instead we use workgroup shared memory reduction + a single atomicAdd of
// the workgroup partial sum encoded as u32 bits.  To make the final global
// sum exact we would need f32 atomics (not in WGSL base).  So we do a
// two-level reduction:
//   Level 1: each workgroup reduces its tile via shared memory → one f32
//   Level 2: atomicAdd the u32-reinterpreted partial sums (NOT correct for
//            general floats, but our values are all positive and of similar
//            magnitude, so we accept a small error).
//
// Actually, the simplest correct approach: reduce within each workgroup to
// shared memory, then store per-workgroup results into an output array that
// the CPU reads back and sums.  Let's do that.

var<workgroup> shared: array<f32, ${WG_SIZE}>;

@compute @workgroup_size(${WG_SIZE})
fn main(
  @builtin(global_invocation_id) gid : vec3<u32>,
  @builtin(local_invocation_id)  lid : vec3<u32>,
  @builtin(workgroup_id)         wid : vec3<u32>,
) {
  let i = gid.x;
  shared[lid.x] = select(0.0, errorBuf[i], i < rParams.count);
  workgroupBarrier();

  // Tree reduction in shared memory
  for (var stride = ${WG_SIZE}u / 2u; stride > 0u; stride /= 2u) {
    if (lid.x < stride) {
      shared[lid.x] += shared[lid.x + stride];
    }
    workgroupBarrier();
  }

  // Thread 0 of each workgroup writes the partial sum
  if (lid.x == 0u) {
    // Store as bit-reinterpreted u32 into the output array (one per workgroup)
    // The CPU will read these f32 partials back.
    // We write into reduceOut as raw u32 bits of the f32 partial sum.
    reduceOut[wid.x] = bitcast<u32>(shared[0]);
  }
}
`;

// ─── Shader: Compute cellEnd from cellStart ─────────────────────────────────
// After the scatter pass corrupts cellStart (atomic increments), we need
// cellEnd.  But we stored the original prefix-sum in a separate copy.
// Strategy: store prefix-sum in cellStart (backup), scatter uses a *copy*
// of cellStart.  Actually, simpler: we keep TWO copies of the cell offsets.
//   cellStartOrig[k] = exclusive prefix sum (read-only after scan)
//   cellStart[k]     = modified by scatter (atomic increments)
// Then cellEnd[k] = cellStart[k] (after scatter, this is start + count = end).
//
// Even simpler: just store count + 1 in the prefix-sum so cellEnd is
// cellStart of the *next* key.  That's what we'll do — cellEnd[k] = cellStart[k+1].
// We allocate tableSize + 1 elements for cellStart.

// ─── WebGPUSPHCompute class ─────────────────────────────────────────────────

export class WebGPUSPHCompute {
  private readonly device: GPUDevice;
  private readonly maxParticles: number;
  private readonly tblSize: number;
  private count = 0;

  // ── GPU Buffers (SoA layout) ────────────────────────────────────────
  private posXBuf!:       GPUBuffer;
  private posYBuf!:       GPUBuffer;
  private velXBuf!:       GPUBuffer;
  private velYBuf!:       GPUBuffer;
  private accXBuf!:       GPUBuffer;
  private accYBuf!:       GPUBuffer;
  private densityBuf!:    GPUBuffer;
  private factorBuf!:     GPUBuffer;
  private densityAdvBuf!: GPUBuffer;
  private errorBuf!:      GPUBuffer;

  // ── Spatial hash buffers ────────────────────────────────────────────
  private cellCountBuf!:   GPUBuffer;   // tableSize u32 — atomics
  private cellStartBuf!:   GPUBuffer;   // (tableSize + 1) u32 — after scan
  private sortedIdxBuf!:   GPUBuffer;   // maxParticles u32

  // ── Prefix-sum scratch ──────────────────────────────────────────────
  private scanBlockSumBuf!: GPUBuffer;

  // ── Uniform buffers ────────────────────────────────────────────────
  private uniformBuf!:        GPUBuffer;
  private correctUniformBuf!: GPUBuffer;
  private scanUniformBuf!:    GPUBuffer;
  private reduceUniformBuf!:  GPUBuffer;

  // ── Reduction readback ─────────────────────────────────────────────
  private reduceOutBuf!:   GPUBuffer;   // workgroup partial sums
  private reduceMapBuf!:   GPUBuffer;   // MAP_READ staging

  // ── Pipelines ──────────────────────────────────────────────────────
  private hashCountPipeline!:      GPUComputePipeline;
  private hashScatterPipeline!:    GPUComputePipeline;
  private prefixSumPipeline!:      GPUComputePipeline;
  private prefixSumAddPipeline!:   GPUComputePipeline;
  private densityFactorPipeline!:  GPUComputePipeline;
  private predictDensityPipeline!: GPUComputePipeline;
  private correctVelocityPipeline!:GPUComputePipeline;
  private applyGravityPipeline!:   GPUComputePipeline;
  private updatePositionPipeline!: GPUComputePipeline;
  private reduceErrorPipeline!:    GPUComputePipeline;

  // ── Bind Group Layouts ─────────────────────────────────────────────
  private uniformBGL!:          GPUBindGroupLayout;
  private uniformCorrectBGL!:   GPUBindGroupLayout;
  private hashBGL!:             GPUBindGroupLayout;
  private hashScatterBGL!:      GPUBindGroupLayout;
  private scanUniformBGL!:      GPUBindGroupLayout;
  private scanDataBGL!:         GPUBindGroupLayout;
  private scanBlockBGL!:        GPUBindGroupLayout;
  private densityBGL!:          GPUBindGroupLayout;
  private hashReadBGL!:         GPUBindGroupLayout;
  private predictBGL!:          GPUBindGroupLayout;
  private correctBGL!:          GPUBindGroupLayout;
  private gravityBGL!:          GPUBindGroupLayout;
  private posUpdateBGL!:        GPUBindGroupLayout;
  private reduceUniformBGL!:    GPUBindGroupLayout;
  private reduceBGL!:           GPUBindGroupLayout;

  // ── Cached Bind Groups ─────────────────────────────────────────────
  private uniformBG!:          GPUBindGroup;
  private uniformCorrectBG!:   GPUBindGroup;
  private hashBG!:             GPUBindGroup;
  private hashScatterBG!:      GPUBindGroup;
  private scanUniformBG!:      GPUBindGroup;
  private scanDataBG!:         GPUBindGroup;
  private scanBlockBG!:        GPUBindGroup;
  private densityBG!:          GPUBindGroup;
  private hashReadBG!:         GPUBindGroup;
  private predictBG!:          GPUBindGroup;
  private correctBG!:          GPUBindGroup;
  private gravityBG!:          GPUBindGroup;
  private posUpdateBG!:        GPUBindGroup;
  private reduceUniformBG!:    GPUBindGroup;
  private reduceBG!:           GPUBindGroup;

  // ═══════════════════════════════════════════════════════════════════
  // Constructor
  // ═══════════════════════════════════════════════════════════════════

  constructor(device: GPUDevice, maxParticles: number) {
    this.device       = device;
    this.maxParticles = maxParticles;
    this.tblSize      = hashTableSize(maxParticles);

    this.createBuffers();
    this.createPipelines();
    this.createBindGroups();
  }

  // ═══════════════════════════════════════════════════════════════════
  // Public API
  // ═══════════════════════════════════════════════════════════════════

  /**
   * Upload particle data from CPU to GPU.
   *
   * The input array uses the interleaved layout defined by PARTICLE_STRIDE:
   *   14 floats per particle × count particles.
   */
  uploadParticles(particles: Float32Array): void {
    const n = Math.floor(particles.length / PARTICLE_STRIDE);
    this.count = Math.min(n, this.maxParticles);
    const c = this.count;
    const dev = this.device;

    // De-interleave into SoA GPU buffers
    const posX = new Float32Array(c);
    const posY = new Float32Array(c);
    const velX = new Float32Array(c);
    const velY = new Float32Array(c);
    const accX = new Float32Array(c);
    const accY = new Float32Array(c);

    for (let i = 0; i < c; i++) {
      const off = i * PARTICLE_STRIDE;
      posX[i] = particles[off + 0];
      posY[i] = particles[off + 1];
      velX[i] = particles[off + 2];
      velY[i] = particles[off + 3];
      accX[i] = particles[off + 4];
      accY[i] = particles[off + 5];
    }

    dev.queue.writeBuffer(this.posXBuf, 0, posX);
    dev.queue.writeBuffer(this.posYBuf, 0, posY);
    dev.queue.writeBuffer(this.velXBuf, 0, velX);
    dev.queue.writeBuffer(this.velYBuf, 0, velY);
    dev.queue.writeBuffer(this.accXBuf, 0, accX);
    dev.queue.writeBuffer(this.accYBuf, 0, accY);
  }

  /**
   * Run one full DFSPH time step on the GPU.
   *
   * Algorithm (Bender & Koschier 2017, Algorithm 3):
   *   1. Build spatial hash grid
   *   2. Compute densities + DFSPH factors
   *   3. Apply non-pressure forces → predict velocity v*
   *   4. Divergence-free solve → correct velocity v**
   *   5. Update positions x += dt * v**
   *   6. Pressure solve → constant-density correction v***
   */
  async step(dt: number, config: SPHConfig): Promise<void> {
    const n = this.count;
    if (n === 0) return;

    // Write uniforms
    this.writeUniforms(dt, config);

    // 1. Build spatial hash
    this.buildSpatialHash();

    // 2. Compute density + DFSPH factor
    this.dispatchDensityFactor();

    // 3. Apply gravity / non-pressure forces → predicted velocity
    this.dispatchApplyGravity();

    // 4. Divergence-free solve (iterative)
    const maxIterDiv  = (config as any).maxIterDiv  ?? 100;
    const maxErrorDiv = (config as any).maxErrorDiv ?? 0.1;
    await this.iterativeSolve(1, maxIterDiv, maxErrorDiv);

    // 5. Update positions
    this.dispatchUpdatePosition();

    // 6. Pressure solve (iterative)
    const maxIterPres  = (config as any).maxIterPres  ?? 100;
    const maxErrorPres = (config as any).maxErrorPres ?? 0.01;
    await this.iterativeSolve(0, maxIterPres, maxErrorPres);
  }

  /**
   * Read back particle data from GPU to CPU.
   *
   * Returns a Float32Array in the same interleaved PARTICLE_STRIDE layout
   * used by uploadParticles.
   */
  async readback(): Promise<Float32Array> {
    const c   = this.count;
    const dev = this.device;

    if (c === 0) return new Float32Array(0);

    const byteSize = c * 4;

    // Create staging buffers for readback
    const stagePosX = dev.createBuffer({ size: byteSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const stagePosY = dev.createBuffer({ size: byteSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const stageVelX = dev.createBuffer({ size: byteSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const stageVelY = dev.createBuffer({ size: byteSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const stageDen  = dev.createBuffer({ size: byteSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const stageFact = dev.createBuffer({ size: byteSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const enc = dev.createCommandEncoder({ label: "readback" });
    enc.copyBufferToBuffer(this.posXBuf,    0, stagePosX, 0, byteSize);
    enc.copyBufferToBuffer(this.posYBuf,    0, stagePosY, 0, byteSize);
    enc.copyBufferToBuffer(this.velXBuf,    0, stageVelX, 0, byteSize);
    enc.copyBufferToBuffer(this.velYBuf,    0, stageVelY, 0, byteSize);
    enc.copyBufferToBuffer(this.densityBuf, 0, stageDen,  0, byteSize);
    enc.copyBufferToBuffer(this.factorBuf,  0, stageFact, 0, byteSize);
    dev.queue.submit([enc.finish()]);

    await Promise.all([
      stagePosX.mapAsync(GPUMapMode.READ),
      stagePosY.mapAsync(GPUMapMode.READ),
      stageVelX.mapAsync(GPUMapMode.READ),
      stageVelY.mapAsync(GPUMapMode.READ),
      stageDen.mapAsync(GPUMapMode.READ),
      stageFact.mapAsync(GPUMapMode.READ),
    ]);

    const posX    = new Float32Array(stagePosX.getMappedRange());
    const posY    = new Float32Array(stagePosY.getMappedRange());
    const velX    = new Float32Array(stageVelX.getMappedRange());
    const velY    = new Float32Array(stageVelY.getMappedRange());
    const density = new Float32Array(stageDen.getMappedRange());
    const factor  = new Float32Array(stageFact.getMappedRange());

    const result = new Float32Array(c * PARTICLE_STRIDE);

    for (let i = 0; i < c; i++) {
      const off = i * PARTICLE_STRIDE;
      result[off + 0]  = posX[i];
      result[off + 1]  = posY[i];
      result[off + 2]  = velX[i];
      result[off + 3]  = velY[i];
      result[off + 4]  = 0; // ax (reset)
      result[off + 5]  = 0; // ay (reset)
      result[off + 6]  = density[i];
      result[off + 7]  = 0; // pressure (not stored separately)
      result[off + 8]  = factor[i];
      result[off + 9]  = 0; // densityAdv
      result[off + 10] = 0; // kappa
      result[off + 11] = 0; // kappaV
      result[off + 12] = 0; // species
      result[off + 13] = 0; // pad
    }

    stagePosX.unmap(); stagePosX.destroy();
    stagePosY.unmap(); stagePosY.destroy();
    stageVelX.unmap(); stageVelX.destroy();
    stageVelY.unmap(); stageVelY.destroy();
    stageDen.unmap();  stageDen.destroy();
    stageFact.unmap(); stageFact.destroy();

    return result;
  }

  /** Release all GPU resources. */
  destroy(): void {
    const bufs = [
      this.posXBuf, this.posYBuf, this.velXBuf, this.velYBuf,
      this.accXBuf, this.accYBuf, this.densityBuf, this.factorBuf,
      this.densityAdvBuf, this.errorBuf,
      this.cellCountBuf, this.cellStartBuf, this.sortedIdxBuf,
      this.scanBlockSumBuf,
      this.uniformBuf, this.correctUniformBuf, this.scanUniformBuf,
      this.reduceUniformBuf,
      this.reduceOutBuf, this.reduceMapBuf,
    ];
    for (const b of bufs) b.destroy();
  }

  // ═══════════════════════════════════════════════════════════════════
  // Buffer creation
  // ═══════════════════════════════════════════════════════════════════

  private createBuffers(): void {
    const dev = this.device;
    const mp  = this.maxParticles;
    const ts  = this.tblSize;

    const storageRW = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

    const f32Buf = (label: string, count: number) => dev.createBuffer({
      label, size: Math.max(count * 4, 4), usage: storageRW,
    });
    const u32Buf = (label: string, count: number) => dev.createBuffer({
      label, size: Math.max(count * 4, 4), usage: storageRW,
    });

    // Particle SoA
    this.posXBuf       = f32Buf("posX",       mp);
    this.posYBuf       = f32Buf("posY",       mp);
    this.velXBuf       = f32Buf("velX",       mp);
    this.velYBuf       = f32Buf("velY",       mp);
    this.accXBuf       = f32Buf("accX",       mp);
    this.accYBuf       = f32Buf("accY",       mp);
    this.densityBuf    = f32Buf("density",    mp);
    this.factorBuf     = f32Buf("factor",     mp);
    this.densityAdvBuf = f32Buf("densityAdv", mp);
    this.errorBuf      = f32Buf("errorBuf",   mp);

    // Spatial hash
    this.cellCountBuf  = u32Buf("cellCount",  ts);
    // cellStart needs tableSize + 1 for the sentinel end entry
    this.cellStartBuf  = u32Buf("cellStart",  ts + 1);
    this.sortedIdxBuf  = u32Buf("sortedIdx",  mp);

    // Prefix-sum scratch: max workgroups = ceil(ts / 512)
    const maxBlocks = Math.ceil(ts / 512);
    this.scanBlockSumBuf = u32Buf("scanBlockSum", Math.max(maxBlocks, 1));

    // Uniforms (padded to 16-byte alignment)
    this.uniformBuf = dev.createBuffer({
      label: "dfsph-uniforms",
      size: 48,    // 12 × 4 bytes
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.correctUniformBuf = dev.createBuffer({
      label: "correct-uniforms",
      size: 16,    // 4 × 4 bytes
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.scanUniformBuf = dev.createBuffer({
      label: "scan-uniforms",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.reduceUniformBuf = dev.createBuffer({
      label: "reduce-uniforms",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Error reduction output (one f32 per workgroup as u32 bits)
    const maxReduceWG = Math.ceil(mp / WG_SIZE);
    this.reduceOutBuf = dev.createBuffer({
      label: "reduce-out",
      size: Math.max(maxReduceWG * 4, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.reduceMapBuf = dev.createBuffer({
      label: "reduce-map",
      size: Math.max(maxReduceWG * 4, 4),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  // ═══════════════════════════════════════════════════════════════════
  // Pipeline creation
  // ═══════════════════════════════════════════════════════════════════

  private createPipelines(): void {
    const dev = this.device;
    const VIS = GPUShaderStage.COMPUTE;

    // ── Bind Group Layouts ──────────────────────────────────────────

    // Uniform BGL (single uniform buffer at binding 0)
    this.uniformBGL = dev.createBindGroupLayout({
      label: "uniform-bgl",
      entries: [{ binding: 0, visibility: VIS, buffer: { type: "uniform" } }],
    });

    // Uniform + correct-mode BGL (two uniform buffers)
    this.uniformCorrectBGL = dev.createBindGroupLayout({
      label: "uniform-correct-bgl",
      entries: [
        { binding: 0, visibility: VIS, buffer: { type: "uniform" } },
        { binding: 1, visibility: VIS, buffer: { type: "uniform" } },
      ],
    });

    // Hash count: posX(r), posY(r), cellCount(rw)
    this.hashBGL = dev.createBindGroupLayout({
      label: "hash-bgl",
      entries: [
        { binding: 0, visibility: VIS, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: VIS, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: VIS, buffer: { type: "storage" } },
      ],
    });

    // Hash scatter: posX(r), posY(r), cellStart(rw atomic), sortedIdx(rw)
    this.hashScatterBGL = dev.createBindGroupLayout({
      label: "hash-scatter-bgl",
      entries: [
        { binding: 0, visibility: VIS, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: VIS, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: VIS, buffer: { type: "storage" } },
        { binding: 3, visibility: VIS, buffer: { type: "storage" } },
      ],
    });

    // Scan BGLs
    this.scanUniformBGL = dev.createBindGroupLayout({
      label: "scan-uniform-bgl",
      entries: [{ binding: 0, visibility: VIS, buffer: { type: "uniform" } }],
    });
    this.scanDataBGL = dev.createBindGroupLayout({
      label: "scan-data-bgl",
      entries: [{ binding: 0, visibility: VIS, buffer: { type: "storage" } }],
    });
    this.scanBlockBGL = dev.createBindGroupLayout({
      label: "scan-block-bgl",
      entries: [{ binding: 0, visibility: VIS, buffer: { type: "storage" } }],
    });

    // Density + factor: posX(r), posY(r), density(rw), factor(rw)
    this.densityBGL = dev.createBindGroupLayout({
      label: "density-bgl",
      entries: [
        { binding: 0, visibility: VIS, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: VIS, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: VIS, buffer: { type: "storage" } },
        { binding: 3, visibility: VIS, buffer: { type: "storage" } },
      ],
    });

    // Hash read: cellStart(r), cellEnd(r), sortedIdx(r)
    // cellEnd is computed as cellStart[key+1], but we'll use cellStart with
    // an extra sentinel entry.  We bind the same buffer twice with different
    // offsets OR use a simpler approach: bind cellStart buffer of size ts+1
    // and use cellStart[key] as start, cellStart[key+1] as end.
    // For the shader we bind: cellStart(r), cellEnd(r), sortedIdx(r)
    // where cellEnd is just the cellStart buffer with a 4-byte offset.
    // Actually WGSL/WebGPU doesn't support buffer offsets within a bind group
    // entry easily; let's just bind the same buffer for both and index as
    // cellStart[key] and cellStart[key+1] in the shader.
    // We'll update the shader to use this pattern.
    this.hashReadBGL = dev.createBindGroupLayout({
      label: "hash-read-bgl",
      entries: [
        { binding: 0, visibility: VIS, buffer: { type: "read-only-storage" } }, // cellStart (ts+1)
        { binding: 1, visibility: VIS, buffer: { type: "read-only-storage" } }, // cellStart again (for end = [key+1])
        { binding: 2, visibility: VIS, buffer: { type: "read-only-storage" } }, // sortedIdx
      ],
    });

    // Predict density: posX(r), posY(r), velX(r), velY(r), density(r),
    //                  factor(r), densityAdv(rw), errorBuf(rw)
    this.predictBGL = dev.createBindGroupLayout({
      label: "predict-bgl",
      entries: [
        { binding: 0, visibility: VIS, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: VIS, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: VIS, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: VIS, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: VIS, buffer: { type: "read-only-storage" } },
        { binding: 5, visibility: VIS, buffer: { type: "read-only-storage" } },
        { binding: 6, visibility: VIS, buffer: { type: "storage" } },
        { binding: 7, visibility: VIS, buffer: { type: "storage" } },
      ],
    });

    // Correct velocity: posX(r), posY(r), velX(rw), velY(rw), density(r),
    //                   factor(r), densityAdv(r)
    this.correctBGL = dev.createBindGroupLayout({
      label: "correct-bgl",
      entries: [
        { binding: 0, visibility: VIS, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: VIS, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: VIS, buffer: { type: "storage" } },
        { binding: 3, visibility: VIS, buffer: { type: "storage" } },
        { binding: 4, visibility: VIS, buffer: { type: "read-only-storage" } },
        { binding: 5, visibility: VIS, buffer: { type: "read-only-storage" } },
        { binding: 6, visibility: VIS, buffer: { type: "read-only-storage" } },
      ],
    });

    // Apply gravity: velX(rw), velY(rw), accX(rw), accY(rw)
    this.gravityBGL = dev.createBindGroupLayout({
      label: "gravity-bgl",
      entries: [
        { binding: 0, visibility: VIS, buffer: { type: "storage" } },
        { binding: 1, visibility: VIS, buffer: { type: "storage" } },
        { binding: 2, visibility: VIS, buffer: { type: "storage" } },
        { binding: 3, visibility: VIS, buffer: { type: "storage" } },
      ],
    });

    // Update position: posX(rw), posY(rw), velX(r), velY(r)
    this.posUpdateBGL = dev.createBindGroupLayout({
      label: "pos-update-bgl",
      entries: [
        { binding: 0, visibility: VIS, buffer: { type: "storage" } },
        { binding: 1, visibility: VIS, buffer: { type: "storage" } },
        { binding: 2, visibility: VIS, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: VIS, buffer: { type: "read-only-storage" } },
      ],
    });

    // Reduce error: uniform, errorBuf(r), reduceOut(rw)
    this.reduceUniformBGL = dev.createBindGroupLayout({
      label: "reduce-uniform-bgl",
      entries: [{ binding: 0, visibility: VIS, buffer: { type: "uniform" } }],
    });
    this.reduceBGL = dev.createBindGroupLayout({
      label: "reduce-bgl",
      entries: [
        { binding: 0, visibility: VIS, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: VIS, buffer: { type: "storage" } },
      ],
    });

    // ── Pipelines ───────────────────────────────────────────────────

    const makePipeline = (label: string, code: string, bgls: GPUBindGroupLayout[]) => {
      const module = dev.createShaderModule({ label: `${label}-shader`, code });
      const layout = dev.createPipelineLayout({
        label: `${label}-layout`,
        bindGroupLayouts: bgls,
      });
      return dev.createComputePipeline({
        label: `${label}-pipeline`,
        layout,
        compute: { module, entryPoint: "main" },
      });
    };

    this.hashCountPipeline      = makePipeline("hash-count",
      HASH_COUNT_SHADER,      [this.uniformBGL, this.hashBGL]);
    this.hashScatterPipeline    = makePipeline("hash-scatter",
      HASH_SCATTER_SHADER,    [this.uniformBGL, this.hashScatterBGL]);
    this.prefixSumPipeline      = makePipeline("prefix-sum",
      PREFIX_SUM_SHADER,      [this.scanUniformBGL, this.scanDataBGL, this.scanBlockBGL]);
    this.prefixSumAddPipeline   = makePipeline("prefix-sum-add",
      PREFIX_SUM_ADD_SHADER,  [this.scanUniformBGL, this.scanDataBGL, this.scanBlockBGL]);
    this.densityFactorPipeline  = makePipeline("density-factor",
      DENSITY_FACTOR_SHADER,  [this.uniformBGL, this.densityBGL, this.hashReadBGL]);
    this.predictDensityPipeline = makePipeline("predict-density",
      PREDICT_DENSITY_SHADER, [this.uniformCorrectBGL, this.predictBGL, this.hashReadBGL]);
    this.correctVelocityPipeline= makePipeline("correct-velocity",
      CORRECT_VELOCITY_SHADER,[this.uniformCorrectBGL, this.correctBGL, this.hashReadBGL]);
    this.applyGravityPipeline   = makePipeline("apply-gravity",
      APPLY_GRAVITY_SHADER,   [this.uniformBGL, this.gravityBGL]);
    this.updatePositionPipeline = makePipeline("update-position",
      UPDATE_POSITION_SHADER, [this.uniformBGL, this.posUpdateBGL]);
    this.reduceErrorPipeline    = makePipeline("reduce-error",
      REDUCE_ERROR_SHADER,    [this.reduceUniformBGL, this.reduceBGL]);
  }

  // ═══════════════════════════════════════════════════════════════════
  // Bind Group creation
  // ═══════════════════════════════════════════════════════════════════

  private createBindGroups(): void {
    const dev = this.device;

    this.uniformBG = dev.createBindGroup({
      label: "uniform-bg",
      layout: this.uniformBGL,
      entries: [{ binding: 0, resource: { buffer: this.uniformBuf } }],
    });

    this.uniformCorrectBG = dev.createBindGroup({
      label: "uniform-correct-bg",
      layout: this.uniformCorrectBGL,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
        { binding: 1, resource: { buffer: this.correctUniformBuf } },
      ],
    });

    this.hashBG = dev.createBindGroup({
      label: "hash-bg",
      layout: this.hashBGL,
      entries: [
        { binding: 0, resource: { buffer: this.posXBuf } },
        { binding: 1, resource: { buffer: this.posYBuf } },
        { binding: 2, resource: { buffer: this.cellCountBuf } },
      ],
    });

    this.hashScatterBG = dev.createBindGroup({
      label: "hash-scatter-bg",
      layout: this.hashScatterBGL,
      entries: [
        { binding: 0, resource: { buffer: this.posXBuf } },
        { binding: 1, resource: { buffer: this.posYBuf } },
        { binding: 2, resource: { buffer: this.cellStartBuf } },
        { binding: 3, resource: { buffer: this.sortedIdxBuf } },
      ],
    });

    this.scanUniformBG = dev.createBindGroup({
      label: "scan-uniform-bg",
      layout: this.scanUniformBGL,
      entries: [{ binding: 0, resource: { buffer: this.scanUniformBuf } }],
    });

    this.scanDataBG = dev.createBindGroup({
      label: "scan-data-bg",
      layout: this.scanDataBGL,
      entries: [{ binding: 0, resource: { buffer: this.cellCountBuf } }],
    });

    this.scanBlockBG = dev.createBindGroup({
      label: "scan-block-bg",
      layout: this.scanBlockBGL,
      entries: [{ binding: 0, resource: { buffer: this.scanBlockSumBuf } }],
    });

    this.densityBG = dev.createBindGroup({
      label: "density-bg",
      layout: this.densityBGL,
      entries: [
        { binding: 0, resource: { buffer: this.posXBuf } },
        { binding: 1, resource: { buffer: this.posYBuf } },
        { binding: 2, resource: { buffer: this.densityBuf } },
        { binding: 3, resource: { buffer: this.factorBuf } },
      ],
    });

    // For hashReadBG, bind cellStartBuf twice (start and end) + sortedIdx.
    // The shaders will use cellStart[key] as start and cellStart[key+1] as end.
    this.hashReadBG = dev.createBindGroup({
      label: "hash-read-bg",
      layout: this.hashReadBGL,
      entries: [
        { binding: 0, resource: { buffer: this.cellStartBuf } },
        { binding: 1, resource: { buffer: this.cellStartBuf } },
        { binding: 2, resource: { buffer: this.sortedIdxBuf } },
      ],
    });

    this.predictBG = dev.createBindGroup({
      label: "predict-bg",
      layout: this.predictBGL,
      entries: [
        { binding: 0, resource: { buffer: this.posXBuf } },
        { binding: 1, resource: { buffer: this.posYBuf } },
        { binding: 2, resource: { buffer: this.velXBuf } },
        { binding: 3, resource: { buffer: this.velYBuf } },
        { binding: 4, resource: { buffer: this.densityBuf } },
        { binding: 5, resource: { buffer: this.factorBuf } },
        { binding: 6, resource: { buffer: this.densityAdvBuf } },
        { binding: 7, resource: { buffer: this.errorBuf } },
      ],
    });

    this.correctBG = dev.createBindGroup({
      label: "correct-bg",
      layout: this.correctBGL,
      entries: [
        { binding: 0, resource: { buffer: this.posXBuf } },
        { binding: 1, resource: { buffer: this.posYBuf } },
        { binding: 2, resource: { buffer: this.velXBuf } },
        { binding: 3, resource: { buffer: this.velYBuf } },
        { binding: 4, resource: { buffer: this.densityBuf } },
        { binding: 5, resource: { buffer: this.factorBuf } },
        { binding: 6, resource: { buffer: this.densityAdvBuf } },
      ],
    });

    this.gravityBG = dev.createBindGroup({
      label: "gravity-bg",
      layout: this.gravityBGL,
      entries: [
        { binding: 0, resource: { buffer: this.velXBuf } },
        { binding: 1, resource: { buffer: this.velYBuf } },
        { binding: 2, resource: { buffer: this.accXBuf } },
        { binding: 3, resource: { buffer: this.accYBuf } },
      ],
    });

    this.posUpdateBG = dev.createBindGroup({
      label: "pos-update-bg",
      layout: this.posUpdateBGL,
      entries: [
        { binding: 0, resource: { buffer: this.posXBuf } },
        { binding: 1, resource: { buffer: this.posYBuf } },
        { binding: 2, resource: { buffer: this.velXBuf } },
        { binding: 3, resource: { buffer: this.velYBuf } },
      ],
    });

    this.reduceUniformBG = dev.createBindGroup({
      label: "reduce-uniform-bg",
      layout: this.reduceUniformBGL,
      entries: [{ binding: 0, resource: { buffer: this.reduceUniformBuf } }],
    });

    this.reduceBG = dev.createBindGroup({
      label: "reduce-bg",
      layout: this.reduceBGL,
      entries: [
        { binding: 0, resource: { buffer: this.errorBuf } },
        { binding: 1, resource: { buffer: this.reduceOutBuf } },
      ],
    });
  }

  // ═══════════════════════════════════════════════════════════════════
  // Uniform upload
  // ═══════════════════════════════════════════════════════════════════

  private writeUniforms(dt: number, config: SPHConfig): void {
    const dev = this.device;

    // Map SPHConfig fields (from sph-kernels.ts) to our uniform layout.
    // SPHConfig uses: smoothingRadius, restDensity, particleMass, dt, gravity
    // DFSPH augmentation adds: gravityX, gravityY, h, mass, rho0
    const h    = (config as any).h    ?? config.smoothingRadius ?? 12;
    const rho0 = (config as any).rho0 ?? config.restDensity     ?? 1000;
    const mass = (config as any).mass ?? config.particleMass    ?? 1.0;
    const gx   = (config as any).gravityX ?? 0;
    const gy   = (config as any).gravityY ?? ((config as any).gravity ?? -9.81);

    // Domain size defaults (the solver uses these for boundary clamping)
    const domainW = (config as any).domainW ?? 200;
    const domainH = (config as any).domainH ?? 200;

    // DFSPHUniforms: 12 × 4 = 48 bytes
    const data = new ArrayBuffer(48);
    const f = new Float32Array(data);
    const u = new Uint32Array(data);
    f[0]  = h;
    f[1]  = rho0;
    f[2]  = mass;
    f[3]  = dt;
    f[4]  = gx;
    f[5]  = gy;
    u[6]  = this.count;
    u[7]  = this.tblSize;
    f[8]  = domainW;
    f[9]  = domainH;
    u[10] = 0;
    u[11] = 0;
    dev.queue.writeBuffer(this.uniformBuf, 0, data);

    // Cache rho0 for error normalisation in reduceError()
    this._cachedRho0 = rho0;

    // Reduce uniforms
    const rData = new Uint32Array([this.count, 0, 0, 0]);
    dev.queue.writeBuffer(this.reduceUniformBuf, 0, rData);
  }

  // ═══════════════════════════════════════════════════════════════════
  // Spatial hash build
  // ═══════════════════════════════════════════════════════════════════

  private buildSpatialHash(): void {
    const dev = this.device;
    const n   = this.count;
    const ts  = this.tblSize;
    const wg  = Math.ceil(n / WG_SIZE);

    const enc = dev.createCommandEncoder({ label: "hash-build" });

    // Zero cellCount buffer
    enc.clearBuffer(this.cellCountBuf, 0, ts * 4);

    // Pass 1a: Hash count
    {
      const pass = enc.beginComputePass({ label: "hash-count" });
      pass.setPipeline(this.hashCountPipeline);
      pass.setBindGroup(0, this.uniformBG);
      pass.setBindGroup(1, this.hashBG);
      pass.dispatchWorkgroups(wg);
      pass.end();
    }

    dev.queue.submit([enc.finish()]);

    // Pass 1b: Prefix sum on cellCount → in-place exclusive scan
    this.dispatchPrefixSum(ts);

    // Copy scanned cellCount → cellStart (we need the original offsets
    // preserved while scatter atomically increments the working copy)
    const enc2 = dev.createCommandEncoder({ label: "copy-cell-start" });
    enc2.copyBufferToBuffer(this.cellCountBuf, 0, this.cellStartBuf, 0, ts * 4);
    // Write sentinel: cellStart[ts] = n (total particle count)
    dev.queue.submit([enc2.finish()]);

    // Write the sentinel value at cellStart[ts]
    const sentinel = new Uint32Array([n]);
    dev.queue.writeBuffer(this.cellStartBuf, ts * 4, sentinel);

    // Pass 1c: Hash scatter — write particle indices into sortedIdx
    // We use the cellStart copy as the atomic counter (it starts at the
    // prefix-sum offset and increments to fill each cell's range).
    // After scatter, cellStart[key] == original cellStart[key] + count[key]
    // which equals cellStart[key+1] (from the prefix-sum), so cellStart is
    // effectively corrupted.  But we already stored the clean copy above.
    //
    // Wait — we need to keep the ORIGINAL cellStart for neighbor queries.
    // Strategy: copy scanned cellCount to cellStart, then use cellCount
    // (which still holds the scanned values after the copy) as the scatter
    // counter.  After scatter, cellCount is corrupted but cellStart is clean.
    //
    // Actually both cellCount and cellStart hold the same values after the copy.
    // Let's use cellStart as the scatter counter and keep cellCount clean.
    // Hmm, but we already submitted the copy.  Let's think again:
    //
    // After prefix sum: cellCount = exclusive scan (clean offsets)
    // Copy: cellStart = cellCount (both clean)
    // Scatter uses cellStart atomically → cellStart corrupted
    // cellCount still has clean offsets ← use cellCount for neighbor queries!
    //
    // But our hashReadBG binds cellStartBuf.  Let's rebind using cellCountBuf
    // instead.  OR: do the scatter using cellCountBuf as the atomic counter,
    // keep cellStartBuf clean.
    //
    // The hashScatterBG binds cellStartBuf at binding 2.  Let's swap:
    // Use cellCountBuf as the scatter counter → rebind hashScatterBG.

    // Actually, let's just re-copy cellCount to cellStart AFTER scatter.
    // That's the simplest approach with no rebinding needed.

    const enc3 = dev.createCommandEncoder({ label: "hash-scatter" });
    {
      const pass = enc3.beginComputePass({ label: "hash-scatter" });
      pass.setPipeline(this.hashScatterPipeline);
      pass.setBindGroup(0, this.uniformBG);
      pass.setBindGroup(1, this.hashScatterBG);
      pass.dispatchWorkgroups(wg);
      pass.end();
    }

    // Restore cellStart from the clean cellCount
    enc3.copyBufferToBuffer(this.cellCountBuf, 0, this.cellStartBuf, 0, ts * 4);
    dev.queue.submit([enc3.finish()]);

    // Re-write the sentinel (it was overwritten by the copy)
    dev.queue.writeBuffer(this.cellStartBuf, ts * 4, sentinel);
  }

  // ── Prefix sum dispatch ───────────────────────────────────────────

  private dispatchPrefixSum(n: number): void {
    const dev         = this.device;
    const BLOCK_ELEMS = 512;
    const numBlocks   = Math.ceil(n / BLOCK_ELEMS);

    // Upload scan uniforms
    const uData = new Uint32Array([n, 0, 0, 0]);
    dev.queue.writeBuffer(this.scanUniformBuf, 0, uData);

    const enc = dev.createCommandEncoder({ label: "prefix-sum" });

    // Phase A: local Blelloch scan per workgroup
    {
      const pass = enc.beginComputePass({ label: "prefix-sum-local" });
      pass.setPipeline(this.prefixSumPipeline);
      pass.setBindGroup(0, this.scanUniformBG);
      pass.setBindGroup(1, this.scanDataBG);
      pass.setBindGroup(2, this.scanBlockBG);
      pass.dispatchWorkgroups(numBlocks);
      pass.end();
    }

    // Phase B: scan block sums (single workgroup)
    if (numBlocks > 1) {
      // Create temporary bind groups for scanning the blockSum array
      const blockBlockSumBuf = dev.createBuffer({
        label: "scan-block-block-sum",
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      const blockBlockSumBG = dev.createBindGroup({
        label: "scan-block-block-sum-bg",
        layout: this.scanBlockBGL,
        entries: [{ binding: 0, resource: { buffer: blockBlockSumBuf } }],
      });
      const blockSumDataBG = dev.createBindGroup({
        label: "scan-block-sum-data-bg",
        layout: this.scanDataBGL,
        entries: [{ binding: 0, resource: { buffer: this.scanBlockSumBuf } }],
      });

      // Re-upload uniforms with n = numBlocks
      const uData2 = new Uint32Array([numBlocks, 0, 0, 0]);
      dev.queue.writeBuffer(this.scanUniformBuf, 0, uData2);

      {
        const pass = enc.beginComputePass({ label: "prefix-sum-block-scan" });
        pass.setPipeline(this.prefixSumPipeline);
        pass.setBindGroup(0, this.scanUniformBG);
        pass.setBindGroup(1, blockSumDataBG);
        pass.setBindGroup(2, blockBlockSumBG);
        pass.dispatchWorkgroups(1);
        pass.end();
      }

      // Restore original n
      dev.queue.writeBuffer(this.scanUniformBuf, 0, uData);

      // Phase C: add block offsets back
      {
        const pass = enc.beginComputePass({ label: "prefix-sum-add" });
        pass.setPipeline(this.prefixSumAddPipeline);
        pass.setBindGroup(0, this.scanUniformBG);
        pass.setBindGroup(1, this.scanDataBG);
        pass.setBindGroup(2, this.scanBlockBG);
        pass.dispatchWorkgroups(numBlocks);
        pass.end();
      }

      Promise.resolve().then(() => blockBlockSumBuf.destroy());
    }

    dev.queue.submit([enc.finish()]);
  }

  // ═══════════════════════════════════════════════════════════════════
  // Compute pass dispatchers
  // ═══════════════════════════════════════════════════════════════════

  private dispatchDensityFactor(): void {
    const dev = this.device;
    const wg  = Math.ceil(this.count / WG_SIZE);

    const enc = dev.createCommandEncoder({ label: "density-factor" });
    {
      const pass = enc.beginComputePass({ label: "density-factor" });
      pass.setPipeline(this.densityFactorPipeline);
      pass.setBindGroup(0, this.uniformBG);
      pass.setBindGroup(1, this.densityBG);
      pass.setBindGroup(2, this.hashReadBG);
      pass.dispatchWorkgroups(wg);
      pass.end();
    }
    dev.queue.submit([enc.finish()]);
  }

  private dispatchApplyGravity(): void {
    const dev = this.device;
    const wg  = Math.ceil(this.count / WG_SIZE);

    const enc = dev.createCommandEncoder({ label: "apply-gravity" });
    {
      const pass = enc.beginComputePass({ label: "apply-gravity" });
      pass.setPipeline(this.applyGravityPipeline);
      pass.setBindGroup(0, this.uniformBG);
      pass.setBindGroup(1, this.gravityBG);
      pass.dispatchWorkgroups(wg);
      pass.end();
    }
    dev.queue.submit([enc.finish()]);
  }

  private dispatchUpdatePosition(): void {
    const dev = this.device;
    const wg  = Math.ceil(this.count / WG_SIZE);

    const enc = dev.createCommandEncoder({ label: "update-position" });
    {
      const pass = enc.beginComputePass({ label: "update-position" });
      pass.setPipeline(this.updatePositionPipeline);
      pass.setBindGroup(0, this.uniformBG);
      pass.setBindGroup(1, this.posUpdateBG);
      pass.dispatchWorkgroups(wg);
      pass.end();
    }
    dev.queue.submit([enc.finish()]);
  }

  // ═══════════════════════════════════════════════════════════════════
  // Iterative DFSPH solve (pressure or divergence-free)
  // ═══════════════════════════════════════════════════════════════════

  /**
   * Run the iterative DFSPH correction loop.
   *
   * @param mode     - 0 = pressure (density error), 1 = divergence
   * @param maxIter  - maximum iteration count
   * @param maxError - convergence threshold (relative to rho0)
   */
  private async iterativeSolve(
    mode: number,
    maxIter: number,
    maxError: number,
  ): Promise<number> {
    const dev = this.device;
    const n   = this.count;
    const wg  = Math.ceil(n / WG_SIZE);

    // Write correction mode uniform
    const modeData = new Uint32Array([mode, 0, 0, 0]);
    dev.queue.writeBuffer(this.correctUniformBuf, 0, modeData);

    let iter = 0;

    while (iter < maxIter) {
      // Step A: Compute predicted density / divergence + per-particle error
      {
        const enc = dev.createCommandEncoder({ label: `predict-${mode}-${iter}` });
        const pass = enc.beginComputePass({ label: "predict" });
        pass.setPipeline(this.predictDensityPipeline);
        pass.setBindGroup(0, this.uniformCorrectBG);
        pass.setBindGroup(1, this.predictBG);
        pass.setBindGroup(2, this.hashReadBG);
        pass.dispatchWorkgroups(wg);
        pass.end();
        dev.queue.submit([enc.finish()]);
      }

      // Step B: Reduce error to get average
      const avgError = await this.reduceError();

      // Check convergence (skip first iteration per original algorithm)
      if (avgError < maxError && iter >= 1) break;

      // Step C: Apply pressure / divergence-free velocity correction
      {
        const enc = dev.createCommandEncoder({ label: `correct-${mode}-${iter}` });
        const pass = enc.beginComputePass({ label: "correct" });
        pass.setPipeline(this.correctVelocityPipeline);
        pass.setBindGroup(0, this.uniformCorrectBG);
        pass.setBindGroup(1, this.correctBG);
        pass.setBindGroup(2, this.hashReadBG);
        pass.dispatchWorkgroups(wg);
        pass.end();
        dev.queue.submit([enc.finish()]);
      }

      iter++;
    }

    return iter;
  }

  // ─── Error reduction readback ─────────────────────────────────────

  private async reduceError(): Promise<number> {
    const dev = this.device;
    const n   = this.count;
    const numWG = Math.ceil(n / WG_SIZE);

    // Dispatch reduction kernel
    const enc = dev.createCommandEncoder({ label: "reduce-error" });
    {
      const pass = enc.beginComputePass({ label: "reduce" });
      pass.setPipeline(this.reduceErrorPipeline);
      pass.setBindGroup(0, this.reduceUniformBG);
      pass.setBindGroup(1, this.reduceBG);
      pass.dispatchWorkgroups(numWG);
      pass.end();
    }

    // Copy partial sums to MAP_READ staging buffer
    const copyBytes = numWG * 4;
    enc.copyBufferToBuffer(this.reduceOutBuf, 0, this.reduceMapBuf, 0, copyBytes);
    dev.queue.submit([enc.finish()]);

    // Read back partial sums
    await this.reduceMapBuf.mapAsync(GPUMapMode.READ, 0, copyBytes);
    const partials = new Uint32Array(
      this.reduceMapBuf.getMappedRange(0, copyBytes),
    );

    // Sum the workgroup partial sums (stored as u32 bit-patterns of f32)
    let totalError = 0;
    for (let i = 0; i < numWG; i++) {
      // Reinterpret u32 → f32
      const view = new DataView(new ArrayBuffer(4));
      view.setUint32(0, partials[i], true);
      totalError += view.getFloat32(0, true);
    }

    this.reduceMapBuf.unmap();

    // Return average relative error (divided by count and rho0).
    // The uniform rho0 was already written; we read it from the uniform
    // buffer concept.  For simplicity, we pass rho0 via the uniform struct
    // and divide here.  We stored rho0 at float offset 1 in the uniform.
    // Rather than read it back, we note the caller's config had it; but
    // we don't have it here.  We'll read from the uniform.
    //
    // Actually we can just approximate: the error values in the buffer are
    // either |ρ* - ρ0| (pressure mode) or |divV| (divergence mode).
    // The CPU dfsph-solver divides by (n * rho0).  We'll do the same,
    // but we need rho0.  Let's cache it.
    //
    // For now, we use a simpler approach: read rho0 from the uploaded
    // uniform data.  Since we wrote it, we know the layout.
    //
    // We'll cache rho0 during writeUniforms.

    return totalError / (n * this._cachedRho0);
  }

  private _cachedRho0 = 1000;

  // ═══════════════════════════════════════════════════════════════════
  // Exported constants for interop
  // ═══════════════════════════════════════════════════════════════════

  /** Number of f32 values per particle in the upload/readback array. */
  static readonly PARTICLE_STRIDE = PARTICLE_STRIDE;
}
