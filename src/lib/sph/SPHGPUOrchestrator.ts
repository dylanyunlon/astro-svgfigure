# === SPHGPUOrchestrator.ts ===
// SPHGPUOrchestrator.ts — WebGPU compute pipeline for SPH

import { GPUBufferSet, SimParams, WORKGROUP_SIZE } from "./types";

// ---------------------------------------------------------------------------
// WGSL Shaders
// ---------------------------------------------------------------------------

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

// group 0 — uniforms
@group(0) @binding(0) var<uniform> params : SimUniforms;

// group 1 — particle buffers (read-only inputs + rw outputs)
@group(1) @binding(0) var<storage, read>       posX    : array<f32>;
@group(1) @binding(1) var<storage, read>       posY    : array<f32>;
@group(1) @binding(2) var<storage, read_write> density : array<f32>;
@group(1) @binding(3) var<storage, read_write> pressure: array<f32>;

// group 2 — neighbor CSR
@group(2) @binding(0) var<storage, read> neighborData: array<i32>;
@group(2) @binding(1) var<storage, read> rowPtr      : array<i32>;

// group 3 — boundary particles
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

// group 0 — uniforms
@group(0) @binding(0) var<uniform> params : SimUniforms;

// group 1 — particle buffers
@group(1) @binding(0) var<storage, read>       posX    : array<f32>;
@group(1) @binding(1) var<storage, read>       posY    : array<f32>;
@group(1) @binding(2) var<storage, read>       velX    : array<f32>;
@group(1) @binding(3) var<storage, read>       velY    : array<f32>;
@group(1) @binding(4) var<storage, read>       density : array<f32>;
@group(1) @binding(5) var<storage, read>       pressure: array<f32>;
@group(1) @binding(6) var<storage, read_write> forceX  : array<f32>;
@group(1) @binding(7) var<storage, read_write> forceY  : array<f32>;

// group 2 — neighbor CSR
@group(2) @binding(0) var<storage, read> neighborData: array<i32>;
@group(2) @binding(1) var<storage, read> rowPtr      : array<i32>;

// group 3 — boundary particles
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

// group 0 — uniforms
@group(0) @binding(0) var<uniform> params : SimUniforms;

// group 1 — particle buffers (all read-write)
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
// Types (local aliases)
// ---------------------------------------------------------------------------

interface NeighborCSR {
  neighborBuf: GPUBuffer; // array<i32>
  rowPtrBuf  : GPUBuffer; // array<i32>, length = N+1
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

export class SPHGPUOrchestrator {
  private readonly device  : GPUDevice;
  private readonly bufs    : GPUBufferSet;

  private uniformBuf!      : GPUBuffer;

  // pipelines
  private densityPipeline  !: GPUComputePipeline;
  private forcePipeline    !: GPUComputePipeline;
  private integratePipeline!: GPUComputePipeline;

  // bind-group layouts (group 0 is uniform, shared)
  private uniformBGL       !: GPUBindGroupLayout;
  private densityParticleBGL  !: GPUBindGroupLayout;
  private forceParticleBGL    !: GPUBindGroupLayout;
  private integrateParticleBGL!: GPUBindGroupLayout;
  private neighborBGL      !: GPUBindGroupLayout;
  private boundaryBGL      !: GPUBindGroupLayout;

  // cached bind groups
  private uniformBG        !: GPUBindGroup;
  private densityParticleBG  !: GPUBindGroup;
  private forceParticleBG    !: GPUBindGroup;
  private integrateParticleBG!: GPUBindGroup;
  private neighborBG       : GPUBindGroup | null = null;
  private boundaryBG       : GPUBindGroup | null = null;

  // last CSR / boundary refs for dirty-checking
  private lastNeighborBuf  : GPUBuffer | null = null;
  private lastRowPtrBuf    : GPUBuffer | null = null;
  private lastBoundaryBuf  : GPUBuffer | null = null;

  constructor(device: GPUDevice, bufs: GPUBufferSet) {
    this.device = device;
    this.bufs   = bufs;
    this.init();
  }

  // -------------------------------------------------------------------------
  private init(): void {
    const dev = this.device;

    // ---------- uniform buffer (48 bytes — 12 × f32/u32) ----------
    this.uniformBuf = dev.createBuffer({
      size : 48,
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
  }

  // -------------------------------------------------------------------------
  private makePipeline(
    label  : string,
    wgsl   : string,
    layouts: GPUBindGroupLayout[],
  ): GPUComputePipeline {
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

  private updateBoundaryBG(boundaryBuf: GPUBuffer): void {
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
    boundaryBuf: GPUBuffer,
  ): void {
    const dev = this.device;
    const n   = simParams.count;
    const wg  = Math.ceil(n / WORKGROUP_SIZE);

    // 1. Upload uniforms
    this.uploadUniforms(simParams);

    // 2. Refresh dynamic bind groups if buffers changed
    this.updateNeighborBG(neighborCSR);
    this.updateBoundaryBG(boundaryBuf);

    const neighborBG  = this.neighborBG!;
    const boundaryBG  = this.boundaryBG!;

    const encoder = dev.createCommandEncoder({ label: "sph-frame" });

    // ------------------------------------------------------------------
    // Pass 1 — density + pressure
    // ------------------------------------------------------------------
    {
      const pass = encoder.beginComputePass({ label: "density-pass" });
      pass.setPipeline(this.densityPipeline);
      pass.setBindGroup(0, this.uniformBG);
      pass.setBindGroup(1, this.densityParticleBG);
      pass.setBindGroup(2, neighborBG);
      pass.setBindGroup(3, boundaryBG);
      pass.dispatchWorkgroups(wg);
      pass.end();
    }

    // ------------------------------------------------------------------
    // Pass 2 — pressure gradient + viscosity → forceX / forceY
    // ------------------------------------------------------------------
    {
      const pass = encoder.beginComputePass({ label: "force-pass" });
      pass.setPipeline(this.forcePipeline);
      pass.setBindGroup(0, this.uniformBG);
      pass.setBindGroup(1, this.forceParticleBG);
      pass.setBindGroup(2, neighborBG);
      pass.setBindGroup(3, boundaryBG);
      pass.dispatchWorkgroups(wg);
      pass.end();
    }

    // ------------------------------------------------------------------
    // Pass 3 — symplectic Euler integration + boundary clamp
    // ------------------------------------------------------------------
    {
      const pass = encoder.beginComputePass({ label: "integrate-pass" });
      pass.setPipeline(this.integratePipeline);
      pass.setBindGroup(0, this.uniformBG);
      pass.setBindGroup(1, this.integrateParticleBG);
      pass.dispatchWorkgroups(wg);
      pass.end();
    }

    dev.queue.submit([encoder.finish()]);
  }

  // -------------------------------------------------------------------------
  /** Release GPU resources owned by this orchestrator. */
  destroy(): void {
    this.uniformBuf.destroy();
    // GPUComputePipelines and GPUBindGroupLayouts are GC'd by the device;
    // no explicit destroy() method exists for them in the WebGPU spec.
    this.neighborBG  = null;
    this.boundaryBG  = null;
    this.lastNeighborBuf = null;
    this.lastRowPtrBuf   = null;
    this.lastBoundaryBuf = null;
  }
}
