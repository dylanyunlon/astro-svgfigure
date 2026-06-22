/**
 * reaction-diffusion.ts
 *
 * Gray-Scott Reaction-Diffusion — WebGPU compute pipeline
 *
 * 架构设计：
 *   • 双缓冲 ping-pong：两张 rgba32float storage texture
 *       ch R = u (activator,  chemical U)
 *       ch G = v (inhibitor,  chemical V)
 *       ch B = reserved (0)
 *       ch A = reserved (1)
 *   • 每次 step() 在 GPU 上执行 N 次 GS compute pass（默认 8 次/帧）
 *   • 最终输出纹理可直接绑定到渲染 pipeline
 *   • parameterSpace(name) 按 Munafo/Pearson/Karl Sims 参数映射返回 (f, k)
 *
 * Gray-Scott 方程（每步 Δt=1.0）：
 *   du/dt = Du·∇²u − u·v² + f·(1−u)
 *   dv/dt = Dv·∇²v + u·v² − (f+k)·v
 *
 * 扩散系数（规范值，源自 Pearson 1993）：
 *   Du = 0.2097,  Dv = 0.1050   (Du/Dv ≈ 2 : 1)
 *
 * 参考：
 *   Pearson, J.E. (1993) Complex Patterns in a Simple System. Science 261.
 *   Munafo, R. — mrob.com/pub/comp/xmorphia  (Pearson extended classes)
 *   Karl Sims — karlsims.com/rd.html  (coral f=0.0545 k=0.062; mitosis f=0.0367 k=0.0649)
 *   Shader: grayscott-species.frag — M550, cell-pubsub-loop branch
 */

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/** Storage texture format for ping-pong buffers (R=u, G=v, B=0, A=1). */
const RD_TEX_FORMAT: GPUTextureFormat = 'rgba32float';

/** Compute workgroup tile size (8×8 = 64 threads per workgroup). */
const RD_WG = 8;

/** Default simulation grid resolution. */
export const RD_DEFAULT_SIZE = 256;

/** Default number of Gray-Scott substeps per animation frame. */
export const RD_DEFAULT_SUBSTEPS = 8;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Named pattern species from the Pearson/Munafo Gray-Scott parameter space.
 *
 * Values verified against:
 *   • Karl Sims — karlsims.com/rd.html
 *   • Munafo — mrob.com/pub/comp/xmorphia/pearson-classes.html
 */
export type GrayScottSpecies =
  | 'coral'
  | 'mitosis'
  | 'dots'
  | 'stripes'
  | 'labyrinth'
  | 'worms'
  | 'spirals'
  | 'bubbles'
  | 'maze'
  | 'chaos';

/** Feed/kill parameter pair for the Gray-Scott model. */
export interface GrayScottParams {
  /** Feed rate f (controls u replenishment). */
  f: number;
  /** Kill rate k (controls v removal). */
  k: number;
  /** Diffusion rate for u (activator). Default 0.2097. */
  Du?: number;
  /** Diffusion rate for v (inhibitor). Default 0.1050. */
  Dv?: number;
}

/** Configuration for ReactionDiffusionSim constructor. */
export interface RDSimConfig {
  /** Grid width in cells. Default: RD_DEFAULT_SIZE. */
  width?: number;
  /** Grid height in cells. Default: RD_DEFAULT_SIZE. */
  height?: number;
  /** Gray-Scott (f, k, Du, Dv) parameters. Default: coral preset. */
  params?: GrayScottParams;
  /** GS substeps executed per step() call. Default: RD_DEFAULT_SUBSTEPS. */
  substeps?: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Parameter space — Munafo / Pearson / Karl Sims canonical values
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Map a named species to its canonical Gray-Scott (f, k) parameters.
 *
 * Parameter sources:
 *   coral     — f=0.0545, k=0.0620  Karl Sims "coral growth" demo
 *   mitosis   — f=0.0367, k=0.0649  Karl Sims "mitosis" demo
 *   dots      — f=0.0350, k=0.0650  Pearson type λ (soliton dots)
 *   stripes   — f=0.0600, k=0.0630  Pearson type κ (labyrinthine)
 *   labyrinth — f=0.0300, k=0.0570  Pearson type δ (Turing stripes)
 *   worms     — f=0.0780, k=0.0610  Pearson type μ (worm/stripe mix)
 *   spirals   — f=0.0100, k=0.0350  Munafo type α (spiral chaos)
 *   bubbles   — f=0.0900, k=0.0590  xmorphia "soap-bubbles"
 *   maze      — f=0.0220, k=0.0510  Pearson type γ (worm maze)
 *   chaos     — f=0.0260, k=0.0510  Pearson type β/γ boundary
 */
export function parameterSpace(species: GrayScottSpecies): GrayScottParams {
  // Canonical Du/Dv from Pearson 1993 (rescaled to ≤1 range for numerical stability)
  const Du = 0.2097;
  const Dv = 0.1050;

  switch (species) {
    // ── Pearson κ — coral / branching loops growing from worm tips
    //    Karl Sims "coral growth" reference: f=0.0545, k=0.062
    case 'coral':
      return { f: 0.0545, k: 0.0620, Du, Dv };

    // ── Pearson λ — mitosis / solitons that grow then divide
    //    Karl Sims "mitosis" reference: f=0.0367, k=0.0649
    case 'mitosis':
      return { f: 0.0367, k: 0.0649, Du, Dv };

    // ── Pearson λ edge — isolated soliton spots, hexagonal packing
    //    xmorphia F350/k650; "dots" in Karl Sims parameter map
    case 'dots':
      return { f: 0.0350, k: 0.0650, Du, Dv };

    // ── Pearson κ — labyrinthine stripe maze (fingerprint / hedgerow)
    //    xmorphia F600/k630 (coral→maze evolution after 250 000 tu)
    case 'stripes':
      return { f: 0.0600, k: 0.0630, Du, Dv };

    // ── Pearson δ — Turing instability: stationary negative-spot hexarray
    //    xmorphia F300/k550
    case 'labyrinth':
      return { f: 0.0300, k: 0.0570, Du, Dv };

    // ── Pearson μ — worms/filaments that elongate without branching
    //    xmorphia high-F stripe zone F780/k610
    case 'worms':
      return { f: 0.0780, k: 0.0610, Du, Dv };

    // ── Munafo type α — spatiotemporal chaos, spirals / wavelets
    //    xmorphia F100/k470
    case 'spirals':
      return { f: 0.0100, k: 0.0350, Du, Dv };

    // ── Munafo high-F "soap bubbles" — negatons in red sea
    //    xmorphia F900/k590 "soap-bubbles" (Karl Sims F=0.090, k=0.059)
    case 'bubbles':
      return { f: 0.0900, k: 0.0590, Du, Dv };

    // ── Pearson γ — worm maze with endless grain-boundary instability
    //    xmorphia F220/k510
    case 'maze':
      return { f: 0.0220, k: 0.0510, Du, Dv };

    // ── Pearson β/γ boundary — localised spatiotemporal chaos
    //    xmorphia F260/k510
    case 'chaos':
      return { f: 0.0260, k: 0.0510, Du, Dv };

    default: {
      // Exhaustiveness guard — TypeScript will error if a case is missing
      const _exhaustive: never = species;
      return { f: 0.0545, k: 0.0620, Du, Dv };
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Gray-Scott compute shader
// ─────────────────────────────────────────────────────────────────────────────
//
// Uses two rgba32float storage textures for ping-pong double-buffering.
//   readTex  — current chemical concentrations (read-only)
//   writeTex — next chemical concentrations (write-only)
//
// Laplacian kernel: Pearson 3×3 isotropic weights
//   edge neighbours (N/S/E/W): weight  0.20
//   corner neighbours:          weight  0.05
//   centre:                     weight −1.00
//   Σ = 4×0.20 + 4×0.05 − 1.00 = 0.00  (conservation)
//
// Gray-Scott step (Euler, Δt=1.0):
//   uvv  = u * v * v
//   u′  = u + Du·∇²u − uvv + f·(1−u)
//   v′  = v + Dv·∇²v + uvv − (f+k)·v
//
// ─────────────────────────────────────────────────────────────────────────────

const GS_COMPUTE_SHADER = /* wgsl */`

// ── Uniforms ────────────────────────────────────────────────────────────────
struct GSUniforms {
  f    : f32,   // feed rate
  k    : f32,   // kill rate
  Du   : f32,   // diffusion rate, activator u
  Dv   : f32,   // diffusion rate, inhibitor v
  // grid dims — used for boundary clamping
  width  : u32,
  height : u32,
  _pad0  : u32,
  _pad1  : u32,
}
@group(0) @binding(0) var<uniform> u_gs : GSUniforms;

// ── Ping-pong storage textures ───────────────────────────────────────────────
@group(1) @binding(0) var readTex  : texture_2d<f32>;
@group(1) @binding(1) var writeTex : texture_storage_2d<rgba32float, write>;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn loadUV(coord: vec2i) -> vec2f {
  // Clamp-to-edge boundary condition (Neumann: zero-flux)
  let c = clamp(coord, vec2i(0, 0),
                vec2i(i32(u_gs.width) - 1, i32(u_gs.height) - 1));
  let s = textureLoad(readTex, c, 0);
  return s.rg;   // R=u, G=v
}

// ── Gray-Scott compute step ──────────────────────────────────────────────────
@compute @workgroup_size(${RD_WG}, ${RD_WG})
fn gs_step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let st = vec2i(i32(gid.x), i32(gid.y));

  // Bounds check
  if (gid.x >= u_gs.width || gid.y >= u_gs.height) { return; }

  // ── Sample 3×3 neighbourhood ──────────────────────────────────────────────
  // Pearson isotropic Laplacian kernel:
  //   [ 0.05  0.20  0.05 ]
  //   [ 0.20 -1.00  0.20 ]
  //   [ 0.05  0.20  0.05 ]
  let c  = loadUV(st);
  let n  = loadUV(st + vec2i( 0,  1));
  let s  = loadUV(st + vec2i( 0, -1));
  let e  = loadUV(st + vec2i( 1,  0));
  let w  = loadUV(st + vec2i(-1,  0));
  let ne = loadUV(st + vec2i( 1,  1));
  let nw = loadUV(st + vec2i(-1,  1));
  let se = loadUV(st + vec2i( 1, -1));
  let sw = loadUV(st + vec2i(-1, -1));

  // ── Discrete Laplacian ────────────────────────────────────────────────────
  let lap = 0.20 * (n + s + e + w)
          + 0.05 * (ne + nw + se + sw)
          - 1.00 * c;

  // ── Gray-Scott reaction ───────────────────────────────────────────────────
  let uu   = c.x;
  let vv   = c.y;
  let uvv  = uu * vv * vv;

  // du/dt = Du·∇²u − u·v² + f·(1−u)
  let du   = u_gs.Du * lap.x - uvv + u_gs.f * (1.0 - uu);
  // dv/dt = Dv·∇²v + u·v² − (f+k)·v
  let dv   = u_gs.Dv * lap.y + uvv - (u_gs.f + u_gs.k) * vv;

  // ── Euler integration (Δt = 1.0) ─────────────────────────────────────────
  let newU = clamp(uu + du, 0.0, 1.0);
  let newV = clamp(vv + dv, 0.0, 1.0);

  textureStore(writeTex, st, vec4f(newU, newV, 0.0, 1.0));
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// ReactionDiffusionSim
// ─────────────────────────────────────────────────────────────────────────────

/**
 * WebGPU compute-based Gray-Scott reaction-diffusion simulator.
 *
 * Usage:
 * ```ts
 * const sim = new ReactionDiffusionSim(device, { width: 512, height: 512 });
 * await sim.init();
 * sim.setParams(parameterSpace('coral'));
 *
 * // Per-frame:
 * const enc = device.createCommandEncoder();
 * sim.step(enc);           // runs substeps GS iterations on GPU
 * device.queue.submit([enc.finish()]);
 *
 * // Bind to a render pipeline:
 * const view = sim.outputTexture.createView();
 * ```
 */
export class ReactionDiffusionSim {

  private readonly device:    GPUDevice;
  private readonly width:     number;
  private readonly height:    number;
  private readonly substeps:  number;

  // Ping-pong textures: A ↔ B
  private texA!: GPUTexture;
  private texB!: GPUTexture;

  // Uniform buffer: GSUniforms (32 bytes — 8 × f32/u32)
  private uniformBuf!: GPUBuffer;

  // Pipeline & bind-group layouts
  private pipeline!:     GPUComputePipeline;
  private uniformBGL!:   GPUBindGroupLayout;
  private ppBGL!:        GPUBindGroupLayout;

  // Prebuilt bind-groups for both ping-pong directions
  private uniformBG!:   GPUBindGroup;
  private bgAtoB!:      GPUBindGroup;  // read A → write B
  private bgBtoA!:      GPUBindGroup;  // read B → write A

  /** Current simulation parameters. */
  private params: Required<GrayScottParams>;

  /** Total number of step() calls executed. */
  private frameIndex = 0;

  /** Whether init() has completed. */
  private ready = false;

  constructor(device: GPUDevice, cfg: RDSimConfig = {}) {
    this.device   = device;
    this.width    = cfg.width    ?? RD_DEFAULT_SIZE;
    this.height   = cfg.height   ?? RD_DEFAULT_SIZE;
    this.substeps = cfg.substeps ?? RD_DEFAULT_SUBSTEPS;

    const base = parameterSpace('coral');
    const p    = cfg.params ?? base;
    this.params = {
      f:  p.f,
      k:  p.k,
      Du: p.Du ?? base.Du!,
      Dv: p.Dv ?? base.Dv!,
    };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Init
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Compile the compute pipeline, allocate textures, and seed the grid.
   *
   * Must be called once (and awaited) before the first step().
   */
  async init(): Promise<void> {
    if (this.ready) return;

    this._createTextures();
    this._createUniformBuffer();
    await this._createPipeline();
    this._createBindGroups();
    this._seedGrid();

    this.ready = true;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Per-frame API
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Execute `substeps` Gray-Scott compute passes on the GPU.
   *
   * The encoder is NOT submitted here — caller must submit at end of frame.
   *
   * @param encoder  Current frame's GPUCommandEncoder.
   */
  step(encoder: GPUCommandEncoder): void {
    if (!this.ready) {
      throw new Error('ReactionDiffusionSim.init() must be awaited before step()');
    }

    this._writeUniforms();

    for (let i = 0; i < this.substeps; i++) {
      const pass = encoder.beginComputePass({
        label: `gs-step-${this.frameIndex * this.substeps + i}`,
      });
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, this.uniformBG);
      // Even i → A→B, odd i → B→A  (ping-pong within the same frame)
      pass.setBindGroup(1, (i & 1) === 0 ? this.bgAtoB : this.bgBtoA);

      const wgX = Math.ceil(this.width  / RD_WG);
      const wgY = Math.ceil(this.height / RD_WG);
      pass.dispatchWorkgroups(wgX, wgY);
      pass.end();
    }

    this.frameIndex++;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Parameter control
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Set Gray-Scott parameters directly.
   *
   * Changes take effect on the next step() call (GPU uniform is re-uploaded).
   */
  setParams(p: GrayScottParams): void {
    this.params.f  = p.f;
    this.params.k  = p.k;
    if (p.Du !== undefined) this.params.Du = p.Du;
    if (p.Dv !== undefined) this.params.Dv = p.Dv;
  }

  /**
   * Set parameters from a named species via the Munafo/Pearson parameter space.
   *
   * Equivalent to `sim.setParams(parameterSpace(name))`.
   */
  setSpecies(name: GrayScottSpecies): void {
    this.setParams(parameterSpace(name));
  }

  /** Read current (f, k, Du, Dv) values. */
  get currentParams(): Readonly<Required<GrayScottParams>> {
    return { ...this.params };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Accessors
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * The GPU texture that contains the *latest* simulation output.
   *
   * After substeps iterations per step():
   *   If substeps is even → last write was to texB (even i final = B).
   *   If substeps is odd  → last write was to texA (odd  i final = A).
   *
   * Channel layout: R=u (activator), G=v (inhibitor), B=0, A=1.
   */
  get outputTexture(): GPUTexture {
    // After N substeps (0-indexed): last write was to
    //   substeps even → texB  (final i = substeps-1 is odd → wrote A)
    //   substeps odd  → texA  (final i = substeps-1 is even → wrote B)
    // Track based on (frameIndex * substeps) parity.
    const totalSteps = this.frameIndex * this.substeps;
    return (totalSteps & 1) === 0 ? this.texA : this.texB;
  }

  /** Grid width in cells. */
  get gridWidth(): number  { return this.width;  }
  /** Grid height in cells. */
  get gridHeight(): number { return this.height; }
  /** Total frames stepped so far. */
  get frame(): number      { return this.frameIndex; }

  // ─────────────────────────────────────────────────────────────────────────
  // Seeding
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Reset and re-seed the simulation grid.
   *
   * Initial condition (Pearson 1993):
   *   Most cells: u=1, v=0  (all-A steady state)
   *   Central square (20% of grid): u=0.5, v=0.25  (perturbation seed)
   *   + small uniform noise on the seed region to break symmetry
   */
  resetSeed(): void {
    if (!this.ready) return;
    this._seedGrid();
    this.frameIndex = 0;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private helpers
  // ─────────────────────────────────────────────────────────────────────────

  private _createTextures(): void {
    const desc: GPUTextureDescriptor = {
      size:   { width: this.width, height: this.height },
      format: RD_TEX_FORMAT,
      usage:  GPUTextureUsage.TEXTURE_BINDING   // sampled read
            | GPUTextureUsage.STORAGE_BINDING   // storage write
            | GPUTextureUsage.COPY_DST          // CPU upload (seed)
            | GPUTextureUsage.COPY_SRC,         // readback (optional)
    };
    this.texA = this.device.createTexture({ ...desc, label: 'rd-tex-A' });
    this.texB = this.device.createTexture({ ...desc, label: 'rd-tex-B' });
  }

  private _createUniformBuffer(): void {
    // GSUniforms: 8 × 4 bytes = 32 bytes
    this.uniformBuf = this.device.createBuffer({
      label:  'rd-uniform',
      size:   32,
      usage:  GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._writeUniforms();
  }

  private _writeUniforms(): void {
    const f32 = new Float32Array(8);
    const u32 = new Uint32Array(f32.buffer);
    f32[0] = this.params.f;
    f32[1] = this.params.k;
    f32[2] = this.params.Du;
    f32[3] = this.params.Dv;
    u32[4] = this.width;
    u32[5] = this.height;
    u32[6] = 0;   // _pad0
    u32[7] = 0;   // _pad1
    this.device.queue.writeBuffer(this.uniformBuf, 0, f32);
  }

  private async _createPipeline(): Promise<void> {
    const shaderModule = this.device.createShaderModule({
      label: 'rd-gs-shader',
      code:  GS_COMPUTE_SHADER,
    });

    // BGL 0: uniform buffer
    this.uniformBGL = this.device.createBindGroupLayout({
      label:   'rd-uniform-bgl',
      entries: [
        {
          binding:    0,
          visibility: GPUShaderStage.COMPUTE,
          buffer:     { type: 'uniform' },
        },
      ],
    });

    // BGL 1: ping-pong (read texture + write storage texture)
    this.ppBGL = this.device.createBindGroupLayout({
      label:   'rd-pingpong-bgl',
      entries: [
        {
          binding:    0,
          visibility: GPUShaderStage.COMPUTE,
          texture:    { sampleType: 'unfilterable-float' },
        },
        {
          binding:        1,
          visibility:     GPUShaderStage.COMPUTE,
          storageTexture: { access: 'write-only', format: RD_TEX_FORMAT },
        },
      ],
    });

    const layout = this.device.createPipelineLayout({
      label:              'rd-pipeline-layout',
      bindGroupLayouts:   [this.uniformBGL, this.ppBGL],
    });

    this.pipeline = await this.device.createComputePipelineAsync({
      label:   'rd-gs-pipeline',
      layout,
      compute: { module: shaderModule, entryPoint: 'gs_step' },
    });
  }

  private _createBindGroups(): void {
    // Uniform bind-group (static — same for both ping-pong directions)
    this.uniformBG = this.device.createBindGroup({
      label:   'rd-uniform-bg',
      layout:  this.uniformBGL,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
      ],
    });

    // A→B: read texA, write texB
    this.bgAtoB = this.device.createBindGroup({
      label:   'rd-bg-AtoB',
      layout:  this.ppBGL,
      entries: [
        { binding: 0, resource: this.texA.createView() },
        { binding: 1, resource: this.texB.createView() },
      ],
    });

    // B→A: read texB, write texA
    this.bgBtoA = this.device.createBindGroup({
      label:   'rd-bg-BtoA',
      layout:  this.ppBGL,
      entries: [
        { binding: 0, resource: this.texB.createView() },
        { binding: 1, resource: this.texA.createView() },
      ],
    });
  }

  private _seedGrid(): void {
    const w = this.width;
    const h = this.height;
    // RGBA32Float — 4 channels × 4 bytes each
    const data = new Float32Array(w * h * 4);

    // Background state: u=1, v=0  (stable "all-A" fixed point)
    for (let i = 0; i < w * h; i++) {
      data[i * 4 + 0] = 1.0;  // u
      data[i * 4 + 1] = 0.0;  // v
      data[i * 4 + 2] = 0.0;
      data[i * 4 + 3] = 1.0;
    }

    // Central seed square: u=0.5, v=0.25 + small noise (Pearson 1993 IC)
    const cx    = Math.floor(w * 0.5);
    const cy    = Math.floor(h * 0.5);
    const half  = Math.floor(Math.min(w, h) * 0.10);  // 10% of grid

    for (let dy = -half; dy <= half; dy++) {
      for (let dx = -half; dx <= half; dx++) {
        const px = cx + dx;
        const py = cy + dy;
        if (px < 0 || px >= w || py < 0 || py >= h) continue;

        const idx = (py * w + px) * 4;
        // Small uniform noise to break spatial symmetry
        const noise = (Math.random() - 0.5) * 0.05;
        data[idx + 0] = 0.50 + noise;   // u
        data[idx + 1] = 0.25 + noise;   // v
      }
    }

    // Upload to texA; texB starts as all-zero (GPU default)
    this.device.queue.writeTexture(
      { texture: this.texA },
      data,
      { bytesPerRow: w * 4 * 4, rowsPerImage: h },
      { width: w, height: h },
    );

    // Ensure texB has a defined initial state too (copy texA → texB)
    const enc = this.device.createCommandEncoder({ label: 'rd-seed-copy' });
    enc.copyTextureToTexture(
      { texture: this.texA },
      { texture: this.texB },
      { width: w, height: h },
    );
    this.device.queue.submit([enc.finish()]);

    // Reset frame counter so outputTexture tracks correctly
    this.frameIndex = 0;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Destroy
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Release all WebGPU resources held by this simulator.
   * The instance must not be used after destroy().
   */
  destroy(): void {
    this.texA.destroy();
    this.texB.destroy();
    this.uniformBuf.destroy();
    this.ready = false;
  }
}
