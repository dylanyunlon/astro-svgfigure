/**
 * lattice-boltzmann-bg.ts
 *
 * 双层流体系统 — Lattice Boltzmann Method (LBM) 宏观背景流场 + SPH 粒子层
 *
 * 架构设计：
 *   Layer 0 — LBM 512×512 grid (WebGPU compute)
 *     • 直接移植 lygia/simulate/latticeBoltzmann.wgsl 规则
 *     • Ping-pong 双缓冲纹理：rgba16float，XY=有序速度，B=无序能量，W=内部质量
 *     • 每帧在 GPU 上执行一次 LBM 步迭代，输出 velocityTex 供下层采样
 *
 *   Layer 1 — SPH 65536 粒子
 *     • 粒子在集成阶段额外采样 LBM velocityTex 作为外力场
 *     • 宏观对流方向由 LBM 驱动；局部压力/粘性细节由 SPH 保留
 *     • 耦合强度由 lbmInfluence 参数控制（0=纯SPH，1=完全跟随LBM流场）
 *
 * 参考：
 *   upstream/lygia/simulate/latticeBoltzmann.wgsl — Patricio Gonzalez Vivo
 *   Wyatt Flanders, "Me And My Neighborhood" (LBM paper)
 *   https://wyattflanders.com/MeAndMyNeighborhood.pdf
 */

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/** LBM 格子分辨率（正方形） */
export const LBM_GRID = 512;

/** SPH 粒子数量上限 */
export const SPH_PARTICLE_COUNT = 65536;

/** LBM 纹理格式：RGBA16Float — XY=速度，B=无序能量，W=质量 */
const LBM_TEX_FORMAT: GPUTextureFormat = "rgba16float";

/** 每 workgroup 处理的格点数（16×16 = 256 线程） */
const LBM_WG = 16;

/** SPH velocity sampling compute workgroup size */
const SPH_WG = 256;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — LBM Compute Shader
//   移植自 lygia/simulate/latticeBoltzmann.wgsl (Patricio Gonzalez Vivo)
//   在 WebGPU compute 环境中以 storageTexture 代替 sampler2D
// ─────────────────────────────────────────────────────────────────────────────

const LBM_SHADER = /* wgsl */`

// ── Uniforms ──────────────────────────────────────────────────────────────────
struct LBMUniforms {
  pixel  : vec2f,   // 1.0 / vec2f(LBM_GRID)
  force  : vec2f,   // 外部体力（鼠标扰动 / 重力偏置）
  dt     : f32,     // 时间步（目前仅做 force scale）
  _pad   : vec3f,
}
@group(0) @binding(0) var<uniform> u : LBMUniforms;

// Ping-pong 读写纹理
@group(1) @binding(0) var readTex  : texture_2d<f32>;
@group(1) @binding(1) var writeTex : texture_storage_2d<rgba16float, write>;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn loadCell(coord: vec2i) -> vec4f {
  let dim = vec2i(${LBM_GRID});
  // 边界反射
  let c = clamp(coord, vec2i(0), dim - vec2i(1));
  return textureLoad(readTex, c, 0);
}

// Rule 1: All my energy moves with me.
//   Find my previous position by backtracking along my own velocity.
fn prevPosSample(st: vec2i) -> vec4f {
  let d      = loadCell(st);
  // offset in pixel units → back-project
  let prevST = vec2f(st) - d.xy;   // d.xy is velocity in pixel-space
  let iST    = vec2i(i32(prevST.x), i32(prevST.y));
  return loadCell(iST);
}

// ── Main LBM step ─────────────────────────────────────────────────────────────
// Implements the four rules from the lygia LBM paper:
//   R1. Ordered energy moves with the cell (advection / semi-Lagrangian)
//   R2. Disordered energy B diffuses symmetrically to neighbours
//   R3. Gradient of B drives ordered velocity XY
//   R4. Divergence of XY feeds back into B (disorder)
// Plus mass conservation and force injection.

@compute @workgroup_size(${LBM_WG}, ${LBM_WG})
fn lbm_step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let st = vec2i(gid.xy);
  if (st.x >= ${LBM_GRID} || st.y >= ${LBM_GRID}) { return; }

  // Neighbourhood (advected samples)
  var d  = prevPosSample(st);
  let pX = prevPosSample(st + vec2i( 1,  0));
  let pY = prevPosSample(st + vec2i( 0,  1));
  let nX = prevPosSample(st + vec2i(-1,  0));
  let nY = prevPosSample(st + vec2i( 0, -1));

  // R2: Disordered energy B diffuses completely from neighbours
  d.z = (pX.z + pY.z + nX.z + nY.z) * 0.25;

  // R3: Gradient of B pushes ordered velocity
  d.x += (nX.z - pX.z) * 0.25;
  d.y += (nY.z - pY.z) * 0.25;

  // R4: Divergence of velocity creates disorder
  d.z += (nX.x - pX.x + nY.y - pY.y) * 0.25;

  // Mass conservation: flux divergence
  d.w += (nX.x * nX.w - pX.x * pX.w + nY.y * nY.w - pY.y * pY.w) * 0.25;

  // External force injection (scaled by saturated mass weight)
  let massW = clamp(d.w, 0.0, 1.0);
  d.x += u.force.x * massW * u.pixel.x;
  d.y += u.force.y * massW * u.pixel.y;

  // Boundary: zero velocity at grid edges (no-slip)
  if (st.x <= 0 || st.y <= 0 || st.x >= ${LBM_GRID} - 1 || st.y >= ${LBM_GRID} - 1) {
    d.x = 0.0;
    d.y = 0.0;
  }

  // Clamp to stable range
  d = clamp(d, vec4f(-0.9999, -0.9999, 0.0, 0.0), vec4f(0.9999, 0.9999, 1.0, 1.0));

  textureStore(writeTex, st, d);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — SPH ↔ LBM Coupling Shader
//   采样 LBM velocity texture，将宏观流速作为外力注入 SPH 粒子速度
// ─────────────────────────────────────────────────────────────────────────────

const SPH_LBM_COUPLE_SHADER = /* wgsl */`

struct CoupleUniforms {
  domainW      : f32,   // SPH 世界宽度（物理单位）
  domainH      : f32,   // SPH 世界高度（物理单位）
  lbmInfluence : f32,   // 耦合强度 [0,1]
  dt           : f32,
  count        : u32,
  _pad         : vec3u,
}
@group(0) @binding(0) var<uniform> cu : CoupleUniforms;

// LBM velocity texture（只读）
@group(1) @binding(0) var lbmTex  : texture_2d<f32>;
@group(1) @binding(1) var lbmSamp : sampler;

// SPH particle buffers
@group(2) @binding(0) var<storage, read>       posX : array<f32>;
@group(2) @binding(1) var<storage, read>       posY : array<f32>;
@group(2) @binding(2) var<storage, read_write> velX : array<f32>;
@group(2) @binding(3) var<storage, read_write> velY : array<f32>;

// LBM velocity is stored in pixels/frame; convert to world-space velocity.
// The LBM grid covers the same physical domain as the SPH world.
fn lbmVelocityAt(px: f32, py: f32) -> vec2f {
  let uv       = vec2f(px / cu.domainW, py / cu.domainH);
  let cell     = textureSampleLevel(lbmTex, lbmSamp, uv, 0.0);
  // XY channel: velocity in normalised pixel-space → scale to world-space
  let worldVel = cell.xy * vec2f(cu.domainW, cu.domainH);
  return worldVel;
}

@compute @workgroup_size(${SPH_WG})
fn couple_sph_lbm(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= cu.count) { return; }

  let macroVel = lbmVelocityAt(posX[i], posY[i]);

  // Blend SPH micro-velocity toward LBM macro-velocity
  //   v_new = lerp(v_sph, v_lbm, influence)
  //   → equivalent to applying correction force: (v_lbm - v_sph) * influence / dt
  let alpha  = clamp(cu.lbmInfluence, 0.0, 1.0);
  velX[i] = velX[i] + (macroVel.x - velX[i]) * alpha;
  velY[i] = velY[i] + (macroVel.y - velY[i]) * alpha;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Public API types
// ─────────────────────────────────────────────────────────────────────────────

export interface LBMConfig {
  /** External body force injected each frame (e.g. mouse drag).
   *  Units: grid-pixels per frame. */
  force?: { x: number; y: number };

  /** How strongly the LBM flow field steers SPH particles.
   *  0 = pure SPH, 1 = SPH fully follows LBM macro-flow.
   *  Recommended range: 0.02–0.15. Default: 0.05. */
  lbmInfluence?: number;

  /** SPH world physical dimensions (same coordinate space as SPH buffers). */
  domainW?: number;
  domainH?: number;
}

export interface LBMBuffers {
  /** SPH position X (f32 array, length = SPH_PARTICLE_COUNT) */
  posX: GPUBuffer;
  /** SPH position Y */
  posY: GPUBuffer;
  /** SPH velocity X (read_write — LBM will add macro-flow correction) */
  velX: GPUBuffer;
  /** SPH velocity Y */
  velY: GPUBuffer;
  /** Actual live particle count */
  count: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// LatticeBoltzmannBackground
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 管理 LBM 背景流场的全 GPU 生命周期：
 *   1. 初始化 512×512 ping-pong float 纹理对
 *   2. 每帧 step()：运行一次 LBM compute pass（更新背景流场）
 *   3. 每帧 coupleSPH()：将 LBM 宏观速度注入 SPH 粒子
 *
 * 使用示例：
 * ```ts
 * const lbm = await LatticeBoltzmannBackground.create(device, { lbmInfluence: 0.06 });
 * // 在每帧渲染循环中：
 * const cmd = device.createCommandEncoder();
 * lbm.step(cmd, mouseForce);
 * lbm.coupleSPH(cmd, sphBuffers);
 * device.queue.submit([cmd.finish()]);
 * // 获取流场纹理（可绑定到渲染 shader 作为背景）：
 * const bgTex = lbm.velocityTexture;
 * ```
 */
export class LatticeBoltzmannBackground {
  readonly device: GPUDevice;

  // ── LBM ping-pong textures ──────────────────────────────────────────────────
  private texA!: GPUTexture;      // read  on even frames, write on odd frames
  private texB!: GPUTexture;      // write on even frames, read  on odd frames
  private viewA!: GPUTextureView;
  private viewB!: GPUTextureView;
  private frameIndex = 0;

  // ── Pipelines ───────────────────────────────────────────────────────────────
  private lbmPipeline!: GPUComputePipeline;
  private couplePipeline!: GPUComputePipeline;

  // ── Bind group layouts ──────────────────────────────────────────────────────
  private lbmUniformBGL!: GPUBindGroupLayout;
  private lbmTexBGL!: GPUBindGroupLayout;

  private coupleUniformBGL!: GPUBindGroupLayout;
  private coupleTexBGL!: GPUBindGroupLayout;
  private coupleParticleBGL!: GPUBindGroupLayout;

  // ── Uniform buffers ─────────────────────────────────────────────────────────
  private lbmUniformBuf!: GPUBuffer;     // LBMUniforms
  private coupleUniformBuf!: GPUBuffer;  // CoupleUniforms

  // ── Sampler for coupling pass ───────────────────────────────────────────────
  private linearSampler!: GPUSampler;

  // ── Cached bind groups (rebuilt when textures swap) ─────────────────────────
  private lbmBG_AtoB!: GPUBindGroup;   // read A, write B
  private lbmBG_BtoA!: GPUBindGroup;   // read B, write A
  private coupleTexBG_A!: GPUBindGroup; // sample A (even frames)
  private coupleTexBG_B!: GPUBindGroup; // sample B (odd  frames)

  // ── Config ──────────────────────────────────────────────────────────────────
  private cfg: Required<LBMConfig>;

  // ─────────────────────────────────────────────────────────────────────────
  // Factory
  // ─────────────────────────────────────────────────────────────────────────

  private constructor(device: GPUDevice, cfg: Required<LBMConfig>) {
    this.device = device;
    this.cfg    = cfg;
  }

  /**
   * 异步初始化 LBM 背景系统。
   * @param device  已初始化的 WebGPU device
   * @param cfg     可选配置，见 LBMConfig
   */
  static async create(
    device: GPUDevice,
    cfg: LBMConfig = {}
  ): Promise<LatticeBoltzmannBackground> {
    const fullCfg: Required<LBMConfig> = {
      force:         cfg.force        ?? { x: 0, y: 0 },
      lbmInfluence:  cfg.lbmInfluence ?? 0.05,
      domainW:       cfg.domainW      ?? 1.0,
      domainH:       cfg.domainH      ?? 1.0,
    };

    const bg = new LatticeBoltzmannBackground(device, fullCfg);
    await bg._init();
    return bg;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Initialisation
  // ─────────────────────────────────────────────────────────────────────────

  private async _init(): Promise<void> {
    this._createTextures();
    this._createSampler();
    this._createUniformBuffers();
    await this._createPipelines();
    this._createBindGroups();
    this._seedInitialState();
  }

  /** Allocate ping-pong texture pair. */
  private _createTextures(): void {
    const desc: GPUTextureDescriptor = {
      size:   { width: LBM_GRID, height: LBM_GRID },
      format: LBM_TEX_FORMAT,
      usage:
        GPUTextureUsage.TEXTURE_BINDING |   // readable as sampler
        GPUTextureUsage.STORAGE_BINDING  |   // writable by compute
        GPUTextureUsage.COPY_DST,            // for initial seed upload
    };
    this.texA  = this.device.createTexture({ ...desc, label: "lbm-texA" });
    this.texB  = this.device.createTexture({ ...desc, label: "lbm-texB" });
    this.viewA = this.texA.createView({ label: "lbm-viewA" });
    this.viewB = this.texB.createView({ label: "lbm-viewB" });
  }

  private _createSampler(): void {
    this.linearSampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });
  }

  private _createUniformBuffers(): void {
    // LBMUniforms: pixel(2f) + force(2f) + dt(1f) + pad(3f) = 32 bytes
    this.lbmUniformBuf = this.device.createBuffer({
      label:  "lbm-uniform",
      size:   32,
      usage:  GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // CoupleUniforms: domainW/H + lbmInfluence + dt + count + pad3u = 32 bytes
    this.coupleUniformBuf = this.device.createBuffer({
      label:  "couple-uniform",
      size:   32,
      usage:  GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  private async _createPipelines(): Promise<void> {
    // ── LBM pipeline ──────────────────────────────────────────────────────────
    this.lbmUniformBGL = this.device.createBindGroupLayout({
      label:   "lbm-uniform-bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE,
          buffer:  { type: "uniform" } },
      ],
    });

    this.lbmTexBGL = this.device.createBindGroupLayout({
      label:   "lbm-tex-bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE,
          texture: { sampleType: "float", viewDimension: "2d" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: "write-only", format: LBM_TEX_FORMAT, viewDimension: "2d" } },
      ],
    });

    const lbmModule = this.device.createShaderModule({
      label: "lbm-shader",
      code:  LBM_SHADER,
    });

    this.lbmPipeline = await this.device.createComputePipelineAsync({
      label:  "lbm-pipeline",
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.lbmUniformBGL, this.lbmTexBGL],
      }),
      compute: { module: lbmModule, entryPoint: "lbm_step" },
    });

    // ── SPH coupling pipeline ─────────────────────────────────────────────────
    this.coupleUniformBGL = this.device.createBindGroupLayout({
      label:   "couple-uniform-bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" } },
      ],
    });

    this.coupleTexBGL = this.device.createBindGroupLayout({
      label:   "couple-tex-bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE,
          texture: { sampleType: "float", viewDimension: "2d" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE,
          sampler: { type: "filtering" } },
      ],
    });

    this.coupleParticleBGL = this.device.createBindGroupLayout({
      label:   "couple-particle-bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" } },
      ],
    });

    const coupleModule = this.device.createShaderModule({
      label: "couple-shader",
      code:  SPH_LBM_COUPLE_SHADER,
    });

    this.couplePipeline = await this.device.createComputePipelineAsync({
      label:  "couple-pipeline",
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.coupleUniformBGL, this.coupleTexBGL, this.coupleParticleBGL],
      }),
      compute: { module: coupleModule, entryPoint: "couple_sph_lbm" },
    });
  }

  /** Build bind groups for both ping-pong directions. */
  private _createBindGroups(): void {
    const dev = this.device;

    const lbmUniformBG = dev.createBindGroup({
      label:  "lbm-uniform-bg",
      layout: this.lbmUniformBGL,
      entries: [{ binding: 0, resource: { buffer: this.lbmUniformBuf } }],
    });
    // Bind group for LBM is shared across both directions (uniforms only).
    // We keep separate tex-BGs per direction.
    this._lbmUniformBG = lbmUniformBG;

    this.lbmBG_AtoB = dev.createBindGroup({
      label:  "lbm-AtoB",
      layout: this.lbmTexBGL,
      entries: [
        { binding: 0, resource: this.viewA },
        { binding: 1, resource: this.viewB },
      ],
    });

    this.lbmBG_BtoA = dev.createBindGroup({
      label:  "lbm-BtoA",
      layout: this.lbmTexBGL,
      entries: [
        { binding: 0, resource: this.viewB },
        { binding: 1, resource: this.viewA },
      ],
    });

    this.coupleTexBG_A = dev.createBindGroup({
      label:  "couple-tex-A",
      layout: this.coupleTexBGL,
      entries: [
        { binding: 0, resource: this.viewA },
        { binding: 1, resource: this.linearSampler },
      ],
    });

    this.coupleTexBG_B = dev.createBindGroup({
      label:  "couple-tex-B",
      layout: this.coupleTexBGL,
      entries: [
        { binding: 0, resource: this.viewB },
        { binding: 1, resource: this.linearSampler },
      ],
    });
  }

  // extra private field for uniform bind group (shared)
  private _lbmUniformBG!: GPUBindGroup;

  /** Seed the LBM grid with a small random thermal noise to kick-start flow. */
  private _seedInitialState(): void {
    // Fill texA with low-energy noise: XY~0, B=small noise, W=1 (full mass)
    const pixels = LBM_GRID * LBM_GRID;
    // rgba16float → 4 channels × 2 bytes = 8 bytes per pixel
    const data = new Float32Array(pixels * 4);
    for (let i = 0; i < pixels; i++) {
      const base  = i * 4;
      data[base + 0] = (Math.random() - 0.5) * 0.01;  // vx
      data[base + 1] = (Math.random() - 0.5) * 0.01;  // vy
      data[base + 2] = Math.random() * 0.05;           // disordered energy B
      data[base + 3] = 0.8 + Math.random() * 0.2;     // mass W
    }

    // Upload via writeTexture (float32 → gpu will convert to float16)
    this.device.queue.writeTexture(
      { texture: this.texA },
      data,
      { bytesPerRow: LBM_GRID * 4 * 4, rowsPerImage: LBM_GRID },
      { width: LBM_GRID, height: LBM_GRID },
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Per-frame API
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * 执行一次 LBM 步迭代。
   * 应在 SPH 步迭代之前调用，以便 coupleSPH 能使用最新的流场。
   *
   * @param encoder  当前帧的 GPUCommandEncoder
   * @param force    可选外部体力（如鼠标速度），单位：格点/帧
   * @param dt       时间步（默认 1.0）
   */
  step(
    encoder: GPUCommandEncoder,
    force:   { x: number; y: number } = this.cfg.force,
    dt      = 1.0,
  ): void {
    const px = 1.0 / LBM_GRID;
    const uniformData = new Float32Array([
      px, px,                 // pixel.xy
      force.x, force.y,       // force.xy
      dt,                     // dt
      0, 0, 0,                // _pad
    ]);
    this.device.queue.writeBuffer(this.lbmUniformBuf, 0, uniformData);

    const pass = encoder.beginComputePass({ label: "lbm-step" });
    pass.setPipeline(this.lbmPipeline);
    pass.setBindGroup(0, this._lbmUniformBG);
    // even frame: A→B, odd frame: B→A
    pass.setBindGroup(1, (this.frameIndex & 1) === 0 ? this.lbmBG_AtoB : this.lbmBG_BtoA);

    const wg = Math.ceil(LBM_GRID / LBM_WG);
    pass.dispatchWorkgroups(wg, wg);
    pass.end();

    this.frameIndex++;
  }

  /**
   * 将 LBM 宏观速度耦合进 SPH 粒子速度缓冲。
   * 在 SPH density/force 步之前或之后调用均可；
   * 推荐在 integrate 步之后调用，让 LBM 校正在下一帧生效（避免刚度震荡）。
   *
   * @param encoder     当前帧的 GPUCommandEncoder
   * @param sphBuffers  SPH 粒子缓冲组
   * @param influence   覆盖默认耦合强度（可选）
   */
  coupleSPH(
    encoder:    GPUCommandEncoder,
    sphBuffers: LBMBuffers,
    influence?: number,
  ): void {
    const alpha = influence ?? this.cfg.lbmInfluence;

    const uniformData = new Float32Array([
      this.cfg.domainW,     // domainW
      this.cfg.domainH,     // domainH
      alpha,                // lbmInfluence
      1.0,                  // dt (normalised)
      sphBuffers.count,     // count (written as float bits → see note below)
      0, 0, 0,              // _pad (3× f32 placeholder)
    ]);
    // 'count' is a u32 in the shader; write raw u32 bits into the f32 slot
    const u32view = new Uint32Array(uniformData.buffer);
    u32view[4] = sphBuffers.count;
    this.device.queue.writeBuffer(this.coupleUniformBuf, 0, uniformData);

    // The "current" LBM output texture is whichever was *written* last step.
    // After N steps: even N → texB was last written; odd N → texA was last written.
    const readCurrent = (this.frameIndex & 1) === 0
      ? this.coupleTexBG_B   // frameIndex just incremented in step(), so "odd step wrote B"
      : this.coupleTexBG_A;

    const particleBG = this.device.createBindGroup({
      label:  "couple-particle-bg",
      layout: this.coupleParticleBGL,
      entries: [
        { binding: 0, resource: { buffer: sphBuffers.posX } },
        { binding: 1, resource: { buffer: sphBuffers.posY } },
        { binding: 2, resource: { buffer: sphBuffers.velX } },
        { binding: 3, resource: { buffer: sphBuffers.velY } },
      ],
    });

    const pass = encoder.beginComputePass({ label: "sph-lbm-couple" });
    pass.setPipeline(this.couplePipeline);
    pass.setBindGroup(0, this._coupleUniformBG());
    pass.setBindGroup(1, readCurrent);
    pass.setBindGroup(2, particleBG);

    const wg = Math.ceil(sphBuffers.count / SPH_WG);
    pass.dispatchWorkgroups(wg);
    pass.end();
  }

  // ── Config update helpers ──────────────────────────────────────────────────

  /** 更新外部体力（例如每帧鼠标速度归一化后传入）。 */
  setForce(x: number, y: number): void {
    this.cfg.force = { x, y };
  }

  /** 动态调整 LBM↔SPH 耦合强度。 */
  setInfluence(alpha: number): void {
    this.cfg.lbmInfluence = Math.max(0, Math.min(1, alpha));
  }

  // ── Accessors ──────────────────────────────────────────────────────────────

  /**
   * 当前帧的 LBM 速度纹理视图（已完成 step() 后的输出）。
   * 可直接绑定到渲染 shader 作为背景流场可视化或粒子着色参考。
   *
   * 通道含义：
   *   RG = 有序速度 XY（[-1,1] 归一化像素空间）
   *   B  = 无序能量（湍流强度代理，[0,1]）
   *   A  = 内部质量（流体密度代理，[0,1]）
   */
  get velocityTexture(): GPUTextureView {
    // After step(), frameIndex has been incremented.
    // Even frameIndex → last write was to texB.
    // Odd  frameIndex → last write was to texA.
    return (this.frameIndex & 1) === 0 ? this.viewB : this.viewA;
  }

  /** 帧计数（从 0 开始，每次 step() +1）。 */
  get frame(): number { return this.frameIndex; }

  // ─────────────────────────────────────────────────────────────────────────
  // Destroy
  // ─────────────────────────────────────────────────────────────────────────

  destroy(): void {
    this.texA.destroy();
    this.texB.destroy();
    this.lbmUniformBuf.destroy();
    this.coupleUniformBuf.destroy();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private helpers
  // ─────────────────────────────────────────────────────────────────────────

  /** Lazily cached couple uniform bind group (uniforms don't change between calls). */
  private _coupleUniformBGCache: GPUBindGroup | null = null;
  private _coupleUniformBG(): GPUBindGroup {
    if (!this._coupleUniformBGCache) {
      this._coupleUniformBGCache = this.device.createBindGroup({
        label:  "couple-uniform-bg",
        layout: this.coupleUniformBGL,
        entries: [{ binding: 0, resource: { buffer: this.coupleUniformBuf } }],
      });
    }
    return this._coupleUniformBGCache;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience factory for the default dual-layer scene
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 创建为标准双层流体场景优化的 LBM 背景系统：
 *   - 512×512 LBM grid（宏观对流）
 *   - 65536 SPH 粒子（局部细节）
 *   - 默认耦合强度 5%（宏观流向轻柔引导 SPH，不压制局部湍流）
 *
 * @param device   WebGPU device
 * @param domainW  SPH 物理域宽度（与 SPHGPUOrchestrator SimParams.domainW 一致）
 * @param domainH  SPH 物理域高度
 */
export async function createDualLayerFluid(
  device:  GPUDevice,
  domainW: number,
  domainH: number,
): Promise<LatticeBoltzmannBackground> {
  return LatticeBoltzmannBackground.create(device, {
    lbmInfluence: 0.05,
    domainW,
    domainH,
    force: { x: 0, y: 0 },
  });
}
