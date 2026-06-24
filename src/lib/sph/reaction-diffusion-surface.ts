/**
 * reaction-diffusion-surface.ts — M762: Per-Cell Reaction-Diffusion Surface
 * ─────────────────────────────────────────────────────────────────────────────
 * Each cell in the Transformer diagram owns a private 64×64 Gray-Scott
 * reaction-diffusion patch that tiles its visual surface.  Species-specific
 * (f, k, Du, Dv) parameters produce distinct morphologies per cell type
 * (coral growth on attention heads, spirals on FFN, dots on add-&-norm, …).
 *
 * 架构设计：
 *   • 每个 Cell 拥有独立的 64×64 RD patch（CellRDSurface 实例）
 *   • 双缓冲 ping-pong：两张 rgba32float storage texture（per cell）
 *       ch R = u (activator)
 *       ch G = v (inhibitor)
 *       ch B = reserved (0)
 *       ch A = 1
 *   • 每个 cell 根据 species 查表获取 (f, k, Du, Dv)
 *   • CellRDSurfaceManager 批量管理所有 cell patches，单次 dispatch 处理全部
 *   • 输出纹理可直接绑定到 cell 渲染 pipeline 作为表面纹理
 *
 * Gray-Scott 方程（每步 Δt=1.0）：
 *   du/dt = Du·∇²u − u·v² + f·(1−u)
 *   dv/dt = Dv·∇²v + u·v² − (f+k)·v
 *
 * 参考：
 *   reaction-diffusion.ts  — global grid RD (M743)
 *   cell-body-bridge.ts    — SPECIES_ORDER, speciesToIndex
 *   cell-material-system.ts — per-species material pipeline
 *
 * Research: xiaodi #M762 — cell-pubsub-loop
 */

import {
  type GrayScottParams,
  type GrayScottSpecies,
  speciesGrayScottParams,
  SPECIES_GRAYSCOTT_MAP,
  RD_MAX_SPECIES,
} from './reaction-diffusion';

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

// [orphan-precise] /** Per-cell surface patch resolution (64×64 texels). */
export const CELL_RD_SIZE = 64;

/** Storage texture format for ping-pong buffers (R=u, G=v, B=0, A=1). */
const SURFACE_TEX_FORMAT: GPUTextureFormat = 'rgba32float';

/** Compute workgroup tile size (8×8 = 64 threads per workgroup). */
const SURFACE_WG = 8;

/** Workgroups per axis for a 64×64 patch: ceil(64/8) = 8. */
const SURFACE_WG_COUNT = Math.ceil(CELL_RD_SIZE / SURFACE_WG);

/** Default substeps per frame for surface RD. */
export const CELL_RD_DEFAULT_SUBSTEPS = 4;

/** Maximum number of cell patches managed by a single CellRDSurfaceManager. */
export const CELL_RD_MAX_CELLS = 64;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/** Configuration for a single cell's RD surface patch. */
export interface CellRDSurfaceConfig {
  /** Unique cell identifier (e.g. "self_attn", "ffn_0"). */
  cellId: string;
  /** Cell species string (e.g. "cil-eye"). Determines GS parameters. */
  species: string;
  /**
   * Optional parameter override. When provided, these (f, k, Du, Dv) values
   * take precedence over the species-based lookup.
   */
  paramsOverride?: GrayScottParams;
}

/** Configuration for CellRDSurfaceManager constructor. */
export interface CellRDManagerConfig {
  /** Number of GS substeps per step() call. Default: CELL_RD_DEFAULT_SUBSTEPS. */
  substeps?: number;
}

/** Read-only snapshot of a single cell patch's state. */
export interface CellRDSnapshot {
  cellId: string;
  species: string;
  params: Readonly<Required<GrayScottParams>>;
  /** The GPU texture containing the latest RD output for this cell. */
  outputTexture: GPUTexture;
  /** Index within the manager's cell array (used for batched dispatch). */
  slotIndex: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Per-cell surface Gray-Scott compute shader
// ─────────────────────────────────────────────────────────────────────────────
//
// Each dispatch covers a single 64×64 cell patch.  The cell's (f,k,Du,Dv)
// are passed via a uniform buffer that the manager updates before dispatch.
//
// Ping-pong is managed externally by swapping bind-group assignments between
// texA ↔ texB on alternating substeps.
//
// Boundary condition: wrap (torus topology) — the cell surface tiles
// seamlessly, avoiding edge artifacts on small patches.
//
// ─────────────────────────────────────────────────────────────────────────────

const SURFACE_GS_SHADER = /* wgsl */`

// ── Constants ───────────────────────────────────────────────────────────────
const GRID_SIZE: u32 = ${CELL_RD_SIZE}u;

// ── Uniforms ────────────────────────────────────────────────────────────────
struct SurfaceGSUniforms {
  f    : f32,   // feed rate
  k    : f32,   // kill rate
  Du   : f32,   // diffusion rate, activator u
  Dv   : f32,   // diffusion rate, inhibitor v
}
@group(0) @binding(0) var<uniform> u_gs : SurfaceGSUniforms;

// ── Ping-pong storage textures ───────────────────────────────────────────────
@group(1) @binding(0) var readTex  : texture_2d<f32>;
@group(1) @binding(1) var writeTex : texture_storage_2d<rgba32float, write>;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn wrapCoord(coord: vec2i) -> vec2i {
  // Toroidal wrap for seamless tiling on the cell surface
  let s = i32(GRID_SIZE);
  return vec2i(
    ((coord.x % s) + s) % s,
    ((coord.y % s) + s) % s,
  );
}

fn loadUV(coord: vec2i) -> vec2f {
  let c = wrapCoord(coord);
  let s = textureLoad(readTex, c, 0);
  return s.rg;   // R=u, G=v
}

// ── Gray-Scott compute step ──────────────────────────────────────────────────
@compute @workgroup_size(${SURFACE_WG}, ${SURFACE_WG})
fn gs_surface_step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let st = vec2i(i32(gid.x), i32(gid.y));

  // Bounds check (grid is exactly 64×64, but guard anyway)
  if (gid.x >= GRID_SIZE || gid.y >= GRID_SIZE) { return; }

  // ── Sample 3×3 neighbourhood (toroidal wrap) ─────────────────────────────
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
// CellRDSurface — single cell patch
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A single cell's 64×64 Gray-Scott reaction-diffusion surface patch.
 *
 * Holds two ping-pong textures and the resolved (f, k, Du, Dv) parameters
 * derived from the cell's species.  The CellRDSurfaceManager orchestrates
 * compute dispatch; this class owns the per-cell GPU resources.
 */
export class CellRDSurface {

  readonly cellId:  string;
  readonly species: string;
  readonly params:  Required<GrayScottParams>;

  /** Ping-pong texture A (64×64 rgba32float). */
  texA: GPUTexture;
  /** Ping-pong texture B (64×64 rgba32float). */
  texB: GPUTexture;

  /** Current ping-pong phase: false = A is read / B is write, true = B is read / A is write. */
  private phase = false;

  constructor(
    private readonly device: GPUDevice,
    cfg: CellRDSurfaceConfig,
  ) {
    this.cellId  = cfg.cellId;
    this.species = cfg.species;

    // Resolve parameters: override > species lookup > coral fallback
    if (cfg.paramsOverride) {
      const base = speciesGrayScottParams(cfg.species);
      this.params = {
        f:  cfg.paramsOverride.f,
        k:  cfg.paramsOverride.k,
        Du: cfg.paramsOverride.Du ?? base.Du,
        Dv: cfg.paramsOverride.Dv ?? base.Dv,
      };
    } else {
      this.params = speciesGrayScottParams(cfg.species);
    }

    // Allocate ping-pong textures
    const desc: GPUTextureDescriptor = {
      size:   { width: CELL_RD_SIZE, height: CELL_RD_SIZE },
      format: SURFACE_TEX_FORMAT,
      usage:  GPUTextureUsage.TEXTURE_BINDING
            | GPUTextureUsage.STORAGE_BINDING
            | GPUTextureUsage.COPY_DST
            | GPUTextureUsage.COPY_SRC,
    };
    this.texA = device.createTexture({ ...desc, label: `rd-surf-${cfg.cellId}-A` });
    this.texB = device.createTexture({ ...desc, label: `rd-surf-${cfg.cellId}-B` });

    this._seed();
  }

  /** The texture that was most recently written to (i.e. the latest output). */
  get outputTexture(): GPUTexture {
    return this.phase ? this.texA : this.texB;
  }

  /** The texture to read from on the current substep. */
  get readTexture(): GPUTexture {
    return this.phase ? this.texB : this.texA;
  }

  /** The texture to write to on the current substep. */
  get writeTexture(): GPUTexture {
    return this.phase ? this.texA : this.texB;
  }

  /** Flip the ping-pong phase after a compute substep. */
  flip(): void {
    this.phase = !this.phase;
  }

  /** Reset phase tracking (called when the manager resets all patches). */
  resetPhase(): void {
    this.phase = false;
  }

  /**
   * Re-seed the patch with Pearson 1993 initial condition.
   *
   * Background: u=1, v=0.
   * Central region (~20% of 64 = 13 texels half-extent): u=0.5, v=0.25 + noise.
   */
  reseed(): void {
    this._seed();
    this.phase = false;
  }

  /** Release GPU resources. */
  destroy(): void {
    this.texA.destroy();
    this.texB.destroy();
  }

  // ── Private ─────────────────────────────────────────────────────────────

  private _seed(): void {
    const w = CELL_RD_SIZE;
    const h = CELL_RD_SIZE;
    const data = new Float32Array(w * h * 4);

    // Background: u=1, v=0
    for (let i = 0; i < w * h; i++) {
      data[i * 4 + 0] = 1.0;   // u
      data[i * 4 + 1] = 0.0;   // v
      data[i * 4 + 2] = 0.0;
      data[i * 4 + 3] = 1.0;
    }

    // Seed: central square with perturbation
    const cx   = w >> 1;        // 32
    const cy   = h >> 1;        // 32
    const half = Math.max(2, Math.floor(Math.min(w, h) * 0.10));  // ~6 texels

    for (let dy = -half; dy <= half; dy++) {
      for (let dx = -half; dx <= half; dx++) {
        const px = cx + dx;
        const py = cy + dy;
        if (px < 0 || px >= w || py < 0 || py >= h) continue;

        const idx = (py * w + px) * 4;
        const noise = (Math.random() - 0.5) * 0.05;
        data[idx + 0] = 0.50 + noise;   // u
        data[idx + 1] = 0.25 + noise;   // v
      }
    }

    // Upload to texA
    this.device.queue.writeTexture(
      { texture: this.texA },
      data,
      { bytesPerRow: w * 4 * 4, rowsPerImage: h },
      { width: w, height: h },
    );

    // Copy texA → texB so both buffers have defined initial state
    const enc = this.device.createCommandEncoder({ label: `rd-surf-seed-${this.cellId}` });
    enc.copyTextureToTexture(
      { texture: this.texA },
      { texture: this.texB },
      { width: w, height: h },
    );
    this.device.queue.submit([enc.finish()]);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// CellRDSurfaceManager — batched compute orchestrator
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Manages per-cell 64×64 Gray-Scott reaction-diffusion surface patches.
 *
 * Owns the shared compute pipeline and iterates over registered cells,
 * dispatching one compute pass per cell per substep.  Each cell has its own
 * (f, k, Du, Dv) uniform derived from its species, and its own pair of
 * ping-pong textures.
 *
 * Usage:
 * ```ts
 * const mgr = new CellRDSurfaceManager(device);
 * await mgr.init();
 *
 * // Register cells
 * mgr.addCell({ cellId: 'self_attn', species: 'cil-eye' });
 * mgr.addCell({ cellId: 'ffn_0',    species: 'cil-bolt' });
 * mgr.addCell({ cellId: 'add_norm',  species: 'cil-plus' });
 *
 * // Per-frame
 * const encoder = device.createCommandEncoder();
 * mgr.step(encoder);
 * device.queue.submit([encoder.finish()]);
 *
 * // Render: use mgr.getSnapshot('self_attn').outputTexture as surface tex
 * ```
 */
export class CellRDSurfaceManager {

  private readonly device:    GPUDevice;
  private readonly substeps:  number;

  /** Registered cell patches, keyed by cellId. */
  private readonly cells: Map<string, CellRDSurface> = new Map();

  /** Ordered slot list for deterministic iteration. */
  private readonly slots: CellRDSurface[] = [];

  // Shared pipeline resources
  private pipeline!:     GPUComputePipeline;
  private uniformBGL!:   GPUBindGroupLayout;
  private ppBGL!:        GPUBindGroupLayout;

  /** Shared uniform buffer — rewritten per cell before each dispatch. */
  private uniformBuf!:   GPUBuffer;
  private uniformBG!:    GPUBindGroup;

  /** Pre-allocated Float32Array for uniform uploads (4 × f32 = 16 bytes). */
  private readonly uniformData = new Float32Array(4);

  /** Frame counter (total step() calls). */
  private frameIndex = 0;

  /** Whether init() has completed. */
  private ready = false;

  constructor(device: GPUDevice, cfg: CellRDManagerConfig = {}) {
    this.device   = device;
    this.substeps = cfg.substeps ?? CELL_RD_DEFAULT_SUBSTEPS;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Init
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Compile the shared compute pipeline and allocate the uniform buffer.
   * Must be called once (and awaited) before addCell() or step().
   */
  async init(): Promise<void> {
    if (this.ready) return;

    this._createUniformBuffer();
    await this._createPipeline();

    this.ready = true;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Cell registration
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Register a new cell patch.
   *
   * Allocates a pair of 64×64 ping-pong textures and resolves (f, k, Du, Dv)
   * from the cell's species via SPECIES_GRAYSCOTT_MAP.
   *
   * @param cfg  Cell configuration (cellId, species, optional paramsOverride).
   * @returns    The created CellRDSurface instance.
   * @throws     If init() has not been called, or cellId is already registered,
   *             or CELL_RD_MAX_CELLS has been reached.
   */
  addCell(cfg: CellRDSurfaceConfig): CellRDSurface {
    if (!this.ready) {
      throw new Error('CellRDSurfaceManager.init() must be awaited before addCell()');
    }
    if (this.cells.has(cfg.cellId)) {
      throw new Error(`Cell '${cfg.cellId}' is already registered`);
    }
    if (this.slots.length >= CELL_RD_MAX_CELLS) {
      throw new Error(`Maximum cell count (${CELL_RD_MAX_CELLS}) reached`);
    }

    const surface = new CellRDSurface(this.device, cfg);
    this.cells.set(cfg.cellId, surface);
    this.slots.push(surface);

    return surface;
  }

  /**
   * Remove a cell patch and release its GPU resources.
   *
   * @param cellId  The cell to remove.
   * @returns       True if the cell was found and removed, false otherwise.
   */
  removeCell(cellId: string): boolean {
    const surface = this.cells.get(cellId);
    if (!surface) return false;

    surface.destroy();
    this.cells.delete(cellId);

    const idx = this.slots.indexOf(surface);
    if (idx >= 0) this.slots.splice(idx, 1);

    return true;
  }

  /** Check whether a cell is registered. */
  hasCell(cellId: string): boolean {
    return this.cells.has(cellId);
  }

  /** Number of registered cell patches. */
  get cellCount(): number {
    return this.slots.length;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Per-frame compute
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Execute `substeps` Gray-Scott compute passes for every registered cell.
   *
   * Each substep iterates all cells: upload per-cell uniforms, bind that
   * cell's ping-pong textures, dispatch 8×8 workgroups, then flip.
   *
   * The encoder is NOT submitted — caller must submit at end of frame.
   *
   * @param encoder  Current frame's GPUCommandEncoder.
   */
  step(encoder: GPUCommandEncoder): void {
    if (!this.ready) {
      throw new Error('CellRDSurfaceManager.init() must be awaited before step()');
    }
    if (this.slots.length === 0) return;

    for (let sub = 0; sub < this.substeps; sub++) {
      for (const cell of this.slots) {
        // Upload this cell's (f, k, Du, Dv)
        this._writeUniformsForCell(cell);

        // Create bind-group for this cell's current ping-pong direction
        const ppBG = this.device.createBindGroup({
          label:   `rd-surf-pp-${cell.cellId}-s${sub}`,
          layout:  this.ppBGL,
          entries: [
            { binding: 0, resource: cell.readTexture.createView() },
            { binding: 1, resource: cell.writeTexture.createView() },
          ],
        });

        const pass = encoder.beginComputePass({
          label: `gs-surf-${cell.cellId}-f${this.frameIndex}-s${sub}`,
        });
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.uniformBG);
        pass.setBindGroup(1, ppBG);
        pass.dispatchWorkgroups(SURFACE_WG_COUNT, SURFACE_WG_COUNT);
        pass.end();

        // Flip this cell's ping-pong after each substep
        cell.flip();
      }
    }

    this.frameIndex++;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Query
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Get a read-only snapshot of a cell's current RD state.
   *
   * @param cellId  Cell identifier.
   * @returns       Snapshot including output texture and resolved params,
   *                or undefined if the cell is not registered.
   */
  getSnapshot(cellId: string): CellRDSnapshot | undefined {
    const cell = this.cells.get(cellId);
    if (!cell) return undefined;

    const slotIndex = this.slots.indexOf(cell);
    return {
      cellId:        cell.cellId,
      species:       cell.species,
      params:        cell.params,
      outputTexture: cell.outputTexture,
      slotIndex,
    };
  }

  /**
   * Iterate over all registered cells' snapshots.
   * Useful for binding each cell's output texture to the render pipeline.
   */
  *snapshots(): IterableIterator<CellRDSnapshot> {
    for (let i = 0; i < this.slots.length; i++) {
      const cell = this.slots[i];
      yield {
        cellId:        cell.cellId,
        species:       cell.species,
        params:        cell.params,
        outputTexture: cell.outputTexture,
        slotIndex:     i,
      };
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Parameter updates
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Update a cell's Gray-Scott parameters at runtime.
   *
   * Replaces the cell's stored params; the new values take effect on the
   * next step() call.  Does NOT re-seed — the current pattern continues
   * evolving under the new parameters (morphology crossfade).
   *
   * @param cellId  Cell identifier.
   * @param params  New (f, k) and optionally (Du, Dv).
   * @throws        If the cell is not registered.
   */
  setCellParams(cellId: string, params: GrayScottParams): void {
    const cell = this.cells.get(cellId);
    if (!cell) {
      throw new Error(`Cell '${cellId}' not registered`);
    }

    // Mutate the cell's params in place (CellRDSurface.params is readonly
    // at the type level but we cast internally for runtime updates)
    const p = cell.params as Required<GrayScottParams>;
    p.f = params.f;
    p.k = params.k;
    if (params.Du !== undefined) p.Du = params.Du;
    if (params.Dv !== undefined) p.Dv = params.Dv;
  }

  /**
   * Update a cell's parameters by species name.
   *
   * Equivalent to `setCellParams(cellId, speciesGrayScottParams(species))`.
   */
  setCellSpecies(cellId: string, species: string): void {
    this.setCellParams(cellId, speciesGrayScottParams(species));
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Seed / reset
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Re-seed a single cell's patch.
   *
   * Resets the 64×64 grid to the Pearson 1993 initial condition and
   * resets the ping-pong phase.
   *
   * @param cellId  Cell identifier.
   * @throws        If the cell is not registered.
   */
  reseedCell(cellId: string): void {
    const cell = this.cells.get(cellId);
    if (!cell) {
      throw new Error(`Cell '${cellId}' not registered`);
    }
    cell.reseed();
  }

  /**
   * Re-seed all registered cell patches and reset the frame counter.
   */
  reseedAll(): void {
    for (const cell of this.slots) {
      cell.reseed();
    }
    this.frameIndex = 0;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Accessors
  // ─────────────────────────────────────────────────────────────────────────

  /** Total frames stepped so far. */
  get frame(): number { return this.frameIndex; }

  /** Number of substeps executed per step() call. */
  get substepsPerFrame(): number { return this.substeps; }

  // ─────────────────────────────────────────────────────────────────────────
  // Destroy
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Release all GPU resources — shared pipeline assets and every cell patch.
   * The instance must not be used after destroy().
   */
  destroy(): void {
    for (const cell of this.slots) {
      cell.destroy();
    }
    this.cells.clear();
    this.slots.length = 0;

    this.uniformBuf.destroy();
    this.ready = false;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Private helpers
  // ─────────────────────────────────────────────────────────────────────────

  private _createUniformBuffer(): void {
    // SurfaceGSUniforms: 4 × f32 = 16 bytes
    this.uniformBuf = this.device.createBuffer({
      label:  'rd-surf-uniform',
      size:   16,
      usage:  GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  private _writeUniformsForCell(cell: CellRDSurface): void {
    this.uniformData[0] = cell.params.f;
    this.uniformData[1] = cell.params.k;
    this.uniformData[2] = cell.params.Du;
    this.uniformData[3] = cell.params.Dv;
    this.device.queue.writeBuffer(this.uniformBuf, 0, this.uniformData);
  }

  private async _createPipeline(): Promise<void> {
    const shaderModule = this.device.createShaderModule({
      label: 'rd-surface-gs-shader',
      code:  SURFACE_GS_SHADER,
    });

    // BGL 0: uniform buffer (SurfaceGSUniforms)
    this.uniformBGL = this.device.createBindGroupLayout({
      label:   'rd-surf-uniform-bgl',
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
      label:   'rd-surf-pingpong-bgl',
      entries: [
        {
          binding:    0,
          visibility: GPUShaderStage.COMPUTE,
          texture:    { sampleType: 'unfilterable-float' },
        },
        {
          binding:        1,
          visibility:     GPUShaderStage.COMPUTE,
          storageTexture: { access: 'write-only', format: SURFACE_TEX_FORMAT },
        },
      ],
    });

    const layout = this.device.createPipelineLayout({
      label:            'rd-surf-pipeline-layout',
      bindGroupLayouts: [this.uniformBGL, this.ppBGL],
    });

    this.pipeline = await this.device.createComputePipelineAsync({
      label:   'rd-surface-gs-pipeline',
      layout,
      compute: { module: shaderModule, entryPoint: 'gs_surface_step' },
    });

    // Create the uniform bind-group (buffer is shared, re-written per cell)
    this.uniformBG = this.device.createBindGroup({
      label:   'rd-surf-uniform-bg',
      layout:  this.uniformBGL,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
      ],
    });
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience — bulk registration from cell-body-bridge output
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Register an array of cell descriptors with a CellRDSurfaceManager.
 *
 * Designed to accept output from `cellsToBodies()` (cell-body-bridge.ts) or
 * any array with `{ id, species }` fields.
 *
 * Cells whose species is 'fluid' or not found in SPECIES_GRAYSCOTT_MAP are
 * skipped (fluid particles don't need a surface RD patch).
 *
 * @param manager  An initialised CellRDSurfaceManager.
 * @param cells    Array of cell descriptors with at least `id` and `species`.
 * @returns        Array of CellRDSurface instances that were successfully added.
 */
export function registerCellSurfaces(
  manager: CellRDSurfaceManager,
  cells: ReadonlyArray<{ id: string; species: string }>,
): CellRDSurface[] {
  const added: CellRDSurface[] = [];

  for (const cell of cells) {
    // Skip fluid and unknown species
    if (cell.species === 'fluid') continue;
    if (!(cell.species in SPECIES_GRAYSCOTT_MAP)) continue;
    // Skip duplicates
    if (manager.hasCell(cell.id)) continue;

    try {
      const surface = manager.addCell({
        cellId:  cell.id,
        species: cell.species,
      });
      added.push(surface);
    } catch {
      // Max cells reached — stop adding
      break;
    }
  }

  return added;
}
