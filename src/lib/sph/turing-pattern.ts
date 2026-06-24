// src/lib/sph/turing-pattern.ts
//
// Gray-Scott reaction-diffusion Turing-pattern generator using WebGPU compute
// shaders.  Inspired by lygia's grayscott.wgsl (Patricio Gonzalez Vivo) but
// fully inlined – no #include pre-processor required.
//
// Theory: two chemical "species" U and V diffuse and react according to:
//   ∂U/∂t =  Du·∇²U  –  U·V²  +  f·(1 – U)
//   ∂V/∂t =  Dv·∇²V  +  U·V²  –  (f + k)·V
//
// Different (f, k) parameter pairs produce qualitatively distinct morphologies:
//   SPOTS    – isolated spots / Turing dots        (coral, leopard)
//   STRIPES  – parallel stripe labyrinth           (zebra, angelfish)
//   MAZE     – connected maze / worm network       (brain coral)
//   SPIRALS  – rotating spiral waves               (chemical Belousov–Zhabotinsky)
//   BUBBLES  – large smooth blobs with halos       (soap foam)
//
// Simulation uses a double-buffer (ping-pong) scheme:
//   • Two rgba16float textures alternate as read ↔ write each step.
//   • R channel = U concentration, G channel = V concentration.
//   • B/A channels are unused during simulation; B is written to 1.0 so the
//     texture can double as a colour source for the render pass.
//
// Usage:
//   const gen = new TuringPatternGenerator(device);
//   const tex = await gen.generate({ width: 512, height: 512, steps: 2000 });
//   // tex is a GPUTexture ready for render binding
//   gen.destroy();

// ─── Species → Turing-pattern mapping ────────────────────────────────────────
// Mirrors the species taxonomy used by NaturalPatternGenerator in
// natural-patterns.ts so both generators share the same species vocabulary.









export type TuringPatternSpecies =
  | 'SPOTS'
  | 'STRIPES'
  | 'MAZE'
  | 'SPIRALS'
  | 'BUBBLES';

/** Map a cell species string to its Turing-pattern species variant. */
export function speciesTuringMode(species: string): TuringPatternSpecies {
  const MAP: Record<string, TuringPatternSpecies> = {
    'fluid':           'BUBBLES',
    'cil-eye':         'SPOTS',
    'cil-bolt':        'SPIRALS',
    'cil-vector':      'STRIPES',
    'cil-plus':        'SPOTS',
    'cil-arrow-right': 'STRIPES',
    'cil-filter':      'MAZE',
    'cil-layers':      'BUBBLES',
    'cil-loop':        'SPIRALS',
    'cil-code':        'MAZE',
    'cil-graph':       'STRIPES',
  };
  return MAP[species] ?? 'SPOTS';
}

/** (f, k, Du, Dv) parameter sets that produce each morphology.
 *  Du / Dv are diffusion rates; f = feed rate; k = kill rate.
 *  Values from Pearson (1993) and Munafo (2014) reference tables. */
const SPECIES_PARAMS: Record<
  TuringPatternSpecies,
  { f: number; k: number; du: number; dv: number }
> = {
  //            f       k       Du     Dv
  SPOTS:   { f: 0.035, k: 0.060, du: 0.210, dv: 0.105 },
  STRIPES: { f: 0.022, k: 0.051, du: 0.210, dv: 0.105 },
  MAZE:    { f: 0.029, k: 0.057, du: 0.210, dv: 0.105 },
  SPIRALS: { f: 0.018, k: 0.051, du: 0.210, dv: 0.105 },
  BUBBLES: { f: 0.098, k: 0.057, du: 0.210, dv: 0.105 },
};

// ─── Public API types ─────────────────────────────────────────────────────────

export interface TuringPatternParams {
  /** Texture width in pixels (power-of-two recommended). Default 512. */
  width?: number;
  /** Texture height in pixels. Default 512. */
  height?: number;
  /** Number of simulation time-steps. Default 2000. More → more evolved pattern. */
  steps?: number;
  /** Time-step size dt. Default 1.0.  Values >1.5 may diverge. */
  dt?: number;
  /** Number of sub-steps dispatched per JS frame (reduces JS overhead). Default 8. */
  stepsPerDispatch?: number;
  /** Turing species variant. Overrides `species` if provided. Default SPOTS. */
  mode?: TuringPatternSpecies;
  /** Optional cell species string from the cell taxonomy – overrides `mode`. */
  species?: string;
  /** Override feed rate f.  Defaults to species preset. */
  f?: number;
  /** Override kill rate k.  Defaults to species preset. */
  k?: number;
  /** Override U diffusion rate Du.  Defaults to species preset. */
  du?: number;
  /** Override V diffusion rate Dv.  Defaults to species preset. */
  dv?: number;
  /** Seed for initial random noise (integer). Default 42. */
  seed?: number;
}

// ─── Inlined WGSL – Gray-Scott compute shader ────────────────────────────────
//
// Laplacian kernel (from lygia grayscott.wgsl, Patricio Gonzalez Vivo):
//   weights:  0.707… 1  0.707…
//             1      -6.828… 1
//             0.707… 1  0.707…
// The kernel integrates diagonal neighbours at 1/√2 weight as an isotropic
// 3×3 discrete Laplacian.  Sum of all weights = 0 (correct for Laplacian).
//
// Ping-pong: the shader reads from `texIn` (TEXTURE_BINDING / sampled) and
// writes to `texOut` (STORAGE_BINDING / write-only).  Each step the host swaps
// the two textures.
//
// Initial condition: U=1 everywhere; V≈0 except small seeded square patches
// of V=0.5 in the centre area.  This matches the lygia convention of starting
// with a "src" (V injection) in the middle.

// Uniform struct layout (aligned to 16-byte WebGPU rules):
//   offset  0 : u32   width
//   offset  4 : u32   height
//   offset  8 : f32   du      (diffusion rate U)
//   offset 12 : f32   dv      (diffusion rate V)
//   offset 16 : f32   f       (feed)
//   offset 20 : f32   k       (kill)
//   offset 24 : f32   dt      (time step)
//   offset 28 : u32   step    (current step index, for seeding on step 0)
//   offset 32 : u32   seed    (RNG seed for initial condition)
//   offset 36 : u32   _pad[3]
//   total: 48 bytes (3 × vec4)

const INIT_SHADER_SRC = /* wgsl */`
// ── uniforms ──────────────────────────────────────────────────────────────────
struct Params {
  width  : u32,
  height : u32,
  du     : f32,
  dv     : f32,
  f      : f32,
  k      : f32,
  dt     : f32,
  step   : u32,
  seed   : u32,
  _pad0  : u32,
  _pad1  : u32,
  _pad2  : u32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var texOut: texture_storage_2d<rgba16float, write>;

// ── Tiny hash RNG (Wang hash) ─────────────────────────────────────────────────
fn wang_hash(seed: u32) -> u32 {
  var s = seed;
  s = (s ^ 61u) ^ (s >> 16u);
  s = s * 9u;
  s = s ^ (s >> 4u);
  s = s * 0x27d4eb2du;
  s = s ^ (s >> 15u);
  return s;
}

fn rand_f32(seed: u32) -> f32 {
  return f32(wang_hash(seed)) / 4294967295.0;
}

// ── Initialisation entry point ────────────────────────────────────────────────
// U = 1.0 everywhere; V = 0 except small square seeds near centre.
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let px = gid.x;
  let py = gid.y;
  if (px >= p.width || py >= p.height) { return; }

  // Normalised coordinates
  let fx = f32(px) / f32(p.width);
  let fy = f32(py) / f32(p.height);

  // Default: U=1, V=0
  var u = 1.0;
  var v = 0.0;

  // Seed multiple square patches of V=0.5 distributed across the grid.
  // 5×5 grid of 6-pixel seed squares (scales with resolution).
  let numSeeds = 5u;
  let patchSize = max(4u, p.width / 64u);
  for (var si: u32 = 0u; si < numSeeds; si++) {
    for (var sj: u32 = 0u; sj < numSeeds; sj++) {
      // Deterministic jitter per patch using Wang hash
      let idx   = si * numSeeds + sj;
      let jx    = rand_f32(p.seed + idx * 2u + 0u) * 0.1 - 0.05;
      let jy    = rand_f32(p.seed + idx * 2u + 1u) * 0.1 - 0.05;
      let cx    = (f32(si) + 0.5) / f32(numSeeds) + jx;
      let cy    = (f32(sj) + 0.5) / f32(numSeeds) + jy;
      let halfP = f32(patchSize) / f32(p.width);
      if (abs(fx - cx) < halfP && abs(fy - cy) < halfP) {
        u = 0.5;
        v = 0.25;
      }
    }
  }

  textureStore(texOut, vec2u(px, py), vec4f(u, v, 1.0, 1.0));
}
`;

// ── Gray-Scott step shader ────────────────────────────────────────────────────
// Reads from texIn (sampled, rgba16float) and writes next state to texOut.
// Laplacian uses the isotropic 3×3 kernel from lygia's grayscott.wgsl.
const STEP_SHADER_SRC = /* wgsl */`
// ── uniforms ──────────────────────────────────────────────────────────────────
struct Params {
  width  : u32,
  height : u32,
  du     : f32,
  dv     : f32,
  f      : f32,
  k      : f32,
  dt     : f32,
  step   : u32,
  seed   : u32,
  _pad0  : u32,
  _pad1  : u32,
  _pad2  : u32,
}
@group(0) @binding(0) var<uniform>             p      : Params;
@group(0) @binding(1) var                      texIn  : texture_2d<f32>;
@group(0) @binding(2) var                      samp   : sampler;
@group(0) @binding(3) var                      texOut : texture_storage_2d<rgba16float, write>;

// ── Laplacian via lygia isotropic 3×3 kernel ──────────────────────────────────
//
//   w = [ 0.707106781,  1.0,  0.707106781,
//          1.0,        -6.828427...,  1.0,
//          0.707106781,  1.0,  0.707106781 ]
//
// Sum = 4×1 + 4×0.707… + 1×(−6.828…) = 4 + 2.828… − 6.828… = 0  ✓

fn laplacian(uv: vec2f, px: vec2f) -> vec2f {
  // Diagonal weight = 1/sqrt(2)
  let d = 0.70710678118f;
  // Centre weight = -(4 + 4*d) = -6.82842712...
  let c = -(4.0 + 4.0 * d);

  var lap = vec2f(0.0);

  // 3×3 neighbourhood — unrolled for WGSL (no dynamic arrays in uniform space)
  lap += textureSampleLevel(texIn, samp, uv + px * vec2f(-1.0, -1.0), 0.0).rg * d;
  lap += textureSampleLevel(texIn, samp, uv + px * vec2f( 0.0, -1.0), 0.0).rg * 1.0;
  lap += textureSampleLevel(texIn, samp, uv + px * vec2f( 1.0, -1.0), 0.0).rg * d;
  lap += textureSampleLevel(texIn, samp, uv + px * vec2f(-1.0,  0.0), 0.0).rg * 1.0;
  lap += textureSampleLevel(texIn, samp, uv + px * vec2f( 0.0,  0.0), 0.0).rg * c;
  lap += textureSampleLevel(texIn, samp, uv + px * vec2f( 1.0,  0.0), 0.0).rg * 1.0;
  lap += textureSampleLevel(texIn, samp, uv + px * vec2f(-1.0,  1.0), 0.0).rg * d;
  lap += textureSampleLevel(texIn, samp, uv + px * vec2f( 0.0,  1.0), 0.0).rg * 1.0;
  lap += textureSampleLevel(texIn, samp, uv + px * vec2f( 1.0,  1.0), 0.0).rg * d;

  return lap;
}

// ── Compute entry point ───────────────────────────────────────────────────────
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let px_u = gid.x;
  let py_u = gid.y;
  if (px_u >= p.width || py_u >= p.height) { return; }

  // UV in [0,1] — textureSampleLevel with repeat sampler handles wrap-around
  let uv  = vec2f(
    (f32(px_u) + 0.5) / f32(p.width),
    (f32(py_u) + 0.5) / f32(p.height),
  );
  let pixel = vec2f(1.0 / f32(p.width), 1.0 / f32(p.height));

  // Current concentrations
  let current = textureSampleLevel(texIn, samp, uv, 0.0).rg;
  let U = current.r;
  let V = current.g;

  // Laplacian for U and V
  let lap = laplacian(uv, pixel);

  // Gray-Scott reaction terms (from lygia grayscott.wgsl, Patricio Gonzalez Vivo)
  let uvv = U * V * V;
  let du  = p.du * lap.r - uvv + p.f * (1.0 - U);
  let dv  = p.dv * lap.g + uvv - (p.f + p.k) * V;

  // Euler integration, clamp to [0,1] for numerical stability
  let uNext = clamp(U + du * p.dt, 0.0, 1.0);
  let vNext = clamp(V + dv * p.dt, 0.0, 1.0);

  // B channel: concentration difference visualisation (bright at pattern edges)
  let diff = abs(uNext - vNext);

  textureStore(texOut, vec2u(px_u, py_u), vec4f(uNext, vNext, diff, 1.0));
}
`;

// ─── TuringPatternGenerator ───────────────────────────────────────────────────

/**
 * GPU-accelerated Gray-Scott reaction-diffusion Turing pattern generator.
 *
 * Simulates the Gray-Scott PDE on the GPU using a double-buffer ping-pong
 * strategy: two {@link GPUTexture}s (rgba16float) alternate as read/write
 * targets across `steps` compute dispatches.  The final texture encodes:
 *   • R – U concentration (substrate chemical)
 *   • G – V concentration (activator chemical)
 *   • B – |U – V| edge contrast (useful for rendering)
 *
 * Different (f, k) parameter pairs produce qualitatively distinct Turing
 * morphologies. Five named {@link TuringPatternSpecies} are pre-configured:
 * SPOTS, STRIPES, MAZE, SPIRALS, BUBBLES.  Parameters can also be overridden
 * per-call for custom species exploration.
 *
 * @example
 * ```ts
 * const gen = new TuringPatternGenerator(device);
 * // Zebra stripes
 * const stripeTex = await gen.generate({ species: 'cil-vector', steps: 3000 });
 * // Custom parameters
 * const customTex = await gen.generate({ f: 0.039, k: 0.058, steps: 4000 });
 * gen.destroy();
 * ```
 */
export class TuringPatternGenerator {
  private readonly device: GPUDevice;

  // Pipelines (lazily initialised)
  private initPipeline: GPUComputePipeline | null = null;
  private stepPipeline: GPUComputePipeline | null = null;

  // Bind group layouts
  private initBGL: GPUBindGroupLayout | null = null;
  private stepBGL: GPUBindGroupLayout | null = null;

  // Sampler for the step shader (repeat wrap for torus topology)
  private sampler: GPUSampler | null = null;

  constructor(device: GPUDevice) {
    this.device = device;
  }

  // ── Lazy pipeline initialisation ─────────────────────────────────────────

  private async ensurePipelines(): Promise<void> {
    if (this.initPipeline && this.stepPipeline) return;

    const dev = this.device;

    // Sampler: clamp-to-edge so pattern wraps seamlessly
    this.sampler = dev.createSampler({
      label:        'turing-sampler',
      addressModeU: 'repeat',
      addressModeV: 'repeat',
      magFilter:    'nearest',
      minFilter:    'nearest',
    });

    // ── Init pipeline ─────────────────────────────────────────────────────
    const initModule = dev.createShaderModule({
      label: 'turing-init-shader',
      code:  INIT_SHADER_SRC,
    });

    this.initBGL = dev.createBindGroupLayout({
      label: 'turing-init-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d' },
        },
      ],
    });

    this.initPipeline = await dev.createComputePipelineAsync({
      label:   'turing-init-pipeline',
      layout:  dev.createPipelineLayout({ bindGroupLayouts: [this.initBGL] }),
      compute: { module: initModule, entryPoint: 'main' },
    });

    // ── Step pipeline ─────────────────────────────────────────────────────
    const stepModule = dev.createShaderModule({
      label: 'turing-step-shader',
      code:  STEP_SHADER_SRC,
    });

    this.stepBGL = dev.createBindGroupLayout({
      label: 'turing-step-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture:  { sampleType: 'float', viewDimension: '2d' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, sampler:  { type: 'filtering' } },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d' },
        },
      ],
    });

    this.stepPipeline = await dev.createComputePipelineAsync({
      label:   'turing-step-pipeline',
      layout:  dev.createPipelineLayout({ bindGroupLayouts: [this.stepBGL] }),
      compute: { module: stepModule, entryPoint: 'main' },
    });
  }

  // ── Uniform buffer helper ─────────────────────────────────────────────────

  private makeUniformBuffer(
    width: number, height: number,
    du: number, dv: number,
    f: number,  k: number,
    dt: number, step: number, seed: number,
  ): GPUBuffer {
    // Struct: 12 × u32/f32 = 48 bytes
    const data = new ArrayBuffer(48);
    const view = new DataView(data);
    view.setUint32 ( 0, width,  true);
    view.setUint32 ( 4, height, true);
    view.setFloat32( 8, du,     true);
    view.setFloat32(12, dv,     true);
    view.setFloat32(16, f,      true);
    view.setFloat32(20, k,      true);
    view.setFloat32(24, dt,     true);
    view.setUint32 (28, step,   true);
    view.setUint32 (32, seed,   true);
    // _pad0, _pad1, _pad2 left as 0
    const buf = this.device.createBuffer({
      label: 'turing-uniforms',
      size:  48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buf, 0, data);
    return buf;
  }

  // ── Public generate ───────────────────────────────────────────────────────

  /**
   * Run a full Gray-Scott simulation and return the final state texture.
   *
   * The returned {@link GPUTexture} is rgba16float with TEXTURE_BINDING and
   * COPY_SRC usage, ready to bind to a render pass or read back via
   * {@link readback}.
   *
   * @param params Simulation parameters (see {@link TuringPatternParams}).
   * @returns Final state texture (ownership transfers to caller).
   */
  async generate(params: TuringPatternParams = {}): Promise<GPUTexture> {
    await this.ensurePipelines();

    const {
      width           = 512,
      height          = 512,
      steps           = 2000,
      dt              = 1.0,
      stepsPerDispatch = 8,
      seed            = 42,
      species,
      mode: modeOverride,
    } = params;

    // Resolve species → (f, k, du, dv)
    const turingMode: TuringPatternSpecies =
      modeOverride ?? (species ? speciesTuringMode(species) : 'SPOTS');
    const preset = SPECIES_PARAMS[turingMode];

    const f  = params.f  ?? preset.f;
    const k  = params.k  ?? preset.k;
    const du = params.du ?? preset.du;
    const dv = params.dv ?? preset.dv;

    const dev = this.device;
    const wx  = Math.ceil(width  / 8);
    const wy  = Math.ceil(height / 8);

    // ── Create ping-pong texture pair ─────────────────────────────────────
    const makeSimTex = (label: string): GPUTexture =>
      dev.createTexture({
        label,
        size:   { width, height, depthOrArrayLayers: 1 },
        format: 'rgba16float',
        usage:
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_SRC,
      });

    let texA = makeSimTex(`turing-texA-${width}x${height}`);
    let texB = makeSimTex(`turing-texB-${width}x${height}`);

    // ── Step 0: initialise texA ───────────────────────────────────────────
    {
      const unifBuf = this.makeUniformBuffer(width, height, du, dv, f, k, dt, 0, seed);
      const bg = dev.createBindGroup({
        label:   'turing-init-bg',
        layout:  this.initBGL!,
        entries: [
          { binding: 0, resource: { buffer: unifBuf } },
          { binding: 1, resource: texA.createView() },
        ],
      });
      const enc  = dev.createCommandEncoder({ label: 'turing-init-enc' });
      const pass = enc.beginComputePass({ label: 'turing-init-pass' });
      pass.setPipeline(this.initPipeline!);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(wx, wy);
      pass.end();
      dev.queue.submit([enc.finish()]);
      unifBuf.destroy();
    }

    // ── Steps 1…N: ping-pong simulation ───────────────────────────────────
    // We batch `stepsPerDispatch` sub-steps into a single command encoder to
    // reduce JS-GPU round-trips.  Each sub-step reads from the "read" texture
    // and writes to the "write" texture, then swaps them.

    let readTex  = texA;
    let writeTex = texB;

    for (let stepBase = 0; stepBase < steps; stepBase += stepsPerDispatch) {
      const batchSize = Math.min(stepsPerDispatch, steps - stepBase);
      const enc       = dev.createCommandEncoder({ label: `turing-step-enc-${stepBase}` });

      for (let b = 0; b < batchSize; b++) {
        const stepIdx = stepBase + b + 1;
        const unifBuf = this.makeUniformBuffer(width, height, du, dv, f, k, dt, stepIdx, seed);

        const bg = dev.createBindGroup({
          label:   `turing-step-bg-${stepIdx}`,
          layout:  this.stepBGL!,
          entries: [
            { binding: 0, resource: { buffer: unifBuf } },
            { binding: 1, resource: readTex.createView()  },
            { binding: 2, resource: this.sampler!         },
            { binding: 3, resource: writeTex.createView() },
          ],
        });

        const pass = enc.beginComputePass({ label: `turing-step-pass-${stepIdx}` });
        pass.setPipeline(this.stepPipeline!);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(wx, wy);
        pass.end();

        // Swap ping-pong pointers
        const tmp = readTex;
        readTex   = writeTex;
        writeTex  = tmp;

        // Uniform buffers are submitted in one batch — schedule destruction
        // after the encoder is finished (micro-task)
        Promise.resolve().then(() => unifBuf.destroy());
      }

      dev.queue.submit([enc.finish()]);

      // Yield to the event loop every batch to avoid GPU timeout
      await dev.queue.onSubmittedWorkDone();
    }

    // `readTex` now holds the latest state after the last step.
    // Destroy the unused write buffer.
    writeTex.destroy();

    return readTex;
  }

  // ── CPU readback ──────────────────────────────────────────────────────────

  /**
   * Read the final texture back to CPU as a Float32Array (RGBA, row-major).
   * Channels: R=U, G=V, B=|U-V|, A=1.
   *
   * @param texture  A texture returned by {@link generate}.
   * @param width    Texture width.
   * @param height   Texture height.
   */
  async readback(
    texture: GPUTexture,
    width: number,
    height: number,
  ): Promise<Float32Array> {
    // rgba16float = 8 bytes per pixel; bytesPerRow must be multiple of 256
    const bytesPerPixel = 8;
    const bytesPerRow   = Math.ceil((width * bytesPerPixel) / 256) * 256;

    const stagingBuf = this.device.createBuffer({
      size:  bytesPerRow * height,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const enc = this.device.createCommandEncoder();
    enc.copyTextureToBuffer(
      { texture },
      { buffer: stagingBuf, bytesPerRow, rowsPerImage: height },
      { width, height, depthOrArrayLayers: 1 },
    );
    this.device.queue.submit([enc.finish()]);

    await stagingBuf.mapAsync(GPUMapMode.READ);
    const raw    = new Uint16Array(stagingBuf.getMappedRange());
    const result = new Float32Array(width * height * 4);

    // Decode float16 → float32 via DataView
    const dv = new DataView(stagingBuf.getMappedRange());
    for (let row = 0; row < height; row++) {
      for (let col = 0; col < width; col++) {
        const srcBase  = row * bytesPerRow + col * bytesPerPixel;
        const dstBase  = (row * width + col) * 4;
        // Simple float16 → float32 (sign/exp/mantissa extraction)
        for (let ch = 0; ch < 4; ch++) {
          const h = dv.getUint16(srcBase + ch * 2, true);
          result[dstBase + ch] = float16ToFloat32(h);
        }
      }
    }

    stagingBuf.unmap();
    stagingBuf.destroy();
    return result;
  }

  // ── Lifecycle ─────────────────────────────────────────────────────────────

  /**
   * Release GPU pipeline resources.  Any textures returned by {@link generate}
   * remain valid until their own {@link GPUTexture.destroy} is called.
   */
  destroy(): void {
    this.initPipeline = null;
    this.stepPipeline = null;
    this.initBGL      = null;
    this.stepBGL      = null;
    this.sampler      = null;
  }
}

// ─── Float16 decoder ─────────────────────────────────────────────────────────
// IEEE 754 half-precision → single-precision conversion.
function float16ToFloat32(h: number): number {
  const sign     = (h >> 15) & 0x1;
  const exponent = (h >> 10) & 0x1f;
  const mantissa =  h        & 0x3ff;
  if (exponent === 0) {
    // Subnormal
    return (sign ? -1 : 1) * Math.pow(2, -14) * (mantissa / 1024);
  } else if (exponent === 31) {
    // Inf / NaN
    return mantissa ? NaN : (sign ? -Infinity : Infinity);
  }
  return (sign ? -1 : 1) * Math.pow(2, exponent - 15) * (1 + mantissa / 1024);
}

// ─── Re-exports for convenience ───────────────────────────────────────────────
export {
  INIT_SHADER_SRC as TURING_INIT_WGSL,
  STEP_SHADER_SRC as TURING_STEP_WGSL,
  SPECIES_PARAMS  as TURING_SPECIES_PARAMS,
};
