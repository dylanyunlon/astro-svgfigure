/**
 * environment-fx.ts — "生物实验室" 环境大气特效
 *
 * 三层特效组合，合力营造赛博生化实验室氛围：
 *
 *   Layer 1 — BRICK TILE GRID (背景网格)
 *     来源：upstream/lygia/space/brickTile.glsl（Patricio Gonzalez Vivo）
 *     砖砌 UV 偏移算法移植为 WGSL。每行横向偏移 0.5，产生交错的矩形网格，
 *     用扫描线绿色 + 深色填充模拟实验室监控界面底纹。
 *
 *   Layer 2 — VORONOISE ATMOSPHERIC SCATTER（大气散射）
 *     来源：upstream/lygia/generative/voronoise.wgsl（Inigo Quilez）
 *     voronoise = voronoi(u=0) … noise(u=1) 之间的平滑插值。
 *     低频慢变 → 仿生物细胞形态的背景粒子散射光晕。
 *
 *   Layer 3 — CHROMATIC ABERRATION 后处理
 *     来源：upstream/lygia/distort/chromaAB.glsl（Patricio Gonzalez Vivo / Johan Ismael）
 *     以到屏幕中心的距离作为 SDF，三通道 RGB 分别向外/向内偏移采样，
 *     边缘产生紫色色散，增强实验室显示器的"老旧 CRT"质感。
 *
 * 管线结构（每帧）：
 *   1. computePass  → BG compute shader 把三层合成写入 rgba8unorm bgTex
 *   2. renderPass   → 全屏四边形读 bgTex，做 chromaAB 后处理，输出到目标 view
 *
 * 用法：
 *   const fx = await EnvironmentFx.create(device, format, width, height);
 *   // 每帧：
 *   fx.setConfig({ gridScale: 12, scatterU: 0.3, aberrationPct: 1.5 });
 *   fx.tick(dt);
 *   fx.render(encoder, dstView);
 */

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/** Tweakable parameters for the three effect layers. */








export interface EnvironmentFxConfig {
  /**
   * Number of brick-tile rows.  Column count = gridScale * aspect.
   * @default 14
   */
  gridScale?: number;

  /**
   * Mortar gap width in UV space (0 = no gap, 1 = all gap).
   * @default 0.04
   */
  mortarWidth?: number;

  /**
   * Primary grid line colour as [r, g, b] in linear sRGB.
   * @default [0.05, 0.28, 0.14]   // lab-green
   */
  gridColor?: [number, number, number];

  /**
   * Background fill colour as [r, g, b] in linear sRGB.
   * @default [0.008, 0.014, 0.024]  // near-black petri-blue
   */
  bgColor?: [number, number, number];

  /**
   * voronoise `u` parameter: 0 = pure voronoi, 1 = pure noise.
   * @default 0.25
   */
  scatterU?: number;

  /**
   * voronoise `v` parameter: controls smoothing/pointiness (0–1).
   * @default 0.5
   */
  scatterV?: number;

  /**
   * Spatial scale of the voronoise pattern.
   * @default 5.0
   */
  scatterScale?: number;

  /**
   * Intensity of the voronoise overlay blended over the grid.
   * @default 0.18
   */
  scatterStrength?: number;

  /**
   * Voronoise hue colour as [r, g, b] in linear sRGB.
   * @default [0.0, 0.9, 0.55]   // bioluminescent cyan-green
   */
  scatterColor?: [number, number, number];

  /**
   * chromaAB aberration amount (maps to CHROMAAB_PCT, typical 0.5–3.0).
   * @default 1.5
   */
  aberrationPct?: number;

  /**
   * Radius in UV units at which aberration starts (0 = whole screen).
   * @default 0.25
   */
  aberrationBuffer?: number;

  /**
   * Slow pulse animation speed for voronoise (radians/second).
   * @default 0.4
   */
  pulseSpeed?: number;
}

/** Snapshot of the processed config (all fields guaranteed). */
export interface EnvironmentFxParams {
  gridScale: number;
  mortarWidth: number;
  gridColor: [number, number, number];
  bgColor: [number, number, number];
  scatterU: number;
  scatterV: number;
  scatterScale: number;
  scatterStrength: number;
  scatterColor: [number, number, number];
  aberrationPct: number;
  aberrationBuffer: number;
  pulseSpeed: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Uniform layout
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Flat Float32Array layout for the compute + render uniform buffer.
 * Total: 24 floats = 96 bytes (6 × vec4 aligned).
 *
 * [ 0] width          [ 1] height         [ 2] time           [ 3] aspect
 * [ 4] gridScale      [ 5] mortarWidth     [ 6] gridR          [ 7] gridG
 * [ 8] gridB          [ 9] bgR             [10] bgG            [11] bgB
 * [12] scatterU       [13] scatterV        [14] scatterScale   [15] scatterStrength
 * [16] scatterR       [17] scatterG        [18] scatterB       [19] pulseSpeed
 * [20] aberrationPct  [21] aberrationBuf   [22] _pad0          [23] _pad1
 */
const UNIFORM_FLOATS = 24;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — compute shader: brick-tile grid + voronoise scatter → bgTex
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_COMPUTE = /* wgsl */`
// ── Uniforms ──────────────────────────────────────────────────────────────────
struct Uni {
  width          : f32,
  height         : f32,
  time           : f32,
  aspect         : f32,

  gridScale      : f32,
  mortarWidth    : f32,
  gridR          : f32,
  gridG          : f32,

  gridB          : f32,
  bgR            : f32,
  bgG            : f32,
  bgB            : f32,

  scatterU       : f32,
  scatterV       : f32,
  scatterScale   : f32,
  scatterStrength: f32,

  scatterR       : f32,
  scatterG       : f32,
  scatterB       : f32,
  pulseSpeed     : f32,

  aberrationPct  : f32,
  aberrationBuf  : f32,
  _pad0          : f32,
  _pad1          : f32,
}

@group(0) @binding(0) var<uniform> u   : Uni;
@group(0) @binding(1) var          out : texture_storage_2d<rgba8unorm, write>;

// ─────────────────────────────────────────────────────────────────────────────
// Lygia — brickTile  (space/brickTile.glsl → WGSL)
//
// sqTile: tile the UV plane into a [0,1)² cell + integer cell index.
// brickTile: offset every other row by 0.5 in X.
//
// Returns vec4f( cellUV.x, cellUV.y, cellID.x, cellID.y )
// ─────────────────────────────────────────────────────────────────────────────
fn sqTile(st: vec2f) -> vec4f {
  return vec4f(fract(st), floor(st));
}

fn brickTile4(t: vec4f) -> vec4f {
  var r = t;
  r.x  += modf(r.w * 0.5 + 0.5).fract;   // offset odd rows by 0.5
  r.z   = floor(r.z + r.x);
  r.x   = fract(r.x);
  return r;
}

fn brickTile(st: vec2f, scale: f32) -> vec4f {
  return brickTile4(sqTile(st * scale));
}

// ─────────────────────────────────────────────────────────────────────────────
// Lygia — voronoise  (generative/voronoise.wgsl → WGSL, 2-D variant)
//
// Inigo Quilez's voronoise: blends voronoi cell centres (u=0) with
// smooth value noise (u=1). v controls the sharpness of the kernel.
// ─────────────────────────────────────────────────────────────────────────────

// 3-component hash → [0,1]³  (cheap but sufficient)
fn hash3_2(p: vec2f) -> vec3f {
  var q = vec3f(
    dot(p, vec2f(127.1, 311.7)),
    dot(p, vec2f(269.5, 183.3)),
    dot(p, vec2f(419.2, 371.9))
  );
  return fract(sin(q) * 43758.5453);
}

fn voronoise2(p: vec2f, vu: f32, vv: f32) -> f32 {
  let k  = 1.0 + 63.0 * pow(1.0 - vv, 6.0);
  let i  = floor(p);
  let f  = fract(p);

  var acc = vec2f(0.0);

  for (var gy: f32 = -2.0; gy <= 2.0; gy += 1.0) {
    for (var gx: f32 = -2.0; gx <= 2.0; gx += 1.0) {
      let g  = vec2f(gx, gy);
      let o  = hash3_2(i + g) * vec3f(vu, vu, 1.0);
      let d  = g - f + o.xy;
      let w  = pow(1.0 - smoothstep(0.0, 1.414, length(d)), k);
      acc   += vec2f(o.z * w, w);
    }
  }
  return acc.x / acc.y;
}

// ─────────────────────────────────────────────────────────────────────────────
// Compute entry point
// ─────────────────────────────────────────────────────────────────────────────
@compute @workgroup_size(8, 8)
fn cs_env(@builtin(global_invocation_id) gid: vec3u) {
  let px = vec2i(i32(gid.x), i32(gid.y));
  let W  = i32(u.width);
  let H  = i32(u.height);
  if (px.x >= W || px.y >= H) { return; }

  // Normalised UV [0,1]
  let uv = (vec2f(f32(px.x), f32(H - 1 - px.y)) + 0.5) / vec2f(f32(W), f32(H));

  // ── Layer 1: brick-tile grid ────────────────────────────────────────────────
  let tiledSt = vec2f(uv.x * u.aspect, uv.y);      // aspect-corrected
  let tile    = brickTile(tiledSt, u.gridScale);
  let cellUV  = tile.xy;                            // position within one brick

  // Mortar mask: 1 inside brick, 0 on mortar joint
  let hw      = u.mortarWidth * 0.5;
  let inside  = step(hw, cellUV.x) * step(hw, cellUV.y)
              * step(cellUV.x, 1.0 - hw) * step(cellUV.y, 1.0 - hw);

  var col = mix(
    vec3f(u.gridR, u.gridG, u.gridB),   // mortar = grid line colour
    vec3f(u.bgR,   u.bgG,   u.bgB),     // brick fill = background colour
    inside
  );

  // Subtle scanline pulse along Y — gives a slow-breathing CRT feel
  let scan = 0.5 + 0.5 * sin(uv.y * u.gridScale * 3.14159 * 2.0 - u.time * 0.6);
  col += vec3f(u.gridR, u.gridG, u.gridB) * scan * 0.04 * inside;

  // ── Layer 2: voronoise atmospheric scatter ──────────────────────────────────
  let pulse     = 0.5 + 0.5 * sin(u.time * u.pulseSpeed);
  let vCoord    = uv * u.scatterScale + vec2f(u.time * 0.07, u.time * 0.04);
  let vn        = voronoise2(vCoord, u.scatterU + pulse * 0.15, u.scatterV);

  // Soft glow: raise to power to concentrate bright spots
  let glow      = pow(vn, 2.2);
  let scatterCol = vec3f(u.scatterR, u.scatterG, u.scatterB);

  col += scatterCol * glow * u.scatterStrength;

  // Vignette — darken corners to focus the eye inward
  let toCenter = uv - 0.5;
  let vign     = 1.0 - dot(toCenter, toCenter) * 1.6;
  col *= clamp(vign, 0.0, 1.0);

  textureStore(out, px, vec4f(col, 1.0));
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — render shader: full-screen quad + chromaAB post-process
//
// Lygia chromaAB (distort/chromaAB.glsl → WGSL):
//   SDF = lengthSq(st - 0.5)  (distance² from centre)
//   R channel samples inward, B channel samples outward,
//   G channel samples at the original UV.
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_RENDER = /* wgsl */`
struct Uni {
  width          : f32,
  height         : f32,
  time           : f32,
  aspect         : f32,
  // (only aberration fields are used in this pass)
  gridScale      : f32,
  mortarWidth    : f32,
  gridR          : f32,
  gridG          : f32,
  gridB          : f32,
  bgR            : f32,
  bgG            : f32,
  bgB            : f32,
  scatterU       : f32,
  scatterV       : f32,
  scatterScale   : f32,
  scatterStrength: f32,
  scatterR       : f32,
  scatterG       : f32,
  scatterB       : f32,
  pulseSpeed     : f32,
  aberrationPct  : f32,
  aberrationBuf  : f32,
  _pad0          : f32,
  _pad1          : f32,
}

@group(0) @binding(0) var<uniform> u   : Uni;
@group(0) @binding(1) var          smp : sampler;
@group(0) @binding(2) var          bgTex : texture_2d<f32>;

// ─────────────────────────────────────────────────────────────────────────────
// Lygia — chromaAB  (distort/chromaAB.glsl → WGSL)
//
// Uses screen-space SDF (squared distance from centre) to drive per-channel
// UV offsets, replicating the two-overload chromaAB(tex, st, sdf, pct) form.
// ─────────────────────────────────────────────────────────────────────────────
fn lengthSq(v: vec2f) -> f32 { return dot(v, v); }

fn chromaAB(st: vec2f, pct: f32, buf: f32) -> vec3f {
  let toCenter = st - 0.5;
  // Apply center buffer: attenuate SDF near the middle so aberration
  // only kicks in toward the edges (mirrors CHROMAAB_CENTER_BUFFER).
  let sdf      = max(lengthSq(toCenter) - buf * buf, 0.0);
  let offset   = vec2f(sdf);

  let stR = st * (1.0 + offset * 0.02 * pct);
  let stB = st * (1.0 - offset * 0.02 * pct);

  let r = textureSample(bgTex, smp, stR).r;
  let g = textureSample(bgTex, smp, st ).g;
  let b = textureSample(bgTex, smp, stB).b;
  return vec3f(r, g, b);
}

// ── Vertex: two triangles covering NDC ───────────────────────────────────────
struct VsOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f }

@vertex
fn vs_quad(@builtin(vertex_index) vi: u32) -> VsOut {
  // Positions for a full-screen triangle pair (6 verts)
  let xs = array<f32,6>(-1., 1., -1.,  1., -1.,  1.);
  let ys = array<f32,6>(-1.,-1.,  1., -1.,  1.,  1.);
  let x  = xs[vi];
  let y  = ys[vi];
  var o: VsOut;
  o.pos = vec4f(x, y, 0., 1.);
  o.uv  = vec2f(x * 0.5 + 0.5, -y * 0.5 + 0.5);
  return o;
}

// ── Fragment: sample bgTex through chromaAB ───────────────────────────────────
@fragment
fn fs_env(in: VsOut) -> @location(0) vec4f {
  let col = chromaAB(in.uv, u.aberrationPct, u.aberrationBuf);
  return vec4f(col, 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Default config
// ─────────────────────────────────────────────────────────────────────────────

const DEFAULTS: Required<EnvironmentFxConfig> = {
  gridScale       : 14,
  mortarWidth     : 0.04,
  gridColor       : [0.05,  0.28,  0.14],
  bgColor         : [0.008, 0.014, 0.024],
  scatterU        : 0.25,
  scatterV        : 0.5,
  scatterScale    : 5.0,
  scatterStrength : 0.18,
  scatterColor    : [0.0,  0.9,  0.55],
  aberrationPct   : 1.5,
  aberrationBuffer: 0.25,
  pulseSpeed      : 0.4,
};

// ─────────────────────────────────────────────────────────────────────────────
// EnvironmentFx — main class
// ─────────────────────────────────────────────────────────────────────────────

export class EnvironmentFx {
  private constructor(
    private readonly device       : GPUDevice,
    private readonly computePipeline : GPUComputePipeline,
    private readonly renderPipeline  : GPURenderPipeline,
    private readonly computeBGL   : GPUBindGroupLayout,
    private readonly renderBGL    : GPUBindGroupLayout,
    private readonly sampler      : GPUSampler,
    private readonly uniformBuf   : GPUBuffer,
    // mutable per resize
    private bgTex                 : GPUTexture,
    private bgView                : GPUTextureView,
    private width                 : number,
    private height                : number,
  ) {}

  // ── Factory ─────────────────────────────────────────────────────────────────

  static async create(
    device : GPUDevice,
    format : GPUTextureFormat,
    width  : number,
    height : number,
  ): Promise<EnvironmentFx> {

    // ── Compute pipeline (bg synthesis) ────────────────────────────────────
    const computeBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: 'write-only', format: 'rgba8unorm' } },
      ],
    });

    const computeMod = device.createShaderModule({ code: WGSL_COMPUTE });
    const computePipeline = await device.createComputePipelineAsync({
      layout : device.createPipelineLayout({ bindGroupLayouts: [computeBGL] }),
      compute: { module: computeMod, entryPoint: 'cs_env' },
    });

    // ── Render pipeline (chromaAB post-process) ─────────────────────────────
    const renderBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT,
          sampler: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT,
          texture: {} },
      ],
    });

    const renderMod = device.createShaderModule({ code: WGSL_RENDER });
    const renderPipeline = await device.createRenderPipelineAsync({
      layout  : device.createPipelineLayout({ bindGroupLayouts: [renderBGL] }),
      vertex  : { module: renderMod, entryPoint: 'vs_quad' },
      fragment: {
        module    : renderMod,
        entryPoint: 'fs_env',
        targets   : [{ format }],
      },
      primitive: { topology: 'triangle-list' },
    });

    // ── Shared resources ────────────────────────────────────────────────────
    const sampler = device.createSampler({
      magFilter   : 'linear',
      minFilter   : 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    const uniformBuf = device.createBuffer({
      size  : UNIFORM_FLOATS * 4,
      usage : GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const [bgTex, bgView] = EnvironmentFx._makeBgTex(device, width, height);

    return new EnvironmentFx(
      device, computePipeline, renderPipeline,
      computeBGL, renderBGL,
      sampler, uniformBuf,
      bgTex, bgView,
      width, height,
    );
  }

  // ── State ───────────────────────────────────────────────────────────────────

  private params: EnvironmentFxParams = { ...DEFAULTS };
  private time = 0;

  /** Merge new config values. Unspecified keys keep their current value. */
  setConfig(cfg: EnvironmentFxConfig): this {
    this.params = {
      gridScale       : cfg.gridScale        ?? this.params.gridScale,
      mortarWidth     : cfg.mortarWidth      ?? this.params.mortarWidth,
      gridColor       : cfg.gridColor        ?? this.params.gridColor,
      bgColor         : cfg.bgColor          ?? this.params.bgColor,
      scatterU        : cfg.scatterU         ?? this.params.scatterU,
      scatterV        : cfg.scatterV         ?? this.params.scatterV,
      scatterScale    : cfg.scatterScale     ?? this.params.scatterScale,
      scatterStrength : cfg.scatterStrength  ?? this.params.scatterStrength,
      scatterColor    : cfg.scatterColor     ?? this.params.scatterColor,
      aberrationPct   : cfg.aberrationPct    ?? this.params.aberrationPct,
      aberrationBuffer: cfg.aberrationBuffer ?? this.params.aberrationBuffer,
      pulseSpeed      : cfg.pulseSpeed       ?? this.params.pulseSpeed,
    };
    return this;
  }

  /** Advance animation clock (call once per frame with delta in seconds). */
  tick(dt: number): this {
    this.time += dt;
    return this;
  }

  /** Recreate the intermediate bgTex when the canvas is resized. */
  resize(width: number, height: number): this {
    if (width === this.width && height === this.height) return this;
    this.bgTex.destroy();
    [this.bgTex, this.bgView] = EnvironmentFx._makeBgTex(this.device, width, height);
    this.width  = width;
    this.height = height;
    return this;
  }

  // ── Per-frame render ────────────────────────────────────────────────────────

  /**
   * Record a compute pass (bg synthesis) and a render pass (chromaAB) into
   * `encoder`, writing the final composited frame to `dstView`.
   *
   * Call `tick()` before this each frame to advance the animation.
   */
  render(encoder: GPUCommandEncoder, dstView: GPUTextureView): void {
    this._uploadUniforms();

    // ── Compute: synthesise brick-tile + voronoise → bgTex ─────────────────
    const computeBG = this.device.createBindGroup({
      layout  : this.computeBGL,
      entries : [
        { binding: 0, resource: { buffer: this.uniformBuf } },
        { binding: 1, resource: this.bgView },
      ],
    });

    const cPass = encoder.beginComputePass();
    cPass.setPipeline(this.computePipeline);
    cPass.setBindGroup(0, computeBG);
    cPass.dispatchWorkgroups(
      Math.ceil(this.width  / 8),
      Math.ceil(this.height / 8),
    );
    cPass.end();

    // ── Render: chromaAB post-process → dstView ─────────────────────────────
    const renderBG = this.device.createBindGroup({
      layout  : this.renderBGL,
      entries : [
        { binding: 0, resource: { buffer: this.uniformBuf } },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: this.bgTex.createView() },
      ],
    });

    const rPass = encoder.beginRenderPass({
      colorAttachments: [{
        view      : dstView,
        loadOp    : 'clear',
        storeOp   : 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    rPass.setPipeline(this.renderPipeline);
    rPass.setBindGroup(0, renderBG);
    rPass.draw(6);   // 2 triangles → full-screen quad
    rPass.end();
  }

  destroy(): void {
    this.uniformBuf.destroy();
    this.bgTex.destroy();
  }

  // ── Internals ───────────────────────────────────────────────────────────────

  private _uploadUniforms(): void {
    const p = this.params;
    const [gR, gG, gB]  = p.gridColor;
    const [bR, bG, bB]  = p.bgColor;
    const [sR, sG, sB]  = p.scatterColor;
    const aspect = this.width / this.height;

    const data = new Float32Array(UNIFORM_FLOATS);
    data[ 0] = this.width;
    data[ 1] = this.height;
    data[ 2] = this.time;
    data[ 3] = aspect;
    data[ 4] = p.gridScale;
    data[ 5] = p.mortarWidth;
    data[ 6] = gR;
    data[ 7] = gG;
    data[ 8] = gB;
    data[ 9] = bR;
    data[10]  = bG;
    data[11]  = bB;
    data[12]  = p.scatterU;
    data[13]  = p.scatterV;
    data[14]  = p.scatterScale;
    data[15]  = p.scatterStrength;
    data[16]  = sR;
    data[17]  = sG;
    data[18]  = sB;
    data[19]  = p.pulseSpeed;
    data[20]  = p.aberrationPct;
    data[21]  = p.aberrationBuffer;
    data[22]  = 0; // _pad0
    data[23]  = 0; // _pad1

    this.device.queue.writeBuffer(this.uniformBuf, 0, data);
  }

  private static _makeBgTex(
    device: GPUDevice,
    width : number,
    height: number,
  ): [GPUTexture, GPUTextureView] {
    const tex = device.createTexture({
      size   : [width, height],
      format : 'rgba8unorm',
      usage  : GPUTextureUsage.STORAGE_BINDING
             | GPUTextureUsage.TEXTURE_BINDING
             | GPUTextureUsage.COPY_SRC,
    });
    return [tex, tex.createView()];
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Preset factory helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Ready-made atmosphere presets for common "bio lab" visual registers.
 * Each mutates the passed EnvironmentFx in-place and returns it for chaining.
 */
export const EnvironmentFxPresets = {

  /**
   * bioLabClassic — 经典生物实验室
   * 深蓝背景 + 翠绿砖格 + 低饱和度青色散射 + 轻微色散
   */
  bioLabClassic(fx: EnvironmentFx): EnvironmentFx {
    return fx.setConfig({
      gridScale       : 14,
      mortarWidth     : 0.04,
      gridColor       : [0.05,  0.28,  0.14],
      bgColor         : [0.008, 0.014, 0.024],
      scatterU        : 0.25,
      scatterV        : 0.5,
      scatterScale    : 5.0,
      scatterStrength : 0.18,
      scatterColor    : [0.0,   0.9,   0.55],
      aberrationPct   : 1.5,
      aberrationBuffer: 0.25,
      pulseSpeed      : 0.4,
    });
  },

  /**
   * bioluminescence — 深海生物发光
   * 极暗背景 + 极细网格 + 高强度青紫散射 + 强色散
   */
  bioluminescence(fx: EnvironmentFx): EnvironmentFx {
    return fx.setConfig({
      gridScale       : 22,
      mortarWidth     : 0.025,
      gridColor       : [0.02,  0.12,  0.22],
      bgColor         : [0.002, 0.004, 0.012],
      scatterU        : 0.15,
      scatterV        : 0.3,
      scatterScale    : 6.5,
      scatterStrength : 0.35,
      scatterColor    : [0.1,   0.6,   1.0],
      aberrationPct   : 2.5,
      aberrationBuffer: 0.18,
      pulseSpeed      : 0.6,
    });
  },

  /**
   * quarantine — 生化隔离舱
   * 琥珀黄警示色调 + 较粗网格 + 橙色散射 + 中等色散
   */
  quarantine(fx: EnvironmentFx): EnvironmentFx {
    return fx.setConfig({
      gridScale       : 10,
      mortarWidth     : 0.06,
      gridColor       : [0.55,  0.28,  0.0],
      bgColor         : [0.018, 0.010, 0.002],
      scatterU        : 0.4,
      scatterV        : 0.6,
      scatterScale    : 4.0,
      scatterStrength : 0.22,
      scatterColor    : [0.9,   0.55,  0.0],
      aberrationPct   : 1.8,
      aberrationBuffer: 0.3,
      pulseSpeed      : 0.3,
    });
  },

  /**
   * cryogenics — 低温冷冻舱
   * 冰蓝细格 + 白蓝散射 + 强边缘色散
   */
  cryogenics(fx: EnvironmentFx): EnvironmentFx {
    return fx.setConfig({
      gridScale       : 18,
      mortarWidth     : 0.03,
      gridColor       : [0.2,   0.5,   0.8],
      bgColor         : [0.005, 0.008, 0.020],
      scatterU        : 0.1,
      scatterV        : 0.65,
      scatterScale    : 7.0,
      scatterStrength : 0.12,
      scatterColor    : [0.6,   0.85,  1.0],
      aberrationPct   : 3.0,
      aberrationBuffer: 0.2,
      pulseSpeed      : 0.2,
    });
  },

} as const;
