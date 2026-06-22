/**
 * post-process.ts — WebGPU全屏后处理管线
 *
 * 三种艺术风格，均可独立启用或叠加：
 *   1. KUWAHARA  — 四象限均值/方差选最平坦区域 → 油画笔触感
 *   2. EDGE      — Sobel梯度检测 → 赛博朋克霓虹描边
 *   3. INK       — 亮度分层 + 随机扰动 → 水墨晕染
 *
 * 设计来源：
 *   - upstream/lygia/filter/kuwahara.glsl  (四象限 Kuwahara)
 *   - upstream/lygia/filter/edge.glsl      (Sobel edge / Prewitt)
 *
 * 用法：
 *   const pp = await PostProcessPipeline.create(device, format, width, height);
 *   // 每帧：
 *   pp.setStyle({ kuwahara: true, edge: true, ink: false });
 *   pp.render(encoder, srcView, dstView);  // srcView = 上一 pass 的输出
 */

// ─────────────────────────────────────────────────────────────────────────────
// WGSL shader source
// ─────────────────────────────────────────────────────────────────────────────

const POST_PROCESS_WGSL = /* wgsl */`
// ─── Uniforms ────────────────────────────────────────────────────────────────
struct PostUniforms {
  // resolution
  width   : f32,
  height  : f32,
  // style toggles  (0.0 = off, 1.0 = on)
  doKuwahara : f32,
  doEdge     : f32,
  doInk      : f32,
  // Kuwahara
  kuwaharaRadius : f32,   // typical: 3–6
  // Edge
  edgeThreshold  : f32,   // [0,1]  sobel magnitude cutoff
  edgeStrength   : f32,   // blend weight for edge overlay
  // Edge colour (cyberpunk neon)
  edgeR : f32,
  edgeG : f32,
  edgeB : f32,
  // Ink
  inkLevels    : f32,   // posterisation steps (3–6)
  inkNoise     : f32,   // perturbation amplitude (0–0.05)
  // time (for ink noise animation)
  time    : f32,
  _pad0   : f32,
  _pad1   : f32,
}

@group(0) @binding(0) var<uniform> u  : PostUniforms;
@group(0) @binding(1) var          smp : sampler;
@group(0) @binding(2) var          src : texture_2d<f32>;

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn luminance(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

// Simple hash for pseudo-random noise (ink wobble)
fn hash21(p: vec2f) -> f32 {
  var p3 = fract(vec3f(p.xyx) * vec3f(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

// ─── Pass 1: Kuwahara oil-painting filter ────────────────────────────────────
// Four-quadrant variant from lygia/filter/kuwahara.glsl:
//   For each of 4 axis-aligned quadrants around st, compute mean & variance.
//   Pick the quadrant with lowest variance (smoothest) → oil-painting stroke.

fn kuwahara(st: vec2f, pixel: vec2f, radius: f32) -> vec4f {
  let n = (radius + 1.0) * (radius + 1.0);

  var m0 = vec4f(0.0); var s0 = vec4f(0.0);
  var m1 = vec4f(0.0); var s1 = vec4f(0.0);
  var m2 = vec4f(0.0); var s2 = vec4f(0.0);
  var m3 = vec4f(0.0); var s3 = vec4f(0.0);

  // Q0: top-left  (-r..0, -r..0)
  for (var j = -radius; j <= 0.0; j += 1.0) {
    for (var i = -radius; i <= 0.0; i += 1.0) {
      let c = textureSample(src, smp, st + vec2f(i, j) * pixel);
      m0 += c;  s0 += c * c;
    }
  }
  // Q1: top-right  (0..r, -r..0)
  for (var j = -radius; j <= 0.0; j += 1.0) {
    for (var i = 0.0; i <= radius; i += 1.0) {
      let c = textureSample(src, smp, st + vec2f(i, j) * pixel);
      m1 += c;  s1 += c * c;
    }
  }
  // Q2: bottom-right  (0..r, 0..r)
  for (var j = 0.0; j <= radius; j += 1.0) {
    for (var i = 0.0; i <= radius; i += 1.0) {
      let c = textureSample(src, smp, st + vec2f(i, j) * pixel);
      m2 += c;  s2 += c * c;
    }
  }
  // Q3: bottom-left  (-r..0, 0..r)
  for (var j = 0.0; j <= radius; j += 1.0) {
    for (var i = -radius; i <= 0.0; i += 1.0) {
      let c = textureSample(src, smp, st + vec2f(i, j) * pixel);
      m3 += c;  s3 += c * c;
    }
  }

  m0 /= n;  let var0 = abs(s0 / n - m0 * m0);
  m1 /= n;  let var1 = abs(s1 / n - m1 * m1);
  m2 /= n;  let var2 = abs(s2 / n - m2 * m2);
  m3 /= n;  let var3 = abs(s3 / n - m3 * m3);

  let sig0 = var0.r + var0.g + var0.b;
  let sig1 = var1.r + var1.g + var1.b;
  let sig2 = var2.r + var2.g + var2.b;
  let sig3 = var3.r + var3.g + var3.b;

  var result = m0;
  var minSig = sig0;
  if (sig1 < minSig) { minSig = sig1; result = m1; }
  if (sig2 < minSig) { minSig = sig2; result = m2; }
  if (sig3 < minSig) {                result = m3; }
  return result;
}

// ─── Pass 2: Sobel edge detection (from lygia/filter/edge/sobel.glsl) ────────
// Returns edge magnitude [0,1].

fn sobelEdge(st: vec2f, pixel: vec2f) -> f32 {
  let tleft  = luminance(textureSample(src, smp, st + vec2f(-pixel.x,  pixel.y)).rgb);
  let left   = luminance(textureSample(src, smp, st + vec2f(-pixel.x,  0.0    )).rgb);
  let bleft  = luminance(textureSample(src, smp, st + vec2f(-pixel.x, -pixel.y)).rgb);
  let top    = luminance(textureSample(src, smp, st + vec2f( 0.0,      pixel.y)).rgb);
  let bottom = luminance(textureSample(src, smp, st + vec2f( 0.0,     -pixel.y)).rgb);
  let tright = luminance(textureSample(src, smp, st + pixel                    ).rgb);
  let right  = luminance(textureSample(src, smp, st + vec2f( pixel.x,  0.0    )).rgb);
  let bright = luminance(textureSample(src, smp, st + vec2f( pixel.x, -pixel.y)).rgb);

  let gx = tleft + 2.0 * left  + bleft  - tright - 2.0 * right  - bright;
  let gy = bleft + 2.0 * bottom + bright - tleft  - 2.0 * top    - tright;
  return sqrt(gx * gx + gy * gy);
}

// ─── Pass 3: Ink / sumi-e wash ───────────────────────────────────────────────
// 1. Convert to greyscale luminance.
// 2. Posterise into N discrete ink "layers."
// 3. Add fibre-like noise to simulate paper grain / brush edge wobble.
// 4. Remap dark→inky-black, light→paper-white with a warm paper tint.

fn inkWash(st: vec2f, baseColor: vec3f) -> vec3f {
  let lum     = luminance(baseColor);
  let levels  = max(u.inkLevels, 2.0);

  // Posterise
  let stepped = floor(lum * levels + 0.5) / levels;

  // Paper-grain perturbation using hash noise
  let noiseUV = st * vec2f(u.width, u.height);   // pixel coords
  let grain   = hash21(noiseUV + vec2f(u.time * 0.13, u.time * 0.07)) * 2.0 - 1.0;
  let noisy   = clamp(stepped + grain * u.inkNoise, 0.0, 1.0);

  // Ink tone: near-black ink on warm parchment
  // dark  → 暗墨色 (0.06, 0.05, 0.07)
  // light → 纸色   (0.96, 0.93, 0.85)
  let ink   = vec3f(0.06, 0.05, 0.07);
  let paper = vec3f(0.96, 0.93, 0.85);
  return mix(ink, paper, noisy);
}

// ─── Full-screen quad vertex shader ──────────────────────────────────────────

struct Vert {
  @builtin(position) pos : vec4f,
  @location(0)       uv  : vec2f,
}

@vertex fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> Vert {
  // Two triangles covering NDC [-1,1]
  var pos = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0),
    vec2f(-1.0, -1.0), vec2f( 1.0,  1.0), vec2f(-1.0,  1.0),
  );
  var uv = array<vec2f, 6>(
    vec2f(0.0, 1.0), vec2f(1.0, 1.0), vec2f(1.0, 0.0),
    vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(0.0, 0.0),
  );
  var out: Vert;
  out.pos = vec4f(pos[vi], 0.0, 1.0);
  out.uv  = uv[vi];
  return out;
}

// ─── Fragment shader — compose all three styles ───────────────────────────────

@fragment fn fs_post(in: Vert) -> @location(0) vec4f {
  let st    = in.uv;
  let pixel = vec2f(1.0 / u.width, 1.0 / u.height);

  // ── Kuwahara
  var color: vec4f;
  if (u.doKuwahara > 0.5) {
    color = kuwahara(st, pixel, u.kuwaharaRadius);
  } else {
    color = textureSample(src, smp, st);
  }

  // ── Ink wash  (operates on kuwahara output for smoother blobs)
  if (u.doInk > 0.5) {
    let inkColor = inkWash(st, color.rgb);
    color = vec4f(inkColor, color.a);
  }

  // ── Cyberpunk Sobel edge overlay
  if (u.doEdge > 0.5) {
    // Sample Sobel from *original* src to preserve sharp edges even after kuwahara
    let mag = sobelEdge(st, pixel);
    if (mag > u.edgeThreshold) {
      // Neon glow: intensity increases with magnitude above threshold
      let intensity = smoothstep(u.edgeThreshold, 1.0, mag) * u.edgeStrength;
      let neon      = vec3f(u.edgeR, u.edgeG, u.edgeB);
      // Additive blend for bloom-like glow
      color = vec4f(color.rgb + neon * intensity, color.a);
    }
  }

  return clamp(color, vec4f(0.0), vec4f(1.0));
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// TypeScript API
// ─────────────────────────────────────────────────────────────────────────────

/** Which art styles to enable this frame. */
export interface PostProcessStyle {
  /** Oil-painting Kuwahara smoothing */
  kuwahara?: boolean;
  /** Sobel edge detection with cyberpunk neon glow */
  edge?: boolean;
  /** Sumi-e ink-wash posterisation */
  ink?: boolean;
}

/** Fine-grained parameter overrides (all optional, fall back to defaults). */
export interface PostProcessParams {
  /** Kuwahara neighbourhood radius in pixels (default 4) */
  kuwaharaRadius?: number;
  /** Sobel magnitude threshold [0,1] below which edges are ignored (default 0.15) */
  edgeThreshold?: number;
  /** Additive strength of edge colour overlay (default 0.9) */
  edgeStrength?: number;
  /** Neon edge colour as [r,g,b] each in [0,1] (default cyan-magenta) */
  edgeColor?: [number, number, number];
  /** Ink posterisation steps (default 4) */
  inkLevels?: number;
  /** Ink paper-grain noise amplitude (default 0.025) */
  inkNoise?: number;
}

// GPU-side uniform layout (float32, std140-compatible, 16 × f32 = 64 bytes)
const UNIFORM_FLOATS = 16;

export class PostProcessPipeline {
  private readonly device   : GPUDevice;
  private readonly pipeline : GPURenderPipeline;
  private readonly bgl      : GPUBindGroupLayout;
  private readonly sampler  : GPUSampler;
  private readonly uniformBuf: GPUBuffer;

  // Cached state
  private style  : Required<PostProcessStyle>  = { kuwahara: false, edge: false, ink: false };
  private params : Required<PostProcessParams>;
  private time   = 0;

  // Bind group cache — keyed by source texture view (invalidated each resize or src change)
  private cachedBG   : GPUBindGroup | null = null;
  private cachedSrc  : GPUTextureView | null = null;

  private constructor(
    device  : GPUDevice,
    pipeline: GPURenderPipeline,
    bgl     : GPUBindGroupLayout,
    sampler : GPUSampler,
    uniformBuf: GPUBuffer,
  ) {
    this.device     = device;
    this.pipeline   = pipeline;
    this.bgl        = bgl;
    this.sampler    = sampler;
    this.uniformBuf = uniformBuf;

    this.params = {
      kuwaharaRadius : 4,
      edgeThreshold  : 0.15,
      edgeStrength   : 0.9,
      edgeColor      : [0.0, 1.0, 0.95],   // cyan-ish neon
      inkLevels      : 4,
      inkNoise       : 0.025,
    };
  }

  // ── Factory ────────────────────────────────────────────────────────────────

  static async create(
    device: GPUDevice,
    format: GPUTextureFormat,
  ): Promise<PostProcessPipeline> {
    const module = device.createShaderModule({ code: POST_PROCESS_WGSL });

    const bgl = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      ],
    });

    const pipeline = await device.createRenderPipelineAsync({
      layout : device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      vertex : { module, entryPoint: 'vs_fullscreen' },
      fragment: {
        module,
        entryPoint: 'fs_post',
        targets: [{ format }],
      },
      primitive: { topology: 'triangle-list' },
    });

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

    return new PostProcessPipeline(device, pipeline, bgl, sampler, uniformBuf);
  }

  // ── Configuration ──────────────────────────────────────────────────────────

  /** Toggle art styles for the next render call. */
  setStyle(style: PostProcessStyle): this {
    this.style = { kuwahara: false, edge: false, ink: false, ...style };
    return this;
  }

  /** Override individual shader parameters. */
  setParams(p: PostProcessParams): this {
    this.params = { ...this.params, ...p };
    return this;
  }

  /** Advance animation clock (call once per frame with delta seconds). */
  tick(dt: number): this {
    this.time += dt;
    return this;
  }

  // ── Per-frame render ───────────────────────────────────────────────────────

  /**
   * Record a single full-screen post-process pass into `encoder`.
   *
   * @param encoder  Active GPUCommandEncoder
   * @param srcView  Input texture view (e.g. scene render target)
   * @param dstView  Output texture view (swap-chain surface or next RT)
   * @param width    Render target width  in pixels
   * @param height   Render target height in pixels
   */
  render(
    encoder: GPUCommandEncoder,
    srcView : GPUTextureView,
    dstView : GPUTextureView,
    width   : number,
    height  : number,
  ): void {
    this._uploadUniforms(width, height);
    const bg = this._bindGroup(srcView);

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view      : dstView,
        loadOp    : 'clear',
        storeOp   : 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bg);
    pass.draw(6);   // two triangles = full-screen quad
    pass.end();
  }

  destroy(): void {
    this.uniformBuf.destroy();
  }

  // ── Internal ───────────────────────────────────────────────────────────────

  private _uploadUniforms(width: number, height: number): void {
    const s = this.style;
    const p = this.params;
    const [er, eg, eb] = p.edgeColor;

    const data = new Float32Array(UNIFORM_FLOATS);
    data[0]  = width;
    data[1]  = height;
    data[2]  = s.kuwahara ? 1.0 : 0.0;
    data[3]  = s.edge     ? 1.0 : 0.0;
    data[4]  = s.ink      ? 1.0 : 0.0;
    data[5]  = p.kuwaharaRadius;
    data[6]  = p.edgeThreshold;
    data[7]  = p.edgeStrength;
    data[8]  = er;
    data[9]  = eg;
    data[10] = eb;
    data[11] = p.inkLevels;
    data[12] = p.inkNoise;
    data[13] = this.time;
    data[14] = 0.0; // _pad0
    data[15] = 0.0; // _pad1

    this.device.queue.writeBuffer(this.uniformBuf, 0, data);
  }

  private _bindGroup(srcView: GPUTextureView): GPUBindGroup {
    if (this.cachedBG && this.cachedSrc === srcView) {
      return this.cachedBG;
    }
    const bg = this.device.createBindGroup({
      layout  : this.bgl,
      entries : [
        { binding: 0, resource: { buffer: this.uniformBuf } },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: srcView },
      ],
    });
    this.cachedBG  = bg;
    this.cachedSrc = srcView;
    return bg;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Preset factory helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Pre-configured style presets for common artistic looks. */
export const PostProcessPresets = {

  /** 浓郁油画感：Kuwahara 平滑 + 水墨晕染 */
  oilPainting(pp: PostProcessPipeline): PostProcessPipeline {
    return pp
      .setStyle({ kuwahara: true, ink: true, edge: false })
      .setParams({ kuwaharaRadius: 5, inkLevels: 5, inkNoise: 0.03 });
  },

  /** 赛博朋克：保留细节 + 青色霓虹描边 */
  cyberpunk(pp: PostProcessPipeline): PostProcessPipeline {
    return pp
      .setStyle({ kuwahara: false, edge: true, ink: false })
      .setParams({
        edgeThreshold : 0.10,
        edgeStrength  : 1.2,
        edgeColor     : [0.0, 1.0, 0.95],
      });
  },

  /** 黄金霓虹变体：暖橙描边 */
  goldenNeon(pp: PostProcessPipeline): PostProcessPipeline {
    return pp
      .setStyle({ kuwahara: false, edge: true, ink: false })
      .setParams({
        edgeThreshold : 0.12,
        edgeStrength  : 1.0,
        edgeColor     : [1.0, 0.65, 0.0],
      });
  },

  /** 纯水墨：无彩色，仅墨色晕染 */
  sumi_e(pp: PostProcessPipeline): PostProcessPipeline {
    return pp
      .setStyle({ kuwahara: true, edge: false, ink: true })
      .setParams({
        kuwaharaRadius : 3,
        inkLevels      : 3,
        inkNoise       : 0.04,
      });
  },

  /** 全效叠加：油画 + 水墨 + 赛博描边 */
  dreamscape(pp: PostProcessPipeline): PostProcessPipeline {
    return pp
      .setStyle({ kuwahara: true, edge: true, ink: true })
      .setParams({
        kuwaharaRadius : 4,
        edgeThreshold  : 0.18,
        edgeStrength   : 0.7,
        edgeColor      : [0.5, 0.9, 1.0],
        inkLevels      : 4,
        inkNoise       : 0.02,
      });
  },

  /** 重置为直通（pass-through）*/
  passthrough(pp: PostProcessPipeline): PostProcessPipeline {
    return pp.setStyle({ kuwahara: false, edge: false, ink: false });
  },
} as const;
