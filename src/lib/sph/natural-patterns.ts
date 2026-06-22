// src/lib/sph/natural-patterns.ts
//
// Natural cell-surface textures via WebGPU compute shaders.
// Voronoi + Worley WGSL ported from lygia (Patricio Gonzalez Vivo):
//   upstream/lygia/generative/voronoi.wgsl
//   upstream/lygia/generative/worley.wgsl
//
// Outputs a GPU texture whose RGBA channels encode:
//   R – base pattern intensity (voronoi cell distance or worley F1)
//   G – secondary distance (voronoi centroid Y or worley F2)
//   B – edge/ridge mask (|F2-F1| for worley, boundary highlight for voronoi)
//   A – species-tinted hue selector (mapped from NaturalPatternParams.species)
//
// Usage:
//   const gen = new NaturalPatternGenerator(device);
//   const tex = await gen.generate({ width: 512, height: 512, scale: 4, ... });
//   // bind tex to your render pipeline as a sampled texture

// ─── Species → pattern-mode mapping ──────────────────────────────────────────
// Each Transformer cell species gets a visual metaphor:
//   CELL_DIVISION  – Voronoi cells expanding from centroids   (cil-eye, cil-bolt)
//   TORTOISE_SHELL – Worley F2–F1 ridges (hexagonal cracking) (cil-vector, cil-plus)
//   LEAF_VEIN      – Multi-octave Worley F1 (branching veins) (cil-arrow-right, cil-filter)
//   FOAM           – Voronoi + Worley blend (soap-bubble foam) (cil-layers, cil-loop)
//   SCALES         – Voronoi with distance modulation          (cil-code, cil-graph)
export type NaturalPatternMode =
  | 'CELL_DIVISION'
  | 'TORTOISE_SHELL'
  | 'LEAF_VEIN'
  | 'FOAM'
  | 'SCALES';

/** Map a cell species string to its natural-pattern visual mode. */
export function speciesPatternMode(species: string): NaturalPatternMode {
  const MAP: Record<string, NaturalPatternMode> = {
    'fluid':           'FOAM',
    'cil-eye':         'CELL_DIVISION',
    'cil-bolt':        'CELL_DIVISION',
    'cil-vector':      'TORTOISE_SHELL',
    'cil-plus':        'TORTOISE_SHELL',
    'cil-arrow-right': 'LEAF_VEIN',
    'cil-filter':      'LEAF_VEIN',
    'cil-layers':      'FOAM',
    'cil-loop':        'FOAM',
    'cil-code':        'SCALES',
    'cil-graph':       'SCALES',
  };
  return MAP[species] ?? 'CELL_DIVISION';
}

/** Numeric mode constant forwarded into the compute shader as a uniform. */
const MODE_INDEX: Record<NaturalPatternMode, number> = {
  CELL_DIVISION:  0,
  TORTOISE_SHELL: 1,
  LEAF_VEIN:      2,
  FOAM:           3,
  SCALES:         4,
};

// ─── Public API types ─────────────────────────────────────────────────────────

export interface NaturalPatternParams {
  /** Texture width in pixels (power-of-two recommended). Default 512. */
  width?: number;
  /** Texture height in pixels. Default 512. */
  height?: number;
  /** Cell frequency – higher = more, smaller cells. Range 1–32. Default 6. */
  scale?: number;
  /** Cell centroid randomness, 0 = grid, 1 = full random. Default 0.85. */
  jitter?: number;
  /** Number of Worley/Voronoi octaves for fBm layering. Range 1–6. Default 3. */
  octaves?: number;
  /** Pattern mode; defaults from species if provided, else CELL_DIVISION. */
  mode?: NaturalPatternMode;
  /** Optional cell species string – overrides `mode` via speciesPatternMode(). */
  species?: string;
  /** Animation time (seconds) for pulsing Voronoi centroids. Default 0. */
  time?: number;
}

// ─── Inlined WGSL – lygia voronoi + worley ───────────────────────────────────
// Adapted for standalone WebGPU (no #include pre-processor):
//   • random22 / random33 replace lygia's random.wgsl dependency
//   • distEuclidean2 / distEuclidean3 replace lygia's dist.wgsl
//   • voronoi and worley bodies are verbatim from the upstream WGSL sources,
//     with the broken WGSL `for(int …)` loops fixed to `for(var i: i32 = …)`
//     and `float(i)` changed to `f32(i)` per WGSL spec.

const COMPUTE_SHADER_SRC = /* wgsl */`
// ── uniforms ──────────────────────────────────────────────────────────────────
struct Params {
  width:   u32,
  height:  u32,
  scale:   f32,   // cell frequency
  jitter:  f32,   // [0,1] centroid jitter
  octaves: u32,   // fBm octaves
  mode:    u32,   // 0=CELL_DIVISION 1=TORTOISE_SHELL 2=LEAF_VEIN 3=FOAM 4=SCALES
  time:    f32,   // animation seconds
  _pad:    f32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var outTex: texture_storage_2d<rgba8unorm, write>;

// ── math helpers ─────────────────────────────────────────────────────────────
const TAU: f32 = 6.28318530717958647692;

// lygia random.wgsl — random22 / random33
fn hash12(p: vec2f) -> f32 {
  var p3 = fract(vec3f(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

fn random22(p: vec2f) -> vec2f {
  var p3 = fract(vec3f(p.xyx) * vec3f(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.xx + p3.yz) * p3.zy);
}

fn random33(p: vec3f) -> vec3f {
  var p3 = fract(p * vec3f(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yxz + 33.33);
  return fract((p3.xxy + p3.yxx) * p3.zyx);
}

// lygia dist.wgsl — Euclidean
fn distEuclidean2(a: vec2f, b: vec2f) -> f32 { return length(a - b); }
fn distEuclidean3(a: vec3f, b: vec3f) -> f32 { return length(a - b); }

// ── lygia voronoi.wgsl (verbatim logic, WGSL-clean loops) ────────────────────
// Returns vec3(centroid.x, centroid.y, min_dist)
fn voronoi2(uv: vec2f, time: f32, jitter: f32) -> vec3f {
  let i_uv = floor(uv);
  let f_uv = fract(uv);
  var min_dist = 10.0;
  var centroid = vec2f(0.0);
  for (var j: i32 = -1; j <= 1; j++) {
    for (var i: i32 = -1; i <= 1; i++) {
      let neighbor = vec2f(f32(i), f32(j));
      var point = random22(i_uv + neighbor);
      // animate centroid position (lygia's 0.5+0.5*sin(time+TAU*random))
      point = 0.5 + 0.5 * sin(time + TAU * point);
      // apply jitter: blend between grid centre (0.5) and animated point
      point = mix(vec2f(0.5), point, jitter);
      let diff = neighbor + point - f_uv;
      let dist = length(diff);
      if (dist < min_dist) {
        min_dist = dist;
        centroid = point;
      }
    }
  }
  return vec3f(centroid, min_dist);
}

// ── lygia worley.wgsl (verbatim logic, WGSL-clean loops) ────────────────────
// Returns vec2(F1, F2) — nearest and second-nearest cell distances
fn worley22(p: vec2f, jitter: f32) -> vec2f {
  let n = floor(p);
  let f = fract(p);
  var distF1 = 1.0;
  var distF2 = 1.0;
  for (var j: i32 = -1; j <= 1; j++) {
    for (var i: i32 = -1; i <= 1; i++) {
      let g = vec2f(f32(i), f32(j));
      let o = random22(n + g) * jitter;
      let wp = g + o;
      let d = distEuclidean2(wp, f);
      if (d < distF1) {
        distF2 = distF1;
        distF1 = d;
      } else if (d < distF2) {
        distF2 = d;
      }
    }
  }
  return vec2f(distF1, distF2);
}

// ── fBm layering ─────────────────────────────────────────────────────────────
fn fbmVoronoi(uv: vec2f, octaves: u32, scale: f32, jitter: f32, time: f32) -> vec3f {
  var result = vec3f(0.0);
  var amp = 0.5;
  var freq = scale;
  var total_amp = 0.0;
  for (var o: u32 = 0u; o < octaves; o++) {
    let v = voronoi2(uv * freq, time, jitter);
    result += v * amp;
    total_amp += amp;
    amp  *= 0.5;
    freq *= 2.0;
  }
  return result / total_amp;
}

fn fbmWorley(uv: vec2f, octaves: u32, scale: f32, jitter: f32) -> vec2f {
  var result = vec2f(0.0);
  var amp = 0.5;
  var freq = scale;
  var total_amp = 0.0;
  for (var o: u32 = 0u; o < octaves; o++) {
    let w = worley22(uv * freq, jitter);
    result += w * amp;
    total_amp += amp;
    amp  *= 0.5;
    freq *= 2.0;
  }
  return result / total_amp;
}

// ── pattern modes ─────────────────────────────────────────────────────────────
// Returns vec4(R, G, B, A) ready to write to rgba8unorm storage texture.

// 0 – CELL_DIVISION: expanding Voronoi cells, pulsing centroids
fn patternCellDivision(uv: vec2f, scale: f32, jitter: f32, octaves: u32, time: f32) -> vec4f {
  let v = fbmVoronoi(uv, octaves, scale, jitter, time);
  let dist  = v.z;                         // distance to centroid [0,1]
  let cx    = v.x;                         // centroid.x
  let cy    = v.y;                         // centroid.y
  let edge  = smoothstep(0.02, 0.06, dist); // soft cell wall
  return vec4f(dist, cy, 1.0 - edge, 1.0);
}

// 1 – TORTOISE_SHELL: Worley F2–F1 ridges (hex cracking pattern)
fn patternTortoiseShell(uv: vec2f, scale: f32, jitter: f32, octaves: u32) -> vec4f {
  let w     = fbmWorley(uv, octaves, scale, jitter);
  let F1    = w.x;
  let F2    = w.y;
  let ridge = F2 - F1;                      // thin ridge where cells meet
  let crack = smoothstep(0.0, 0.12, ridge); // sharp crack lines
  return vec4f(F1, F2, crack, 1.0);
}

// 2 – LEAF_VEIN: multi-octave Worley F1 builds branching vein network
fn patternLeafVein(uv: vec2f, scale: f32, jitter: f32, octaves: u32) -> vec4f {
  let w      = fbmWorley(uv, octaves, scale, jitter);
  let vein   = 1.0 - w.x;                  // invert: bright at cell centres
  let branch = pow(vein, 2.5);             // narrow to branching veins
  let mid    = fbmWorley(uv, 1u, scale * 0.5, jitter * 0.6).x;
  return vec4f(branch, mid, 1.0 - w.y, 1.0);
}

// 3 – FOAM: blend Voronoi + Worley for soap-bubble foam
fn patternFoam(uv: vec2f, scale: f32, jitter: f32, octaves: u32, time: f32) -> vec4f {
  let v  = fbmVoronoi(uv, octaves, scale, jitter, time);
  let w  = fbmWorley(uv, octaves, scale, jitter);
  let bubble = mix(v.z, w.x, 0.5);
  let wall   = smoothstep(0.03, 0.08, bubble);
  return vec4f(bubble, w.y, wall, 1.0);
}

// 4 – SCALES: Voronoi distance modulated to create overlapping scales
fn patternScales(uv: vec2f, scale: f32, jitter: f32, octaves: u32, time: f32) -> vec4f {
  let offset = vec2f(0.0, 0.25); // shift second layer for overlap
  let v1 = fbmVoronoi(uv,          octaves, scale,       jitter, time);
  let v2 = fbmVoronoi(uv + offset, octaves, scale * 1.1, jitter, time * 0.7);
  let scale_cell = smoothstep(0.0, 0.5, v1.z) * (1.0 - smoothstep(0.4, 0.7, v2.z));
  return vec4f(scale_cell, v2.z, v1.z, 1.0);
}

// ── compute entry point ───────────────────────────────────────────────────────
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let px = gid.x;
  let py = gid.y;
  if (px >= p.width || py >= p.height) { return; }

  // UV in [0, 1] — Y flipped so (0,0) is top-left
  let uv = vec2f(
    f32(px) / f32(p.width),
    1.0 - f32(py) / f32(p.height),
  );

  var color: vec4f;
  switch (p.mode) {
    case 0u: { color = patternCellDivision (uv, p.scale, p.jitter, p.octaves, p.time); }
    case 1u: { color = patternTortoiseShell(uv, p.scale, p.jitter, p.octaves); }
    case 2u: { color = patternLeafVein     (uv, p.scale, p.jitter, p.octaves); }
    case 3u: { color = patternFoam         (uv, p.scale, p.jitter, p.octaves, p.time); }
    case 4u: { color = patternScales       (uv, p.scale, p.jitter, p.octaves, p.time); }
    default: { color = patternCellDivision (uv, p.scale, p.jitter, p.octaves, p.time); }
  }

  textureStore(outTex, vec2u(px, py), color);
}
`;

// ─── NaturalPatternGenerator ─────────────────────────────────────────────────

/**
 * GPU-accelerated natural cell-surface texture generator.
 *
 * Creates a {@link GPUTexture} (rgba8unorm, TEXTURE_BINDING | COPY_SRC)
 * populated by a WebGPU compute dispatch.  The texture can be bound directly
 * to a render pipeline or read back to CPU via {@link readback}.
 *
 * @example
 * ```ts
 * const gen = new NaturalPatternGenerator(device);
 * const tex = await gen.generate({ species: 'cil-eye', scale: 6, octaves: 3 });
 * // tex is a GPUTexture ready for render binding
 * gen.destroy(); // free pipelines when done
 * ```
 */
export class NaturalPatternGenerator {
  private readonly device: GPUDevice;
  private pipeline: GPUComputePipeline | null = null;
  private bindGroupLayout: GPUBindGroupLayout | null = null;

  constructor(device: GPUDevice) {
    this.device = device;
  }

  // ── lazy pipeline init ────────────────────────────────────────────────────

  private async ensurePipeline(): Promise<void> {
    if (this.pipeline) return;

    const module = this.device.createShaderModule({
      label: 'natural-patterns-compute',
      code: COMPUTE_SHADER_SRC,
    });

    this.bindGroupLayout = this.device.createBindGroupLayout({
      label: 'natural-patterns-bgl',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform' },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: {
            access: 'write-only',
            format: 'rgba8unorm',
            viewDimension: '2d',
          },
        },
      ],
    });

    this.pipeline = await this.device.createComputePipelineAsync({
      label: 'natural-patterns-pipeline',
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.bindGroupLayout],
      }),
      compute: { module, entryPoint: 'main' },
    });
  }

  // ── public generate ───────────────────────────────────────────────────────

  /**
   * Generate a natural-pattern texture.  Resolves once the GPU work is
   * submitted; the returned texture is valid immediately for binding.
   */
  async generate(params: NaturalPatternParams = {}): Promise<GPUTexture> {
    await this.ensurePipeline();

    const {
      width   = 512,
      height  = 512,
      scale   = 6,
      jitter  = 0.85,
      octaves = 3,
      time    = 0,
      species,
      mode: modeOverride,
    } = params;

    const mode: NaturalPatternMode =
      modeOverride ?? (species ? speciesPatternMode(species) : 'CELL_DIVISION');

    // Uniform buffer: Params struct (8 × 4 bytes = 32 bytes)
    const uniformData = new ArrayBuffer(32);
    const uView = new DataView(uniformData);
    uView.setUint32 ( 0, width,                  true);
    uView.setUint32 ( 4, height,                 true);
    uView.setFloat32( 8, Math.max(1, scale),     true);
    uView.setFloat32(12, Math.min(1, Math.max(0, jitter)), true);
    uView.setUint32 (16, Math.min(6, Math.max(1, octaves)), true);
    uView.setUint32 (20, MODE_INDEX[mode],        true);
    uView.setFloat32(24, time,                   true);
    uView.setFloat32(28, 0,                      true); // _pad

    const uniformBuf = this.device.createBuffer({
      label:  'natural-patterns-uniforms',
      size:   32,
      usage:  GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(uniformBuf, 0, uniformData);

    // Output texture
    const texture = this.device.createTexture({
      label:  `natural-pattern-${mode}-${width}x${height}`,
      size:   { width, height, depthOrArrayLayers: 1 },
      format: 'rgba8unorm',
      usage:  GPUTextureUsage.STORAGE_BINDING
            | GPUTextureUsage.TEXTURE_BINDING
            | GPUTextureUsage.COPY_SRC,
    });

    const bindGroup = this.device.createBindGroup({
      label:  'natural-patterns-bg',
      layout: this.bindGroupLayout!,
      entries: [
        { binding: 0, resource: { buffer: uniformBuf } },
        { binding: 1, resource: texture.createView() },
      ],
    });

    // Dispatch
    const enc = this.device.createCommandEncoder({ label: 'natural-patterns-enc' });
    const pass = enc.beginComputePass({ label: 'natural-patterns-pass' });
    pass.setPipeline(this.pipeline!);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
      Math.ceil(width  / 8),
      Math.ceil(height / 8),
    );
    pass.end();
    this.device.queue.submit([enc.finish()]);

    // Uniform buffer is safe to destroy after submit
    uniformBuf.destroy();

    return texture;
  }

  // ── convenience: CPU readback ─────────────────────────────────────────────

  /**
   * Read generated texture back to CPU as a {@link Uint8ClampedArray} RGBA
   * buffer (row-major, top-to-bottom).  Useful for canvas rendering or
   * server-side image export.
   */
  async readback(
    texture: GPUTexture,
    width: number,
    height: number,
  ): Promise<Uint8ClampedArray> {
    const bytesPerRow = Math.ceil((width * 4) / 256) * 256; // WebGPU alignment
    const stagingBuf  = this.device.createBuffer({
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
    const raw  = new Uint8Array(stagingBuf.getMappedRange());
    const rgba = new Uint8ClampedArray(width * height * 4);

    for (let row = 0; row < height; row++) {
      const src = raw.subarray(row * bytesPerRow, row * bytesPerRow + width * 4);
      rgba.set(src, row * width * 4);
    }

    stagingBuf.unmap();
    stagingBuf.destroy();
    return rgba;
  }

  // ── lifecycle ─────────────────────────────────────────────────────────────

  /** Free GPU pipeline resources.  Any previously generated textures remain
   *  valid until their own {@link GPUTexture.destroy} is called. */
  destroy(): void {
    // WebGPU pipelines / bind-group-layouts are GC'd; no explicit destroy API.
    this.pipeline        = null;
    this.bindGroupLayout = null;
  }
}

// ─── Re-exports for convenience ───────────────────────────────────────────────
export { COMPUTE_SHADER_SRC as NATURAL_PATTERNS_WGSL };
