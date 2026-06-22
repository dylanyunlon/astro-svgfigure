# === ParticleRenderer.ts ===
// ParticleRenderer.ts — WebGPU render pipeline (instanced quads)

const RENDER_SHADER = /* wgsl */`
struct Camera {
  pointSize  : f32,
  maxSpeed   : f32,
  domainScale: f32,
  _pad       : f32,
  scaleX     : f32,
  scaleY     : f32,
  offsetX    : f32,
  offsetY    : f32,
}
@group(0) @binding(0) var<uniform>      cam   : Camera;
@group(0) @binding(1) var<storage,read> posX  : array<f32>;
@group(0) @binding(2) var<storage,read> posY  : array<f32>;
@group(0) @binding(3) var<storage,read> velX  : array<f32>;
@group(0) @binding(4) var<storage,read> velY  : array<f32>;

struct VertOut {
  @builtin(position) pos   : vec4f,
  @location(0)       uv    : vec2f,
  @location(1)       speed : f32,
}

@vertex fn vs_main(
  @builtin(vertex_index)   vi : u32,
  @builtin(instance_index) ii : u32,
) -> VertOut {
  // Two-triangle quad; uv in [-1, 1]
  var quadUV = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0),
    vec2f(-1.0, -1.0), vec2f( 1.0,  1.0), vec2f(-1.0,  1.0),
  );
  let uv = quadUV[vi];

  let px = posX[ii];
  let py = posY[ii];
  let vx = velX[ii];
  let vy = velY[ii];
  let spd = length(vec2f(vx, vy));

  // half-size in NDC: pointSize is in domain units, convert via domainScale
  let halfNDC = (cam.pointSize / cam.domainScale) * 0.5;

  let ndcX = px * cam.scaleX + cam.offsetX + uv.x * halfNDC;
  let ndcY = py * cam.scaleY + cam.offsetY + uv.y * halfNDC;

  var out : VertOut;
  out.pos   = vec4f(ndcX, ndcY, 0.0, 1.0);
  out.uv    = uv;
  out.speed = spd;
  return out;
}

// Velocity → color: blue (0) → white (0.5) → red (1)
fn speedColor(t: f32) -> vec3f {
  let tc = clamp(t, 0.0, 1.0);
  let blue  = vec3f(0.1, 0.3, 1.0);
  let white = vec3f(1.0, 1.0, 1.0);
  let red   = vec3f(1.0, 0.15, 0.05);
  if (tc < 0.5) {
    return mix(blue, white, tc * 2.0);
  } else {
    return mix(white, red, (tc - 0.5) * 2.0);
  }
}

@fragment fn fs_main(in: VertOut) -> @location(0) vec4f {
  // Circular disc: discard corners of the quad
  if (length(in.uv) > 1.0) { discard; }

  // Soft radial falloff for a smooth particle look
  let rim   = 1.0 - smoothstep(0.7, 1.0, length(in.uv));
  let t     = clamp(in.speed / cam.maxSpeed, 0.0, 1.0);
  let color = speedColor(t);
  return vec4f(color * rim, rim * 0.9);
}
`;

// ─── GPU buffer set expected by render() ────────────────────────────────────
export interface GPUBufferSet {
  posX: GPUBuffer;
  posY: GPUBuffer;
  velX: GPUBuffer;
  velY: GPUBuffer;
}

// ─── Uniform layout (std140 / vec4 aligned, 8 × f32 = 32 bytes) ─────────────
//  offset 0 : pointSize
//  offset 4 : maxSpeed
//  offset 8 : domainScale
//  offset 12: _pad
//  offset 16: scaleX
//  offset 20: scaleY
//  offset 24: offsetX
//  offset 28: offsetY

export interface CameraUniforms {
  /** particle diameter in domain units */
  pointSize  : number;
  /** speed that maps to full red */
  maxSpeed   : number;
  /** domain width (used to convert pointSize to NDC) */
  domainScale: number;
  scaleX     : number;
  scaleY     : number;
  offsetX    : number;
  offsetY    : number;
}

export class ParticleRenderer {
  private readonly device : GPUDevice;
  private readonly format : GPUTextureFormat;

  private pipeline!     : GPURenderPipeline;
  private camBuffer!    : GPUBuffer;
  private bindGroup!    : GPUBindGroup;
  private bindGroupLayout!: GPUBindGroupLayout;

  // Cached buffer references so we can detect when they change
  private lastBufs: GPUBufferSet | null = null;

  // Default camera / uniform values
  private uniforms: CameraUniforms = {
    pointSize  : 0.02,
    maxSpeed   : 5.0,
    domainScale: 1.0,
    scaleX     : 2.0,
    scaleY     : 2.0,
    offsetX    : -1.0,
    offsetY    : -1.0,
  };

  constructor(device: GPUDevice, format: GPUTextureFormat) {
    this.device = device;
    this.format = format;
  }

  // ── Public: update uniform values before next render() call ──────────────
  setUniforms(u: Partial<CameraUniforms>): void {
    Object.assign(this.uniforms, u);
  }

  // ── Build (or rebuild) the pipeline ──────────────────────────────────────
  async buildPipeline(): Promise<void> {
    const d = this.device;

    const shaderModule = d.createShaderModule({
      label: "particle-shader",
      code : RENDER_SHADER,
    });

    this.bindGroupLayout = d.createBindGroupLayout({
      label  : "particle-bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX,
          buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.VERTEX,
          buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.VERTEX,
          buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.VERTEX,
          buffer: { type: "read-only-storage" } },
      ],
    });

    const pipelineLayout = d.createPipelineLayout({
      label             : "particle-pipeline-layout",
      bindGroupLayouts  : [this.bindGroupLayout],
    });

    this.pipeline = await d.createRenderPipelineAsync({
      label : "particle-pipeline",
      layout: pipelineLayout,
      vertex: {
        module    : shaderModule,
        entryPoint: "vs_main",
      },
      fragment: {
        module    : shaderModule,
        entryPoint: "fs_main",
        targets   : [
          {
            format: this.format,
            blend : {
              color: {
                srcFactor: "src-alpha",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
              alpha: {
                srcFactor: "one",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
            },
          },
        ],
      },
      primitive: {
        topology: "triangle-list",
      },
    });

    // Camera uniform buffer (32 bytes, 8 × f32)
    this.camBuffer = d.createBuffer({
      label: "particle-cam-uniform",
      size : 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Write initial uniform values
    this.uploadUniforms();
  }

  // ── Upload camera / uniform data to GPU ──────────────────────────────────
  private uploadUniforms(): void {
    const u   = this.uniforms;
    const arr = new Float32Array([
      u.pointSize, u.maxSpeed, u.domainScale, 0.0 /* _pad */,
      u.scaleX,    u.scaleY,   u.offsetX,     u.offsetY,
    ]);
    this.device.queue.writeBuffer(this.camBuffer, 0, arr);
  }

  // ── (Re)build bind group when particle buffers change ────────────────────
  private ensureBindGroup(bufs: GPUBufferSet): void {
    if (
      this.bindGroup &&
      this.lastBufs?.posX === bufs.posX &&
      this.lastBufs?.posY === bufs.posY &&
      this.lastBufs?.velX === bufs.velX &&
      this.lastBufs?.velY === bufs.velY
    ) {
      return; // already up-to-date
    }

    this.bindGroup = this.device.createBindGroup({
      label  : "particle-bg",
      layout : this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.camBuffer } },
        { binding: 1, resource: { buffer: bufs.posX } },
        { binding: 2, resource: { buffer: bufs.posY } },
        { binding: 3, resource: { buffer: bufs.velX } },
        { binding: 4, resource: { buffer: bufs.velY } },
      ],
    });
    this.lastBufs = bufs;
  }

  // ── Main render call ──────────────────────────────────────────────────────
  render(
    encoder : GPUCommandEncoder,
    view    : GPUTextureView,
    bufs    : GPUBufferSet,
    count   : number,
  ): void {
    if (!this.pipeline) {
      console.warn("ParticleRenderer: pipeline not built yet — call buildPipeline() first");
      return;
    }
    if (count === 0) return;

    this.uploadUniforms();
    this.ensureBindGroup(bufs);

    const pass = encoder.beginRenderPass({
      label          : "particle-pass",
      colorAttachments: [
        {
          view      : view,
          loadOp    : "load",   // preserve background / previous passes
          storeOp   : "store",
        },
      ],
    });

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(6, count); // 6 vertices per quad, `count` instances
    pass.end();
  }
}
