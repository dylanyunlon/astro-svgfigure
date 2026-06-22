# === ParticleRenderer.ts ===
// ParticleRenderer.ts — WebGPU render pipeline
// Two-mode rendering:
//   1. PARTICLES  — instanced quads, velocity-coloured discs (original)
//   2. METABALL   — screen-space fluid surface via metaball iso-surface
//                   extraction, normal-map reconstruction, and PBR shading

// ─────────────────────────────────────────────────────────────────────────────
// Pass A — Particle splat shader
// Each particle splats a smooth radial field value into the accumulation texture.
// We draw instanced quads exactly as before, but the fragment writes field
// strength (scalar) rather than colour.  The quad radius is enlarged so we
// capture the full metaball falloff kernel (2× smoothing length).
// ─────────────────────────────────────────────────────────────────────────────
const SPLAT_SHADER = /* wgsl */`
struct Camera {
  pointSize  : f32,   // particle diameter in domain units
  maxSpeed   : f32,
  domainScale: f32,
  splatRadius: f32,   // splat quad half-size in domain units (≥ 2×pointSize)
  scaleX     : f32,
  scaleY     : f32,
  offsetX    : f32,
  offsetY    : f32,
}

@group(0) @binding(0) var<uniform>      cam  : Camera;
@group(0) @binding(1) var<storage,read> posX : array<f32>;
@group(0) @binding(2) var<storage,read> posY : array<f32>;
@group(0) @binding(3) var<storage,read> velX : array<f32>;
@group(0) @binding(4) var<storage,read> velY : array<f32>;

struct SplatVert {
  @builtin(position) pos   : vec4f,
  @location(0)       uv    : vec2f,   // [-1, 1] quad local coords
  @location(1)       speed : f32,
}

@vertex fn vs_splat(
  @builtin(vertex_index)   vi : u32,
  @builtin(instance_index) ii : u32,
) -> SplatVert {
  var quadUV = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0),
    vec2f(-1.0, -1.0), vec2f( 1.0,  1.0), vec2f(-1.0,  1.0),
  );
  let uv = quadUV[vi];

  let px  = posX[ii];
  let py  = posY[ii];
  let spd = length(vec2f(velX[ii], velY[ii]));

  // Use splatRadius for the quad size (larger than a display point)
  let halfNDC = (cam.splatRadius / cam.domainScale) * 0.5;

  let ndcX = px * cam.scaleX + cam.offsetX + uv.x * halfNDC;
  let ndcY = py * cam.scaleY + cam.offsetY + uv.y * halfNDC;

  var out : SplatVert;
  out.pos   = vec4f(ndcX, ndcY, 0.0, 1.0);
  out.uv    = uv;
  out.speed = spd;
  return out;
}

// Metaball kernel: cubic falloff from Müller et al. 2003 (Poly6)
//   W(r,h) = max(0, (1 - (r/h)^2)^3)
// We pass r² in [0, 1] range (uv is already normalised to that square).
fn metaballKernel(r2: f32) -> f32 {
  let t = clamp(1.0 - r2, 0.0, 1.0);
  return t * t * t;
}

struct SplatFrag {
  @location(0) field : vec4f,   // rg = fieldStrength + speed packed into two channels
}

@fragment fn fs_splat(in: SplatVert) -> SplatFrag {
  // r² in [0, 1] where 1 = quad corner
  let r2     = dot(in.uv, in.uv);
  if (r2 > 1.0) { discard; }  // discard outside unit circle

  let field  = metaballKernel(r2);
  let t      = clamp(in.speed / cam.maxSpeed, 0.0, 1.0);

  // Pack: .r = field value, .g = speed·field (for weighted colour blend)
  return SplatFrag(vec4f(field, t * field, 0.0, field));
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Pass B — Metaball composite / surface-extraction fragment shader
//
// This full-screen pass reads the accumulated field texture and performs:
//   1. Iso-surface test  : field ≥ threshold → inside fluid
//   2. Screen-space normal: finite differences on the field gradient
//   3. PBR shading        : Cook-Torrance BRDF (dielectric, roughness 0.05)
//   4. Refraction hint    : subsurface tint based on depth proxy (field value)
//   5. Specular highlight : sharp directional + environment rim
//   6. Edge foam          : field near threshold → white specular streak
// ─────────────────────────────────────────────────────────────────────────────
const COMPOSITE_SHADER = /* wgsl */`
struct CompositeUniforms {
  threshold    : f32,   // iso-value (0.25–0.55 works well)
  normalStrength: f32,  // gradient amplification for normals
  roughness    : f32,   // PBR roughness (0 = mirror, 1 = diffuse)
  fresnelBase  : f32,   // F0 for Schlick approximation
  lightX       : f32,   // directional light (NDC-ish)
  lightY       : f32,
  lightZ       : f32,
  _pad         : f32,
  deepColor    : vec4f, // fluid deep colour  (linear sRGB + alpha)
  shallowColor : vec4f, // fluid surface colour
  foamColor    : vec4f, // foam / edge colour
}

@group(0) @binding(0) var<uniform> uni : CompositeUniforms;
@group(0) @binding(1) var fieldTex     : texture_2d<f32>;
@group(0) @binding(2) var fieldSampler : sampler;

struct CompVert {
  @builtin(position) pos : vec4f,
  @location(0)       uv  : vec2f,
}

@vertex fn vs_comp(
  @builtin(vertex_index) vi : u32,
) -> CompVert {
  // Full-screen triangle trick: three vertices cover the entire NDC quad
  let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
  let y = f32( vi         & 2u) * 2.0 - 1.0;
  var out : CompVert;
  out.pos = vec4f(x, y, 0.0, 1.0);
  out.uv  = vec2f(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
  return out;
}

// ── Helpers ────────────────────────────────────────────────────────────────

// Sample field value (r channel) with a pixel offset
fn sampleField(uv: vec2f, offset: vec2f) -> f32 {
  let dims = vec2f(textureDimensions(fieldTex, 0));
  return textureSampleLevel(fieldTex, fieldSampler, uv + offset / dims, 0.0).r;
}

// Sobel gradient — returns approximate dF/dx, dF/dy in screen space
fn sobelGrad(uv: vec2f) -> vec2f {
  let tl = sampleField(uv, vec2f(-1.0,  1.0));
  let tc = sampleField(uv, vec2f( 0.0,  1.0));
  let tr = sampleField(uv, vec2f( 1.0,  1.0));
  let ml = sampleField(uv, vec2f(-1.0,  0.0));
  let mr = sampleField(uv, vec2f( 1.0,  0.0));
  let bl = sampleField(uv, vec2f(-1.0, -1.0));
  let bc = sampleField(uv, vec2f( 0.0, -1.0));
  let br = sampleField(uv, vec2f( 1.0, -1.0));

  let gx = (tr + 2.0*mr + br) - (tl + 2.0*ml + bl);
  let gy = (tl + 2.0*tc + tr) - (bl + 2.0*bc + br);
  return vec2f(gx, gy);
}

// Schlick Fresnel approximation
fn fresnelSchlick(cosTheta: f32, f0: f32) -> f32 {
  let c1 = 1.0 - cosTheta;
  let c2 = c1 * c1;
  return f0 + (1.0 - f0) * c2 * c2 * c1;
}

// GGX / Trowbridge-Reitz NDF
fn distributionGGX(nDotH: f32, roughness: f32) -> f32 {
  let a  = roughness * roughness;
  let a2 = a * a;
  let d  = nDotH * nDotH * (a2 - 1.0) + 1.0;
  return a2 / (3.14159265 * d * d);
}

// Smith geometry term (simplified Schlick-GGX)
fn geometrySmith(nDotV: f32, nDotL: f32, roughness: f32) -> f32 {
  let k  = (roughness + 1.0) * (roughness + 1.0) / 8.0;
  let gv = nDotV / (nDotV * (1.0 - k) + k);
  let gl = nDotL / (nDotL * (1.0 - k) + k);
  return gv * gl;
}

// ── Fragment ───────────────────────────────────────────────────────────────

@fragment fn fs_comp(in: CompVert) -> @location(0) vec4f {
  let sample   = textureSampleLevel(fieldTex, fieldSampler, in.uv, 0.0);
  let field    = sample.r;           // accumulated metaball field
  let speedW   = sample.g;           // speed-weighted field (for colour)

  // Discard background — no fluid here
  if (field < uni.threshold * 0.3) { discard; }

  // ── 1. Inside / outside test ─────────────────────────────────────────────
  let inside = field >= uni.threshold;

  // Smooth transition weight near the surface
  let surfaceT = smoothstep(uni.threshold * 0.5, uni.threshold, field);

  // ── 2. Screen-space normal ───────────────────────────────────────────────
  let grad2D   = sobelGrad(in.uv) * uni.normalStrength;
  // Treat the field gradient in 2-D screen space, synthesise a Z component
  // using the assumption that the normal mostly points toward the viewer.
  let gradLen  = length(grad2D);
  let nXY      = -normalize(grad2D + vec2f(0.0001, 0.0001));
  let nZ       = sqrt(max(0.0, 1.0 - dot(nXY, nXY)));
  let normal   = normalize(vec3f(nXY * 0.6, nZ));   // attenuate lateral bulge

  // ── 3. Lighting setup ────────────────────────────────────────────────────
  let L = normalize(vec3f(uni.lightX, uni.lightY, uni.lightZ));
  let V = vec3f(0.0, 0.0, 1.0);   // viewer always looks down Z in screen space
  let H = normalize(L + V);

  let nDotL = max(dot(normal, L), 0.0);
  let nDotV = max(dot(normal, V), 0.001);
  let nDotH = max(dot(normal, H), 0.0);
  let hDotV = max(dot(H,      V), 0.0);

  // ── 4. Speed-weighted colour ─────────────────────────────────────────────
  // speedW/field gives average normalised speed in this texel
  let avgSpeed  = select(0.0, speedW / max(field, 0.001), field > 0.001);
  let baseColor = mix(uni.deepColor.rgb, uni.shallowColor.rgb, surfaceT)
                + vec3f(0.05, 0.15, 0.35) * avgSpeed; // velocity tint (blue shift)
  let baseColor4 = vec4f(clamp(baseColor, vec3f(0.0), vec3f(1.0)), 1.0);

  // ── 5. PBR Cook-Torrance specular ────────────────────────────────────────
  let D   = distributionGGX(nDotH, uni.roughness);
  let G   = geometrySmith(nDotV, nDotL, uni.roughness);
  let F   = fresnelSchlick(hDotV, uni.fresnelBase);
  let specular = (D * G * F) / max(4.0 * nDotV * nDotL, 0.001);

  // ── 6. Diffuse + specular + ambient ──────────────────────────────────────
  let kD      = (1.0 - F) * (1.0 - uni.fresnelBase); // energy conservation
  let diffuse = kD * baseColor4.rgb * max(nDotL, 0.0);
  let spec    = vec3f(specular) * nDotL;
  let ambient = baseColor4.rgb * 0.12;
  var color   = diffuse + spec + ambient;

  // ── 7. Rim / environment reflection ──────────────────────────────────────
  let rim     = pow(1.0 - nDotV, 3.0);
  let rimCol  = vec3f(0.6, 0.85, 1.0);  // sky-blue rim
  color      += rim * rimCol * 0.35;

  // ── 8. Foam at the iso-surface edge ──────────────────────────────────────
  let edgeDist = abs(field - uni.threshold);
  let foamT    = smoothstep(0.08, 0.0, edgeDist) * surfaceT;
  color        = mix(color, uni.foamColor.rgb, foamT * 0.7);

  // ── 9. Alpha: fully opaque inside, fade at boundary ──────────────────────
  let alpha = smoothstep(uni.threshold * 0.4, uni.threshold * 1.1, field);

  return vec4f(color, alpha * baseColor4.a);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Original particle pass (kept for PARTICLES render mode)
// ─────────────────────────────────────────────────────────────────────────────
const PARTICLE_SHADER = /* wgsl */`
struct Camera {
  pointSize  : f32,
  maxSpeed   : f32,
  domainScale: f32,
  splatRadius: f32,
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
  let halfNDC = (cam.pointSize / cam.domainScale) * 0.5;
  let ndcX = px * cam.scaleX + cam.offsetX + uv.x * halfNDC;
  let ndcY = py * cam.scaleY + cam.offsetY + uv.y * halfNDC;
  var out : VertOut;
  out.pos   = vec4f(ndcX, ndcY, 0.0, 1.0);
  out.uv    = uv;
  out.speed = spd;
  return out;
}

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
  if (length(in.uv) > 1.0) { discard; }
  let rim   = 1.0 - smoothstep(0.7, 1.0, length(in.uv));
  let t     = clamp(in.speed / cam.maxSpeed, 0.0, 1.0);
  let color = speedColor(t);
  return vec4f(color * rim, rim * 0.9);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

export interface GPUBufferSet {
  posX: GPUBuffer;
  posY: GPUBuffer;
  velX: GPUBuffer;
  velY: GPUBuffer;
}

export interface CameraUniforms {
  /** particle diameter in domain units */
  pointSize   : number;
  /** speed that maps to full red */
  maxSpeed    : number;
  /** domain width (used to convert pointSize to NDC) */
  domainScale : number;
  /** metaball splat quad radius in domain units (default 3× pointSize) */
  splatRadius?: number;
  scaleX      : number;
  scaleY      : number;
  offsetX     : number;
  offsetY     : number;
}

export interface MetaballUniforms {
  /** iso-surface threshold — lower = thicker fluid surface */
  threshold     : number;
  /** Sobel gradient amplification for normals */
  normalStrength: number;
  /** PBR roughness 0 (mirror) … 1 (diffuse) */
  roughness     : number;
  /** Schlick F0 (water ≈ 0.02, glass ≈ 0.04) */
  fresnelBase   : number;
  /** Directional light vector (screen-space) */
  light         : [number, number, number];
  /** Deep-water RGBA [0..1] */
  deepColor     : [number, number, number, number];
  /** Surface highlight RGBA */
  shallowColor  : [number, number, number, number];
  /** Foam / edge streak RGBA */
  foamColor     : [number, number, number, number];
}

export type RenderMode = 'PARTICLES' | 'METABALL';

// ─────────────────────────────────────────────────────────────────────────────
// ParticleRenderer
// ─────────────────────────────────────────────────────────────────────────────

export class ParticleRenderer {
  private readonly device : GPUDevice;
  private readonly format : GPUTextureFormat;

  // ── Particle (original) pipeline ──────────────────────────────────────────
  private particlePipeline  !: GPURenderPipeline;
  private particleBGL       !: GPUBindGroupLayout;
  private camBuffer         !: GPUBuffer;
  private particleBG        !: GPUBindGroup;
  private lastBufs: GPUBufferSet | null = null;

  // ── Metaball pipelines ─────────────────────────────────────────────────────
  private splatPipeline     !: GPURenderPipeline;   // Pass A
  private compositePipeline !: GPURenderPipeline;   // Pass B
  private splatBGL          !: GPUBindGroupLayout;
  private compositeBGL      !: GPUBindGroupLayout;
  private splatBG           !: GPUBindGroup;
  private lastSplatBufs: GPUBufferSet | null = null;

  // Field accumulation render target (off-screen, float16 for precision)
  private fieldTexture  !: GPUTexture;
  private fieldView     !: GPUTextureView;
  private fieldSampler  !: GPUSampler;
  private fieldWidth    = 0;
  private fieldHeight   = 0;

  // Composite uniform buffer
  private compUniBuf    !: GPUBuffer;
  private compositeBG   !: GPUBindGroup;

  // Current render mode
  mode: RenderMode = 'METABALL';

  // Uniform values
  private uniforms: CameraUniforms = {
    pointSize  : 0.02,
    maxSpeed   : 5.0,
    domainScale: 1.0,
    splatRadius: 0.06,
    scaleX     : 2.0,
    scaleY     : 2.0,
    offsetX    : -1.0,
    offsetY    : -1.0,
  };

  private metaUniforms: MetaballUniforms = {
    threshold     : 0.35,
    normalStrength: 6.0,
    roughness     : 0.06,
    fresnelBase   : 0.04,
    light         : [0.4, 0.7, 1.0],
    deepColor     : [0.04, 0.18, 0.55, 1.0],
    shallowColor  : [0.18, 0.55, 0.90, 1.0],
    foamColor     : [0.92, 0.97, 1.00, 1.0],
  };

  constructor(device: GPUDevice, format: GPUTextureFormat) {
    this.device = device;
    this.format = format;
  }

  // ── Public: update uniform values ─────────────────────────────────────────
  setUniforms(u: Partial<CameraUniforms>): void {
    Object.assign(this.uniforms, u);
  }

  setMetaballUniforms(u: Partial<MetaballUniforms>): void {
    Object.assign(this.metaUniforms, u);
  }

  setMode(m: RenderMode): void {
    this.mode = m;
  }

  // ── Build (or rebuild) both pipelines ─────────────────────────────────────
  async buildPipeline(): Promise<void> {
    await Promise.all([
      this._buildParticlePipeline(),
      this._buildMetaballPipelines(),
    ]);
  }

  // ─────────────────────────────────────────────────────────────────────────
  private async _buildParticlePipeline(): Promise<void> {
    const d = this.device;

    const sm = d.createShaderModule({ label: "particle-shader", code: PARTICLE_SHADER });

    this.particleBGL = d.createBindGroupLayout({
      label  : "particle-bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      ],
    });

    this.particlePipeline = await d.createRenderPipelineAsync({
      label : "particle-pipeline",
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.particleBGL] }),
      vertex  : { module: sm, entryPoint: "vs_main" },
      fragment: {
        module : sm,
        entryPoint: "fs_main",
        targets: [{
          format: this.format,
          blend : {
            color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
            alpha: { srcFactor: "one",       dstFactor: "one-minus-src-alpha", operation: "add" },
          },
        }],
      },
      primitive: { topology: "triangle-list" },
    });

    // Shared camera uniform buffer (32 bytes, 8 × f32)
    this.camBuffer = d.createBuffer({
      label: "cam-uniform",
      size : 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._uploadCameraUniforms();
  }

  // ─────────────────────────────────────────────────────────────────────────
  private async _buildMetaballPipelines(): Promise<void> {
    const d = this.device;

    // ── Pass A — splat ──────────────────────────────────────────────────────
    const splatSM = d.createShaderModule({ label: "splat-shader", code: SPLAT_SHADER });

    this.splatBGL = d.createBindGroupLayout({
      label  : "splat-bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      ],
    });

    this.splatPipeline = await d.createRenderPipelineAsync({
      label : "splat-pipeline",
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.splatBGL] }),
      vertex  : { module: splatSM, entryPoint: "vs_splat" },
      fragment: {
        module    : splatSM,
        entryPoint: "fs_splat",
        targets   : [{
          // Additive blending into the field accumulation texture
          format: "rgba16float",
          blend : {
            color: { srcFactor: "one", dstFactor: "one", operation: "add" },
            alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
          },
        }],
      },
      primitive: { topology: "triangle-list" },
    });

    // ── Pass B — composite ──────────────────────────────────────────────────
    const compSM = d.createShaderModule({ label: "composite-shader", code: COMPOSITE_SHADER });

    this.compositeBGL = d.createBindGroupLayout({
      label  : "composite-bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "float", viewDimension: "2d" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT,
          sampler: { type: "filtering" } },
      ],
    });

    this.compositePipeline = await d.createRenderPipelineAsync({
      label : "composite-pipeline",
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.compositeBGL] }),
      vertex  : { module: compSM, entryPoint: "vs_comp" },
      fragment: {
        module    : compSM,
        entryPoint: "fs_comp",
        targets   : [{
          format: this.format,
          blend : {
            color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
            alpha: { srcFactor: "one",       dstFactor: "one-minus-src-alpha", operation: "add" },
          },
        }],
      },
      primitive: { topology: "triangle-list" },
    });

    // Composite uniform buffer (80 bytes: 4×f32 scalars + light(3f+pad) + 3×vec4f)
    this.compUniBuf = d.createBuffer({
      label: "comp-uniform",
      size : 96,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Sampler for the field texture
    this.fieldSampler = d.createSampler({
      label     : "field-sampler",
      magFilter : "linear",
      minFilter : "linear",
    });

    this._uploadMetaballUniforms();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Ensure the field accumulation texture matches the swap-chain dimensions
  // ─────────────────────────────────────────────────────────────────────────
  private _ensureFieldTexture(width: number, height: number): void {
    if (this.fieldTexture && this.fieldWidth === width && this.fieldHeight === height) return;

    this.fieldTexture?.destroy();

    this.fieldTexture = this.device.createTexture({
      label  : "field-accum",
      size   : { width, height },
      format : "rgba16float",
      usage  : GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.fieldView    = this.fieldTexture.createView({ label: "field-view" });
    this.fieldWidth   = width;
    this.fieldHeight  = height;

    // Rebuild composite bind group whenever the texture changes
    if (this.compositePipeline && this.compUniBuf) {
      this._rebuildCompositeBG();
    }
  }

  private _rebuildCompositeBG(): void {
    this.compositeBG = this.device.createBindGroup({
      label  : "composite-bg",
      layout : this.compositeBGL,
      entries: [
        { binding: 0, resource: { buffer: this.compUniBuf } },
        { binding: 1, resource: this.fieldView },
        { binding: 2, resource: this.fieldSampler },
      ],
    });
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Uniform uploads
  // ─────────────────────────────────────────────────────────────────────────
  private _uploadCameraUniforms(): void {
    const u = this.uniforms;
    this.device.queue.writeBuffer(this.camBuffer, 0, new Float32Array([
      u.pointSize,
      u.maxSpeed,
      u.domainScale,
      u.splatRadius ?? (u.pointSize * 3),
      u.scaleX, u.scaleY, u.offsetX, u.offsetY,
    ]));
  }

  private _uploadMetaballUniforms(): void {
    const m = this.metaUniforms;
    // Layout (96 bytes = 24 × f32):
    //  [0]  threshold       [1]  normalStrength  [2]  roughness   [3]  fresnelBase
    //  [4]  lightX          [5]  lightY          [6]  lightZ      [7]  _pad
    //  [8..11]  deepColor   (vec4f)
    //  [12..15] shallowColor(vec4f)
    //  [16..19] foamColor   (vec4f)
    const arr = new Float32Array(24);
    arr[0]  = m.threshold;
    arr[1]  = m.normalStrength;
    arr[2]  = m.roughness;
    arr[3]  = m.fresnelBase;
    arr[4]  = m.light[0];
    arr[5]  = m.light[1];
    arr[6]  = m.light[2];
    arr[7]  = 0.0;
    arr.set(m.deepColor,    8);
    arr.set(m.shallowColor, 12);
    arr.set(m.foamColor,    16);
    this.device.queue.writeBuffer(this.compUniBuf, 0, arr);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Bind group management
  // ─────────────────────────────────────────────────────────────────────────
  private _ensureParticleBG(bufs: GPUBufferSet): void {
    if (
      this.particleBG &&
      this.lastBufs?.posX === bufs.posX &&
      this.lastBufs?.posY === bufs.posY
    ) { return; }

    this.particleBG = this.device.createBindGroup({
      label  : "particle-bg",
      layout : this.particleBGL,
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

  private _ensureSplatBG(bufs: GPUBufferSet): void {
    if (
      this.splatBG &&
      this.lastSplatBufs?.posX === bufs.posX &&
      this.lastSplatBufs?.posY === bufs.posY
    ) { return; }

    this.splatBG = this.device.createBindGroup({
      label  : "splat-bg",
      layout : this.splatBGL,
      entries: [
        { binding: 0, resource: { buffer: this.camBuffer } },
        { binding: 1, resource: { buffer: bufs.posX } },
        { binding: 2, resource: { buffer: bufs.posY } },
        { binding: 3, resource: { buffer: bufs.velX } },
        { binding: 4, resource: { buffer: bufs.velY } },
      ],
    });
    this.lastSplatBufs = bufs;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Main render entry point
  // ─────────────────────────────────────────────────────────────────────────
  render(
    encoder : GPUCommandEncoder,
    view    : GPUTextureView,
    bufs    : GPUBufferSet,
    count   : number,
    /** Canvas pixel dimensions — required for metaball mode */
    canvasWidth  = 512,
    canvasHeight = 512,
  ): void {
    if (!this.particlePipeline) {
      console.warn("ParticleRenderer: pipeline not built yet — call buildPipeline() first");
      return;
    }
    if (count === 0) return;

    this._uploadCameraUniforms();

    if (this.mode === 'METABALL') {
      this._renderMetaball(encoder, view, bufs, count, canvasWidth, canvasHeight);
    } else {
      this._renderParticles(encoder, view, bufs, count);
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  private _renderParticles(
    encoder: GPUCommandEncoder,
    view   : GPUTextureView,
    bufs   : GPUBufferSet,
    count  : number,
  ): void {
    this._ensureParticleBG(bufs);

    const pass = encoder.beginRenderPass({
      label           : "particle-pass",
      colorAttachments: [{
        view  : view,
        loadOp: "load",
        storeOp: "store",
      }],
    });
    pass.setPipeline(this.particlePipeline);
    pass.setBindGroup(0, this.particleBG);
    pass.draw(6, count);
    pass.end();
  }

  // ─────────────────────────────────────────────────────────────────────────
  private _renderMetaball(
    encoder     : GPUCommandEncoder,
    view        : GPUTextureView,
    bufs        : GPUBufferSet,
    count       : number,
    canvasWidth : number,
    canvasHeight: number,
  ): void {
    this._ensureFieldTexture(canvasWidth, canvasHeight);
    this._ensureSplatBG(bufs);
    this._uploadMetaballUniforms();

    // Ensure composite bind group is fresh
    if (!this.compositeBG) { this._rebuildCompositeBG(); }

    // ── Pass A: splat particles into field texture ──────────────────────────
    {
      const splatPass = encoder.beginRenderPass({
        label           : "metaball-splat-pass",
        colorAttachments: [{
          view    : this.fieldView,
          loadOp  : "clear",          // clear field each frame
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          storeOp : "store",
        }],
      });
      splatPass.setPipeline(this.splatPipeline);
      splatPass.setBindGroup(0, this.splatBG);
      splatPass.draw(6, count);       // 6 verts × count instances
      splatPass.end();
    }

    // ── Pass B: full-screen composite / iso-surface extraction ─────────────
    {
      const compPass = encoder.beginRenderPass({
        label           : "metaball-composite-pass",
        colorAttachments: [{
          view    : view,
          loadOp  : "load",           // preserve background
          storeOp : "store",
        }],
      });
      compPass.setPipeline(this.compositePipeline);
      compPass.setBindGroup(0, this.compositeBG);
      compPass.draw(3);               // full-screen triangle (3 vertices)
      compPass.end();
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  /** Release all GPU resources. */
  destroy(): void {
    this.camBuffer?.destroy();
    this.compUniBuf?.destroy();
    this.fieldTexture?.destroy();
  }
}
