/**
 * gpu-culling.ts — M793: GPU Frustum + Occlusion Culling
 * ─────────────────────────────────────────────────────────────────────────────
 * WebGPU compute shader pipeline that determines per-cell and per-particle-
 * group visibility on the GPU, writing results into an indirect draw buffer
 * so the CPU never touches per-object visibility logic.
 *
 * Two-phase culling
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   Phase 1 — FRUSTUM CULLING
 *     Each compute invocation tests one AABB (cell bounding box or particle
 *     group AABB) against the six planes of the view-projection frustum.
 *     Objects fully outside any plane are marked invisible.  The test uses
 *     the "positive vertex" method (Gribb & Hartmann 2001) for tight AABB ↔
 *     half-plane classification.
 *
 *   Phase 2 — HIERARCHICAL Z-BUFFER OCCLUSION CULLING (HiZ)
 *     Surviving objects from Phase 1 are tested against a hierarchical depth
 *     pyramid (HiZ) built from the previous frame's depth buffer.  The AABB
 *     is projected to screen-space to find the tightest mip level whose
 *     texel covers the projected extent.  If the object's nearest depth is
 *     behind the conservative max-depth in that mip texel, the object is
 *     occluded and marked invisible.
 *
 *     The HiZ pyramid is built via a separate compute pass
 *     (`buildHiZPyramid`) that down-samples depth using a max-filter,
 *     producing a power-of-two mip chain stored in a single texture.
 *
 * Indirect draw compaction
 * ─────────────────────────────────────────────────────────────────────────────
 * A third compute pass (`compactIndirectArgs`) scans the visibility bit-mask
 * and appends visible object indices into a compacted index buffer, then
 * writes the final `DrawIndirectArgs.instanceCount` (for cells) or
 * `DrawIndexedIndirectArgs.instanceCount` (for particle groups) so the GPU
 * draw call automatically skips culled geometry.
 *
 * Performance characteristics
 * ─────────────────────────────────────────────────────────────────────────────
 *   • Zero CPU readback — all culling decisions and indirect counts stay on
 *     the GPU.  The CPU dispatches a fixed set of passes and never touches
 *     per-object data.
 *   • Single atomic counter — the compaction pass uses one `atomicAdd` to
 *     build the output index, avoiding prefix-sum overhead for the typical
 *     cell counts (< 1024).
 *   • HiZ reuse — the previous frame's depth is repurposed; no extra
 *     depth-pre-pass is needed.
 *
 * Integration points
 * ─────────────────────────────────────────────────────────────────────────────
 *   • instanced-cell-renderer.ts  — supplies CellBBox AABBs; reads back the
 *     compacted index buffer + indirect draw args for the single instanced
 *     draw call.
 *   • particle-instancing.ts      — supplies per-group AABBs; reads the
 *     particle-group indirect draw args.
 *   • adaptive-lod.ts             — LODCamera supplies view/projection data
 *     and viewport dimensions for frustum plane extraction.
 *   • render-compositor.ts        — provides the previous frame's depth
 *     texture for HiZ pyramid construction.
 *   • performance-budget.ts       — when tier < HIGH, occlusion culling is
 *     skipped (frustum-only mode) to save the HiZ build cost.
 *
 * Upstream references
 * ─────────────────────────────────────────────────────────────────────────────
 *   src/lib/sph/instanced-cell-renderer.ts  — CellBBox, FLOATS_PER_CELL
 *   src/lib/sph/particle-instancing.ts      — INSTANCE_STRIDE
 *   src/lib/sph/adaptive-lod.ts             — LODCamera
 *   src/lib/sph/performance-budget.ts       — Tier, getGlobalBudget
 *   src/lib/sph/render-compositor.ts        — depth texture, SceneMatrices
 *   src/lib/sph/gpu-particle-sort.ts        — compute shader patterns
 *   src/lib/sph/types.ts                    — MAX_PARTICLES, WORKGROUP_SIZE
 *
 * Research: xiaodi #M793 — cell-pubsub-loop
 */




// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/** Maximum number of cullable objects (cells + particle groups). */



import type { CellBBox }       from './instanced-cell-renderer';
import type { LODCamera }      from './adaptive-lod';
import type { Tier }            from './performance-budget';
import { getGlobalBudget }     from './performance-budget';
import { WORKGROUP_SIZE }      from './types';

const MAX_OBJECTS      = 2048;

/** Workgroup size for culling + compaction shaders. */
const WG               = WORKGROUP_SIZE;   // 256

/** Bytes per DrawIndirectArgs (vertexCount, instanceCount, firstVertex, firstInstance). */
const DRAW_INDIRECT_SIZE = 4 * 4;  // 16 bytes

/** HiZ pyramid max mip levels (covers up to 4096 × 4096 depth). */
const MAX_MIP_LEVELS   = 12;

/** Floats per AABB in the object buffer: minX, minY, maxX, maxY, nearZ. */
const AABB_STRIDE      = 8;  // padded to 8 for alignment (5 used + 3 pad)

// ─────────────────────────────────────────────────────────────────────────────
// Frustum plane extraction (CPU-side, from viewProj matrix)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A frustum plane in Hessian normal form: ax + by + cz + d = 0.
 * Points with dot(n, p) + d ≥ 0 are on the inside (visible) half-space.
 */
export interface FrustumPlane {
  a: number; b: number; c: number; d: number;
}

/**
 * Extract the six frustum planes from a column-major 4×4 view-projection
 * matrix using the Gribb–Hartmann method.
 *
 * Plane order: Left, Right, Bottom, Top, Near, Far.
 * Each plane is normalised so (a,b,c) is unit-length.
 */
export function extractFrustumPlanes(vp: Float32Array): FrustumPlane[] {
  //  column-major indexing: M[row][col] = vp[col*4 + row]
  const m = (r: number, c: number) => vp[c * 4 + r];

  const raw: [number, number, number, number][] = [
    // Left:    row3 + row0
    [m(3,0)+m(0,0), m(3,1)+m(0,1), m(3,2)+m(0,2), m(3,3)+m(0,3)],
    // Right:   row3 - row0
    [m(3,0)-m(0,0), m(3,1)-m(0,1), m(3,2)-m(0,2), m(3,3)-m(0,3)],
    // Bottom:  row3 + row1
    [m(3,0)+m(1,0), m(3,1)+m(1,1), m(3,2)+m(1,2), m(3,3)+m(1,3)],
    // Top:     row3 - row1
    [m(3,0)-m(1,0), m(3,1)-m(1,1), m(3,2)-m(1,2), m(3,3)-m(1,3)],
    // Near:    row3 + row2
    [m(3,0)+m(2,0), m(3,1)+m(2,1), m(3,2)+m(2,2), m(3,3)+m(2,3)],
    // Far:     row3 - row2
    [m(3,0)-m(2,0), m(3,1)-m(2,1), m(3,2)-m(2,2), m(3,3)-m(2,3)],
  ];

  return raw.map(([a, b, c, d]) => {
    const len = Math.sqrt(a * a + b * b + c * c) || 1;
    return { a: a / len, b: b / len, c: c / len, d: d / len };
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// ViewProjection helpers (2-D → pseudo-3-D for frustum math)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build a column-major 4×4 orthographic view-projection matrix from a
 * LODCamera.  The 2-D SPH world is treated as a thin Z-slab [−1, 1].
 */
export function buildViewProjFromCamera(cam: LODCamera): Float32Array {
  const halfW = (cam.viewportW * 0.5) / cam.zoom;
  const halfH = (cam.viewportH * 0.5) / cam.zoom;

  const l = cam.x - halfW;
  const r = cam.x + halfW;
  const b = cam.y - halfH;
  const t = cam.y + halfH;
  const n = -1.0;
  const f =  1.0;

  // Column-major orthographic projection
  const out = new Float32Array(16);
  out[0]  =  2.0 / (r - l);
  out[5]  =  2.0 / (t - b);
  out[10] = -2.0 / (f - n);
  out[12] = -(r + l) / (r - l);
  out[13] = -(t + b) / (t - b);
  out[14] = -(f + n) / (f - n);
  out[15] =  1.0;

  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// AABB descriptor (CPU upload format)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Axis-aligned bounding box for a cullable object.
 * `nearZ` is the closest depth value (for occlusion testing).
 * In the 2-D SPH world nearZ is typically 0.
 */
export interface CullAABB {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  nearZ: number;
}

/** Convert a CellBBox to a CullAABB. */
export function cellBBoxToAABB(b: CellBBox): CullAABB {
  return {
    minX:  b.x,
    minY:  b.y,
    maxX:  b.x + b.w,
    maxY:  b.y + b.h,
    nearZ: 0.0,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Frustum + Occlusion culling compute shader
// ─────────────────────────────────────────────────────────────────────────────

const CULL_SHADER_SOURCE = /* wgsl */ `

// ── Uniforms ─────────────────────────────────────────────────────────────────

struct CullUniforms {
  // Frustum planes (6 × vec4<f32>) — each (a,b,c,d)
  planes : array<vec4<f32>, 6>,
  // View-projection matrix (column-major)
  viewProj : mat4x4<f32>,
  // Viewport dimensions
  viewportW : f32,
  viewportH : f32,
  // Total object count this dispatch
  objectCount : u32,
  // HiZ mip count (0 = frustum-only, skip occlusion)
  hizMipCount : u32,
}

// ── AABB storage ─────────────────────────────────────────────────────────────

struct AABB {
  minXY   : vec2<f32>,
  maxXY   : vec2<f32>,
  nearZ   : f32,
  _pad0   : f32,
  _pad1   : f32,
  _pad2   : f32,
}

@group(0) @binding(0) var<uniform>              u : CullUniforms;
@group(0) @binding(1) var<storage, read>        aabbs : array<AABB>;
@group(0) @binding(2) var<storage, read_write>  visibility : array<u32>;

// HiZ texture (bound only when occlusion culling is active)
@group(1) @binding(0) var hizTexture : texture_2d<f32>;
@group(1) @binding(1) var hizSampler : sampler;

// ── Frustum test ─────────────────────────────────────────────────────────────

fn frustumTestAABB(aabb: AABB) -> bool {
  // For each of the 6 planes, find the "positive vertex" (the AABB corner
  // most in the direction of the plane normal).  If it's behind the plane
  // the entire AABB is outside.
  for (var i = 0u; i < 6u; i = i + 1u) {
    let plane = u.planes[i];
    let n = plane.xyz;

    // Select positive vertex components
    var pv = aabb.minXY;
    if (n.x >= 0.0) { pv.x = aabb.maxXY.x; }
    if (n.y >= 0.0) { pv.y = aabb.maxXY.y; }

    // Z component: use nearZ for positive normal, 0.0 for negative
    let pvZ = select(0.0, aabb.nearZ, n.z >= 0.0);

    let dist = dot(n, vec3<f32>(pv, pvZ)) + plane.w;
    if (dist < 0.0) {
      return false;  // fully outside this plane
    }
  }
  return true;
}

// ── Occlusion test (HiZ) ────────────────────────────────────────────────────

fn occlusionTestAABB(aabb: AABB) -> bool {
  // Project all 4 corners of the 2-D AABB to clip space
  let corners = array<vec2<f32>, 4>(
    aabb.minXY,
    vec2<f32>(aabb.maxXY.x, aabb.minXY.y),
    aabb.maxXY,
    vec2<f32>(aabb.minXY.x, aabb.maxXY.y),
  );

  var screenMin = vec2<f32>( 1e10,  1e10);
  var screenMax = vec2<f32>(-1e10, -1e10);
  var nearestZ  = 1.0;

  for (var i = 0u; i < 4u; i = i + 1u) {
    let clip = u.viewProj * vec4<f32>(corners[i], aabb.nearZ, 1.0);
    // Perspective divide (orthographic: w ≈ 1, but handle general case)
    let w = max(clip.w, 0.0001);
    let ndc = clip.xyz / w;

    // NDC → [0, 1] UV space
    let uv = ndc.xy * 0.5 + 0.5;
    screenMin = min(screenMin, uv);
    screenMax = max(screenMax, uv);
    nearestZ  = min(nearestZ, ndc.z * 0.5 + 0.5);
  }

  // Clamp to viewport
  screenMin = clamp(screenMin, vec2<f32>(0.0), vec2<f32>(1.0));
  screenMax = clamp(screenMax, vec2<f32>(0.0), vec2<f32>(1.0));

  // Determine the HiZ mip level: pick the level where the projected rect
  // covers ≤ 2×2 texels → conservative single-texel lookup.
  let extentPx = (screenMax - screenMin) * vec2<f32>(u.viewportW, u.viewportH);
  let maxExtent = max(extentPx.x, extentPx.y);
  let mipF = max(ceil(log2(maxExtent)), 0.0);
  let mip = min(u32(mipF), u.hizMipCount - 1u);

  // Sample the max-depth from the HiZ texture at the chosen mip
  let center = (screenMin + screenMax) * 0.5;
  let hizDepth = textureSampleLevel(hizTexture, hizSampler, center, f32(mip)).r;

  // Object is occluded if its nearest depth is farther than the HiZ depth
  return nearestZ <= hizDepth;
}

// ── Main compute entry ───────────────────────────────────────────────────────

@compute @workgroup_size(${WG})
fn cs_cull(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= u.objectCount) {
    return;
  }

  let aabb = aabbs[idx];

  // Phase 1: frustum test
  var visible = frustumTestAABB(aabb);

  // Phase 2: occlusion test (only if frustum-visible and HiZ is available)
  if (visible && u.hizMipCount > 0u) {
    visible = occlusionTestAABB(aabb);
  }

  visibility[idx] = select(0u, 1u, visible);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — Indirect draw compaction compute shader
// ─────────────────────────────────────────────────────────────────────────────

const COMPACT_SHADER_SOURCE = /* wgsl */ `

struct CompactUniforms {
  objectCount        : u32,
  baseVertexCount    : u32,  // e.g. 6 for particle quads, or cell index count
  baseFirstVertex    : u32,
  baseFirstInstance  : u32,
}

// DrawIndirectArgs: vertexCount, instanceCount, firstVertex, firstInstance
struct DrawIndirectArgs {
  vertexCount    : u32,
  instanceCount  : atomic<u32>,
  firstVertex    : u32,
  firstInstance  : u32,
}

@group(0) @binding(0) var<uniform>              cu : CompactUniforms;
@group(0) @binding(1) var<storage, read>        visibility : array<u32>;
@group(0) @binding(2) var<storage, read_write>  compactedIndices : array<u32>;
@group(0) @binding(3) var<storage, read_write>  drawArgs : DrawIndirectArgs;

@compute @workgroup_size(${WG})
fn cs_compact(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= cu.objectCount) {
    return;
  }

  if (visibility[idx] == 1u) {
    // Atomically allocate a slot in the compacted output
    let slot = atomicAdd(&drawArgs.instanceCount, 1u);
    compactedIndices[slot] = idx;
  }
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — HiZ pyramid builder (max-depth downsample)
// ─────────────────────────────────────────────────────────────────────────────

const HIZ_BUILD_SHADER_SOURCE = /* wgsl */ `

struct HiZUniforms {
  srcMipW : u32,
  srcMipH : u32,
}

@group(0) @binding(0) var<uniform>      hu : HiZUniforms;
@group(0) @binding(1) var srcMip : texture_2d<f32>;
@group(0) @binding(2) var dstMip : texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn cs_hiz_build(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dstCoord = vec2<i32>(gid.xy);
  let dstSize  = vec2<i32>(textureDimensions(dstMip));

  if (dstCoord.x >= dstSize.x || dstCoord.y >= dstSize.y) {
    return;
  }

  // Fetch 2×2 block from the source mip and take the maximum depth
  let srcBase = dstCoord * 2;
  let d00 = textureLoad(srcMip, srcBase + vec2<i32>(0, 0), 0).r;
  let d10 = textureLoad(srcMip, srcBase + vec2<i32>(1, 0), 0).r;
  let d01 = textureLoad(srcMip, srcBase + vec2<i32>(0, 1), 0).r;
  let d11 = textureLoad(srcMip, srcBase + vec2<i32>(1, 1), 0).r;

  let maxDepth = max(max(d00, d10), max(d01, d11));

  textureStorageBarrier();
  textureStore(dstMip, dstCoord, vec4<f32>(maxDepth, 0.0, 0.0, 1.0));
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// CullUniforms CPU layout (matches WGSL struct)
// ─────────────────────────────────────────────────────────────────────────────

/** Byte size of the CullUniforms uniform buffer. */
const CULL_UNIFORM_BYTES =
    6 * 16          // planes: 6 × vec4<f32>
  + 16 * 4          // viewProj: mat4x4<f32>
  + 4 + 4           // viewportW, viewportH
  + 4 + 4;          // objectCount, hizMipCount
  // Total: 96 + 64 + 16 = 176 bytes → aligned to 256 for WebGPU uniform

const CULL_UNIFORM_ALIGNED = 256;  // minUniformBufferOffsetAlignment

// ─────────────────────────────────────────────────────────────────────────────
// GPUCullingPipeline
// ─────────────────────────────────────────────────────────────────────────────

export interface GPUCullingConfig {
  /** Maximum objects (cells + particle groups) supported. Default MAX_OBJECTS. */
  maxObjects?: number;
  /** Force frustum-only mode (skip HiZ occlusion). Default false. */
  frustumOnly?: boolean;
}

/**
 * Encapsulates the full GPU culling pipeline: frustum test, HiZ occlusion
 * test, and indirect draw argument compaction.
 *
 * Lifecycle:
 * ```ts
 *   const culler = new GPUCullingPipeline(device);
 *   await culler.init(config);
 *
 *   // Each frame:
 *   culler.uploadAABBs(aabbList);
 *   culler.updateCamera(camera);
 *   culler.dispatch(encoder, depthTexture);   // optional depth for HiZ
 *
 *   // Bind for rendering:
 *   pass.drawIndirect(culler.drawIndirectBuffer, 0);
 *   // culler.compactedIndexBuffer holds visible object indices.
 * ```
 */
export class GPUCullingPipeline {

  // ── WebGPU handles ─────────────────────────────────────────────────────

  private device: GPUDevice;

  // Pipelines
  private cullPipeline!:    GPUComputePipeline;
  private compactPipeline!: GPUComputePipeline;
  private hizPipeline!:     GPUComputePipeline;

  // Buffers
  private uniformBuffer!:          GPUBuffer;
  private aabbBuffer!:             GPUBuffer;
  private visibilityBuffer!:       GPUBuffer;
  private compactUniformBuffer!:   GPUBuffer;
  private compactedIndexBuffer_!:  GPUBuffer;
  private drawIndirectBuffer_!:    GPUBuffer;

  // Bind groups (rebuilt per frame for HiZ texture changes)
  private cullBindGroup0!:     GPUBindGroup;
  private cullBindGroup1!:     GPUBindGroup | null;
  private compactBindGroup!:   GPUBindGroup;

  // HiZ
  private hizTexture:    GPUTexture  | null = null;
  private hizViews:      GPUTextureView[] = [];
  private hizSampler!:   GPUSampler;
  private hizBindGroups: GPUBindGroup[]   = [];
  private hizMipCount = 0;

  // Bind group layouts
  private cullBGL0!:    GPUBindGroupLayout;
  private cullBGL1!:    GPUBindGroupLayout;
  private compactBGL!:  GPUBindGroupLayout;
  private hizBGL!:      GPUBindGroupLayout;

  // Config
  private maxObjects = MAX_OBJECTS;
  private frustumOnly = false;

  // Stats (CPU-side, updated via readback only when debug is on)
  private _lastVisibleCount = 0;

  constructor(device: GPUDevice) {
    this.device = device;
  }

  // ── Public accessors ───────────────────────────────────────────────────

  /** The indirect draw args buffer. Bind to pass.drawIndirect(). */
  get drawIndirectBuffer(): GPUBuffer { return this.drawIndirectBuffer_; }

  /** Compacted visible-object index buffer (storage, read-only in VS). */
  get compactedIndexBuffer(): GPUBuffer { return this.compactedIndexBuffer_; }

  /** Last known visible count (only valid if debug readback is enabled). */
  get lastVisibleCount(): number { return this._lastVisibleCount; }

  // ── Initialisation ─────────────────────────────────────────────────────

  async init(config: GPUCullingConfig = {}): Promise<void> {
    this.maxObjects  = config.maxObjects  ?? MAX_OBJECTS;
    this.frustumOnly = config.frustumOnly ?? false;

    // Auto-disable occlusion on low-tier devices
    const budget = getGlobalBudget();
    if (budget) {
      const tier: Tier = budget.tier;
      if (tier === 'LOW' || tier === 'MEDIUM') {
        this.frustumOnly = true;
      }
    }

    this.createBuffers();
    await this.createPipelines();
    this.createSampler();
  }

  // ── Buffer creation ────────────────────────────────────────────────────

  private createBuffers(): void {
    const dev = this.device;

    // Uniform buffer for cull shader
    this.uniformBuffer = dev.createBuffer({
      label: 'gpu-cull-uniforms',
      size:  CULL_UNIFORM_ALIGNED,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // AABB storage buffer
    this.aabbBuffer = dev.createBuffer({
      label: 'gpu-cull-aabbs',
      size:  this.maxObjects * AABB_STRIDE * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Visibility output (one u32 per object)
    this.visibilityBuffer = dev.createBuffer({
      label: 'gpu-cull-visibility',
      size:  this.maxObjects * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Compaction uniform buffer
    this.compactUniformBuffer = dev.createBuffer({
      label: 'gpu-cull-compact-uniforms',
      size:  16,  // 4 × u32
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Compacted index buffer
    this.compactedIndexBuffer_ = dev.createBuffer({
      label: 'gpu-cull-compacted-indices',
      size:  this.maxObjects * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX,
    });

    // DrawIndirectArgs buffer (4 × u32 = 16 bytes)
    this.drawIndirectBuffer_ = dev.createBuffer({
      label: 'gpu-cull-draw-indirect',
      size:  DRAW_INDIRECT_SIZE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT
           | GPUBufferUsage.COPY_DST,
    });
  }

  // ── Pipeline creation ──────────────────────────────────────────────────

  private async createPipelines(): Promise<void> {
    const dev = this.device;

    // ── Cull pipeline bind group layouts ──────────────────────────────────

    this.cullBGL0 = dev.createBindGroupLayout({
      label: 'cull-bgl-0',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' } },
      ],
    });

    this.cullBGL1 = dev.createBindGroupLayout({
      label: 'cull-bgl-1',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE,
          texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE,
          sampler: { type: 'filtering' } },
      ],
    });

    const cullPipelineLayout = dev.createPipelineLayout({
      label: 'cull-pipeline-layout',
      bindGroupLayouts: [this.cullBGL0, this.cullBGL1],
    });

    const cullModule = dev.createShaderModule({
      label: 'cull-shader',
      code: CULL_SHADER_SOURCE,
    });

    this.cullPipeline = await dev.createComputePipelineAsync({
      label: 'cull-pipeline',
      layout: cullPipelineLayout,
      compute: { module: cullModule, entryPoint: 'cs_cull' },
    });

    // ── Compaction pipeline ──────────────────────────────────────────────

    this.compactBGL = dev.createBindGroupLayout({
      label: 'compact-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' } },
      ],
    });

    const compactPipelineLayout = dev.createPipelineLayout({
      label: 'compact-pipeline-layout',
      bindGroupLayouts: [this.compactBGL],
    });

    const compactModule = dev.createShaderModule({
      label: 'compact-shader',
      code: COMPACT_SHADER_SOURCE,
    });

    this.compactPipeline = await dev.createComputePipelineAsync({
      label: 'compact-pipeline',
      layout: compactPipelineLayout,
      compute: { module: compactModule, entryPoint: 'cs_compact' },
    });

    // ── HiZ pyramid build pipeline ───────────────────────────────────────

    this.hizBGL = dev.createBindGroupLayout({
      label: 'hiz-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE,
          texture: { sampleType: 'unfilterable-float' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: 'write-only', format: 'r32float' } },
      ],
    });

    const hizPipelineLayout = dev.createPipelineLayout({
      label: 'hiz-pipeline-layout',
      bindGroupLayouts: [this.hizBGL],
    });

    const hizModule = dev.createShaderModule({
      label: 'hiz-shader',
      code: HIZ_BUILD_SHADER_SOURCE,
    });

    this.hizPipeline = await dev.createComputePipelineAsync({
      label: 'hiz-pipeline',
      layout: hizPipelineLayout,
      compute: { module: hizModule, entryPoint: 'cs_hiz_build' },
    });
  }

  // ── Sampler ────────────────────────────────────────────────────────────

  private createSampler(): void {
    this.hizSampler = this.device.createSampler({
      label: 'hiz-sampler',
      magFilter: 'nearest',
      minFilter: 'nearest',
      mipmapFilter: 'nearest',
    });
  }

  // ── AABB upload ────────────────────────────────────────────────────────

  /**
   * Upload an array of CullAABBs for the current frame.
   * Call once per frame before `dispatch()`.
   */
  uploadAABBs(aabbs: CullAABB[]): void {
    const count = Math.min(aabbs.length, this.maxObjects);
    const data  = new Float32Array(count * AABB_STRIDE);

    for (let i = 0; i < count; i++) {
      const a   = aabbs[i];
      const off = i * AABB_STRIDE;
      data[off + 0] = a.minX;
      data[off + 1] = a.minY;
      data[off + 2] = a.maxX;
      data[off + 3] = a.maxY;
      data[off + 4] = a.nearZ;
      // [5..7] padding — left as 0
    }

    this.device.queue.writeBuffer(this.aabbBuffer, 0, data);
  }

  // ── Camera / frustum update ────────────────────────────────────────────

  /**
   * Update the culling uniforms from a LODCamera.
   * Extracts frustum planes and writes the viewProj matrix.
   */
  updateCamera(cam: LODCamera, objectCount: number): void {
    const viewProj = buildViewProjFromCamera(cam);
    const planes   = extractFrustumPlanes(viewProj);

    const mipCount = this.frustumOnly ? 0 : this.hizMipCount;

    // Pack into the uniform layout
    const buf = new Float32Array(CULL_UNIFORM_ALIGNED / 4);
    let off = 0;

    // 6 × vec4 planes (each padded to vec4)
    for (const p of planes) {
      buf[off++] = p.a;
      buf[off++] = p.b;
      buf[off++] = p.c;
      buf[off++] = p.d;
    }

    // mat4x4 viewProj (16 floats)
    buf.set(viewProj, off);
    off += 16;

    // viewportW, viewportH
    buf[off++] = cam.viewportW;
    buf[off++] = cam.viewportH;

    // objectCount, hizMipCount (write as u32 into Float32Array via DataView)
    const dv = new DataView(buf.buffer);
    dv.setUint32(off * 4, objectCount, true);
    dv.setUint32((off + 1) * 4, mipCount, true);

    this.device.queue.writeBuffer(this.uniformBuffer, 0, buf);
  }

  // ── HiZ pyramid ───────────────────────────────────────────────────────

  /**
   * Build or rebuild the HiZ pyramid texture from a depth texture.
   * Must be called when the depth texture dimensions change (e.g. resize).
   */
  ensureHiZTexture(depthW: number, depthH: number): void {
    // Compute power-of-two dimensions
    const w = nextPow2(depthW);
    const h = nextPow2(depthH);
    const mipCount = Math.floor(Math.log2(Math.max(w, h))) + 1;

    if (this.hizTexture && this.hizMipCount === mipCount) {
      return; // already the right size
    }

    this.hizTexture?.destroy();

    this.hizTexture = this.device.createTexture({
      label: 'hiz-pyramid',
      size:  { width: w, height: h },
      mipLevelCount: mipCount,
      format: 'r32float',
      usage: GPUTextureUsage.TEXTURE_BINDING
           | GPUTextureUsage.STORAGE_BINDING
           | GPUTextureUsage.COPY_DST,
    });

    this.hizMipCount = mipCount;

    // Create per-mip views
    this.hizViews = [];
    for (let m = 0; m < mipCount; m++) {
      this.hizViews.push(this.hizTexture.createView({
        label: `hiz-mip-${m}`,
        baseMipLevel: m,
        mipLevelCount: 1,
      }));
    }
  }

  /**
   * Dispatch compute passes to downsample the depth into the HiZ pyramid.
   */
  buildHiZPyramid(
    encoder:      GPUCommandEncoder,
    depthTexture: GPUTexture,
  ): void {
    if (this.frustumOnly || !this.hizTexture) return;

    const dev = this.device;

    // Mip 0 ← copy from the depth texture (requires matching sizes or blit)
    // For simplicity, we treat the depth texture as mip-0 source directly
    // via a compute downsample from depth → hiz mip 0.

    // Build bind groups for each mip-to-mip downsample
    this.hizBindGroups = [];

    // First pass: depth → hiz mip 0
    // Subsequent passes: hiz mip N → hiz mip N+1
    for (let m = 0; m < this.hizMipCount - 1; m++) {
      const srcView = m === 0
        ? depthTexture.createView({ label: 'depth-src-for-hiz' })
        : this.hizViews[m];

      const dstView = this.hizViews[m === 0 ? 0 : m + 1];
      const srcW = Math.max(1, (this.hizTexture!.width)  >> m);
      const srcH = Math.max(1, (this.hizTexture!.height) >> m);

      const ub = dev.createBuffer({
        label: `hiz-uniform-mip-${m}`,
        size:  8,  // 2 × u32
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Uint32Array(ub.getMappedRange()).set([srcW, srcH]);
      ub.unmap();

      const bg = dev.createBindGroup({
        label: `hiz-bg-mip-${m}`,
        layout: this.hizBGL,
        entries: [
          { binding: 0, resource: { buffer: ub } },
          { binding: 1, resource: srcView },
          { binding: 2, resource: m === 0 ? this.hizViews[0] : this.hizViews[m + 1] },
        ],
      });

      this.hizBindGroups.push(bg);
    }

    // Dispatch each mip level
    for (let m = 0; m < this.hizBindGroups.length; m++) {
      const dstW = Math.max(1, (this.hizTexture!.width)  >> (m + 1));
      const dstH = Math.max(1, (this.hizTexture!.height) >> (m + 1));

      const pass = encoder.beginComputePass({ label: `hiz-build-mip-${m + 1}` });
      pass.setPipeline(this.hizPipeline);
      pass.setBindGroup(0, this.hizBindGroups[m]);
      pass.dispatchWorkgroups(
        Math.ceil(dstW / 8),
        Math.ceil(dstH / 8),
      );
      pass.end();
    }
  }

  // ── Main dispatch ──────────────────────────────────────────────────────

  /**
   * Run the full culling + compaction pipeline.
   *
   * @param encoder       Active command encoder.
   * @param objectCount   Number of objects uploaded via `uploadAABBs`.
   * @param depthTexture  Previous frame's depth texture (optional — if null,
   *                      occlusion culling is skipped for this frame).
   * @param baseVertexCount  Vertex count for the base draw call (e.g. 6 for
   *                         quad instancing, or index count for cells).
   */
  dispatch(
    encoder:          GPUCommandEncoder,
    objectCount:      number,
    depthTexture:     GPUTexture | null = null,
    baseVertexCount   = 6,
  ): void {
    const dev   = this.device;
    const count = Math.min(objectCount, this.maxObjects);
    if (count === 0) return;

    // ── Reset indirect draw args ─────────────────────────────────────────
    // instanceCount = 0; vertexCount = baseVertexCount; firstVertex = 0; firstInstance = 0
    const resetData = new Uint32Array([baseVertexCount, 0, 0, 0]);
    dev.queue.writeBuffer(this.drawIndirectBuffer_, 0, resetData);

    // Clear visibility buffer
    const zeroVis = new Uint32Array(count);
    dev.queue.writeBuffer(this.visibilityBuffer, 0, zeroVis);

    // ── HiZ build (if occlusion enabled) ─────────────────────────────────
    if (!this.frustumOnly && depthTexture) {
      this.ensureHiZTexture(depthTexture.width, depthTexture.height);
      this.buildHiZPyramid(encoder, depthTexture);
    }

    // ── Rebuild bind groups ──────────────────────────────────────────────

    this.cullBindGroup0 = dev.createBindGroup({
      label: 'cull-bg-0',
      layout: this.cullBGL0,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.aabbBuffer } },
        { binding: 2, resource: { buffer: this.visibilityBuffer } },
      ],
    });

    // HiZ bind group — use a dummy 1×1 texture if occlusion is off
    if (!this.frustumOnly && this.hizTexture) {
      this.cullBindGroup1 = dev.createBindGroup({
        label: 'cull-bg-1',
        layout: this.cullBGL1,
        entries: [
          { binding: 0, resource: this.hizTexture.createView() },
          { binding: 1, resource: this.hizSampler },
        ],
      });
    } else {
      // Create a 1×1 dummy texture for the bind group
      this.cullBindGroup1 = this.makeDummyHiZBindGroup();
    }

    this.compactBindGroup = dev.createBindGroup({
      label: 'compact-bg',
      layout: this.compactBGL,
      entries: [
        { binding: 0, resource: { buffer: this.compactUniformBuffer } },
        { binding: 1, resource: { buffer: this.visibilityBuffer } },
        { binding: 2, resource: { buffer: this.compactedIndexBuffer_ } },
        { binding: 3, resource: { buffer: this.drawIndirectBuffer_ } },
      ],
    });

    // Upload compaction uniforms
    const compactU = new Uint32Array([count, baseVertexCount, 0, 0]);
    dev.queue.writeBuffer(this.compactUniformBuffer, 0, compactU);

    // ── Phase 1 + 2: frustum + occlusion cull ────────────────────────────

    const cullPass = encoder.beginComputePass({ label: 'gpu-cull' });
    cullPass.setPipeline(this.cullPipeline);
    cullPass.setBindGroup(0, this.cullBindGroup0);
    cullPass.setBindGroup(1, this.cullBindGroup1!);
    cullPass.dispatchWorkgroups(Math.ceil(count / WG));
    cullPass.end();

    // ── Phase 3: compaction ──────────────────────────────────────────────

    const compactPass = encoder.beginComputePass({ label: 'gpu-compact' });
    compactPass.setPipeline(this.compactPipeline);
    compactPass.setBindGroup(0, this.compactBindGroup);
    compactPass.dispatchWorkgroups(Math.ceil(count / WG));
    compactPass.end();
  }

  // ── Dummy HiZ bind group (for frustum-only mode) ──────────────────────

  private dummyTexture: GPUTexture | null = null;

  private makeDummyHiZBindGroup(): GPUBindGroup {
    if (!this.dummyTexture) {
      this.dummyTexture = this.device.createTexture({
        label: 'hiz-dummy-1x1',
        size: { width: 1, height: 1 },
        format: 'r32float',
        usage: GPUTextureUsage.TEXTURE_BINDING,
      });
    }

    return this.device.createBindGroup({
      label: 'cull-bg-1-dummy',
      layout: this.cullBGL1,
      entries: [
        { binding: 0, resource: this.dummyTexture.createView() },
        { binding: 1, resource: this.hizSampler },
      ],
    });
  }

  // ── Debug readback ─────────────────────────────────────────────────────

  /**
   * Read back the instance count from the indirect draw buffer.
   * **Only for debugging** — causes a GPU→CPU sync stall.
   */
  async readbackVisibleCount(): Promise<number> {
    const dev = this.device;
    const staging = dev.createBuffer({
      label: 'cull-readback',
      size: DRAW_INDIRECT_SIZE,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(this.drawIndirectBuffer_, 0, staging, 0, DRAW_INDIRECT_SIZE);
    dev.queue.submit([enc.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const data = new Uint32Array(staging.getMappedRange());
    this._lastVisibleCount = data[1];  // instanceCount
    staging.unmap();
    staging.destroy();

    return this._lastVisibleCount;
  }

  // ── Disposal ───────────────────────────────────────────────────────────

  destroy(): void {
    this.uniformBuffer.destroy();
    this.aabbBuffer.destroy();
    this.visibilityBuffer.destroy();
    this.compactUniformBuffer.destroy();
    this.compactedIndexBuffer_.destroy();
    this.drawIndirectBuffer_.destroy();
    this.hizTexture?.destroy();
    this.dummyTexture?.destroy();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience: batch cell culling helper
// ─────────────────────────────────────────────────────────────────────────────

/**
 * High-level helper that converts a Map of cell bounding boxes into the
 * format expected by `GPUCullingPipeline.uploadAABBs` and runs the cull.
 *
 * Returns the indirect draw buffer + compacted index buffer ready for
 * binding to the instanced cell renderer.
 */
export function prepareCellCullDispatch(
  culler:       GPUCullingPipeline,
  encoder:      GPUCommandEncoder,
  cellBBoxes:   Map<string, CellBBox>,
  camera:       LODCamera,
  depthTexture: GPUTexture | null = null,
  baseVertexCount = 6,
): { drawIndirectBuffer: GPUBuffer; compactedIndexBuffer: GPUBuffer; cellOrder: string[] } {

  const cellOrder: string[] = [];
  const aabbs: CullAABB[]   = [];

  for (const [id, bbox] of cellBBoxes) {
    cellOrder.push(id);
    aabbs.push(cellBBoxToAABB(bbox));
  }

  culler.uploadAABBs(aabbs);
  culler.updateCamera(camera, aabbs.length);
  culler.dispatch(encoder, aabbs.length, depthTexture, baseVertexCount);

  return {
    drawIndirectBuffer:  culler.drawIndirectBuffer,
    compactedIndexBuffer: culler.compactedIndexBuffer,
    cellOrder,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

function nextPow2(v: number): number {
  v--;
  v |= v >> 1; v |= v >> 2; v |= v >> 4;
  v |= v >> 8; v |= v >> 16;
  return v + 1;
}
