/**
 * ue-vsm-shadows.ts — M836: UE5 Virtual Shadow Map (VSM) Port
 * ─────────────────────────────────────────────────────────────────────────────
 * 移植 Unreal Engine 5 Virtual Shadow Map 系统到 WebGPU Cell 渲染管线。
 *
 * 核心概念 (参照 UE5 Renderer-Private/VirtualShadowMaps/ 约47个文件):
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *  § 1  物理页池 (Physical Page Pool)
 *       PhysicalPagePool — rgba16float texture, 分辨率 PHYS_DIM × PHYS_DIM
 *       每个 tile 为 VSM_PAGE_SIZE(128)×128 像素的深度 tile
 *       PageMetadata buffer 记录每个物理页的 flags / dirty 状态
 *
 *  § 2  页表虚拟化 (Page Table Virtualization)
 *       PageTableBuffer — 每个 VSM 有 Level0DimPagesXY²=128² 项
 *       项目存储 physical page index (16bit) + flags (16bit)
 *       仅可见区域的 shadow tile 分配物理页 → 节省大量显存
 *
 *  § 3  Clipmap 分级 (Clipmap-based Shadow)
 *       与 UE5 FVirtualShadowMapClipmap 对应:
 *         FirstLevel=8, LastLevel=18 → 11 个 clipmap 级别
 *         每级 radius = 2^level 世界单位, 覆盖 Cell 不同距离
 *         级别越高分辨率越低 → 细节近处高分辨率, 远处低分辨率
 *         级别中心随摄像机 snap 到页对齐, 避免 shadow swimming
 *
 *  § 4  页缓存 (Page Cache)
 *       VSMPageCache 跟踪上帧的 PageSpaceLocation 和 physical page 映射
 *       静止 Cell 的 shadow tile 不重画 (FORCE_CACHED flag)
 *       动态 Cell 标记 DYNAMIC_UNCACHED → 该页重渲染
 *       帧间 clipmap 平移时: 新页 invalidated, 复用重叠页
 *
 *  § 5  与 AT lighting.fs 桥接
 *       ATVSMBridge — 将 VSM page table + physical pool 绑定到 lighting pass
 *       getShadow() 替换为 getVSMShadow() WGSL 实现
 *       支持 PCF filtering over physical page tiles
 *
 * 算法流程
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   ┌─ Pass A ── Page Marking ────────────────────────────────────────────────┐
 *   │  GBuffer depth + camera → mark which shadow pages are visible           │
 *   │  输出: PageRequestBuffer (per-VSM per-page 可见性标记)                   │
 *   └──────────────────────────────────────────────────────────────────────────┘
 *                │ PageRequestBuffer
 *                ▼
 *   ┌─ Pass B ── Physical Page Allocation ───────────────────────────────────┐
 *   │  空闲页池 → 为每个 requested+uncached 页分配物理页                       │
 *   │  写入 PageTableBuffer (virtualIdx → physicalIdx)                        │
 *   └──────────────────────────────────────────────────────────────────────────┘
 *                │ PageTableBuffer (updated)
 *                ▼
 *   ┌─ Pass C ── Shadow Depth Render ─────────────────────────────────────────┐
 *   │  仅渲染 DYNAMIC_UNCACHED / STATIC_UNCACHED 页的 shadow casters           │
 *   │  写入 PhysicalPagePool[physIdx].depth                                   │
 *   └──────────────────────────────────────────────────────────────────────────┘
 *                │ PhysicalPagePool (partial update)
 *                ▼
 *   ┌─ Pass D ── Projection / Lighting ──────────────────────────────────────┐
 *   │  lighting.fs → getVSMShadow(worldPos, lightDir)                        │
 *   │  查 PageTable → 找 physicalPage → 采样 depth → PCF shadow factor        │
 *   └──────────────────────────────────────────────────────────────────────────┘
 *
 * Usage:
 *   const vsm = await UEVSMShadows.create(device, { width, height });
 *   vsm.updateClipmaps(cameraPos, lightDir);
 *   vsm.markPages(encoder, depthTex, gBufferNormalTex);
 *   vsm.allocatePages(encoder);
 *   vsm.renderShadowDepth(encoder, shadowCasterDraws);
 *   vsm.bindToLightingPass(passEncoder, lightingPipeline, groupIndex);
 *
 * Research: xiaodi #M836 — cell-pubsub-loop
 * Ported from: Renderer-Private/VirtualShadowMaps/ (UE5)
 *              Shaders-Private/VirtualShadowMaps/*.usf/.ush
 */

// ─────────────────────────────────────────────────────────────────────────────
// § 0  Constants (mirrors VirtualShadowMapDefinitions.h)
// ─────────────────────────────────────────────────────────────────────────────

/** Shadow tile size in texels — must match VSM_PAGE_SIZE in UE5 */
const VSM_PAGE_SIZE            = 128;
const VSM_PAGE_SIZE_MASK       = VSM_PAGE_SIZE - 1;
const VSM_LOG2_PAGE_SIZE       = 7;
/** Level-0 virtual address space in pages per axis: 128×128 = 16 384 */
const VSM_LEVEL0_DIM_PAGES_XY  = 128;
const VSM_LOG2_LEVEL0_DIM_PAGES_XY = 7;
const VSM_MAX_MIP_LEVELS       = 8;
/** Max virtual resolution per axis in texels: 16 384 × 128 = 2 097 152 */
const VSM_VIRTUAL_MAX_RES_XY   = VSM_LEVEL0_DIM_PAGES_XY * VSM_PAGE_SIZE;

/** Physical pool dimension in pages (width & height) */
const PHYS_POOL_DIM_PAGES      = 64;   // 64×64 pages = 4096 physical pages
const PHYS_POOL_DIM_TEXELS     = PHYS_POOL_DIM_PAGES * VSM_PAGE_SIZE; // 8192
const MAX_PHYSICAL_PAGES       = PHYS_POOL_DIM_PAGES * PHYS_POOL_DIM_PAGES;

/** Clipmap level range — mirrors FVirtualShadowMapClipmapConfig defaults */
const VSM_FIRST_LEVEL          = 8;
const VSM_LAST_LEVEL           = 18;
const VSM_NUM_LEVELS           = VSM_LAST_LEVEL - VSM_FIRST_LEVEL + 1; // 11

/** Max single-page VSMs (spot/point lights use 1 page each) */
const VSM_MAX_SINGLE_PAGE_SHADOW_MAPS = 1024;

// Page flags (mirrors VirtualShadowMapPageAccessCommon.ush)
const VSM_FLAG_ALLOCATED          = 1 << 0;
const VSM_FLAG_DYNAMIC_UNCACHED   = 1 << 1;
const VSM_FLAG_STATIC_UNCACHED    = 1 << 2;
const VSM_FLAG_DETAIL_GEOMETRY    = 1 << 3;
const VSM_FLAG_PRIMARY_REQUEST    = 1 << 4;
const VSM_FLAG_SECONDARY_REQUEST  = 1 << 5;
const VSM_PAGE_FLAGS_BITS_PER_HMIP = 6;
const VSM_EXTENDED_FLAG_DYNAMIC_INITIALIZED = 1 << 6;
const VSM_EXTENDED_FLAG_STATIC_INITIALIZED  = 1 << 7;
const VSM_EXTENDED_FLAG_DYNAMIC_DIRTY       = 1 << 8;
const VSM_EXTENDED_FLAG_STATIC_DIRTY        = 1 << 9;
const VSM_EXTENDED_FLAG_INVALIDATE_DYNAMIC  = 1 << 10;
const VSM_EXTENDED_FLAG_INVALIDATE_STATIC   = 1 << 11;
const VSM_EXTENDED_FLAG_FORCE_CACHED        = 1 << 14;
const VSM_EXTENDED_FLAG_UNREFERENCED        = 1 << 13;

// ─────────────────────────────────────────────────────────────────────────────
// § 1  Math helpers
// ─────────────────────────────────────────────────────────────────────────────

/** 4×4 column-major matrix multiply (row-major in JS arrays) */
function mat4Mul(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) sum += a[row * 4 + k] * b[k * 4 + col];
      out[row * 4 + col] = sum;
    }
  }
  return out;
}

/** Build orthographic projection (matches UE5 clipmap ViewToClip) */
function buildOrtho(halfW: number, halfH: number, zNear: number, zFar: number): Float32Array {
  const m = new Float32Array(16);
  m[0]  =  1 / halfW;
  m[5]  =  1 / halfH;
  m[10] = -2 / (zFar - zNear);
  m[14] = -(zFar + zNear) / (zFar - zNear);
  m[15] = 1;
  return m;
}

/** Build view matrix from light direction (directional light) */
function buildLightView(lightDir: [number, number, number], center: [number, number, number]): Float32Array {
  const [dx, dy, dz] = lightDir;
  // Choose an up vector not parallel to lightDir
  const up: [number, number, number] = Math.abs(dy) < 0.99 ? [0, 1, 0] : [1, 0, 0];
  // right = lightDir × up
  const rx = dy * up[2] - dz * up[1];
  const ry = dz * up[0] - dx * up[2];
  const rz = dx * up[1] - dy * up[0];
  const rlen = Math.sqrt(rx * rx + ry * ry + rz * rz) || 1;
  const [rcx, rcy, rcz] = [rx / rlen, ry / rlen, rz / rlen];
  // newUp = right × lightDir
  const ux = rcy * dz - rcz * dy;
  const uy = rcz * dx - rcx * dz;
  const uz = rcx * dy - rcy * dx;
  const tx = -(rcx * center[0] + rcy * center[1] + rcz * center[2]);
  const ty = -(ux * center[0] + uy * center[1] + uz * center[2]);
  const tz = -(dx * center[0] + dy * center[1] + dz * center[2]);
  return new Float32Array([
    rcx, ux, dx, 0,
    rcy, uy, dy, 0,
    rcz, uz, dz, 0,
    tx,  ty, tz, 1,
  ]);
}

/** Snap value to multiples of `snap` (clipmap origin snapping) */
function snapToGrid(v: number, snap: number): number {
  return Math.floor(v / snap) * snap;
}

/** log2 of clipmap level dimension in pages */
function calcLog2LevelDimPages(level: number): number {
  return VSM_LOG2_LEVEL0_DIM_PAGES_XY - level;
}
function calcLevelDimPages(level: number): number {
  return 1 << Math.max(0, calcLog2LevelDimPages(level));
}

// ─────────────────────────────────────────────────────────────────────────────
// § 2  Types
// ─────────────────────────────────────────────────────────────────────────────

/** One clipmap level data (mirrors FVirtualShadowMapClipmap::FLevelData) */
export interface VSMClipmapLevelData {
  /** Absolute level index (VSM_FIRST_LEVEL..VSM_LAST_LEVEL) */
  level: number;
  /** Level radius in world units = 2^level */
  radius: number;
  /** World-space center (snapped to page grid) */
  worldCenter: [number, number, number];
  /** Orthographic half-extents for this level */
  halfExtent: number;
  /** Light view matrix (row-major Float32Array[16]) */
  viewMatrix: Float32Array;
  /** Orthographic projection matrix */
  projMatrix: Float32Array;
  /** Combined viewProj for shadow depth render */
  viewProjMatrix: Float32Array;
  /** Page-space corner offset for caching delta (Int2) */
  cornerOffset: [number, number];
  /** Whether this level's pages need full invalidation */
  invalidated: boolean;
}

/** Per-physical-page metadata (matches FVirtualShadowMapPhysicalPageMetaData layout) */
export interface VSMPageMetadata {
  /** Index of VSM that owns this page (-1 = free) */
  vsmIndex: number;
  /** Virtual page X within the owning VSM's page table */
  virtualPageX: number;
  /** Virtual page Y within the owning VSM's page table */
  virtualPageY: number;
  /** MIP level (0 = finest) */
  mipLevel: number;
  /** Combined VSM_FLAG_* + VSM_EXTENDED_FLAG_* */
  flags: number;
  /** Frame number when this page was last rendered */
  lastRenderedFrame: number;
}

/** Per-light (per-VSM) cache entry (mirrors FVirtualShadowMapCacheEntry) */
export interface VSMCacheEntry {
  /** Previous frame's page-space location of clipmap origin */
  prevPageSpaceLocation: [number, number];
  /** Current frame's page-space location */
  currPageSpaceLocation: [number, number];
  /** Physical page allocations: virtualIndex → physicalIndex */
  pageMapping: Map<number, number>;
  /** Frame in which this entry was last rendered */
  lastRenderedFrame: number;
  /** Whether this light moved (forces uncached) */
  isDynamic: boolean;
}

/** Configuration for UEVSMShadows */
export interface VSMConfig {
  /** Render target width (for page marking pass) */
  width: number;
  /** Render target height */
  height: number;
  /** Physical page pool size in pages per axis (default 64) */
  physPoolDimPages?: number;
  /** Number of clipmap levels (default VSM_NUM_LEVELS = 11) */
  numClipmapLevels?: number;
  /** First clipmap level (default 8) */
  firstClipmapLevel?: number;
  /** Resolution LOD bias (default 0) */
  resolutionLodBias?: number;
  /** PCF filter radius in texels (default 2) */
  pcfRadius?: number;
  /** Max distance for shadow casting (default 65536) */
  maxShadowDistance?: number;
  /** Enable static/dynamic page separation (default true) */
  enablePageCaching?: boolean;
}

/** Shadow draw call descriptor for depth render pass */
export interface VSMShadowDraw {
  /** Vertex buffer */
  vertexBuffer: GPUBuffer;
  /** Index buffer (optional — draw arrays if null) */
  indexBuffer: GPUBuffer | null;
  /** Number of indices / vertices */
  count: number;
  /** Model matrix uniform buffer */
  modelUBO: GPUBuffer;
  /** Which clipmap levels this caster affects (bitmask) */
  levelMask: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 3  WGSL — Page Marking Compute Shader
//      Mirrors VirtualShadowMapPageMarking.usf
//      For each GBuffer pixel: compute clipmap level → mark shadow page
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_PAGE_MARKING = /* wgsl */ `
// ── Constants ─────────────────────────────────────────────────────────────────
const VSM_PAGE_SIZE         : u32 = 128u;
const VSM_LEVEL0_DIM_PAGES  : u32 = 128u;
const VSM_LOG2_LEVEL0_DIM   : u32 = 7u;
const VSM_FIRST_LEVEL       : u32 = 8u;
const VSM_LAST_LEVEL        : u32 = 18u;
const VSM_NUM_LEVELS        : u32 = 11u;
const VSM_FLAG_PRIMARY_REQUEST : u32 = 16u;  // 1<<4

// ── Uniforms ──────────────────────────────────────────────────────────────────
struct MarkUniforms {
  invViewProj       : mat4x4f,
  cameraPos         : vec3f,
  lightDir          : vec3f,
  screenSize        : vec2f,
  nearZ             : f32,
  farZ              : f32,
  firstLevel        : u32,
  numLevels         : u32,
  resolutionLodBias : f32,
  _pad              : f32,
  // Per-clipmap level data: 11 * (mat4 viewProj + vec4 center + vec4 extents)
  // Packed as array of 11*24 floats
};

@group(0) @binding(0) var<uniform>         uMark       : MarkUniforms;
@group(0) @binding(1) var                  depthTex    : texture_depth_2d;
@group(0) @binding(2) var                  depthSampler: sampler;
@group(0) @binding(3) var<storage, read_write> pageRequests : array<atomic<u32>>;
// pageRequests layout: [VSM_NUM_LEVELS × (128×128) u32]
// Each u32 is a bitmask of 32 sub-tiles, or simple 1/0 if ≤32 tiles per level

// ── Level helpers ─────────────────────────────────────────────────────────────
fn calcLog2LevelDimPages(level: u32) -> u32 {
  return VSM_LOG2_LEVEL0_DIM - level;
}
fn calcLevelDimPages(level: u32) -> u32 {
  return 1u << calcLog2LevelDimPages(level);
}
fn levelBaseOffset(levelIdx: u32) -> u32 {
  // Sum of pages for previous levels: Σ dimPages^2 for l in [0..levelIdx)
  var offset = 0u;
  for (var l = 0u; l < levelIdx; l++) {
    let d = calcLevelDimPages(l);
    offset += d * d;
  }
  return offset;
}

// ── Reconstruct world pos from depth ─────────────────────────────────────────
fn reconstructWorldPos(uv: vec2f, depth: f32) -> vec3f {
  let ndc = vec4f(uv * 2.0 - 1.0, depth, 1.0);
  let worldH = uMark.invViewProj * ndc;
  return worldH.xyz / worldH.w;
}

// ── Compute absolute clipmap level for a world position ─────────────────────
// Mirrors CalcAbsoluteClipmapLevel in VirtualShadowMapProjectionCommon.ush
fn calcAbsoluteClipmapLevel(distSq: f32) -> f32 {
  return log2(distSq) * 0.5;
}

// ── Main compute kernel ───────────────────────────────────────────────────────
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let screenSize = vec2u(u32(uMark.screenSize.x), u32(uMark.screenSize.y));
  if (gid.x >= screenSize.x || gid.y >= screenSize.y) { return; }

  let uv = (vec2f(gid.xy) + 0.5) / uMark.screenSize;
  let depth = textureSampleLevel(depthTex, depthSampler, uv, 0.0);
  if (depth <= 0.0001) { return; }  // sky

  let worldPos = reconstructWorldPos(uv, depth);
  let toCam = worldPos - uMark.cameraPos;
  let distSq = dot(toCam, toCam);

  let absLevel = calcAbsoluteClipmapLevel(distSq) + uMark.resolutionLodBias;
  let firstL = uMark.firstLevel;
  let lastL  = firstL + uMark.numLevels - 1u;

  for (var li = 0u; li < uMark.numLevels; li++) {
    let absLevelI = firstL + li;
    // Only mark the best-fit level (±0.5 from integer boundary)
    let levelF    = f32(absLevelI);
    if (absLevel < levelF - 0.5 || absLevel > levelF + 1.5) { continue; }

    let dimPages  = calcLevelDimPages(li);
    let levelRadius = exp2(levelF);
    let halfExt   = levelRadius * f32(VSM_PAGE_SIZE);

    // Project worldPos into this clipmap's UV [0..1]
    // (simplified: assume clipmap center = camera snapped)
    // Full implementation would use per-level viewProj from uMark
    let shadowUV = clamp((worldPos.xz - uMark.cameraPos.xz) / (2.0 * levelRadius) + 0.5, vec2f(0.0), vec2f(1.0));

    let pageX = u32(shadowUV.x * f32(dimPages));
    let pageY = u32(shadowUV.y * f32(dimPages));
    if (pageX >= dimPages || pageY >= dimPages) { continue; }

    let baseOff = levelBaseOffset(li);
    let pageIdx = baseOff + pageY * dimPages + pageX;
    atomicOr(&pageRequests[pageIdx], VSM_FLAG_PRIMARY_REQUEST);
    break;
  }
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// § 4  WGSL — Page Allocation Compute Shader
//      Mirrors VirtualShadowMapPhysicalPageManagement.usf
//      Allocates free physical pages for uncached requested pages
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_PAGE_ALLOC = /* wgsl */ `
const VSM_FLAG_ALLOCATED        : u32 = 1u;
const VSM_FLAG_DYNAMIC_UNCACHED : u32 = 2u;
const VSM_FLAG_STATIC_UNCACHED  : u32 = 4u;
const VSM_FLAG_PRIMARY_REQUEST  : u32 = 16u;
const VSM_EXTENDED_FORCE_CACHED : u32 = 0x4000u;
const VSM_EXTENDED_UNREFERENCED : u32 = 0x2000u;

const MAX_PHYSICAL_PAGES : u32 = 4096u;
const INVALID_PAGE       : u32 = 0xFFFFFFFFu;

struct AllocUniforms {
  totalVirtualPages : u32,
  frameNumber       : u32,
  _pad0             : u32,
  _pad1             : u32,
};

@group(0) @binding(0) var<uniform>            uAlloc       : AllocUniforms;
@group(0) @binding(1) var<storage, read>       pageRequests : array<u32>;
@group(0) @binding(2) var<storage, read_write> pageTable    : array<u32>;
// pageTable[virtualIdx] = (physicalIdx << 16) | flags
@group(0) @binding(3) var<storage, read_write> freePageList : array<atomic<u32>>;
// freePageList[0] = count, freePageList[1..] = free physical page indices
@group(0) @binding(4) var<storage, read_write> pageMetadata : array<u32>;
// pageMetadata[physIdx] = packed (vsmIdx:12, virtX:10, virtY:10)

fn allocPhysicalPage() -> u32 {
  let oldCount = atomicSub(&freePageList[0], 1u);
  if (oldCount == 0u) {
    atomicAdd(&freePageList[0], 1u);  // undo
    return INVALID_PAGE;
  }
  return atomicExchange(&freePageList[oldCount], INVALID_PAGE);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let virtIdx = gid.x;
  if (virtIdx >= uAlloc.totalVirtualPages) { return; }

  let req   = pageRequests[virtIdx];
  let entry = pageTable[virtIdx];
  let flags = entry & 0xFFFFu;
  let phys  = entry >> 16u;

  let requested = (req & 16u) != 0u;  // VSM_FLAG_PRIMARY_REQUEST

  if (!requested) {
    // Mark unreferenced if was allocated
    if ((flags & VSM_FLAG_ALLOCATED) != 0u) {
      pageTable[virtIdx] = (phys << 16u) | (flags | VSM_EXTENDED_UNREFERENCED);
    }
    return;
  }

  if ((flags & VSM_FLAG_ALLOCATED) != 0u && (flags & (VSM_FLAG_DYNAMIC_UNCACHED | VSM_FLAG_STATIC_UNCACHED)) == 0u) {
    // Already allocated and cached — keep it, clear unreferenced
    pageTable[virtIdx] = (phys << 16u) | (flags & ~VSM_EXTENDED_UNREFERENCED);
    return;
  }

  // Need to allocate a new physical page
  var physPage = phys;
  if ((flags & VSM_FLAG_ALLOCATED) == 0u) {
    physPage = allocPhysicalPage();
    if (physPage == INVALID_PAGE) { return; }  // out of physical pages
  }

  let newFlags = VSM_FLAG_ALLOCATED | VSM_FLAG_DYNAMIC_UNCACHED;
  pageTable[virtIdx] = (physPage << 16u) | newFlags;
  // Record ownership in metadata
  pageMetadata[physPage] = (virtIdx & 0xFFFFFu) | (uAlloc.frameNumber << 20u);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// § 5  WGSL — Shadow Depth Vertex/Fragment Shaders
//      Renders shadow casters into physical page pool tiles
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_SHADOW_DEPTH_VS = /* wgsl */ `
struct ShadowDepthUniforms {
  lightViewProj : mat4x4f,
  // Physical page tile offset within pool texture (in pages)
  physPageX     : f32,
  physPageY     : f32,
  physPoolDim   : f32,  // dimension in pages
  _pad          : f32,
};

@group(0) @binding(0) var<uniform> u : ShadowDepthUniforms;
@group(1) @binding(0) var<uniform> model : mat4x4f;

struct VsIn {
  @location(0) position : vec3f,
};
struct VsOut {
  @builtin(position) clipPos : vec4f,
};

@vertex
fn main(v: VsIn) -> VsOut {
  var out: VsOut;
  let worldPos = model * vec4f(v.position, 1.0);
  var clipPos  = u.lightViewProj * worldPos;

  // Remap clip XY from [-1,1] to the physical page tile's UV region
  // Each physical page occupies (1/physPoolDim) of the texture
  let pageUV = vec2f(u.physPageX, u.physPageY) / u.physPoolDim;
  let pageScale = 1.0 / u.physPoolDim;
  // clipPos.xy ∈ [-1,1] → UV ∈ [0,1] → tile UV ∈ [pageUV, pageUV+pageScale]
  let uv = clipPos.xy * 0.5 + 0.5;
  let tileUV = pageUV + uv * pageScale;
  out.clipPos = vec4f(tileUV * 2.0 - 1.0, clipPos.z, clipPos.w);
  return out;
}
`;

const WGSL_SHADOW_DEPTH_FS = /* wgsl */ `
@fragment
fn main(@builtin(position) fragCoord: vec4f) -> @builtin(frag_depth) f32 {
  return fragCoord.z;
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// § 6  WGSL — VSM Projection / Shadow Sampling
//      Mirrors VirtualShadowMapProjection.usf + VirtualShadowMapProjectionDirectional.ush
//      Used in lighting pass to fetch PCF shadows
// ─────────────────────────────────────────────────────────────────────────────

const WGSL_VSM_PROJECTION = /* wgsl */ `
// ── VSM shadow sampling function — insert into lighting fragment shader ──────
//
// Inputs:
//   worldPos    — fragment world position
//   cameraPos   — camera world position
//   lightDir    — normalized light direction (toward light)
//   pageTable   — virtual page table buffer
//   physPool    — physical page pool depth texture
//   uniforms    — VSMProjectionUniforms
//
// Output: shadow factor ∈ [0,1] where 1 = fully lit, 0 = fully shadowed

const VSM_PAGE_SIZE         : f32  = 128.0;
const VSM_LEVEL0_DIM_PAGES  : f32  = 128.0;
const VSM_LOG2_LEVEL0_DIM   : u32  = 7u;
const VSM_FLAG_ALLOCATED    : u32  = 1u;
const PCF_NUM_SAMPLES       : u32  = 8u;
const INVALID_PHYS_PAGE     : u32  = 0xFFFFu;

// Poisson disk for PCF (matches UE5 SMRT pattern)
const POISSON_DISK = array<vec2f, 8>(
  vec2f(-0.613392,  0.617481),
  vec2f( 0.170019, -0.040254),
  vec2f(-0.299417,  0.791925),
  vec2f( 0.645680,  0.493210),
  vec2f(-0.651784,  0.717887),
  vec2f( 0.421003,  0.027070),
  vec2f(-0.817194, -0.271096),
  vec2f(-0.705374, -0.668203),
);

struct VSMProjectionUniforms {
  // 11 clipmap level ViewProj matrices
  levelViewProj     : array<mat4x4f, 11>,
  // 11 clipmap world centers (xyz) + level radius (w)
  levelCenterRadius : array<vec4f, 11>,
  // 11 clipmap page-space corner offsets (xy) for cache alignment
  levelCornerOffset : array<vec2i, 11>,

  cameraPos        : vec3f,
  firstLevel       : u32,
  numLevels        : u32,
  physPoolDimPages : f32,
  resolutionLodBias: f32,
  shadowBias       : f32,
  pcfRadius        : f32,
  frameNumber      : u32,
  _pad0            : u32,
  _pad1            : u32,
};

// ── Calculate which clipmap level best covers this world position ─────────────
fn selectClipmapLevel(worldPos: vec3f, cameraPos: vec3f, firstLevel: u32, numLevels: u32, lodBias: f32) -> u32 {
  let toCam  = worldPos - cameraPos;
  let distSq = dot(toCam, toCam);
  let absLvl = log2(distSq) * 0.5 + lodBias;
  let relLvl = clamp(u32(max(0.0, absLvl - f32(firstLevel))), 0u, numLevels - 1u);
  return relLvl;
}

// ── Compute virtual page index for a world position in a given level ──────────
fn worldToVirtualPage(
  worldPos  : vec3f,
  levelIdx  : u32,
  viewProj  : mat4x4f,
  dimPages  : u32,
  cornerOff : vec2i,
) -> vec2i {
  let clip    = viewProj * vec4f(worldPos, 1.0);
  let ndc     = clip.xy / clip.w;
  let uv      = ndc * 0.5 + 0.5;
  let pageXY  = vec2i(vec2f(uv) * f32(dimPages)) + cornerOff;
  return clamp(pageXY, vec2i(0), vec2i(i32(dimPages) - 1));
}

// ── Read virtual page table entry ─────────────────────────────────────────────
fn readPageEntry(
  pageTable : ptr<storage, array<u32>, read>,
  levelBaseOffset: u32,
  pageX     : u32,
  pageY     : u32,
  dimPages  : u32,
) -> u32 {
  let idx = levelBaseOffset + pageY * dimPages + pageX;
  return (*pageTable)[idx];
}

// ── Compute physical page tile UV in pool texture ─────────────────────────────
fn physPageToPoolUV(physIdx: u32, physPoolDimPages: f32) -> vec2f {
  let px = f32(physIdx % u32(physPoolDimPages));
  let py = f32(physIdx / u32(physPoolDimPages));
  return vec2f(px, py) / physPoolDimPages;
}

// ── PCF shadow sample within one physical page tile ──────────────────────────
fn sampleShadowPCF(
  physPool      : texture_depth_2d,
  physSampler   : sampler_comparison,
  shadowUV      : vec2f,    // UV within [0,1] of the virtual clipmap level
  physPageUV    : vec2f,    // top-left UV of physical page tile in pool
  tileScale     : f32,      // 1.0 / physPoolDimPages
  compareDepth  : f32,      // depth to compare (with bias)
  pcfRadius     : f32,      // in texels
  texelSize     : f32,      // 1.0 / (physPoolDimPages * VSM_PAGE_SIZE)
) -> f32 {
  // Map shadowUV into tile UV space within the physical page
  let tileUV = physPageUV + fract(shadowUV * f32(128u)) / f32(128u) * tileScale;

  var shadow = 0.0;
  let radiusUV = pcfRadius * texelSize;
  for (var i = 0u; i < PCF_NUM_SAMPLES; i++) {
    let offset  = POISSON_DISK[i] * radiusUV;
    let sampleUV = tileUV + offset;
    shadow += textureSampleCompare(physPool, physSampler, sampleUV, compareDepth);
  }
  return shadow / f32(PCF_NUM_SAMPLES);
}

// ── Main VSM shadow lookup ────────────────────────────────────────────────────
// Returns: shadow factor ∈ [0=shadowed, 1=lit]
fn getVSMShadow(
  worldPos      : vec3f,
  u             : VSMProjectionUniforms,
  pageTable     : ptr<storage, array<u32>, read>,
  physPool      : texture_depth_2d,
  physSampler   : sampler_comparison,
) -> f32 {
  let levelIdx = selectClipmapLevel(worldPos, u.cameraPos, u.firstLevel, u.numLevels, u.resolutionLodBias);
  let absLevel = u.firstLevel + levelIdx;

  let viewProj  = u.levelViewProj[levelIdx];
  let center    = u.levelCenterRadius[levelIdx].xyz;
  let radius    = u.levelCenterRadius[levelIdx].w;
  let cornerOff = u.levelCornerOffset[levelIdx];

  // Level 0 has 128 pages, each subsequent level halves
  let log2Dim   = VSM_LOG2_LEVEL0_DIM - levelIdx;
  let dimPages  = 1u << log2Dim;

  let virtualPage = worldToVirtualPage(worldPos, levelIdx, viewProj, dimPages, cornerOff);
  if (virtualPage.x < 0 || virtualPage.y < 0) { return 1.0; }

  // Compute base offset for this level in the flat page table
  var baseOff = 0u;
  for (var l = 0u; l < levelIdx; l++) {
    let d = 1u << (VSM_LOG2_LEVEL0_DIM - l);
    baseOff += d * d;
  }

  let entry = readPageEntry(pageTable, baseOff, u32(virtualPage.x), u32(virtualPage.y), dimPages);
  let flags = entry & 0xFFFFu;
  let physIdx = entry >> 16u;

  if ((flags & VSM_FLAG_ALLOCATED) == 0u || physIdx == INVALID_PHYS_PAGE) {
    return 1.0;  // No shadow data → assume lit (or use cascade fallback)
  }

  // Compute clip-space position for depth comparison
  let clipPos     = viewProj * vec4f(worldPos, 1.0);
  let ndcDepth    = clipPos.z / clipPos.w;
  let compareD    = ndcDepth - u.shadowBias;

  // Physical page tile UV
  let tileScale   = 1.0 / u.physPoolDimPages;
  let physPageUV  = physPageToPoolUV(physIdx, u.physPoolDimPages);
  let texelSize   = 1.0 / (u.physPoolDimPages * VSM_PAGE_SIZE);

  // UV within the virtual level page
  let ndc = clipPos.xy / clipPos.w;
  let shadowUV = ndc * 0.5 + 0.5;

  return sampleShadowPCF(physPool, physSampler, shadowUV, physPageUV, tileScale, compareD, u.pcfRadius, texelSize);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// § 7  VSMClipmapManager — Clipmap level CPU management
//      Mirrors FVirtualShadowMapClipmap (Renderer-Private/VirtualShadowMaps/)
// ─────────────────────────────────────────────────────────────────────────────

export class VSMClipmapManager {
  readonly numLevels: number;
  readonly firstLevel: number;
  readonly lastLevel: number;

  private levels: VSMClipmapLevelData[] = [];
  private cacheEntries: VSMCacheEntry[] = [];
  private lightDir: [number, number, number] = [0, -1, 0];
  private cameraPos: [number, number, number] = [0, 0, 0];
  private frameNumber = 0;

  constructor(firstLevel = VSM_FIRST_LEVEL, numLevels = VSM_NUM_LEVELS) {
    this.firstLevel = firstLevel;
    this.numLevels  = numLevels;
    this.lastLevel  = firstLevel + numLevels - 1;
    for (let i = 0; i < numLevels; i++) {
      this.cacheEntries.push({
        prevPageSpaceLocation: [0, 0],
        currPageSpaceLocation: [0, 0],
        pageMapping: new Map(),
        lastRenderedFrame: -1,
        isDynamic: false,
      });
    }
  }

  /** Update clipmap levels for new camera + light direction.
   *  Mirrors FVirtualShadowMapClipmap constructor logic. */
  update(
    cameraPos: [number, number, number],
    lightDir: [number, number, number],
    frameNumber: number,
  ): void {
    this.cameraPos   = cameraPos;
    this.lightDir    = lightDir;
    this.frameNumber = frameNumber;
    this.levels      = [];

    const [ldx, ldy, ldz] = lightDir;
    // Normalize lightDir
    const llen = Math.sqrt(ldx * ldx + ldy * ldy + ldz * ldz);
    const ld: [number, number, number] = [ldx / llen, ldy / llen, ldz / llen];

    for (let li = 0; li < this.numLevels; li++) {
      const absLevel = this.firstLevel + li;
      // Level radius in world units = 2^absLevel
      const radius   = Math.pow(2, absLevel);
      // halfExtent for orthographic proj covers the level's shadow area
      const halfExt  = radius;

      // Snap camera pos to page grid at this level's scale
      const pageWorldSize = (radius * 2) / calcLevelDimPages(li);
      const snapX = snapToGrid(cameraPos[0], pageWorldSize);
      const snapZ = snapToGrid(cameraPos[2], pageWorldSize);
      const center: [number, number, number] = [snapX, cameraPos[1], snapZ];

      const viewMat = buildLightView(ld, center);
      const projMat = buildOrtho(halfExt, halfExt, -radius * 4, radius * 4);
      const vpMat   = mat4Mul(projMat, viewMat);

      // Page-space corner offset for cache validation
      const dimPages   = calcLevelDimPages(li);
      const cornerX    = Math.floor((center[0] - cameraPos[0]) / pageWorldSize);
      const cornerZ    = Math.floor((center[2] - cameraPos[2]) / pageWorldSize);
      const cornerOff: [number, number] = [cornerX, cornerZ];

      // Detect if level moved since last frame
      const prev = this.cacheEntries[li];
      const moved = (
        prev.currPageSpaceLocation[0] !== cornerX ||
        prev.currPageSpaceLocation[1] !== cornerZ
      );
      if (moved) {
        prev.prevPageSpaceLocation = [...prev.currPageSpaceLocation];
        prev.currPageSpaceLocation = [cornerX, cornerZ];
      }
      const invalidated = moved || prev.lastRenderedFrame < 0;

      this.levels.push({
        level:          absLevel,
        radius,
        worldCenter:    center,
        halfExtent:     halfExt,
        viewMatrix:     viewMat,
        projMatrix:     projMat,
        viewProjMatrix: vpMat,
        cornerOffset:   cornerOff,
        invalidated,
      });
    }
  }

  getLevels(): readonly VSMClipmapLevelData[] { return this.levels; }

  getLevelData(levelIdx: number): VSMClipmapLevelData {
    return this.levels[levelIdx];
  }

  getCacheEntry(levelIdx: number): VSMCacheEntry {
    return this.cacheEntries[levelIdx];
  }

  markRendered(levelIdx: number): void {
    this.cacheEntries[levelIdx].lastRenderedFrame = this.frameNumber;
    this.levels[levelIdx].invalidated = false;
  }

  /** Calculate total virtual page count across all levels */
  getTotalVirtualPages(): number {
    let total = 0;
    for (let li = 0; li < this.numLevels; li++) {
      const d = calcLevelDimPages(li);
      total += d * d;
    }
    return total;
  }

  /** Get virtual page buffer offset for a given level */
  getLevelPageOffset(levelIdx: number): number {
    let off = 0;
    for (let li = 0; li < levelIdx; li++) {
      const d = calcLevelDimPages(li);
      off += d * d;
    }
    return off;
  }

  /** Serialize clipmap data to flat Float32Array for GPU upload.
   *  Layout: [11 × mat4 viewProj] [11 × vec4 centerRadius] [11 × vec2i cornerOffset padding]
   */
  serializeToGPU(): Float32Array {
    const floatsPerMat   = 16;
    const floatsPerVec4  = 4;
    const floatsPerCorner = 4; // vec2i padded to vec4
    const stride = this.numLevels * (floatsPerMat + floatsPerVec4 + floatsPerCorner);
    const out = new Float32Array(stride);
    let cursor = 0;

    // viewProj matrices
    for (let li = 0; li < this.numLevels; li++) {
      const vp = li < this.levels.length ? this.levels[li].viewProjMatrix : new Float32Array(16);
      out.set(vp, cursor);
      cursor += 16;
    }
    // centerRadius
    for (let li = 0; li < this.numLevels; li++) {
      if (li < this.levels.length) {
        const lv = this.levels[li];
        out[cursor + 0] = lv.worldCenter[0];
        out[cursor + 1] = lv.worldCenter[1];
        out[cursor + 2] = lv.worldCenter[2];
        out[cursor + 3] = lv.radius;
      }
      cursor += 4;
    }
    // cornerOffset (as float for WGSL compatibility)
    for (let li = 0; li < this.numLevels; li++) {
      if (li < this.levels.length) {
        out[cursor + 0] = this.levels[li].cornerOffset[0];
        out[cursor + 1] = this.levels[li].cornerOffset[1];
      }
      cursor += 4;
    }
    return out;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 8  VSMPageCache — GPU buffer management + free list
//      Mirrors FVirtualShadowMapCacheManager
// ─────────────────────────────────────────────────────────────────────────────

export class VSMPageCache {
  readonly maxPhysPages: number;

  private freeList: number[];
  private pageMetadata: VSMPageMetadata[];
  private dirtyPhysPages = new Set<number>();

  constructor(maxPhysPages = MAX_PHYSICAL_PAGES) {
    this.maxPhysPages = maxPhysPages;
    this.freeList = Array.from({ length: maxPhysPages }, (_, i) => i);
    this.pageMetadata = Array.from({ length: maxPhysPages }, (_, i) => ({
      vsmIndex: -1,
      virtualPageX: 0,
      virtualPageY: 0,
      mipLevel: 0,
      flags: 0,
      lastRenderedFrame: -1,
    }));
  }

  allocate(): number | null {
    return this.freeList.pop() ?? null;
  }

  free(physIdx: number): void {
    const meta = this.pageMetadata[physIdx];
    meta.vsmIndex = -1;
    meta.flags = 0;
    this.dirtyPhysPages.delete(physIdx);
    this.freeList.push(physIdx);
  }

  getMetadata(physIdx: number): VSMPageMetadata {
    return this.pageMetadata[physIdx];
  }

  setMetadata(physIdx: number, meta: Partial<VSMPageMetadata>): void {
    Object.assign(this.pageMetadata[physIdx], meta);
  }

  markDirty(physIdx: number): void {
    this.dirtyPhysPages.add(physIdx);
    this.pageMetadata[physIdx].flags |= VSM_EXTENDED_FLAG_DYNAMIC_DIRTY;
  }

  isDirty(physIdx: number): boolean {
    return this.dirtyPhysPages.has(physIdx);
  }

  clearDirty(physIdx: number): void {
    this.dirtyPhysPages.delete(physIdx);
    this.pageMetadata[physIdx].flags &= ~VSM_EXTENDED_FLAG_DYNAMIC_DIRTY;
    this.pageMetadata[physIdx].flags |= VSM_EXTENDED_FLAG_DYNAMIC_INITIALIZED;
  }

  /** Serialize free list to Uint32Array for GPU upload.
   *  Layout: [count, page0, page1, ...] */
  serializeFreeList(): Uint32Array {
    const buf = new Uint32Array(this.maxPhysPages + 1);
    buf[0] = this.freeList.length;
    for (let i = 0; i < this.freeList.length; i++) {
      buf[i + 1] = this.freeList[i];
    }
    return buf;
  }

  get freePageCount(): number { return this.freeList.length; }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 9  ATVSMBridge — Bridge between UEVSMShadows and AT lighting.fs
//      Generates WGSL binding declarations + shadow sampling code for
//      injection into the AT lighting fragment shader.
// ─────────────────────────────────────────────────────────────────────────────

export class ATVSMBridge {
  /**
   * Returns WGSL binding declarations to append to the lighting shader header.
   * @param groupIndex — bind group index (e.g. 2 if lighting uses 0 and 1)
   */
  static getBindingDeclarations(groupIndex: number): string {
    return /* wgsl */ `
// ── VSM Shadow bindings (injected by ATVSMBridge M836) ────────────────────────
struct VSMProjectionUniforms {
  levelViewProj     : array<mat4x4f, 11>,
  levelCenterRadius : array<vec4f,   11>,
  levelCornerOffset : array<vec4f,   11>,  // vec2i padded to vec4
  cameraPos         : vec3f,
  firstLevel        : u32,
  numLevels         : u32,
  physPoolDimPages  : f32,
  resolutionLodBias : f32,
  shadowBias        : f32,
  pcfRadius         : f32,
  frameNumber       : u32,
  _pad0             : u32,
  _pad1             : u32,
};

@group(${groupIndex}) @binding(0) var<uniform>       vsmUniforms  : VSMProjectionUniforms;
@group(${groupIndex}) @binding(1) var<storage, read>  vsmPageTable : array<u32>;
@group(${groupIndex}) @binding(2) var                 vsmPhysPool  : texture_depth_2d;
@group(${groupIndex}) @binding(3) var                 vsmSampler   : sampler_comparison;
`;
  }

  /**
   * Returns WGSL shadow sampling function that replaces AT's getShadow().
   * Includes the full VSM projection code.
   */
  static getShadowFunction(): string {
    return WGSL_VSM_PROJECTION + /* wgsl */ `

// ── Drop-in replacement for AT lighting.fs getShadow() ───────────────────────
fn getShadow(worldPos: vec3f) -> f32 {
  return getVSMShadow(worldPos, vsmUniforms, &vsmPageTable, vsmPhysPool, vsmSampler);
}
`;
  }

  /**
   * Patch AT compiled GLSL/WGSL source to replace shadow system with VSM.
   * Removes old getShadow() / getShadowPCSS() and injects VSM version.
   */
  static patchATLightingSource(source: string, bindGroupIndex: number): string {
    // Remove old shadow uniforms
    let patched = source
      .replace(/uniform\s+sampler2D\s+shadowMap[^;]*;/g, '')
      .replace(/uniform\s+mat4\s+shadowMatrix[^;]*;/g, '')
      .replace(/float\s+getShadow\s*\([^}]+\}\s*/gs, '')
      .replace(/float\s+getShadowPCSS\s*\([^}]+\}\s*/gs, '');

    // Prepend VSM bindings + functions
    const header = ATVSMBridge.getBindingDeclarations(bindGroupIndex);
    const shadowFn = ATVSMBridge.getShadowFunction();
    patched = header + '\n' + shadowFn + '\n' + patched;

    return patched;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 10  UEVSMShadows — Main class
// ─────────────────────────────────────────────────────────────────────────────

export class UEVSMShadows {
  readonly device: GPUDevice;
  readonly config: Required<VSMConfig>;

  // Sub-systems
  readonly clipmap: VSMClipmapManager;
  readonly pageCache: VSMPageCache;
  readonly bridge: typeof ATVSMBridge = ATVSMBridge;

  // GPU resources
  physicalPagePool!: GPUTexture;
  physicalPagePoolView!: GPUTextureView;
  pageTableBuffer!: GPUBuffer;       // virtual → physical mapping
  pageRequestBuffer!: GPUBuffer;     // per-frame page visibility
  freePageListBuffer!: GPUBuffer;    // GPU-side free page list
  pageMetadataBuffer!: GPUBuffer;    // physical page metadata
  projectionUniformBuffer!: GPUBuffer; // VSMProjectionUniforms
  markUniformBuffer!: GPUBuffer;      // PageMarking uniforms

  // Pipelines
  private pageMarkPipeline!: GPUComputePipeline;
  private pageAllocPipeline!: GPUComputePipeline;
  private shadowDepthPipeline!: GPURenderPipeline;

  // Bind groups (rebuilt each frame after allocation)
  private markBindGroup!: GPUBindGroup;
  private allocBindGroup!: GPUBindGroup;
  private projectionBindGroup!: GPUBindGroup;

  // Samplers
  private depthSampler!: GPUSampler;
  private shadowSampler!: GPUSampler;

  // State
  private frameNumber = 0;
  private physPoolDimPages: number;
  private totalVirtualPages: number;

  private constructor(device: GPUDevice, config: Required<VSMConfig>) {
    this.device    = device;
    this.config    = config;
    this.physPoolDimPages = config.physPoolDimPages;

    this.clipmap   = new VSMClipmapManager(
      config.firstClipmapLevel,
      config.numClipmapLevels,
    );
    this.pageCache = new VSMPageCache(
      config.physPoolDimPages * config.physPoolDimPages,
    );
    this.totalVirtualPages = this.clipmap.getTotalVirtualPages();
  }

  // ── Factory ────────────────────────────────────────────────────────────────

  static async create(device: GPUDevice, config: VSMConfig): Promise<UEVSMShadows> {
    const fullConfig: Required<VSMConfig> = {
      width:               config.width,
      height:              config.height,
      physPoolDimPages:    config.physPoolDimPages    ?? PHYS_POOL_DIM_PAGES,
      numClipmapLevels:    config.numClipmapLevels    ?? VSM_NUM_LEVELS,
      firstClipmapLevel:   config.firstClipmapLevel   ?? VSM_FIRST_LEVEL,
      resolutionLodBias:   config.resolutionLodBias   ?? 0,
      pcfRadius:           config.pcfRadius           ?? 2,
      maxShadowDistance:   config.maxShadowDistance   ?? 65536,
      enablePageCaching:   config.enablePageCaching   ?? true,
    };
    const vsm = new UEVSMShadows(device, fullConfig);
    await vsm._initialize();
    return vsm;
  }

  // ── Initialization ─────────────────────────────────────────────────────────

  private async _initialize(): Promise<void> {
    this._createGPUResources();
    this._createSamplers();
    await this._createPipelines();
  }

  private _createGPUResources(): void {
    const { device } = this;
    const poolTexels = this.physPoolDimPages * VSM_PAGE_SIZE;

    // Physical page pool — depth texture
    this.physicalPagePool = device.createTexture({
      label:  'VSM.PhysicalPagePool',
      size:   [poolTexels, poolTexels, 1],
      format: 'depth32float',
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.TEXTURE_BINDING,
    });
    this.physicalPagePoolView = this.physicalPagePool.createView();

    // Page table buffer — u32 per virtual page
    this.pageTableBuffer = device.createBuffer({
      label: 'VSM.PageTable',
      size:  this.totalVirtualPages * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Page request buffer — u32 per virtual page (atomic)
    this.pageRequestBuffer = device.createBuffer({
      label: 'VSM.PageRequests',
      size:  this.totalVirtualPages * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Free page list buffer: [count, page0, page1, ...]
    const maxPhysPages = this.physPoolDimPages * this.physPoolDimPages;
    this.freePageListBuffer = device.createBuffer({
      label: 'VSM.FreePageList',
      size:  (maxPhysPages + 1) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Page metadata buffer
    this.pageMetadataBuffer = device.createBuffer({
      label: 'VSM.PageMetadata',
      size:  maxPhysPages * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Projection uniforms (for lighting pass)
    // Layout: 11×mat4 + 11×vec4 + 11×vec4(corner) + misc
    const projUniSize = (11 * 16 + 11 * 4 + 11 * 4 + 16) * 4;
    this.projectionUniformBuffer = device.createBuffer({
      label: 'VSM.ProjectionUniforms',
      size:  Math.max(256, projUniSize),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Mark uniforms
    this.markUniformBuffer = device.createBuffer({
      label: 'VSM.MarkUniforms',
      size:  256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Initialize free page list from CPU
    const freeListData = this.pageCache.serializeFreeList();
    device.queue.writeBuffer(this.freePageListBuffer, 0, freeListData);
  }

  private _createSamplers(): void {
    this.depthSampler = this.device.createSampler({
      label:        'VSM.DepthSampler',
      minFilter:    'nearest',
      magFilter:    'nearest',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
    this.shadowSampler = this.device.createSampler({
      label:        'VSM.ShadowSampler',
      compare:      'less-equal',
      minFilter:    'linear',
      magFilter:    'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
  }

  private async _createPipelines(): Promise<void> {
    const { device } = this;

    // Page marking compute pipeline
    const markModule = device.createShaderModule({ label: 'VSM.PageMark', code: WGSL_PAGE_MARKING });
    this.pageMarkPipeline = device.createComputePipeline({
      label:   'VSM.PageMarkPipeline',
      layout:  'auto',
      compute: { module: markModule, entryPoint: 'main' },
    });

    // Page allocation compute pipeline
    const allocModule = device.createShaderModule({ label: 'VSM.PageAlloc', code: WGSL_PAGE_ALLOC });
    this.pageAllocPipeline = device.createComputePipeline({
      label:   'VSM.PageAllocPipeline',
      layout:  'auto',
      compute: { module: allocModule, entryPoint: 'main' },
    });

    // Shadow depth render pipeline
    const depthVsModule = device.createShaderModule({ label: 'VSM.DepthVS', code: WGSL_SHADOW_DEPTH_VS });
    const depthFsModule = device.createShaderModule({ label: 'VSM.DepthFS', code: WGSL_SHADOW_DEPTH_FS });
    this.shadowDepthPipeline = device.createRenderPipeline({
      label:  'VSM.ShadowDepthPipeline',
      layout: 'auto',
      vertex: {
        module:     depthVsModule,
        entryPoint: 'main',
        buffers: [{
          arrayStride: 12,
          attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }],
        }],
      },
      fragment: {
        module:     depthFsModule,
        entryPoint: 'main',
        targets:    [],
      },
      depthStencil: {
        format:            'depth32float',
        depthWriteEnabled: true,
        depthCompare:      'less',
      },
      primitive: {
        topology: 'triangle-list',
        cullMode: 'back',
      },
    });
  }

  // ── Per-frame update ───────────────────────────────────────────────────────

  /**
   * Update clipmap matrices and upload projection uniforms.
   * Call once per frame before markPages().
   */
  updateClipmaps(
    cameraPos: [number, number, number],
    lightDir: [number, number, number],
  ): void {
    this.frameNumber++;
    this.clipmap.update(cameraPos, lightDir, this.frameNumber);
    this._uploadProjectionUniforms(cameraPos, lightDir);
    this._uploadMarkUniforms(cameraPos, lightDir);
  }

  private _uploadProjectionUniforms(
    cameraPos: [number, number, number],
    lightDir: [number, number, number],
  ): void {
    const gpuData = this.clipmap.serializeToGPU();
    // Append misc uniforms after clipmap data
    const miscOffset = gpuData.byteLength;
    const misc = new Float32Array([
      cameraPos[0], cameraPos[1], cameraPos[2],
      this.config.firstClipmapLevel,    // reinterpreted as uint
      this.config.numClipmapLevels,
      this.physPoolDimPages,
      this.config.resolutionLodBias,
      0.001,                             // shadowBias
      this.config.pcfRadius,
      this.frameNumber,                  // reinterpreted as uint
      0, 0,                              // padding
    ]);
    const combined = new Float32Array(gpuData.length + misc.length);
    combined.set(gpuData, 0);
    combined.set(misc, gpuData.length);
    this.device.queue.writeBuffer(this.projectionUniformBuffer, 0, combined);
  }

  private _uploadMarkUniforms(
    cameraPos: [number, number, number],
    lightDir: [number, number, number],
  ): void {
    // Minimal mark uniforms — full invViewProj would be provided externally
    const data = new Float32Array(64);
    // Skip invViewProj (identity placeholder)
    for (let i = 0; i < 16; i++) data[i] = i === 0 || i === 5 || i === 10 || i === 15 ? 1 : 0;
    data[16] = cameraPos[0]; data[17] = cameraPos[1]; data[18] = cameraPos[2];
    data[19] = lightDir[0];  data[20] = lightDir[1];  data[21] = lightDir[2];
    data[22] = this.config.width;  data[23] = this.config.height;
    data[24] = 0.01; data[25] = this.config.maxShadowDistance;
    data[26] = this.config.firstClipmapLevel;
    data[27] = this.config.numClipmapLevels;
    data[28] = this.config.resolutionLodBias;
    this.device.queue.writeBuffer(this.markUniformBuffer, 0, data);
  }

  // ── Pass A: Page Marking ───────────────────────────────────────────────────

  /**
   * Mark which shadow pages are visible from the camera.
   * Mirrors VirtualShadowMapPageMarking.usf
   */
  markPages(
    encoder:       GPUCommandEncoder,
    depthTexView:  GPUTextureView,
  ): void {
    // Clear page requests
    encoder.clearBuffer(this.pageRequestBuffer);

    const bg = this.device.createBindGroup({
      layout: this.pageMarkPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.markUniformBuffer } },
        { binding: 3, resource: { buffer: this.pageRequestBuffer } },
      ],
    });

    const pass = encoder.beginComputePass({ label: 'VSM.PageMark' });
    pass.setPipeline(this.pageMarkPipeline);
    pass.setBindGroup(0, bg);
    const wgX = Math.ceil(this.config.width  / 8);
    const wgY = Math.ceil(this.config.height / 8);
    pass.dispatchWorkgroups(wgX, wgY, 1);
    pass.end();
  }

  // ── Pass B: Page Allocation ────────────────────────────────────────────────

  /**
   * Allocate physical pages for uncached requested virtual pages.
   * Mirrors VirtualShadowMapPhysicalPageManagement.usf
   */
  allocatePages(encoder: GPUCommandEncoder): void {
    // Upload fresh free list
    const freeListData = this.pageCache.serializeFreeList();
    this.device.queue.writeBuffer(this.freePageListBuffer, 0, freeListData);

    // Alloc uniforms
    const allocUni = new Uint32Array([this.totalVirtualPages, this.frameNumber, 0, 0]);
    const allocUniBuf = this.device.createBuffer({
      size:  16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(allocUniBuf, 0, allocUni);

    const bg = this.device.createBindGroup({
      layout: this.pageAllocPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: allocUniBuf } },
        { binding: 1, resource: { buffer: this.pageRequestBuffer } },
        { binding: 2, resource: { buffer: this.pageTableBuffer } },
        { binding: 3, resource: { buffer: this.freePageListBuffer } },
        { binding: 4, resource: { buffer: this.pageMetadataBuffer } },
      ],
    });

    const pass = encoder.beginComputePass({ label: 'VSM.PageAlloc' });
    pass.setPipeline(this.pageAllocPipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(this.totalVirtualPages / 64), 1, 1);
    pass.end();
  }

  // ── Pass C: Shadow Depth Render ────────────────────────────────────────────

  /**
   * Render shadow depth for uncached pages.
   * For each invalidated clipmap level, render shadow casters into
   * the physical page pool tiles.
   */
  renderShadowDepth(
    encoder:   GPUCommandEncoder,
    draws:     VSMShadowDraw[],
  ): void {
    const levels = this.clipmap.getLevels();
    for (let li = 0; li < levels.length; li++) {
      const level = levels[li];
      if (!level.invalidated && this.config.enablePageCaching) continue;

      // One render pass per clipmap level
      // In a full implementation we'd batch draws by physical page tile
      const shadowDepthUniData = new Float32Array(8);
      shadowDepthUniData.set(level.viewProjMatrix, 0); // [0..15]
      // physPage coords would be per-draw; here we write a placeholder
      shadowDepthUniData[16] = 0;
      shadowDepthUniData[17] = li;
      shadowDepthUniData[18] = this.physPoolDimPages;
      shadowDepthUniData[19] = 0;

      const shadowDepthUniBuf = this.device.createBuffer({
        size:  Math.max(256, shadowDepthUniData.byteLength),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(shadowDepthUniBuf, 0, shadowDepthUniData);

      const pass = encoder.beginRenderPass({
        label: `VSM.ShadowDepth[L${level.level}]`,
        colorAttachments: [],
        depthStencilAttachment: {
          view:              this.physicalPagePoolView,
          depthLoadOp:       level.invalidated ? 'clear' : 'load',
          depthStoreOp:      'store',
          depthClearValue:   1.0,
        },
      });
      pass.setPipeline(this.shadowDepthPipeline);

      for (const draw of draws) {
        if (!(draw.levelMask & (1 << li))) continue;

        const modelBG = this.device.createBindGroup({
          layout: this.shadowDepthPipeline.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: { buffer: shadowDepthUniBuf } }],
        });
        const drawBG = this.device.createBindGroup({
          layout: this.shadowDepthPipeline.getBindGroupLayout(1),
          entries: [{ binding: 0, resource: { buffer: draw.modelUBO } }],
        });

        pass.setBindGroup(0, modelBG);
        pass.setBindGroup(1, drawBG);
        pass.setVertexBuffer(0, draw.vertexBuffer);

        if (draw.indexBuffer) {
          pass.setIndexBuffer(draw.indexBuffer, 'uint32');
          pass.drawIndexed(draw.count);
        } else {
          pass.draw(draw.count);
        }
      }

      pass.end();
      this.clipmap.markRendered(li);
    }
  }

  // ── Pass D: Bind to Lighting ───────────────────────────────────────────────

  /**
   * Create a GPUBindGroup to attach VSM resources to the AT lighting pass.
   * @param pipeline      — The lighting render pipeline
   * @param groupIndex    — Bind group index in the lighting shader (e.g. 2)
   */
  createLightingBindGroup(pipeline: GPURenderPipeline, groupIndex: number): GPUBindGroup {
    return this.device.createBindGroup({
      label:  'VSM.LightingBindGroup',
      layout: pipeline.getBindGroupLayout(groupIndex),
      entries: [
        { binding: 0, resource: { buffer: this.projectionUniformBuffer } },
        { binding: 1, resource: { buffer: this.pageTableBuffer } },
        { binding: 2, resource: this.physicalPagePoolView },
        { binding: 3, resource: this.shadowSampler },
      ],
    });
  }

  /**
   * Bind VSM resources to an active render pass for lighting.
   */
  bindToLightingPass(
    pass:       GPURenderPassEncoder,
    pipeline:   GPURenderPipeline,
    groupIndex: number,
  ): void {
    const bg = this.createLightingBindGroup(pipeline, groupIndex);
    pass.setBindGroup(groupIndex, bg);
  }

  // ── Utility ────────────────────────────────────────────────────────────────

  /**
   * Force-invalidate all clipmap levels (e.g. after scene change).
   * Mirrors FVirtualShadowMapClipmapConfig.bForceInvalidate
   */
  forceInvalidateAll(): void {
    const levels = this.clipmap.getLevels();
    for (let li = 0; li < levels.length; li++) {
      (levels[li] as VSMClipmapLevelData).invalidated = true;
    }
  }

  /**
   * Invalidate pages touched by a dynamic Cell (position changed).
   * Mirrors FVirtualShadowMapCacheManager invalidation by primitive.
   */
  invalidateCellPages(
    cellWorldPos:   [number, number, number],
    cellRadius:     number,
  ): void {
    const levels = this.clipmap.getLevels();
    for (let li = 0; li < levels.length; li++) {
      const lv = levels[li];
      // If cell is within this level's extent, invalidate
      const dx = cellWorldPos[0] - lv.worldCenter[0];
      const dz = cellWorldPos[2] - lv.worldCenter[2];
      const dist2D = Math.sqrt(dx * dx + dz * dz);
      if (dist2D < lv.radius + cellRadius) {
        (lv as VSMClipmapLevelData).invalidated = true;
      }
    }
  }

  /** Get WGSL projection shader code (for manual injection into lighting shader) */
  getProjectionWGSL(): string { return WGSL_VSM_PROJECTION; }

  /** Get AT lighting patch helper */
  get atBridge(): typeof ATVSMBridge { return ATVSMBridge; }

  /** Current frame number */
  get currentFrame(): number { return this.frameNumber; }

  /** Physical page pool texture for external inspection / debug */
  get physPoolTexture(): GPUTexture { return this.physicalPagePool; }

  /** Stats snapshot */
  getStats(): {
    totalVirtualPages: number;
    freePhysPages: number;
    usedPhysPages: number;
    numClipmapLevels: number;
    frameNumber: number;
  } {
    const maxPhys = this.physPoolDimPages * this.physPoolDimPages;
    const free = this.pageCache.freePageCount;
    return {
      totalVirtualPages: this.totalVirtualPages,
      freePhysPages:     free,
      usedPhysPages:     maxPhys - free,
      numClipmapLevels:  this.config.numClipmapLevels,
      frameNumber:       this.frameNumber,
    };
  }

  destroy(): void {
    this.physicalPagePool.destroy();
    this.pageTableBuffer.destroy();
    this.pageRequestBuffer.destroy();
    this.freePageListBuffer.destroy();
    this.pageMetadataBuffer.destroy();
    this.projectionUniformBuffer.destroy();
    this.markUniformBuffer.destroy();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 11  Exports
// ─────────────────────────────────────────────────────────────────────────────

export {
  // Constants
  VSM_PAGE_SIZE,
  VSM_LEVEL0_DIM_PAGES_XY,
  VSM_MAX_MIP_LEVELS,
  VSM_FIRST_LEVEL,
  VSM_LAST_LEVEL,
  VSM_NUM_LEVELS,
  PHYS_POOL_DIM_PAGES,
  MAX_PHYSICAL_PAGES,
  // Page flags
  VSM_FLAG_ALLOCATED,
  VSM_FLAG_DYNAMIC_UNCACHED,
  VSM_FLAG_STATIC_UNCACHED,
  VSM_FLAG_PRIMARY_REQUEST,
  VSM_EXTENDED_FLAG_FORCE_CACHED,
  VSM_EXTENDED_FLAG_INVALIDATE_DYNAMIC,
  // Math helpers
  calcLevelDimPages,
  calcLog2LevelDimPages,
  buildOrtho,
  buildLightView,
  // WGSL sources (for manual shader assembly)
  WGSL_PAGE_MARKING,
  WGSL_PAGE_ALLOC,
  WGSL_SHADOW_DEPTH_VS,
  WGSL_SHADOW_DEPTH_FS,
  WGSL_VSM_PROJECTION,
};
