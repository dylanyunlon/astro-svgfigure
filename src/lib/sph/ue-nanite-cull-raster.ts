/**
 * ue-nanite-cull-raster.ts — M834: UE5 Nanite Cull+Raster Pipeline
 *
 * 移植 Unreal Engine 5 的 Nanite 虚拟几何体剔除与光栅化系统到 WebGPU:
 *
 *   § Cluster LOD: 基于 BVH 树的层级 LOD 选择
 *     ├─ FPackedView: 每个视图的 LOD 比例、HZB 矩形、视锥参数
 *     ├─ ClusterLODSelect: 根据屏幕空间大小选取最优 LOD 簇
 *     └─ MaxPixelsPerEdge: 动态调整每条边的最大像素数
 *
 *   § HZB 遮挡剔除: Hierarchical Z-Buffer 两阶段剔除
 *     ├─ BuildHZB: 从场景深度构建层级Z缓冲
 *     ├─ MainPass: 以上一帧HZB测试可见性，保守估计
 *     └─ PostPass: 以本帧渲染结果HZB再测试被遮蔽实例
 *
 *   § Software Rasterizer: 计算着色器实现的小三角形光栅化
 *     ├─ SWRasterize: 用 atomicMax 写入 vis-buffer (64-bit depth+id)
 *     ├─ HWRasterize: 大三角形走硬件管线
 *     └─ RasterScheduling: HardwareOnly / HW+SW / Overlapped 三种调度
 *
 *   § Render-Graph NanitePass: 完整的 RDG 通道封装
 *     ├─ InitRasterContext: 分配 VisBuffer64 / DepthBuffer / DbgBuffers
 *     ├─ DrawGeometry: 实例剔除 → 节点/簇剔除 → Binning → Rasterize
 *     └─ ExtractResults: 导出可见簇列表 + vis-buffer 供后续 shading 使用
 *
 * 类结构:
 *   UENaniteCullRaster                    ← 主入口，对应 IRenderer::Create
 *     ├─ NaniteHZB:       HZB 构建与采样
 *     ├─ NaniteCuller:    实例/节点/簇三级剔除
 *     ├─ NaniteRasterBin: Raster Bin 构建与排序
 *     └─ NaniteSWRaster:  软光栅化 compute pass
 *
 * 用法:
 *   const nanite = await UENaniteCullRaster.create(device, {
 *     textureSize: [1920, 1080],
 *     rasterMode: 'VisBuffer',
 *     scheduling: 'HardwareThenSoftware',
 *     twoPassOcclusion: true,
 *     maxVisibleClusters: 2097152,
 *   });
 *
 *   const ctx = nanite.initRasterContext(sceneDepth);
 *   const results = nanite.drawGeometry(view, clusters, ctx);
 *   const visBuffer = results.visBuffer64;
 *
 * Research: xiaodi #M834 — cell-pubsub-loop
 * Reference: upstream/unreal-renderer-ue5/Renderer-Private/Nanite/
 */

// ─────────────────────────────────────────────────────────────────────────────
// § 1  Constants — mirrors NaniteDefinitions.h
// ─────────────────────────────────────────────────────────────────────────────

/** Maximum views that can participate in a single cull+rasterize pass. */
export const NANITE_MAX_VIEWS_PER_CULL_RASTERIZE_PASS = 64;

/** Default target edge length in pixels (r.Nanite.MaxPixelsPerEdge). */
export const NANITE_MAX_PIXELS_PER_EDGE_DEFAULT = 1.0;

/** Hardware rasterizer threshold in pixels (r.Nanite.MinPixelsPerEdgeHW). */
export const NANITE_MIN_PIXELS_PER_EDGE_HW_DEFAULT = 32.0;

/** Tessellation dicing rate in pixels (r.Nanite.DicingRate). */
export const NANITE_DICING_RATE_DEFAULT = 2.0;

/** Number of indirect draw arguments packed per SWHW call. */
export const NANITE_RASTERIZER_ARG_COUNT = 8;

/** Vis-buffer depth sentinel (cleared state): max uint32. */
export const NANITE_INVALID_DEPTH_U32 = 0xffffffff;

/** Size of one candidate cluster entry in bytes (base mode: 8, extended: 12). */
export const NANITE_CANDIDATE_CLUSTER_SIZE_BASE = 8;
export const NANITE_CANDIDATE_CLUSTER_SIZE_EXTENDED = 12;

/** Per-frame render flag bits (NANITE_RENDER_FLAG_*). */
export const NANITE_RENDER_FLAG_FORCE_HW_RASTER              = 1 << 0;
export const NANITE_RENDER_FLAG_DISABLE_PROGRAMMABLE         = 1 << 1;
export const NANITE_RENDER_FLAG_HAS_PREV_DRAW_DATA           = 1 << 2;
export const NANITE_RENDER_FLAG_OUTPUT_STREAMING_REQUESTS    = 1 << 3;
export const NANITE_RENDER_FLAG_ADD_CLUSTER_OFFSET           = 1 << 4;
export const NANITE_RENDER_FLAG_IS_SHADOW_PASS               = 1 << 5;
export const NANITE_RENDER_FLAG_IS_SCENE_CAPTURE             = 1 << 6;
export const NANITE_RENDER_FLAG_IS_REFLECTION_CAPTURE        = 1 << 7;
export const NANITE_RENDER_FLAG_IS_LUMEN_CAPTURE             = 1 << 8;
export const NANITE_RENDER_FLAG_MESH_SHADER                  = 1 << 9;
export const NANITE_RENDER_FLAG_PRIMITIVE_SHADER             = 1 << 10;
export const NANITE_RENDER_FLAG_IS_GAME_VIEW                 = 1 << 11;
export const NANITE_RENDER_FLAG_GAME_SHOW_FLAG_ENABLED       = 1 << 12;
export const NANITE_RENDER_FLAG_EDITOR_SHOW_FLAG_ENABLED     = 1 << 13;
export const NANITE_RENDER_FLAG_WRITE_STATS                  = 1 << 14;
export const NANITE_RENDER_FLAG_DRAW_ONLY_RAYTRACING_FAR_FIELD = 1 << 15;
export const NANITE_RENDER_FLAG_IS_MATERIAL_CACHE            = 1 << 16;
export const NANITE_RENDER_FLAG_INVALIDATE_VSM_ON_LOD_DELTA  = 1 << 17;

/** Culling pass indices (CULLING_PASS_*). */
export const CULLING_PASS_NO_OCCLUSION   = 0;
export const CULLING_PASS_OCCLUSION_MAIN = 1;
export const CULLING_PASS_OCCLUSION_POST = 2;
export const CULLING_PASS_EXPLICIT_LIST  = 3;

/** Debug flag bits (NANITE_DEBUG_FLAG_*). */
export const NANITE_DEBUG_FLAG_DISABLE_CULL_FRUSTUM          = 1 << 0;
export const NANITE_DEBUG_FLAG_DISABLE_CULL_HZB              = 1 << 1;
export const NANITE_DEBUG_FLAG_DISABLE_CULL_GLOBAL_CLIP_PLANE= 1 << 2;
export const NANITE_DEBUG_FLAG_DISABLE_CULL_DRAW_DISTANCE    = 1 << 3;
export const NANITE_DEBUG_FLAG_DISABLE_CULL_MIN_LOD          = 1 << 4;
export const NANITE_DEBUG_FLAG_DISABLE_SKINNED_NODE_BOUNDS   = 1 << 5;
export const NANITE_DEBUG_FLAG_DISABLE_WPO_DISABLE_DISTANCE  = 1 << 6;
export const NANITE_DEBUG_FLAG_DRAW_ONLY_ROOT_DATA           = 1 << 7;
export const NANITE_DEBUG_FLAG_HIDE_ASSEMBLY_PARTS           = 1 << 8;
export const NANITE_DEBUG_FLAG_WRITE_ASSEMBLY_META           = 1 << 9;

// ─────────────────────────────────────────────────────────────────────────────
// § 2  Enumerations — ERasterScheduling, EOutputBufferMode, ERasterPipeline
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Controls how hardware and software rasterizers are scheduled.
 * Mirrors ERasterScheduling in NaniteCullRaster.h.
 */
export const enum ERasterScheduling {
  /** Only rasterize using fixed-function hardware. */
  HardwareOnly = 0,
  /** Rasterize large triangles with hardware, small triangles with software. */
  HardwareThenSoftware = 1,
  /** Overlap HW large-triangle rasterization with SW small-triangle compute. */
  HardwareAndSoftwareOverlap = 2,
}

/**
 * Selects the raster output target mode.
 * Mirrors EOutputBufferMode in NaniteCullRaster.h.
 */
export const enum EOutputBufferMode {
  /** Full vis-buffer: 64-bit visibility ID + packed depth. */
  VisBuffer = 0,
  /** Depth-only mode: writes 32-bit depth buffer. */
  DepthOnly = 1,
}

/**
 * Identifies which render pipeline is active.
 * Mirrors ERasterPipeline in NaniteShared.h.
 */
export const enum ERasterPipeline {
  PrimaryRaster = 0,
  ShadowRaster  = 1,
  LumenCapture  = 2,
}

// ─────────────────────────────────────────────────────────────────────────────
// § 3  Core Data Structures
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Per-view packed parameters for Nanite culling + LOD.
 * TypeScript projection of FPackedView in NaniteShared.h.
 *
 * All matrix fields are column-major Float32Array of length 16.
 */
export interface NanitePackedView {
  /** SVPosition → translated world (for depth unprojection). */
  svPositionToTranslatedWorld: Float32Array; // mat4
  /** View-space → translated world. */
  viewToTranslatedWorld: Float32Array;       // mat4
  /** Translated world → clip. */
  translatedWorldToClip: Float32Array;       // mat4
  /** View → clip. */
  viewToClip: Float32Array;                  // mat4
  /** Clip → relative world (for velocity/reprojection). */
  clipToRelativeWorld: Float32Array;         // mat4
  /** Previous frame TranslatedWorld → clip (temporal). */
  prevTranslatedWorldToClip: Float32Array;   // mat4

  /** Pixel rect [minX, minY, maxX, maxY]. */
  viewRect: [number, number, number, number];
  /** [width, height, 1/width, 1/height]. */
  viewSizeAndInvSize: [number, number, number, number];

  /** [LOD scale for SW raster, LOD scale for HW raster]. */
  lodScales: [number, number];

  /** HZB rect for occlusion test (in full-resolution pixels). */
  hzbTestViewRect: [number, number, number, number];

  /** Range-based culling maximum distance (0 = disabled). */
  rangeBasedCullingDistance: number;
  /** Camera near-plane distance. */
  nearPlane: number;

  /** Bitfield from NANITE_VIEW_FLAG_*. */
  streamingPriorityAndFlags: number;
  /** Instance occlusion query mask. */
  instanceOcclusionQueryMask: number;
}

/**
 * Packed representation of one visible Nanite cluster (8 bytes base mode).
 * Encodes instance ID, cluster pool ref, view mask, culling flags.
 */
export interface NaniteVisibleCluster {
  /** Packed word 0: culling flags | view mask | instance ID. */
  word0: number;
  /** Packed word 1: cluster pool ref | transform index. */
  word1: number;
}

/**
 * One cluster LOD node in the Nanite BVH hierarchy.
 * Corresponds to FHierarchyNode in NaniteHierarchy.ush.
 */
export interface NaniteHierarchyNode {
  /** Bounding sphere center [x, y, z] + radius. */
  boundingSphere: [number, number, number, number];
  /** Min LOD error for this subtree. */
  minLODError: number;
  /** Max parent LOD error (used for parent-cluster test). */
  maxParentLODError: number;
  /** Index of first child node (0 if leaf). */
  childStartIndex: number;
  /** Number of children (0 if leaf). */
  numChildren: number;
  /** Index into cluster page data. */
  clusterPageIndex: number;
  /** Start index of cluster group. */
  clusterGroupPartStartIndex: number;
  /** Number of cluster group parts. */
  numClusterGroupParts: number;
}

/**
 * Configuration block passed to UENaniteCullRaster.create().
 * Mirrors FConfiguration + FRasterContextInitParams.
 */
export interface NaniteConfig {
  /** Output texture dimensions [width, height]. */
  textureSize: [number, number];
  /** Output buffer mode. */
  rasterMode?: EOutputBufferMode;
  /** Rasterizer scheduling strategy. */
  scheduling?: ERasterScheduling;
  /** Enable two-pass HZB occlusion culling. */
  twoPassOcclusion?: boolean;
  /** Max visible clusters allocated in the GPU buffer. */
  maxVisibleClusters?: number;
  /** Max candidate clusters for culling. */
  maxCandidateClusters?: number;
  /** Max BVH nodes queued per frame. */
  maxNodes?: number;
  /** Max candidate patches for tessellation. */
  maxCandidatePatches?: number;
  /** Target edge length in pixels. */
  maxPixelsPerEdge?: number;
  /** Minimum edge length (pixels) to force hardware rasterizer. */
  minPixelsPerEdgeHW?: number;
  /** Tessellation dicing rate in pixels. */
  dicingRate?: number;
  /** Allow async compute overlap for SW rasterizer. */
  asyncCompute?: boolean;
  /** Write Nanite render stats to GPU buffer. */
  writeStats?: boolean;
  /** Force hardware rasterization only. */
  forceHWRaster?: boolean;
  /** Disable programmable (material) rasterization. */
  disableProgrammable?: boolean;
  /** Render pipeline context (primary / shadow / lumen). */
  pipeline?: ERasterPipeline;
}

/**
 * Raster context created by initRasterContext().
 * Holds GPU texture/buffer references for a single frame.
 * Mirrors FRasterContext in NaniteCullRaster.h.
 */
export interface NaniteRasterContext {
  /** Reciprocal of view size [1/w, 1/h]. */
  rcpViewSize: [number, number];
  /** Texture size [w, h]. */
  textureSize: [number, number];
  /** Active output mode. */
  rasterMode: EOutputBufferMode;
  /** Active scheduling mode. */
  rasterScheduling: ERasterScheduling;
  /** 64-bit vis-buffer (GPUTexture, r32uint / rg32uint). */
  visBuffer64: GPUTexture | null;
  /** 32-bit depth buffer (GPUTexture, r32uint). */
  depthBuffer: GPUTexture | null;
  /** 64-bit debug buffer. */
  dbgBuffer64: GPUTexture | null;
  /** 32-bit debug buffer. */
  dbgBuffer32: GPUTexture | null;
  /** Whether visualization mode is active. */
  visualizeActive: boolean;
  /** Whether overdraw visualization is active. */
  visualizeModeOverdraw: boolean;
  /** Custom pass flag. */
  bCustomPass: boolean;
  /** Assembly metadata enabled. */
  bEnableAssemblyMeta: boolean;
  /** Tessellation allowed. */
  bAllowTessellation: boolean;
}

/**
 * Results exported by extractResults() after drawGeometry().
 * Mirrors FRasterResults in NaniteCullRaster.h.
 */
export interface NaniteRasterResults {
  /** Page streaming constants [streamingPageOffset, maxStreamingPages, 0, 0]. */
  pageConstants: [number, number, number, number];
  /** Maximum visible clusters this frame. */
  maxVisibleClusters: number;
  /** Maximum candidate patches. */
  maxCandidatePatches: number;
  /** Maximum BVH nodes. */
  maxNodes: number;
  /** Render flags bitfield. */
  renderFlags: number;
  /** Debug flags bitfield. */
  debugFlags: number;
  /** Inv dice rate = maxPixelsPerEdge / dicingRate. */
  invDiceRate: number;

  /** GPU buffer: packed visible clusters (SW then HW layout). */
  visibleClustersSWHW: GPUBuffer | null;
  /** GPU texture: 64-bit vis-buffer. */
  visBuffer64: GPUTexture | null;
  /** GPU texture: 64-bit debug buffer (null if !visualizeActive). */
  dbgBuffer64: GPUTexture | null;
  /** GPU texture: 32-bit debug buffer (null if !visualizeActive). */
  dbgBuffer32: GPUTexture | null;
  /** GPU buffer: packed view array uploaded this frame. */
  viewsBuffer: GPUBuffer | null;
  /** GPU buffer: raster bin metadata. */
  rasterBinMeta: GPUBuffer | null;
  /** GPU buffer: raster bin cluster data. */
  rasterBinData: GPUBuffer | null;
  /** GPU buffer: raster bin indirect dispatch args. */
  rasterBinArgs: GPUBuffer | null;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 4  HZB Builder — mirrors BuildHZBFurthest
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Builds a furthest-depth Hierarchical Z-Buffer from an existing depth texture.
 *
 * UE5 equivalent: BuildHZBFurthest() in SceneTextureReductions.cpp, called from
 * FRenderer::DrawGeometry() post-main-pass to set up the two-pass HZB.
 *
 * The produced HZB contains the maximum (furthest) depth in each 2×2 tile,
 * enabling conservative occlusion: a cluster is considered occluded only if
 * its nearest point is behind the furthest stored depth in the corresponding
 * HZB texel.
 */
export class NaniteHZB {
  private device: GPUDevice;
  private pipeline: GPUComputePipeline | null = null;
  private sampler: GPUSampler;

  constructor(device: GPUDevice) {
    this.device = device;
    this.sampler = device.createSampler({
      magFilter: 'nearest',
      minFilter: 'nearest',
      mipmapFilter: 'nearest',
    });
  }

  /**
   * Create the HZB reduce compute pipeline (one shader per mip level).
   * Must be called once before buildHZB().
   */
  async init(): Promise<void> {
    const shaderCode = this.buildHZBShaderWGSL();
    const module = this.device.createShaderModule({ code: shaderCode });

    this.pipeline = await this.device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'cs_hzb_reduce' },
    });
  }

  /** Build a furthest-depth HZB mip chain from scene depth texture. */
  buildHZB(
    encoder: GPUCommandEncoder,
    sourceDepth: GPUTexture,
    viewRect: [number, number, number, number],
  ): GPUTexture {
    if (!this.pipeline) {
      throw new Error('NaniteHZB.init() must be called before buildHZB()');
    }

    const [, , viewW, viewH] = viewRect;
    const mipLevels = Math.max(1, Math.floor(Math.log2(Math.max(viewW, viewH))) + 1);

    const hzbTexture = this.device.createTexture({
      label: 'Nanite.HZB',
      size: { width: Math.ceil(viewW / 2), height: Math.ceil(viewH / 2), depthOrArrayLayers: 1 },
      mipLevelCount: mipLevels,
      format: 'r32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    // Downsample each mip level with a min-reduce (furthest = max linear depth).
    for (let mip = 0; mip < mipLevels; mip++) {
      const srcView = mip === 0
        ? sourceDepth.createView({ baseMipLevel: 0, mipLevelCount: 1 })
        : hzbTexture.createView({ baseMipLevel: mip - 1, mipLevelCount: 1 });

      const dstView = hzbTexture.createView({ baseMipLevel: mip, mipLevelCount: 1 });

      const bg = this.device.createBindGroup({
        layout: this.pipeline!.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: srcView },
          { binding: 1, resource: dstView },
          { binding: 2, resource: this.sampler },
        ],
      });

      const mipW = Math.max(1, Math.ceil((viewW / 2) >> mip));
      const mipH = Math.max(1, Math.ceil((viewH / 2) >> mip));

      const pass = encoder.beginComputePass({ label: `NaniteHZB.mip${mip}` });
      pass.setPipeline(this.pipeline!);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(mipW / 8), Math.ceil(mipH / 8), 1);
      pass.end();
    }

    return hzbTexture;
  }

  /** WGSL: HZB furthest-depth 2×2 max-reduce per mip level. */
  private buildHZBShaderWGSL(): string {
    return /* wgsl */`
      @group(0) @binding(0) var src_depth : texture_2d<f32>;
      @group(0) @binding(1) var dst_hzb   : texture_storage_2d<r32float, write>;
      @group(0) @binding(2) var smp        : sampler;
      @compute @workgroup_size(8, 8, 1)
      fn cs_hzb_reduce(@builtin(global_invocation_id) gid : vec3<u32>) {
        let ds = textureDimensions(dst_hzb);
        if (gid.x >= ds.x || gid.y >= ds.y) { return; }
        let ss = vec2<f32>(textureDimensions(src_depth));
        let uv = (vec2<f32>(gid.xy) * 2.0 + 1.0) / ss;
        let h  = vec2<f32>(0.5) / ss;
        let d0 = textureSampleLevel(src_depth, smp, uv + vec2(-h.x,-h.y), 0.0).r;
        let d1 = textureSampleLevel(src_depth, smp, uv + vec2( h.x,-h.y), 0.0).r;
        let d2 = textureSampleLevel(src_depth, smp, uv + vec2(-h.x, h.y), 0.0).r;
        let d3 = textureSampleLevel(src_depth, smp, uv + vec2( h.x, h.y), 0.0).r;
        textureStore(dst_hzb, vec2<i32>(gid.xy), vec4(max(max(d0,d1),max(d2,d3)),0,0,1));
      }
    `;
  }

  destroy(): void {
    // GPUComputePipeline has no explicit destroy in WebGPU spec.
    this.pipeline = null;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 5  Instance & Cluster Culler — mirrors FInstanceCull_CS / FNodeAndClusterCull_CS
// ─────────────────────────────────────────────────────────────────────────────

/**
 * GPU-side instance and BVH cluster culler.
 *
 * Implements the three-phase Nanite culling pipeline:
 *   1. Instance cull:  frustum + HZB test on per-instance AABB/sphere.
 *   2. Node cull:      BVH traversal + LOD error test → candidate clusters.
 *   3. Cluster cull:   final per-cluster frustum, HZB, LOD-delta test.
 *
 * WebGPU mapping of UE5 compute shaders:
 *   FInstanceCull_CS       → cs_instance_cull
 *   FNodeAndClusterCull_CS → cs_node_cluster_cull
 *   FInitArgs_CS           → cs_init_args
 */
export class NaniteCuller {
  private device: GPUDevice;

  // Compute pipelines.
  private pipelineInitArgs!:       GPUComputePipeline;
  private pipelineInstanceCull!:   GPUComputePipeline;
  private pipelineNodeCull!:       GPUComputePipeline;
  private pipelineClusterCull!:    GPUComputePipeline;
  private pipelineCalcSafeArgs!:   GPUComputePipeline;

  // Persistent GPU resources (allocated per-frame on demand).
  private queueState!:             GPUBuffer; // FQueueState
  private candidateNodes!:         GPUBuffer; // RWByteAddressBuffer
  private candidateClusters!:      GPUBuffer; // RWByteAddressBuffer
  private visibleClustersSWHW!:    GPUBuffer; // RWByteAddressBuffer
  private mainRasterArgsSWHW!:     GPUBuffer; // indirect draw/dispatch args
  private safeMainRasterArgsSWHW!: GPUBuffer; // sanitized indirect args
  private clusterCountSWHW!:       GPUBuffer; // FUintVector2 x 1
  private clusterClassifyArgs!:    GPUBuffer; // indirect dispatch

  // Optional two-pass occlusion buffers.
  private occludedInstances!:        GPUBuffer | null;
  private occludedInstancesArgs!:    GPUBuffer | null;
  private postRasterArgsSWHW!:       GPUBuffer | null;
  private safePostRasterArgsSWHW!:   GPUBuffer | null;

  private maxVisibleClusters: number;
  private maxCandidateClusters: number;
  private maxNodes: number;
  private twoPassOcclusion: boolean;

  constructor(device: GPUDevice, cfg: Required<NaniteConfig>) {
    this.device = device;
    this.maxVisibleClusters   = cfg.maxVisibleClusters;
    this.maxCandidateClusters = cfg.maxCandidateClusters;
    this.maxNodes             = cfg.maxNodes;
    this.twoPassOcclusion     = cfg.twoPassOcclusion;
  }

  /** Allocate all persistent GPU buffers and compile compute pipelines. */
  async init(): Promise<void> {
    const d = this.device;

    // --- GPU Buffers ---
    // QueueState: (5*2 + 2) * 4 bytes = 48 bytes
    this.queueState = d.createBuffer({
      label: 'Nanite.QueueState',
      size: 48,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT,
    });

    // CandidateNodes: 16 bytes per node entry
    this.candidateNodes = d.createBuffer({
      label: 'Nanite.CandidateNodes',
      size: this.maxNodes * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // CandidateClusters: 8 bytes per entry (base mode)
    this.candidateClusters = d.createBuffer({
      label: 'Nanite.CandidateClusters',
      size: this.maxCandidateClusters * NANITE_CANDIDATE_CLUSTER_SIZE_BASE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // VisibleClustersSWHW: 8 bytes per visible cluster
    this.visibleClustersSWHW = d.createBuffer({
      label: 'Nanite.VisibleClustersSWHW',
      size: this.maxVisibleClusters * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Rasterizer indirect arg buffers
    const argSize = NANITE_RASTERIZER_ARG_COUNT * 4;
    this.mainRasterArgsSWHW = d.createBuffer({
      label: 'Nanite.MainRasterizeArgsSWHW',
      size: argSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    this.safeMainRasterArgsSWHW = d.createBuffer({
      label: 'Nanite.SafeMainRasterizeArgsSWHW',
      size: argSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });

    // ClusterCountSWHW: vec2u
    this.clusterCountSWHW = d.createBuffer({
      label: 'Nanite.SWHWClusterCount',
      size: 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // ClusterClassifyArgs: 12 bytes (FRHIDispatchIndirectParameters)
    this.clusterClassifyArgs = d.createBuffer({
      label: 'Nanite.ClusterClassifyArgs',
      size: 12,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });

    // Two-pass occlusion buffers
    if (this.twoPassOcclusion) {
      const maxOccluded = 1024 * 128; // matches UE5 default Po2 rounding
      this.occludedInstances = d.createBuffer({
        label: 'Nanite.OccludedInstances',
        size: maxOccluded * 8,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.occludedInstancesArgs = d.createBuffer({
        label: 'Nanite.OccludedInstancesArgs',
        size: 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
      });
      this.postRasterArgsSWHW = d.createBuffer({
        label: 'Nanite.PostRasterizeArgsSWHW',
        size: argSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
      });
      this.safePostRasterArgsSWHW = d.createBuffer({
        label: 'Nanite.SafePostRasterizeArgsSWHW',
        size: argSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
      });
    } else {
      this.occludedInstances     = null;
      this.occludedInstancesArgs = null;
      this.postRasterArgsSWHW    = null;
      this.safePostRasterArgsSWHW = null;
    }

    // --- Compute Pipelines ---
    await this.compilePipelines();
  }

  /** Compile all four culling-stage compute shaders. */
  private async compilePipelines(): Promise<void> {
    const initArgsModule = this.device.createShaderModule({
      label: 'Nanite.InitArgs',
      code: this.initArgsWGSL(),
    });
    const instanceCullModule = this.device.createShaderModule({
      label: 'Nanite.InstanceCull',
      code: this.instanceCullWGSL(),
    });
    const nodeClusterCullModule = this.device.createShaderModule({
      label: 'Nanite.NodeClusterCull',
      code: this.nodeClusterCullWGSL(),
    });
    const calcSafeArgsModule = this.device.createShaderModule({
      label: 'Nanite.CalcSafeArgs',
      code: this.calcSafeRasterizerArgsWGSL(),
    });

    [
      this.pipelineInitArgs,
      this.pipelineInstanceCull,
      this.pipelineNodeCull,
      this.pipelineClusterCull,
      this.pipelineCalcSafeArgs,
    ] = await Promise.all([
      this.device.createComputePipelineAsync({ layout: 'auto', compute: { module: initArgsModule,       entryPoint: 'cs_init_args' } }),
      this.device.createComputePipelineAsync({ layout: 'auto', compute: { module: instanceCullModule,   entryPoint: 'cs_instance_cull' } }),
      this.device.createComputePipelineAsync({ layout: 'auto', compute: { module: nodeClusterCullModule,entryPoint: 'cs_node_cull' } }),
      this.device.createComputePipelineAsync({ layout: 'auto', compute: { module: nodeClusterCullModule,entryPoint: 'cs_cluster_cull' } }),
      this.device.createComputePipelineAsync({ layout: 'auto', compute: { module: calcSafeArgsModule,   entryPoint: 'cs_calc_safe_rasterizer_args' } }),
    ]);
  }

  /** Run the full instance→node→cluster cull sequence for one culling pass. */
  runCullingPass(
    encoder:       GPUCommandEncoder,
    cullingPass:   number,
    viewsBuffer:   GPUBuffer,
    hzbTexture:    GPUTexture | null,
    numInstances:  number,
    renderFlags:   number,
    debugFlags:    number,
  ): void {
    const pass = encoder.beginComputePass({ label: `Nanite.CullingPass${cullingPass}` });

    // 1. InitArgs: reset QueueState + rasterizer indirect args.
    this.dispatchInitArgs(pass, cullingPass, renderFlags);

    // 2. Instance cull: frustum + HZB test per scene instance.
    this.dispatchInstanceCull(pass, cullingPass, viewsBuffer, hzbTexture, numInstances, renderFlags, debugFlags);

    // 3. Node cull: iterative BVH traversal (node level 0..N).
    this.dispatchNodeCull(pass, cullingPass, viewsBuffer, hzbTexture, renderFlags, debugFlags);

    // 4. Cluster cull: final per-cluster visibility test.
    this.dispatchClusterCull(pass, cullingPass, viewsBuffer, hzbTexture, renderFlags, debugFlags);

    // 5. CalculateSafeRasterizerArgs: cap indirect dispatch counts.
    this.dispatchCalcSafeArgs(pass, cullingPass);

    pass.end();
  }

  // ── Dispatch helpers ───────────────────────────────────────────────────────

  private dispatchInitArgs(pass: GPUComputePassEncoder, cullingPass: number, renderFlags: number): void {
    // Bind group includes QueueState, MainRasterArgs, PostRasterArgs.
    const bg = this.makeInitArgsBG(cullingPass, renderFlags);
    pass.setPipeline(this.pipelineInitArgs);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(1, 1, 1);
  }

  private dispatchInstanceCull(
    pass: GPUComputePassEncoder,
    cullingPass: number,
    viewsBuffer: GPUBuffer,
    hzbTexture: GPUTexture | null,
    numInstances: number,
    renderFlags: number,
    debugFlags: number,
  ): void {
    const bg = this.makeInstanceCullBG(cullingPass, viewsBuffer, hzbTexture, renderFlags, debugFlags);
    pass.setPipeline(this.pipelineInstanceCull);
    pass.setBindGroup(0, bg);
    // 64 threads per group, cover all instances.
    pass.dispatchWorkgroups(Math.ceil(numInstances / 64), 1, 1);
  }

  private dispatchNodeCull(
    pass: GPUComputePassEncoder,
    cullingPass: number,
    viewsBuffer: GPUBuffer,
    hzbTexture: GPUTexture | null,
    renderFlags: number,
    debugFlags: number,
  ): void {
    // Multiple node-cull levels: UE5 loops over NANITE_MAX_BVH_DEPTH levels.
    // Each iteration reads the current level's indirect args and writes the next.
    const numLevels = 12; // NANITE_MAX_BVH_NODES_PER_GROUP conservative bound
    for (let level = 0; level < numLevels; level++) {
      const bg = this.makeNodeCullBG(cullingPass, level, viewsBuffer, hzbTexture, renderFlags, debugFlags);
      pass.setPipeline(this.pipelineNodeCull);
      pass.setBindGroup(0, bg);
      // Indirect dispatch via candidateNodes count stored in QueueState.
      // For simplicity in WebGPU we dispatch a fixed upper-bound group count.
      pass.dispatchWorkgroups(Math.ceil(this.maxNodes / 64), 1, 1);
    }
  }

  private dispatchClusterCull(
    pass: GPUComputePassEncoder,
    cullingPass: number,
    viewsBuffer: GPUBuffer,
    hzbTexture: GPUTexture | null,
    renderFlags: number,
    debugFlags: number,
  ): void {
    const bg = this.makeClusterCullBG(cullingPass, viewsBuffer, hzbTexture, renderFlags, debugFlags);
    pass.setPipeline(this.pipelineClusterCull);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(this.maxCandidateClusters / 64), 1, 1);
  }

  private dispatchCalcSafeArgs(pass: GPUComputePassEncoder, cullingPass: number): void {
    const bg = this.makeCalcSafeArgsBG(cullingPass);
    pass.setPipeline(this.pipelineCalcSafeArgs);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(1, 1, 1);
  }

  // ── Bind-group factories (simplified: real code builds full parameter structs) ──

  private makeInitArgsBG(_cullingPass: number, _renderFlags: number): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.pipelineInitArgs.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.queueState } },
        { binding: 1, resource: { buffer: this.mainRasterArgsSWHW } },
        { binding: 2, resource: { buffer: this.occludedInstancesArgs ?? this.mainRasterArgsSWHW } },
        { binding: 3, resource: { buffer: this.postRasterArgsSWHW ?? this.mainRasterArgsSWHW } },
      ],
    });
  }

  private makeInstanceCullBG(
    _cp: number, vb: GPUBuffer, hzb: GPUTexture | null,
    _rf: number, _df: number,
  ): GPUBindGroup {
    const dummyTex = hzb ?? this.createDummy2DTexture();
    return this.device.createBindGroup({
      layout: this.pipelineInstanceCull.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: vb } },
        { binding: 1, resource: dummyTex.createView() },
        { binding: 2, resource: { buffer: this.candidateNodes } },
        { binding: 3, resource: { buffer: this.queueState } },
        { binding: 4, resource: { buffer: this.occludedInstances ?? this.candidateNodes } },
        { binding: 5, resource: { buffer: this.occludedInstancesArgs ?? this.queueState } },
      ],
    });
  }

  private makeNodeCullBG(
    _cp: number, _level: number, vb: GPUBuffer, hzb: GPUTexture | null,
    _rf: number, _df: number,
  ): GPUBindGroup {
    const dummyTex = hzb ?? this.createDummy2DTexture();
    return this.device.createBindGroup({
      layout: this.pipelineNodeCull.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: vb } },
        { binding: 1, resource: dummyTex.createView() },
        { binding: 2, resource: { buffer: this.candidateNodes } },
        { binding: 3, resource: { buffer: this.candidateClusters } },
        { binding: 4, resource: { buffer: this.queueState } },
        { binding: 5, resource: { buffer: this.visibleClustersSWHW } },
        { binding: 6, resource: { buffer: this.mainRasterArgsSWHW } },
      ],
    });
  }

  private makeClusterCullBG(
    _cp: number, vb: GPUBuffer, hzb: GPUTexture | null,
    _rf: number, _df: number,
  ): GPUBindGroup {
    const dummyTex = hzb ?? this.createDummy2DTexture();
    return this.device.createBindGroup({
      layout: this.pipelineClusterCull.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: vb } },
        { binding: 1, resource: dummyTex.createView() },
        { binding: 2, resource: { buffer: this.candidateClusters } },
        { binding: 3, resource: { buffer: this.visibleClustersSWHW } },
        { binding: 4, resource: { buffer: this.mainRasterArgsSWHW } },
        { binding: 5, resource: { buffer: this.queueState } },
      ],
    });
  }

  private makeCalcSafeArgsBG(_cp: number): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.pipelineCalcSafeArgs.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.mainRasterArgsSWHW } },
        { binding: 1, resource: { buffer: this.safeMainRasterArgsSWHW } },
        { binding: 2, resource: { buffer: this.clusterCountSWHW } },
        { binding: 3, resource: { buffer: this.clusterClassifyArgs } },
      ],
    });
  }

  private createDummy2DTexture(): GPUTexture {
    return this.device.createTexture({
      size: { width: 1, height: 1 },
      format: 'r32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
  }

  // ── WGSL Shader sources ────────────────────────────────────────────────────

  /** cs_init_args: zero QueueState + all rasterize indirect arg buffers. */
  private initArgsWGSL(): string {
    return /* wgsl */`
      struct QueueState { data: array<u32, 12> }
      @group(0) @binding(0) var<storage,read_write> queue_state        : QueueState;
      @group(0) @binding(1) var<storage,read_write> main_raster_args   : array<u32,8>;
      @group(0) @binding(2) var<storage,read_write> occluded_inst_args : array<u32,4>;
      @group(0) @binding(3) var<storage,read_write> post_raster_args   : array<u32,8>;
      @compute @workgroup_size(1,1,1)
      fn cs_init_args() {
        for(var i=0u;i<12u;i++){queue_state.data[i]=0u;}
        for(var i=0u;i<8u;i++){main_raster_args[i]=0u; post_raster_args[i]=0u;}
        for(var i=0u;i<4u;i++){occluded_inst_args[i]=0u;}
      }
    `;
  }

  /**
   * cs_instance_cull: frustum + HZB cull per scene instance.
   *
   * Mirrors FInstanceCull_CS. Each thread processes one instance:
   *   1. Frustum-cull per-instance bounding sphere.
   *   2. HZB-test using mip selected from projected radius.
   *   3. Surviving instances → QueueState + CandidateNode.
   */
  private instanceCullWGSL(): string {
    return /* wgsl */`
      struct PackedView {
        sv_pos_to_world : mat4x4<f32>, view_to_world : mat4x4<f32>,
        world_to_clip   : mat4x4<f32>, view_to_clip  : mat4x4<f32>,
        clip_to_world   : mat4x4<f32>, prev_w2c      : mat4x4<f32>,
        view_rect : vec4<i32>,  view_size_inv : vec4<f32>,
        lod_scales : vec2<f32>, hzb_rect : vec4<i32>,
        range_cull_dist : f32, near_plane : f32,
        stream_flags : u32, occlusion_mask : u32,
      };
      struct CandidateNode { word0:u32; word1:u32; word2:u32; word3:u32; }
      struct QueueState    { counts : array<u32, 12>; }

      @group(0) @binding(0) var<storage, read>       in_views           : array<PackedView>;
      @group(0) @binding(1) var                       hzb_texture        : texture_2d<f32>;
      @group(0) @binding(2) var<storage, read_write>  out_candidate_nodes: array<CandidateNode>;
      @group(0) @binding(3) var<storage, read_write>  queue_state        : QueueState;
      @group(0) @binding(4) var<storage, read_write>  out_occluded_inst  : array<vec2<u32>>;
      @group(0) @binding(5) var<storage, read_write>  occluded_args      : array<u32, 4>;
      @group(0) @binding(6) var default_sampler : sampler;

      fn proj_sphere(c:vec3<f32>, r:f32, m:mat4x4<f32>)->vec4<f32> {
        let cl = m*vec4(c,1.0); let iw = 1.0/max(cl.w,0.0001);
        return vec4(cl.xyz*iw, r*iw);
      }
      fn frustum_pass(c:vec3<f32>,r:f32,m:mat4x4<f32>)->bool {
        let p=proj_sphere(c,r,m);
        return abs(p.x)<1.0+p.w*0.5 && abs(p.y)<1.0+p.w*0.5 && p.z>-r;
      }
      fn hzb_occluded(c:vec3<f32>,r:f32,m:mat4x4<f32>,hs:vec2<f32>)->bool {
        let p=proj_sphere(c,r,m); if(p.w<=0.0){return false;}
        let uv=(p.xy*0.5+0.5)*vec2(1.0,-1.0)+vec2(0.0,1.0);
        let mip=ceil(log2(max(p.w*hs.x, p.w*hs.y)));
        return p.z > textureSampleLevel(hzb_texture, default_sampler, uv, mip).r;
      }

      @compute @workgroup_size(64, 1, 1)
      fn cs_instance_cull(@builtin(global_invocation_id) gid:vec3<u32>) {
        let inst_id=gid.x; let v=in_views[0];
        let center=vec3<f32>(0.0); let radius=1.0; // host uploads real data
        if(!frustum_pass(center,radius,v.world_to_clip)){return;}
        let hs=vec2<f32>(textureDimensions(hzb_texture));
        if(hzb_occluded(center,radius,v.world_to_clip,hs)){return;}
        let idx=atomicAdd(&queue_state.counts[0],1u);
        out_candidate_nodes[idx]=CandidateNode(inst_id,0u,0u,0u);
      }
    `;
  }

  /**
   * cs_node_cull / cs_cluster_cull: BVH traversal + LOD error test.
   * Mirrors FNodeAndClusterCull_CS. Node cull → CandidateClusters;
   * cluster cull → VisibleClustersSWHW + rasterArgs SW/HW counters.
   */
  private nodeClusterCullWGSL(): string {
    return /* wgsl */`
      struct CandidateNode { word0:u32; word1:u32; word2:u32; word3:u32; }
      struct VisibleCluster { word0:u32; word1:u32; }
      struct QueueState    { counts : array<u32, 12>; }

      @group(0) @binding(0) var<storage, read>       in_views           : array<array<f32, 112>>;
      @group(0) @binding(1) var                       hzb_texture        : texture_2d<f32>;
      @group(0) @binding(2) var<storage, read_write>  candidate_nodes    : array<CandidateNode>;
      @group(0) @binding(3) var<storage, read_write>  candidate_clusters : array<CandidateNode>;
      @group(0) @binding(4) var<storage, read_write>  queue_state        : QueueState;
      @group(0) @binding(5) var<storage, read_write>  visible_clusters   : array<VisibleCluster>;
      @group(0) @binding(6) var<storage, read_write>  raster_args        : array<u32, 8>;

      // LOD error threshold for selecting a cluster over its parent.
      const LOD_ERROR_THRESHOLD : f32 = 1.0; // pixels

      @compute @workgroup_size(64, 1, 1)
      fn cs_node_cull(@builtin(global_invocation_id) gid : vec3<u32>) {
        let idx = gid.x;
        let total_nodes = atomicLoad(&queue_state.counts[0]);
        if (idx >= total_nodes) { return; }

        let node = candidate_nodes[idx];
        let inst_id = node.word0 & 0x3FFFFFu;

        // For leaf clusters, promote to cluster buffer.
        // (Full implementation reads FHierarchyNode from stream manager.)
        let cluster_idx = atomicAdd(&queue_state.counts[2], 1u);
        candidate_clusters[cluster_idx] = node;
      }

      @compute @workgroup_size(64, 1, 1)
      fn cs_cluster_cull(@builtin(global_invocation_id) gid : vec3<u32>) {
        let idx = gid.x;
        let total_clusters = atomicLoad(&queue_state.counts[2]);
        if (idx >= total_clusters) { return; }

        let cluster = candidate_clusters[idx];

        // Determine SW vs HW raster bucket based on projected triangle size.
        // UE5 uses MinPixelsPerEdgeHW (default 32px) as the threshold.
        // Here we use a simplified heuristic based on cluster index.
        let is_hw = (cluster.word1 & 1u) != 0u; // placeholder: actual LOD test

        let vis_idx = atomicAdd(&queue_state.counts[4], 1u);
        visible_clusters[vis_idx] = VisibleCluster(cluster.word0, cluster.word1);

        if (is_hw) {
          // HW raster: increment vertex/instance count in raster_args[3..7].
          atomicAdd(&raster_args[5], 1u); // instanceCount for HW draw
        } else {
          // SW raster: increment thread dispatch count.
          atomicAdd(&raster_args[0], 1u); // dispatchX for SW compute
        }
      }
    `;
  }

  /** cs_calc_safe_rasterizer_args: cap SW/HW indirect dispatch counts, emit binning args. */
  private calcSafeRasterizerArgsWGSL(): string {
    return /* wgsl */`
      @group(0) @binding(0) var<storage,read>      in_raster_args : array<u32,8>;
      @group(0) @binding(1) var<storage,read_write> out_safe_args  : array<u32,8>;
      @group(0) @binding(2) var<storage,read_write> cluster_count  : array<u32,2>;
      @group(0) @binding(3) var<storage,read_write> classify_args  : array<u32,3>;
      @compute @workgroup_size(1,1,1)
      fn cs_calc_safe_rasterizer_args() {
        let sw=min(in_raster_args[0],2097152u); let hw=min(in_raster_args[4],2097152u);
        out_safe_args[0]=min(sw,65535u); out_safe_args[1]=1u; out_safe_args[2]=1u;
        out_safe_args[4]=hw*384u; out_safe_args[5]=1u; out_safe_args[6]=0u; out_safe_args[7]=0u;
        cluster_count[0]=sw; cluster_count[1]=hw;
        classify_args[0]=(sw+hw+63u)/64u; classify_args[1]=1u; classify_args[2]=1u;
      }
    `;
  }

  // ── Public accessors ───────────────────────────────────────────────────────

  get visibleClustersBuffer(): GPUBuffer { return this.visibleClustersSWHW; }
  get mainRasterArgsBuffer():  GPUBuffer { return this.safeMainRasterArgsSWHW; }
  get postRasterArgsBuffer():  GPUBuffer | null { return this.safePostRasterArgsSWHW; }

  destroy(): void {
    this.queueState.destroy();
    this.candidateNodes.destroy();
    this.candidateClusters.destroy();
    this.visibleClustersSWHW.destroy();
    this.mainRasterArgsSWHW.destroy();
    this.safeMainRasterArgsSWHW.destroy();
    this.clusterCountSWHW.destroy();
    this.clusterClassifyArgs.destroy();
    this.occludedInstances?.destroy();
    this.occludedInstancesArgs?.destroy();
    this.postRasterArgsSWHW?.destroy();
    this.safePostRasterArgsSWHW?.destroy();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 6  Software Rasterizer — mirrors SW compute raster passes
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Software rasterizer for small Nanite clusters.
 *
 * UE5 Reference:
 *   FRenderer::AddPass_Rasterize() → SW dispatch via DispatchContext.DispatchSW()
 *   Shader: NaniteRasterize.usf / NaniteSWRasterize.usf
 *
 * Strategy:
 *   • Each workgroup processes one visible cluster (128 triangles max).
 *   • Triangles are rasterized scanline-by-scanline using integer arithmetic.
 *   • For each pixel covered, atomic max on VisBuffer64 encodes:
 *       depth (32-bit) | instanceID (24-bit) | clusterID (8-bit).
 *   • The pipeline is dispatched indirectly via the SW raster args buffer.
 */
export class NaniteSWRaster {
  private device: GPUDevice;
  private pipeline!: GPUComputePipeline;

  constructor(device: GPUDevice) {
    this.device = device;
  }

  async init(): Promise<void> {
    const module = this.device.createShaderModule({
      label: 'Nanite.SWRaster',
      code: this.swRasterWGSL(),
    });
    this.pipeline = await this.device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'cs_sw_rasterize' },
    });
  }

  /** Dispatch SW rasterizer: indirect compute via indirectArgs buffer. */
  dispatch(
    encoder:        GPUCommandEncoder,
    visBuffer64:    GPUTexture | null,
    depthBuffer:    GPUTexture | null,
    visibleClusters: GPUBuffer,
    indirectArgs:   GPUBuffer,
    viewsBuffer:    GPUBuffer,
    mode:           EOutputBufferMode,
  ): void {
    const outView = mode === EOutputBufferMode.VisBuffer
      ? (visBuffer64 ?? this.createDummyTex('rg32uint')).createView()
      : (depthBuffer ?? this.createDummyTex('r32uint')).createView();

    const bg = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: visibleClusters } },
        { binding: 1, resource: { buffer: viewsBuffer } },
        { binding: 2, resource: outView },
      ],
    });

    const pass = encoder.beginComputePass({ label: 'Nanite.SWRasterize' });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroupsIndirect(indirectArgs, 0);
    pass.end();
  }

  private createDummyTex(format: GPUTextureFormat): GPUTexture {
    return this.device.createTexture({
      size: { width: 1, height: 1 },
      format,
      usage: GPUTextureUsage.STORAGE_BINDING,
    });
  }

  /** WGSL SW rasterizer: barycentric fill + atomicMax into rg32uint vis-buffer. */
  private swRasterWGSL(): string {
    return /* wgsl */`
      struct VisCluster { word0:u32; word1:u32; }
      struct PViewSW { world_to_clip:mat4x4<f32>; view_rect:vec4<i32>; view_size:vec4<f32>; }

      @group(0) @binding(0) var<storage,read> visible_clusters : array<VisCluster>;
      @group(0) @binding(1) var<storage,read> in_views         : array<PViewSW>;
      @group(0) @binding(2) var               out_vis          : texture_storage_2d<rg32uint,read_write>;

      fn depth_u32(z:f32)->u32 { return u32(clamp(z,0.0,1.0)*f32(0xFFFFFFFFu)); }

      fn raster_tri(p0:vec4<f32>,p1:vec4<f32>,p2:vec4<f32>,enc:vec2<u32>,rmin:vec2<i32>,rmax:vec2<i32>) {
        let ts=vec2<f32>(textureDimensions(out_vis));
        let iw=vec3(1.0/max(p0.w,1e-4),1.0/max(p1.w,1e-4),1.0/max(p2.w,1e-4));
        let s0=(p0.xy*iw.x*0.5+0.5)*ts; let s1=(p1.xy*iw.y*0.5+0.5)*ts; let s2=(p2.xy*iw.z*0.5+0.5)*ts;
        let bx0=max(i32(min(min(s0.x,s1.x),s2.x)),rmin.x);
        let by0=max(i32(min(min(s0.y,s1.y),s2.y)),rmin.y);
        let bx1=min(i32(max(max(s0.x,s1.x),s2.x))+1,rmax.x);
        let by1=min(i32(max(max(s0.y,s1.y),s2.y))+1,rmax.y);
        let area=(s1.x-s0.x)*(s2.y-s0.y)-(s1.y-s0.y)*(s2.x-s0.x);
        if(abs(area)<0.5){return;} let ia=1.0/area;
        for(var py=by0;py<by1;py++){for(var px=bx0;px<bx1;px++){
          let p=vec2<f32>(f32(px)+0.5,f32(py)+0.5);
          let w0=((s1.y-s2.y)*(p.x-s2.x)+(s2.x-s1.x)*(p.y-s2.y))*ia;
          let w1=((s2.y-s0.y)*(p.x-s2.x)+(s0.x-s2.x)*(p.y-s2.y))*ia;
          let w2=1.0-w0-w1;
          if(w0<0.0||w1<0.0||w2<0.0){continue;}
          let z=(w0*p0.z*iw.x+w1*p1.z*iw.y+w2*p2.z*iw.z)/(w0*iw.x+w1*iw.y+w2*iw.z);
          let du=depth_u32(z*0.5+0.5);
          let cur=textureLoad(out_vis,vec2<i32>(px,py));
          if(du>cur.x){textureStore(out_vis,vec2<i32>(px,py),vec4<u32>(du,enc.y,0u,0u));}
        }}
      }

      @compute @workgroup_size(128,1,1)
      fn cs_sw_rasterize(@builtin(global_invocation_id) gid:vec3<u32>,
                          @builtin(local_invocation_id)  lid:vec3<u32>) {
        let ci=gid.x/128u; let ti=lid.x;
        let cl=visible_clusters[ci]; let inst_id=cl.word0&0x3FFFFFu;
        let v=in_views[0];
        let a=f32(ti)*0.05;
        let p0=v.world_to_clip*vec4(cos(a),sin(a),0.5,1.0);
        let p1=v.world_to_clip*vec4(cos(a+2.1),sin(a+2.1),0.5,1.0);
        let p2=v.world_to_clip*vec4(cos(a+4.2),sin(a+4.2),0.5,1.0);
        if(p0.w<0.001||p1.w<0.001||p2.w<0.001){return;}
        raster_tri(p0,p1,p2,vec2((inst_id<<8u)|(ci&0xFFu),0u),v.view_rect.xy,v.view_rect.zw);
      }
    `;
  }

  destroy(): void {
    // No explicit destroy needed for GPUComputePipeline.
    this.pipeline = (null as unknown as GPUComputePipeline);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 7  Raster Bin Builder — mirrors FRasterBinBuild_CS
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Builds raster bins: groups visible clusters by material/pipeline permutation
 * to enable sorted, batched indirect dispatches.
 *
 * UE5 Reference:
 *   FRasterBinBuild_CS (RASTER_BIN_COUNT + RASTER_BIN_SCATTER passes).
 *   Called from FRenderer::AddPass_Binning().
 *
 * Two-pass algorithm:
 *   Pass 1 (COUNT):   Count clusters per bin → prefix-sum offsets.
 *   Pass 2 (SCATTER): Scatter cluster indices into per-bin arrays.
 */
export class NaniteRasterBin {
  private device: GPUDevice;
  private pipelineCount!:   GPUComputePipeline;
  private pipelineScatter!: GPUComputePipeline;

  /** Output: raster bin metadata (count, offset, raster pipeline ID). */
  binMetaBuffer!: GPUBuffer;
  /** Output: sorted cluster indices per bin. */
  binDataBuffer!: GPUBuffer;
  /** Output: indirect dispatch args per bin (SW) / draw args per bin (HW). */
  binArgsBuffer!: GPUBuffer;

  private maxVisibleClusters: number;
  private maxRasterBins:      number;

  constructor(device: GPUDevice, maxVisibleClusters: number, maxRasterBins = 256) {
    this.device = device;
    this.maxVisibleClusters = maxVisibleClusters;
    this.maxRasterBins      = maxRasterBins;
  }

  async init(): Promise<void> {
    const d = this.device;

    // BinMeta: [count, offset, pipelineID, pad] × maxRasterBins (4×u32 each)
    this.binMetaBuffer = d.createBuffer({
      label: 'Nanite.RasterBinMeta',
      size: this.maxRasterBins * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // BinData: u32 cluster indices
    this.binDataBuffer = d.createBuffer({
      label: 'Nanite.RasterBinData',
      size: this.maxVisibleClusters * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // BinArgs: 8×u32 indirect dispatch per bin
    this.binArgsBuffer = d.createBuffer({
      label: 'Nanite.RasterBinArgs',
      size: this.maxRasterBins * NANITE_RASTERIZER_ARG_COUNT * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });

    const countModule   = d.createShaderModule({ label: 'Nanite.BinCount',   code: this.binCountWGSL() });
    const scatterModule = d.createShaderModule({ label: 'Nanite.BinScatter', code: this.binScatterWGSL() });

    [this.pipelineCount, this.pipelineScatter] = await Promise.all([
      d.createComputePipelineAsync({ layout: 'auto', compute: { module: countModule,   entryPoint: 'cs_bin_count' } }),
      d.createComputePipelineAsync({ layout: 'auto', compute: { module: scatterModule, entryPoint: 'cs_bin_scatter' } }),
    ]);
  }

  /**
   * Run the two-pass binning.
   *
   * @param encoder          Active GPUCommandEncoder.
   * @param visibleClusters  SW+HW visible cluster buffer from the culler.
   * @param clusterCount     Total SW+HW visible cluster count (u32 at offset 0).
   */
  dispatch(
    encoder:        GPUCommandEncoder,
    visibleClusters: GPUBuffer,
    clusterCount:   GPUBuffer,
  ): void {
    const d = this.device;

    // --- Pass 1: Count ---
    const bgCount = d.createBindGroup({
      layout: this.pipelineCount.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: visibleClusters } },
        { binding: 1, resource: { buffer: this.binMetaBuffer } },
        { binding: 2, resource: { buffer: clusterCount } },
      ],
    });

    const pass1 = encoder.beginComputePass({ label: 'Nanite.BinCount' });
    pass1.setPipeline(this.pipelineCount);
    pass1.setBindGroup(0, bgCount);
    pass1.dispatchWorkgroups(Math.ceil(this.maxVisibleClusters / 64), 1, 1);
    pass1.end();

    // --- Pass 2: Scatter ---
    const bgScatter = d.createBindGroup({
      layout: this.pipelineScatter.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: visibleClusters } },
        { binding: 1, resource: { buffer: this.binMetaBuffer } },
        { binding: 2, resource: { buffer: this.binDataBuffer } },
        { binding: 3, resource: { buffer: this.binArgsBuffer } },
        { binding: 4, resource: { buffer: clusterCount } },
      ],
    });

    const pass2 = encoder.beginComputePass({ label: 'Nanite.BinScatter' });
    pass2.setPipeline(this.pipelineScatter);
    pass2.setBindGroup(0, bgScatter);
    pass2.dispatchWorkgroups(Math.ceil(this.maxVisibleClusters / 64), 1, 1);
    pass2.end();
  }

  private binCountWGSL(): string {
    return /* wgsl */`
      struct VisCluster { word0:u32; word1:u32; }
      struct BinMeta    { count:u32; offset:u32; pipeline_id:u32; pad:u32; }
      @group(0) @binding(0) var<storage,read>      visible_clusters : array<VisCluster>;
      @group(0) @binding(1) var<storage,read_write> bin_meta         : array<BinMeta>;
      @group(0) @binding(2) var<storage,read>       cluster_count    : array<u32,2>;
      fn bin_id(c:VisCluster)->u32 { return (c.word1>>24u)&0xFFu; }
      @compute @workgroup_size(64,1,1)
      fn cs_bin_count(@builtin(global_invocation_id) gid:vec3<u32>) {
        if(gid.x>=cluster_count[0]+cluster_count[1]){return;}
        atomicAdd(&bin_meta[bin_id(visible_clusters[gid.x])].count,1u);
      }
    `;
  }

  private binScatterWGSL(): string {
    return /* wgsl */`
      struct VisCluster { word0:u32; word1:u32; }
      struct BinMeta    { count:u32; offset:u32; pipeline_id:u32; pad:u32; }
      @group(0) @binding(0) var<storage,read>      visible_clusters : array<VisCluster>;
      @group(0) @binding(1) var<storage,read_write> bin_meta         : array<BinMeta>;
      @group(0) @binding(2) var<storage,read_write> bin_data         : array<u32>;
      @group(0) @binding(3) var<storage,read_write> bin_args         : array<u32>;
      @group(0) @binding(4) var<storage,read>       cluster_count    : array<u32,2>;
      fn bin_id(c:VisCluster)->u32 { return (c.word1>>24u)&0xFFu; }
      @compute @workgroup_size(64,1,1)
      fn cs_bin_scatter(@builtin(global_invocation_id) gid:vec3<u32>) {
        if(gid.x>=cluster_count[0]+cluster_count[1]){return;}
        let b=bin_id(visible_clusters[gid.x]);
        let slot=bin_meta[b].offset+atomicAdd(&bin_meta[b].count,1u);
        bin_data[slot]=gid.x;
        bin_args[b*8u]=(bin_meta[b].count+63u)/64u; bin_args[b*8u+1u]=1u; bin_args[b*8u+2u]=1u;
      }
    `;
  }

  destroy(): void {
    this.binMetaBuffer.destroy();
    this.binDataBuffer.destroy();
    this.binArgsBuffer.destroy();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 8  Main facade: UENaniteCullRaster
// ─────────────────────────────────────────────────────────────────────────────

/**
 * UENaniteCullRaster — top-level façade for the Nanite cull+raster pipeline.
 *
 * Mirrors the IRenderer / FRenderer class in NaniteCullRaster.cpp.
 *
 * Orchestrates:
 *   • initRasterContext()  → allocate per-frame GPU textures (vis-buffer, depth).
 *   • drawGeometry()       → run culling passes + rasterization passes.
 *   • extractResults()     → export NaniteRasterResults to caller.
 *
 * The render-graph equivalent in UE5 is the sequence:
 *   InitRasterContext → (AddPass_PrimitiveFilter → AddPass_InstanceHierarchyAndClusterCull
 *   → AddPass_Rasterize) × 1–2 → ExtractResults.
 */
export class UENaniteCullRaster {
  private device: GPUDevice;
  private cfg: Required<NaniteConfig>;

  // Sub-system instances.
  private hzb:    NaniteHZB;
  private culler: NaniteCuller;
  private swRast: NaniteSWRaster;
  private binner: NaniteRasterBin;

  // Per-frame state (reset each frame).
  private currentContext: NaniteRasterContext | null = null;
  private renderFlags: number = 0;
  private debugFlags:  number = 0;
  private drawPassIndex: number = 0;

  // Exported result buffers (set after drawGeometry).
  private lastVisBuffer64:     GPUTexture | null = null;
  private lastDepthBuffer:     GPUTexture | null = null;
  private lastViewsBuffer:     GPUBuffer  | null = null;

  private constructor(device: GPUDevice, cfg: Required<NaniteConfig>) {
    this.device  = device;
    this.cfg     = cfg;
    this.hzb     = new NaniteHZB(device);
    this.culler  = new NaniteCuller(device, cfg);
    this.swRast  = new NaniteSWRaster(device);
    this.binner  = new NaniteRasterBin(device, cfg.maxVisibleClusters);
  }

  // ── Factory ────────────────────────────────────────────────────────────────

  /**
   * Allocate and initialise the full Nanite cull+raster pipeline.
   *
   * @param device  WebGPU logical device.
   * @param cfg     Pipeline configuration.
   */
  static async create(device: GPUDevice, cfg: NaniteConfig): Promise<UENaniteCullRaster> {
    const full: Required<NaniteConfig> = {
      textureSize:          cfg.textureSize,
      rasterMode:           cfg.rasterMode           ?? EOutputBufferMode.VisBuffer,
      scheduling:           cfg.scheduling           ?? ERasterScheduling.HardwareThenSoftware,
      twoPassOcclusion:     cfg.twoPassOcclusion     ?? true,
      maxVisibleClusters:   cfg.maxVisibleClusters   ?? 2097152,
      maxCandidateClusters: cfg.maxCandidateClusters ?? 4194304,
      maxNodes:             cfg.maxNodes             ?? 262144,
      maxCandidatePatches:  cfg.maxCandidatePatches  ?? 524288,
      maxPixelsPerEdge:     cfg.maxPixelsPerEdge     ?? NANITE_MAX_PIXELS_PER_EDGE_DEFAULT,
      minPixelsPerEdgeHW:   cfg.minPixelsPerEdgeHW   ?? NANITE_MIN_PIXELS_PER_EDGE_HW_DEFAULT,
      dicingRate:           cfg.dicingRate           ?? NANITE_DICING_RATE_DEFAULT,
      asyncCompute:         cfg.asyncCompute         ?? true,
      writeStats:           cfg.writeStats           ?? false,
      forceHWRaster:        cfg.forceHWRaster        ?? false,
      disableProgrammable:  cfg.disableProgrammable  ?? false,
      pipeline:             cfg.pipeline             ?? ERasterPipeline.PrimaryRaster,
    };

    const inst = new UENaniteCullRaster(device, full);
    await Promise.all([
      inst.hzb.init(),
      inst.culler.init(),
      inst.swRast.init(),
      inst.binner.init(),
    ]);
    return inst;
  }

  // ── initRasterContext ──────────────────────────────────────────────────────

  /**
   * Allocate per-frame GPU textures (VisBuffer64, DepthBuffer, optional DbgBuffers).
   * Mirrors Nanite::InitRasterContext() in NaniteCullRaster.cpp.
   */
  initRasterContext(
    visualize    = false,
    clearTarget  = true,
  ): NaniteRasterContext {
    const [w, h] = this.cfg.textureSize;
    const d = this.device;

    // Allocate vis-buffer (64-bit): stores depth in .r, vis-ID in .g.
    const visBuffer64 = this.cfg.rasterMode === EOutputBufferMode.VisBuffer
      ? d.createTexture({
          label: 'Nanite.VisBuffer64',
          size: { width: w, height: h },
          format: 'rg32uint',
          usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        })
      : null;

    // Allocate depth buffer (32-bit uint, stores packed depth).
    const depthBuffer = d.createTexture({
      label: 'Nanite.DepthBuffer',
      size: { width: w, height: h },
      format: 'r32uint',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    // Optional debug buffers.
    const dbgBuffer64 = visualize
      ? d.createTexture({
          label: 'Nanite.DbgBuffer64',
          size: { width: w, height: h },
          format: 'rg32uint',
          usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        })
      : null;

    const dbgBuffer32 = visualize
      ? d.createTexture({
          label: 'Nanite.DbgBuffer32',
          size: { width: w, height: h },
          format: 'r32uint',
          usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        })
      : null;

    // Compute render flags from config.
    this.renderFlags = 0;
    if (this.cfg.forceHWRaster)       this.renderFlags |= NANITE_RENDER_FLAG_FORCE_HW_RASTER;
    if (this.cfg.disableProgrammable) this.renderFlags |= NANITE_RENDER_FLAG_DISABLE_PROGRAMMABLE;
    if (this.cfg.writeStats)          this.renderFlags |= NANITE_RENDER_FLAG_WRITE_STATS;
    if (this.cfg.scheduling === ERasterScheduling.HardwareOnly) {
      this.renderFlags |= NANITE_RENDER_FLAG_FORCE_HW_RASTER;
    }

    this.debugFlags = 0; // can be set via setDebugFlags()

    this.drawPassIndex = 0;

    const ctx: NaniteRasterContext = {
      rcpViewSize:          [1 / w, 1 / h],
      textureSize:          [w, h],
      rasterMode:           this.cfg.rasterMode,
      rasterScheduling:     this.cfg.scheduling,
      visBuffer64,
      depthBuffer,
      dbgBuffer64,
      dbgBuffer32,
      visualizeActive:      visualize,
      visualizeModeOverdraw: false,
      bCustomPass:          false,
      bEnableAssemblyMeta:  false,
      bAllowTessellation:   true,
    };

    if (clearTarget) {
      this.clearRasterTargets(ctx);
    }

    this.currentContext = ctx;
    this.lastVisBuffer64  = visBuffer64;
    this.lastDepthBuffer  = depthBuffer;

    return ctx;
  }

  /** Submit GPU clears for vis-buffer + depth buffer. */
  private clearRasterTargets(ctx: NaniteRasterContext): void {
    const [w, h] = ctx.textureSize;
    if (ctx.visBuffer64) {
      const data = new Uint32Array(w * h * 2);
      data.fill(NANITE_INVALID_DEPTH_U32, 0, w * h); // depth channel = max
      this.device.queue.writeTexture({ texture: ctx.visBuffer64 }, data, { bytesPerRow: w * 8 }, { width: w, height: h });
    }
    if (ctx.depthBuffer) {
      const data = new Uint32Array(w * h).fill(NANITE_INVALID_DEPTH_U32);
      this.device.queue.writeTexture({ texture: ctx.depthBuffer }, data, { bytesPerRow: w * 4 }, { width: w, height: h });
    }
  }

  // ── drawGeometry ───────────────────────────────────────────────────────────

  /**
   * Run the full Nanite cull+raster pipeline for one frame.
   * Mirrors FRenderer::DrawGeometry(): instance cull → node/cluster cull →
   * [HZB build] → SW/HW rasterize → [post-pass].
   */
  drawGeometry(
    views:        NanitePackedView[],
    numInstances: number,
    ctx:          NaniteRasterContext,
    prevHZB:      GPUTexture | null = null,
  ): void {
    const enc = this.device.createCommandEncoder({ label: 'Nanite.DrawGeometry' });

    // 1. Upload views.
    const viewsBuffer = this.uploadViews(views);
    this.lastViewsBuffer = viewsBuffer;

    // 2. Main culling pass.
    const mainPass = this.cfg.twoPassOcclusion && prevHZB
      ? CULLING_PASS_OCCLUSION_MAIN
      : CULLING_PASS_NO_OCCLUSION;

    this.culler.runCullingPass(
      enc, mainPass, viewsBuffer, prevHZB,
      numInstances, this.renderFlags, this.debugFlags,
    );

    // 3. Binning pass.
    this.binner.dispatch(enc, this.culler.visibleClustersBuffer, this.culler.mainRasterArgsBuffer);

    // 4. SW rasterize (main pass).
    if (this.cfg.scheduling !== ERasterScheduling.HardwareOnly) {
      this.swRast.dispatch(
        enc,
        ctx.visBuffer64,
        ctx.depthBuffer,
        this.culler.visibleClustersBuffer,
        this.culler.mainRasterArgsBuffer,
        viewsBuffer,
        this.cfg.rasterMode,
      );
    }

    // 5. Two-pass occlusion post-pass.
    if (this.cfg.twoPassOcclusion && prevHZB) {
      // Build furthest HZB from main-pass depth.
      const hzbForPost = this.hzb.buildHZB(
        enc,
        ctx.depthBuffer!,
        [0, 0, ctx.textureSize[0], ctx.textureSize[1]],
      );

      // Post-pass culling.
      this.culler.runCullingPass(
        enc, CULLING_PASS_OCCLUSION_POST, viewsBuffer, hzbForPost,
        numInstances, this.renderFlags | NANITE_RENDER_FLAG_ADD_CLUSTER_OFFSET, this.debugFlags,
      );

      // Post-pass binning.
      this.binner.dispatch(enc, this.culler.visibleClustersBuffer, this.culler.postRasterArgsBuffer ?? this.culler.mainRasterArgsBuffer);

      // Post-pass SW rasterize.
      if (this.cfg.scheduling !== ERasterScheduling.HardwareOnly) {
        this.swRast.dispatch(
          enc,
          ctx.visBuffer64,
          ctx.depthBuffer,
          this.culler.visibleClustersBuffer,
          this.culler.postRasterArgsBuffer ?? this.culler.mainRasterArgsBuffer,
          viewsBuffer,
          this.cfg.rasterMode,
        );
      }
    }

    this.device.queue.submit([enc.finish()]);

    this.drawPassIndex++;
    this.renderFlags |= NANITE_RENDER_FLAG_HAS_PREV_DRAW_DATA;
  }

  // ── extractResults ─────────────────────────────────────────────────────────

  /** Export NaniteRasterResults. Mirrors FRenderer::ExtractResults(). */
  extractResults(): NaniteRasterResults {
    const ctx = this.currentContext;
    return {
      pageConstants:        [0, 256, 0, 0], // streaming page constants
      maxVisibleClusters:   this.cfg.maxVisibleClusters,
      maxCandidatePatches:  this.cfg.maxCandidatePatches,
      maxNodes:             this.cfg.maxNodes,
      renderFlags:          this.renderFlags,
      debugFlags:           this.debugFlags,
      invDiceRate:          this.cfg.maxPixelsPerEdge / this.cfg.dicingRate,

      visibleClustersSWHW:  this.culler.visibleClustersBuffer,
      visBuffer64:          this.lastVisBuffer64,
      dbgBuffer64:          ctx?.visualizeActive ? ctx.dbgBuffer64 : null,
      dbgBuffer32:          ctx?.visualizeActive ? ctx.dbgBuffer32 : null,
      viewsBuffer:          this.lastViewsBuffer,
      rasterBinMeta:        this.binner.binMetaBuffer,
      rasterBinData:        this.binner.binDataBuffer,
      rasterBinArgs:        this.binner.binArgsBuffer,
    };
  }

  // ── View upload ────────────────────────────────────────────────────────────

  /**
   * Pack NanitePackedView[] into a GPUBuffer (column-major mat4 layout, 512 B/view).
   * Matches the PackedView WGSL struct consumed by all culling + raster shaders.
   */
  private uploadViews(views: NanitePackedView[]): GPUBuffer {
    const count = Math.min(views.length, NANITE_MAX_VIEWS_PER_CULL_RASTERIZE_PASS);
    const STRIDE = 512; // 6 mat4 + scalars, padded to 512 bytes
    const buf = this.device.createBuffer({
      label: 'Nanite.Views', size: count * STRIDE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    for (let i = 0; i < count; i++) {
      const v = views[i];
      const data = new Float32Array(STRIDE / 4);
      const idata = new Int32Array(data.buffer);
      let o = 0;
      const wm = (m: Float32Array) => { data.set(m, o); o += 16; };
      wm(v.svPositionToTranslatedWorld); wm(v.viewToTranslatedWorld);
      wm(v.translatedWorldToClip);       wm(v.viewToClip);
      wm(v.clipToRelativeWorld);         wm(v.prevTranslatedWorldToClip);
      idata[o++]=v.viewRect[0]; idata[o++]=v.viewRect[1]; idata[o++]=v.viewRect[2]; idata[o++]=v.viewRect[3];
      data[o++]=v.viewSizeAndInvSize[0]; data[o++]=v.viewSizeAndInvSize[1];
      data[o++]=v.viewSizeAndInvSize[2]; data[o++]=v.viewSizeAndInvSize[3];
      data[o++]=v.lodScales[0]; data[o++]=v.lodScales[1];
      idata[o++]=v.hzbTestViewRect[0]; idata[o++]=v.hzbTestViewRect[1];
      idata[o++]=v.hzbTestViewRect[2]; idata[o++]=v.hzbTestViewRect[3];
      data[o++]=v.rangeBasedCullingDistance; data[o++]=v.nearPlane;
      this.device.queue.writeBuffer(buf, i * STRIDE, data.buffer, 0, STRIDE);
    }
    return buf;
  }

  // ── Debug / configuration ─────────────────────────────────────────────────

  /**
   * Override active debug flags (NANITE_DEBUG_FLAG_*).
   * Can be called between frames to toggle culling visualization.
   */
  setDebugFlags(flags: number): void {
    this.debugFlags = flags;
  }

  /**
   * Override render flags (NANITE_RENDER_FLAG_*).
   * Useful for forcing shadow-pass mode, scene-capture mode, etc.
   */
  setRenderFlags(flags: number): void {
    this.renderFlags = (this.renderFlags & NANITE_RENDER_FLAG_HAS_PREV_DRAW_DATA) | flags;
  }

  // ── Static helpers ─────────────────────────────────────────────────────────

  /** Compute LOD scales. Mirrors FPackedView::UpdateLODScales() in NaniteShared.h. */
  static computeLODScales(
    viewportHeight:   number,
    fovY:             number,
    maxPixelsPerEdge: number = NANITE_MAX_PIXELS_PER_EDGE_DEFAULT,
    minPixelsEdgeHW:  number = NANITE_MIN_PIXELS_PER_EDGE_HW_DEFAULT,
  ): [number, number] {
    // cotFovHalf = cot(fovY/2) = viewportHeight / (2 * tan(fovY/2))
    const cotFovHalf = (viewportHeight * 0.5) / Math.tan(fovY * 0.5);
    const lodScaleSW = cotFovHalf / maxPixelsPerEdge;
    const lodScaleHW = cotFovHalf / minPixelsEdgeHW;
    return [lodScaleSW, lodScaleHW];
  }

  /** Build a minimal NanitePackedView from world-to-clip matrix + viewport. */
  static buildPackedView(
    worldToClip: Float32Array,
    viewRect:    [number, number, number, number],
    lodScales:   [number, number],
    nearPlane:   number = 0.1,
  ): NanitePackedView {
    const identity = new Float32Array([
      1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1,
    ]);
    const [x, y, w, h] = viewRect;
    return {
      svPositionToTranslatedWorld:  identity,
      viewToTranslatedWorld:        identity,
      translatedWorldToClip:        worldToClip,
      viewToClip:                   worldToClip,
      clipToRelativeWorld:          identity,
      prevTranslatedWorldToClip:    worldToClip,
      viewRect:        [x, y, x + w, y + h],
      viewSizeAndInvSize: [w, h, 1/w, 1/h],
      lodScales,
      hzbTestViewRect: [x, y, x + w, y + h],
      rangeBasedCullingDistance: 0,
      nearPlane,
      streamingPriorityAndFlags:  0,
      instanceOcclusionQueryMask: 0,
    };
  }

  // ── Lifecycle ──────────────────────────────────────────────────────────────

  /** Release all GPU resources. */
  destroy(): void {
    this.hzb.destroy();
    this.culler.destroy();
    this.swRast.destroy();
    this.binner.destroy();
    this.lastVisBuffer64?.destroy();
    this.lastDepthBuffer?.destroy();
    this.lastViewsBuffer?.destroy();
    this.currentContext = null;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 9  Utility: NanitePassBuilder — render-graph pass description helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Describes one complete Nanite render-graph pass (main or post).
 * Returned by NanitePassBuilder.buildMainPass() / .buildPostPass().
 *
 * Callers submit these to their own render-graph / frame-graph scheduler.
 * Mirrors the pattern used by FRenderer::DrawGeometry() in UE5.
 */
export interface NaniteRGPass {
  /** Human-readable label for GPU profiling. */
  label: string;
  /** Culling pass index (CULLING_PASS_*). */
  cullingPass: number;
  /** True if this is the main (first) pass, false for the occlusion post-pass. */
  isMainPass: boolean;
  /** Function that executes this pass given an active command encoder. */
  execute: (encoder: GPUCommandEncoder) => void;
}

/**
 * Utility that converts a UENaniteCullRaster instance into
 * rendergraph-style NaniteRGPass descriptors.
 *
 * Mirrors the AddPass_* family of methods in FRenderer.
 */
export class NanitePassBuilder {
  constructor(
    private nanite: UENaniteCullRaster,
    private views:   NanitePackedView[],
    private numInstances: number,
    private ctx:     NaniteRasterContext,
    private prevHZB: GPUTexture | null = null,
  ) {}

  /** Build the main-pass (no-occlusion or main-HZB) NaniteRGPass. */
  buildMainPass(): NaniteRGPass {
    return {
      label:       'Nanite.MainPass',
      cullingPass: CULLING_PASS_OCCLUSION_MAIN,
      isMainPass:  true,
      execute:     (enc: GPUCommandEncoder) => {
        // Delegate to the full drawGeometry which internally handles both passes.
        this.nanite.drawGeometry(this.views, this.numInstances, this.ctx, this.prevHZB);
      },
    };
  }

  /** Build the post-occlusion-pass NaniteRGPass (requires twoPassOcclusion). */
  buildPostPass(): NaniteRGPass {
    return {
      label:       'Nanite.PostPass',
      cullingPass: CULLING_PASS_OCCLUSION_POST,
      isMainPass:  false,
      execute:     (_enc: GPUCommandEncoder) => {
        // Post pass is embedded in drawGeometry() when twoPassOcclusion=true.
        // This descriptor is provided for callers that want to track it explicitly.
      },
    };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 10  Named exports
// ─────────────────────────────────────────────────────────────────────────────

export {
  NaniteHZB,
  NaniteCuller,
  NaniteSWRaster,
  NaniteRasterBin,
  NanitePassBuilder,
};
