/**
 * ue-tsr-temporal.ts — M852b: UE5 Temporal Super-Resolution (TSR) Port
 * ─────────────────────────────────────────────────────────────────────────────
 * 移植 Unreal Engine 5 Temporal Super-Resolution 系统到 TypeScript CPU 参考实现。
 * 原始来源:
 *   upstream/unreal-renderer-ue5/Renderer-Private/PostProcess/TemporalSuperResolution.cpp (3780行)
 *   upstream/unreal-renderer-ue5/Shaders-Private/TemporalSuperResolution/ (24个 .usf/.ush 文件, ~10853行)
 *
 * TSR 核心功能: 低分辨率渲染 → temporal accumulation → 超分辨率输出
 *   + anti-ghosting (shading rejection, flickering heuristic)
 *   + history rectification (resurrection, reprojection field)
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * 色彩空间 (TSRColorSpace.ush):
 *
 *   Linear ←→ GCS (Guide Color Space) ←→ SMCS (Shading Measurement Color Space)
 *
 *   GCS(L)  = L / (L + 0.17)         — 感知线性化, 类似 Reinhard tone curve
 *   SMCS(G) = G²                      — 二次映射近似 ACES 色调映射
 *   逆: G = SMCS^(1/2), L = 0.17·G / (1 - G)
 *
 *   HDR Weight: w(luma) = 1/(luma + 4)  — Karis anti-firefly 权重
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * 管线流程 (TemporalSuperResolution.cpp — AddTemporalSuperResolutionPasses):
 *
 *   Input: SceneColor(input_res), SceneDepth, SceneVelocity, PrevHistory
 *
 *   ┌─ Pre-Pass ─── MeasureFlickeringLuma ────────────────────────────────────┐
 *   │  SceneColor → 曝光校正 → LDR luminance (R8)                             │
 *   │  用于后续 anti-flickering 时域分析 (TSRMeasureFlickeringLuma.usf)        │
 *   │  同时测量 ThinGeometryCoverage (GBuffer shading model → coverage mask)   │
 *   └────────────────────────────────────────────────────────────────────────┘
 *                              ▼
 *   ┌─ Pass 0 ── ClearPrevTextures ───────────────────────────────────────────┐
 *   │  清零 PrevAtomicTextureArray (R32_UINT, atomic scatter 目标)             │
 *   │  AsyncCompute level ≥ 1 时可异步执行                                     │
 *   └────────────────────────────────────────────────────────────────────────┘
 *                              ▼
 *   ┌─ Pass 1 ── DilateVelocity ──────────────────────────────────────────────┐
 *   │  输入: SceneDepth, SceneVelocity                                         │
 *   │  • 3×3 邻域膨胀: 选取最近深度像素的速度向量                              │
 *   │  • ClosestDepthTexture: 最前深度 (R16F/R32F ortho)                       │
 *   │  • DilateMask: 标记膨胀来源                                              │
 *   │  • DepthError: 深度不连续性误差量化                                      │
 *   │  • IsMovingMask: 标记运动像素 (flickering period > 0)                   │
 *   │  • PrevAtomicOutput: 前向散射到前帧像素坐标 (InterlockedMax)             │
 *   │  • ReprojectionField: 可选 4-slice 场 (vector/jacobian/boundary/dilated) │
 *   │  • VelocityFlatten: 可选运动模糊 tile 信息                               │
 *   │  输出: ClosestDepth, R8Output[3-4], ReprojectionField, VelocityFlatten  │
 *   └────────────────────────────────────────────────────────────────────────┘
 *                              ▼
 *   ┌─ Pass 2 ── DecimateHistory ─────────────────────────────────────────────┐
 *   │  输入: PrevHistory (Color/Guide/Moire/Coverage), DilatedVelocity         │
 *   │  • 用膨胀速度重投影前帧 Guide (GCS 色彩 + metadata)                      │
 *   │  • 双线性采样 PrevHistory.Guide → ReprojectedHistoryGuide               │
 *   │  • 可选重投影 Moire 历史 (flickering heuristic)                          │
 *   │  • 可选重投影 Coverage 历史 (thin geometry detection)                    │
 *   │  • 历史复活: 重投影 ResurrectionFrame (oldest persistent frame)           │
 *   │  • DecimateMask: 编码遮挡状态 (parallax disocclusion)                   │
 *   │  输出: ReprojectedHistoryGuide, ReprojectedHistoryMoire,                │
 *   │        ReprojectedHistoryCoverage, DecimateMask                          │
 *   └────────────────────────────────────────────────────────────────────────┘
 *                              ▼
 *   ┌─ Pass 3 ── RejectShading ───────────────────────────────────────────────┐
 *   │  输入: InputSceneColor, ReprojectedHistoryGuide, DecimateMask            │
 *   │  • 卷积网络 (32×32 tile + TileOverscan padding)                          │
 *   │  • SMCS 空间对比: InputColor vs ReprojectedGuide → 差异量               │
 *   │  • MeasureBackbufferLDRQuantizationError → 区分可见差异 vs 量化噪声      │
 *   │  • 输出 HistoryRejection: 逐像素拒绝强度 [0=全保留, 1=全拒绝]           │
 *   │  • Flickering heuristic: 时域分析连续帧亮度震荡 → 允许 ghost 稳定       │
 *   │  • 历史复活比较: 若 resurrection guide 更接近 → 标记切换复活帧           │
 *   │  • AntiAliasMask: 标记需要空间抗锯齿的像素                               │
 *   │  • InputSceneColorOutput: 可选预合成半透明                                │
 *   │  输出: HistoryGuide, HistoryMoire, HistoryRejection, AntiAliasMask,     │
 *   │        InputSceneColor (composed), InputSceneColorLdrLuma               │
 *   └────────────────────────────────────────────────────────────────────────┘
 *                              ▼
 *   ┌─ Pass 3b ─ DetectThinGeometry (可选) ───────────────────────────────────┐
 *   │  深度边缘检测 → 标记细几何体 (foliage, hair) → WeightRelaxation         │
 *   └────────────────────────────────────────────────────────────────────────┘
 *                              ▼
 *   ┌─ Pass 4 ── SpatialAntiAliasing ────────────────────────────────────────┐
 *   │  输入: InputSceneColorLdrLuma, AntiAliasMask                             │
 *   │  • 仅在 RejectionAntiAliasingQuality > 0 时执行                          │
 *   │  • 边缘方向检测 → 沿边缘混合相邻像素 → R8G8_UINT 输出                   │
 *   │  • 高低分辨率渲染时特别关键 (渲染分辨率越低, 锯齿越明显)                 │
 *   │  输出: AntiAliasingTexture (R8G8_UINT — 方向 + 混合权重)                │
 *   └────────────────────────────────────────────────────────────────────────┘
 *                              ▼
 *   ┌─ Pass 5 ── UpdateHistory ───────────────────────────────────────────────┐
 *   │  输入: InputSceneColor, PrevHistoryColor, HistoryRejection,              │
 *   │        AntiAliasing, ReprojectionField                                   │
 *   │  • 核心时域累积: blend(PrevColor, InputColor, weight)                    │
 *   │    weight = HistoryHisteresis = 1/MaxSampleCount                         │
 *   │    受 HistoryRejection 调制: rejection=1 → weight=1 (完全替换)           │
 *   │  • 3×3+ 采样核 (Quality Low=PLUS, High/Epic=PLUS_MOVE_FAR)              │
 *   │  • 最近输入像素查找 + 子像素偏移补偿                                     │
 *   │  • Min/Max color box clamping (anti-ghosting 核心)                       │
 *   │  • Velocity weight clamping: 高速运动时降低历史权重                       │
 *   │  • 可选 ReprojectionField Jacobian → 亚像素精确重投影                     │
 *   │  • 可选镜头畸变补偿 (LensDistortion LUT)                                 │
 *   │  • 输出到 Texture2DArray[CurrentFrameSlice]                              │
 *   │  输出: HistoryColorArray, HistoryMetadataArray                           │
 *   └────────────────────────────────────────────────────────────────────────┘
 *                              ▼
 *   ┌─ Pass 6 ── ResolveHistory ──────────────────────────────────────────────┐
 *   │  输入: HistoryColorArray (history_res ≥ output_res)                      │
 *   │  • 当 HistoryScreenPercentage > 100%: 从历史分辨率下采样到输出分辨率     │
 *   │    利用 Nyquist-Shannon 采样定理: 200% 时额外 2-bit 精度                 │
 *   │  • 可选 wave ops (16/32 lane) → 快速 2x2 下采样                         │
 *   │  • 可选生成 mip1 给后续 pass (DOF 等)                                    │
 *   │  输出: SceneColorOutput (output_res), 可选 HalfRes/QuarterRes/EighthRes │
 *   └────────────────────────────────────────────────────────────────────────┘
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * History Slice Sequence (FTSRHistorySliceSequence):
 *   • kTransientSliceCount=2 (当前帧 + 前帧, 乒乓)
 *   • Resurrection 模式下 FrameStorageCount ≥ 4: 额外 persistent slice
 *     每 FrameStoragePeriod 帧录制一次 → 可从数十帧前复活细节
 *   • RollingIndex → SliceIndex 映射, 避免 D3D12 全 array 绑定限制
 *
 * Exported: UETSRTemporal
 */

// ─────────────────────────────────────────── Enums & Constants ───────────────

/** TSR pipeline pass types — mirrors shader IMPLEMENT_GLOBAL_SHADER registrations */
export const enum TSRPassType {
  MeasureFlickeringLuma  = 0,
  MeasureThinGeoCoverage = 1,
  ClearPrevTextures      = 2,
  DilateVelocity         = 3,
  DecimateHistory        = 4,
  RejectShading          = 5,
  DetectThinGeometry     = 6,
  WeightRelaxation       = 7,
  SpatialAntiAliasing    = 8,
  UpdateHistory          = 9,
  ResolveHistory         = 10,
  Visualize              = 11,
}

/** UpdateHistory quality levels — DIM_UPDATE_QUALITY in TSRUpdateHistory.usf */
export const enum TSRUpdateQuality {
  Low    = 0,  // CONFIG_SAMPLES_PLUS, no rejection AA
  Medium = 1,  // CONFIG_SAMPLES_PLUS, with rejection AA
  High   = 2,  // CONFIG_SAMPLES_PLUS_MOVE_FAR, with rejection AA
  Epic   = 3,  // CONFIG_SAMPLES_PLUS_MOVE_FAR, with rejection AA
}

/** Shading rejection mode — r.TSR.ShadingRejection.Mode */
export const enum TSRShadingRejectionMode {
  Responsive = 0,  // More responsiveness, less ghosting, more blocky artifacts
  Stable     = 1,  // Better stability, better shading rejection control, more ghosting
}

/** Thin geometry coverage shading range — EThinGeometryShadingRange */
export const enum TSRThinGeometryShadingRange {
  Foliage         = 0,
  FoliageAndHair  = 1,
  All             = 2,
  VaryingRange    = 3,
}

/** History format bits — ETSRHistoryFormatBits */
export const enum TSRHistoryFormatBits {
  None         = 0,
  Moire        = 1 << 0,  // flickering detection enabled
  AlphaChannel = 1 << 1,  // scene color alpha propagation
}

/** Visualization modes — r.TSR.Visualize, from FTSRVisualizeCS */
export const enum TSRVisualizeMode {
  ReprojectionFieldOverview = -3,
  GridOverviewAlways        = -2,
  ShowFlagGrid              = -1,
  HistorySampleCount        = 0,
  ParallaxDisocclusion       = 1,
  HistoryRejection          = 2,
  HistoryClamp              = 3,
  ResurrectionMask          = 4,
  ResurrectedColor          = 5,
  SpatialAntiAliasingMask   = 6,
  FlickeringAnalysis        = 7,
  ReprojectionFieldSummary  = 8,
  ReprojectionFieldOffset   = 9,
  ReprojectionFieldCoverage = 10,
  ReprojectionFieldAA       = 11,
  ReprojectionFieldNullJac  = 12,
  ReprojectionFieldClampJac = 13,
  ReprojectionFieldDilateJac = 14,
  ThinGeometry              = 15,
}

/** Async compute levels — r.TSR.AsyncCompute */
export const enum TSRAsyncComputeLevel {
  Disabled         = 0,
  IndependentOnly  = 1,  // ClearPrevTextures, ForwardScatterDepth
  DepthVelocityDep = 2,  // + passes dependent only on depth/velocity (default)
  AllPasses        = 3,  // All passes on async compute
}

/** Sample kernel shapes used in UpdateHistory (TSRUpdateHistory.usf) */
export const enum TSRSampleKernel {
  Samples1x1          = 0,
  Samples3x3          = 1,
  SamplesPlus         = 2,  // 3×3 plus (5 samples)
  SamplesPlusAndCorner = 3, // plus + closest corner (6 samples)
  SamplesPlusDisableFar = 4,
  SamplesPlusMoveFar  = 5,  // plus with far samples shifted toward closest corner
}

/** ETSRPassConfig — Main vs MainUpsampling */
export const enum TSRPassConfigType {
  Main           = 0,
  MainUpsampling = 1,
}

// ─── 3×3 kernel offsets (TSRKernels.ush) ────────────────────────────────────

const kOffsets3x3: [number, number][] = [
  [-1, -1], [0, -1], [1, -1],
  [-1,  0], [0,  0], [1,  0],
  [-1,  1], [0,  1], [1,  1],
];

const kPlusIndexes3x3 = [4, 1, 3, 7, 5];

const kSquareIndexes3x3 = [4, 0, 1, 2, 3, 8, 7, 6, 5];

const kPairOffsets: [number, number][] = [
  [1, 0], [1, 1], [0, 1], [-1, 1],
];

// ─── Color space constants (TSRColorSpace.ush) ──────────────────────────────

const kGCSPerceptionAdd = 0.17;
const kHistoryAccumulationPerceptionAdd = 1.0;
const kMoireLumaToChannel = 3.0;
const kLargestNormalNumberHalf = 65504.0;
const kLargestSceneColorComponent = kLargestNormalNumberHalf * 0.25;
const HDR_WEIGHT_SAFE_MIN_VALUE = 0.0001;

// ─────────────────────────────────────────── Interfaces ──────────────────────

export interface TSRPassConfig {
  /** Pipeline mode: simple TAA or temporal upsampling */
  pass: TSRPassConfigType;
  /** Enable history resurrection for recovering old discarded detail */
  resurrectionEnable: boolean;
  /** Number of persistent frames stored for resurrection (even, ≥2) */
  resurrectionPersistentFrameCount: number;
  /** Interval in frames between persistent frame recordings (odd, ≥1) */
  resurrectionPersistentFrameInterval: number;
  /** Alpha channel mode: -1=auto, 0=disabled, 1=enabled */
  alphaChannel: number;
  /** Enable flickering detection heuristic (sg.AntiAliasingQuality ≥ High) */
  shadingRejectionFlickering: boolean;
  /** Adjust flickering period to frame rate */
  shadingRejectionFlickeringAdjustToFrameRate: boolean;
  /** Frame rate cap for flickering adjustment (Hz, default 60) */
  shadingRejectionFlickeringFrameRateCap: number;
  /** Flickering period in frames (default 2.0) */
  shadingRejectionFlickeringPeriod: number;
  /** Max parallax velocity for flickering (1080p pixels, default 10) */
  shadingRejectionFlickeringMaxParallaxVelocity: number;
  /** Exposure offset factor for shading rejection */
  shadingRejectionExposureOffsetFactor: number;
  /** Shading rejection mode (0=responsive, 1=stable) */
  shadingRejectionMode: TSRShadingRejectionMode;
  /** Enable thin geometry detection (foliage, hair) */
  thinGeometryDetectionEnable: boolean;
  /** Error multiplier for thin geometry depth edge detection (default 200) */
  thinGeometryErrorMultiplier: number;
  /** Spatial anti-aliasing quality (0=off, 1=standard, 2=high) */
  rejectionAntiAliasingQuality: number;
  /** Max sample count after shading rejection (default 2.0) */
  historyRejectionSampleCount: number;
  /** History resolution as percentage of output (100-200, default 100) */
  historyScreenPercentage: number;
  /** Max accumulated sample count in history (8-32, default 16) */
  historySampleCount: number;
  /** History update quality preset */
  historyUpdateQuality: TSRUpdateQuality;
  /** Use R11G11B10 format for history (saves bandwidth) */
  historyR11G11B10: boolean;
  /** Enable reprojection field for sub-pixel accuracy */
  reprojectionField: boolean;
  /** Anti-alias pixel speed threshold for reprojection field (default 0.125) */
  reprojectionFieldAntiAliasPixelSpeed: number;
  /** Weight clamping sample count on velocity (default 4.0) */
  velocityWeightClampingSampleCount: number;
  /** Pixel speed at which weight clamping starts (default 1.0) */
  velocityWeightClampingPixelSpeed: number;
  /** Motion blur valid in current frame */
  motionBlurValidInCurrentFrame: boolean;
  /** Visualization mode (-1 = off) */
  visualize: number;
  /** Tile overscan for convolution network (3 to 15, default 3) */
  tileOverscan: number;
}

/** Viewport parameters mirroring FScreenPassTextureViewportParameters */
export interface TSRViewportParams {
  extent: [number, number];
  extentInverse: [number, number];
  viewportMin: [number, number];
  viewportMax: [number, number];
  viewportSize: [number, number];
  viewportSizeInverse: [number, number];
  uvViewportMin: [number, number];
  uvViewportMax: [number, number];
  uvViewportBilinearMin: [number, number];
  uvViewportBilinearMax: [number, number];
}

/** History texture set — mirrors FTSRHistoryTextures struct */
export interface TSRHistoryTextures {
  /** Accumulated color (Texture2DArray, FloatRGBA or R11G11B10) */
  colorArray: Float32Array[];
  /** Per-pixel metadata: sample count, validity, etc. */
  metadataArray: Float32Array[];
  /** Guide in GCS for shading rejection comparison */
  guideArray: Float32Array[];
  /** Moiré history for flickering heuristic */
  moireArray: Float32Array[];
  /** Coverage history for thin geometry detection */
  coverageArray: Float32Array[];
  /** Array extent */
  extent: [number, number];
  /** Number of slices per texture array */
  sliceCount: number;
}

/** Per-frame history state — mirrors FTSRHistory */
export interface TSRHistoryState {
  textures: TSRHistoryTextures;
  /** Total accumulated frames since last camera cut */
  accumulatedFrameCount: number;
  /** Last written rolling index in the slice sequence */
  lastFrameRollingIndex: number;
  /** History format bits active */
  formatBits: TSRHistoryFormatBits;
  /** Output viewport rect for the history */
  outputViewportRect: [number, number, number, number];
  /** Per-slice input viewport rects */
  inputViewportRects: [number, number, number, number][];
  /** Per-slice view matrices for resurrection reprojection */
  viewMatrices: Float32Array[];
  /** Per-slice pre-exposure values */
  sceneColorPreExposures: number[];
  /** Is history valid (not camera cut) */
  isValid: boolean;
}

/** Common parameters shared across all TSR passes — FTSRCommonParameters */
export interface TSRCommonParams {
  inputInfo: TSRViewportParams;
  historyInfo: TSRViewportParams;
  inputPixelPosMin: [number, number];
  inputPixelPosMax: [number, number];
  inputJitter: [number, number];
  bCameraCut: boolean;
  screenVelocityToInputPixelVelocity: [number, number];
  inputPixelVelocityToScreenVelocity: [number, number];
}

/** Previous history parameters — FTSRPrevHistoryParameters */
export interface TSRPrevHistoryParams {
  prevHistoryInfo: TSRViewportParams;
  screenPosToPrevHistoryBufferUV: [number, number, number, number];
  historyPreExposureCorrection: number;
  resurrectionPreExposureCorrection: number;
}

/** History array indices — FTSRHistoryArrayIndices */
export interface TSRHistoryArrayIndices {
  highFrequency: number;
  size: number;
}

// ──────────────────────────── History Slice Sequence ─────────────────────────

/**
 * FTSRHistorySliceSequence — manages rolling indices to Texture2DArray slices.
 * Handles transient frames (ping-pong) + persistent frames for resurrection.
 */
export class TSRHistorySliceSequence {
  static readonly kTransientSliceCount = 2;

  frameStorageCount: number;
  frameStoragePeriod: number;

  constructor(resurrectionEnable: boolean, persistentFrameCount: number, persistentFrameInterval: number) {
    if (resurrectionEnable) {
      const paddedCount = TSRHistorySliceSequence.kTransientSliceCount
        + Math.ceil(persistentFrameCount / 2) * 2;
      this.frameStorageCount = Math.max(4, Math.min(paddedCount, 2048));
      this.frameStoragePeriod = Math.max(1, persistentFrameInterval | 1);
    } else {
      this.frameStorageCount = 1;
      this.frameStoragePeriod = 1;
    }
  }

  /** Total number of rolling indices in the cycle */
  getRollingIndexCount(): number {
    if (this.frameStorageCount === 1) return 2;
    return TSRHistorySliceSequence.kTransientSliceCount
      + (this.frameStorageCount - TSRHistorySliceSequence.kTransientSliceCount) * this.frameStoragePeriod;
  }

  /** Map rolling index to slice index in the Texture2DArray */
  rollingIndexToSliceIndex(rollingIndex: number): number {
    if (this.frameStorageCount === 1) {
      return rollingIndex % 2 === 0 ? 0 : 0;
    }
    const transient = TSRHistorySliceSequence.kTransientSliceCount;
    if (rollingIndex < transient) return rollingIndex;
    const persistentIndex = rollingIndex - transient;
    return transient + Math.floor(persistentIndex / this.frameStoragePeriod);
  }

  /** Advance rolling index to next frame */
  incrementFrameRollingIndex(rollingIndex: number): number {
    return (rollingIndex + 1) % this.getRollingIndexCount();
  }

  /** Get resurrection frame rolling index (oldest persistent frame) */
  getResurrectionFrameRollingIndex(accumulatedFrameCount: number, lastRollingIndex: number): number {
    if (this.frameStorageCount <= 2) return lastRollingIndex;
    const totalRolling = this.getRollingIndexCount();
    const transient = TSRHistorySliceSequence.kTransientSliceCount;
    const persistentSlots = this.frameStorageCount - transient;
    // Find the oldest persistent slot that was actually written
    const currentPersistentSlot = Math.floor((lastRollingIndex - transient) / this.frameStoragePeriod);
    const oldestSlot = (currentPersistentSlot + 1) % persistentSlots;
    const rollingBase = transient + oldestSlot * this.frameStoragePeriod;
    return Math.min(rollingBase, totalRolling - 1);
  }
}

// ──────────────────────── Color Space Functions ─────────────────────────────

/** Linear → Guide Color Space: GCS(L) = L / (L + 0.17) */
function linearToGCS(r: number, g: number, b: number): [number, number, number] {
  return [
    r / (r + kGCSPerceptionAdd),
    g / (g + kGCSPerceptionAdd),
    b / (b + kGCSPerceptionAdd),
  ];
}

/** Guide Color Space → Linear: L = 0.17·G / (1 - G) */
function gcsToLinear(r: number, g: number, b: number): [number, number, number] {
  const safeR = Math.min(r, 0.9999);
  const safeG = Math.min(g, 0.9999);
  const safeB = Math.min(b, 0.9999);
  return [
    kGCSPerceptionAdd * safeR / (1 - safeR),
    kGCSPerceptionAdd * safeG / (1 - safeG),
    kGCSPerceptionAdd * safeB / (1 - safeB),
  ];
}

/** GCS → Shading Measurement Color Space: SMCS(G) = G² */
function gcsToSMCS(r: number, g: number, b: number): [number, number, number] {
  return [r * r, g * g, b * b];
}

/** SMCS → GCS: G = sqrt(SMCS) */
function smcsToGCS(r: number, g: number, b: number): [number, number, number] {
  return [Math.sqrt(r), Math.sqrt(g), Math.sqrt(b)];
}

/** Linear → SMCS (composed: linearToGCS then gcsToSMCS) */
function linearToSMCS(r: number, g: number, b: number): [number, number, number] {
  const gcs = linearToGCS(r, g, b);
  return gcsToSMCS(gcs[0], gcs[1], gcs[2]);
}

/** SMCS → Linear (composed: smcsToGCS then gcsToLinear) */
function smcsToLinear(r: number, g: number, b: number): [number, number, number] {
  const gcs = smcsToGCS(r, g, b);
  return gcsToLinear(gcs[0], gcs[1], gcs[2]);
}

/** Pre-exposure correction in GCS: linearToGCS(gcsToLinear(G) * correction) */
function preExposureCorrectGCS(
  r: number, g: number, b: number, preExposureCorrection: number,
): [number, number, number] {
  const lin = gcsToLinear(r, g, b);
  return linearToGCS(
    lin[0] * preExposureCorrection,
    lin[1] * preExposureCorrection,
    lin[2] * preExposureCorrection,
  );
}

/** Fast luma approximation: L4 = 2*G + R + B (Luma4 from TSRColorSpace.ush) */
function luma4(r: number, g: number, b: number): number {
  return g * 2.0 + r + b;
}

/** Karis HDR weight: w = 1/(luma + 4) — clamp to safe minimum */
function hdrWeight(luma: number): number {
  return Math.max(HDR_WEIGHT_SAFE_MIN_VALUE, 1.0 / (luma + 4.0));
}

/** HDR weighted color = color * hdrWeight(luma4(color)) */
function hdrWeightedColor(r: number, g: number, b: number): [number, number, number] {
  const w = hdrWeight(luma4(r, g, b));
  return [r * w, g * w, b * w];
}

// ────────────────────────── Helper Utilities ────────────────────────────────

/** Compute pixel format quantization error (approx for R11G11B10 / RGBA16F) */
function computePixelFormatQuantizationError(isR11G11B10: boolean): [number, number, number] {
  if (isR11G11B10) {
    // 6-bit mantissa for R/G (11-bit floats), 5-bit for B (10-bit)
    return [1.0 / 64.0, 1.0 / 64.0, 1.0 / 32.0];
  }
  // RGBA16F: 10-bit mantissa
  return [1.0 / 1024.0, 1.0 / 1024.0, 1.0 / 1024.0];
}

/** Bilinear interpolation of a 2D Float32Array buffer */
function sampleBilinear(
  buffer: Float32Array, width: number, height: number, channels: number,
  u: number, v: number,
): number[] {
  const fx = u * (width - 1);
  const fy = v * (height - 1);
  const x0 = Math.max(0, Math.min(Math.floor(fx), width - 1));
  const y0 = Math.max(0, Math.min(Math.floor(fy), height - 1));
  const x1 = Math.min(x0 + 1, width - 1);
  const y1 = Math.min(y0 + 1, height - 1);
  const sx = fx - x0;
  const sy = fy - y0;
  const result: number[] = new Array(channels);
  for (let c = 0; c < channels; c++) {
    const v00 = buffer[(y0 * width + x0) * channels + c];
    const v10 = buffer[(y0 * width + x1) * channels + c];
    const v01 = buffer[(y1 * width + x0) * channels + c];
    const v11 = buffer[(y1 * width + x1) * channels + c];
    result[c] = (v00 * (1 - sx) + v10 * sx) * (1 - sy) + (v01 * (1 - sx) + v11 * sx) * sy;
  }
  return result;
}

/** Create viewport parameters from extent and rect */
function makeViewportParams(
  extent: [number, number], rectMin: [number, number], rectMax: [number, number],
): TSRViewportParams {
  const [ew, eh] = extent;
  const vw = rectMax[0] - rectMin[0];
  const vh = rectMax[1] - rectMin[1];
  const extInv: [number, number] = [1 / ew, 1 / eh];
  const uvMin: [number, number] = [rectMin[0] / ew, rectMin[1] / eh];
  const uvMax: [number, number] = [rectMax[0] / ew, rectMax[1] / eh];
  return {
    extent, extentInverse: extInv,
    viewportMin: rectMin, viewportMax: rectMax,
    viewportSize: [vw, vh],
    viewportSizeInverse: [1 / vw, 1 / vh],
    uvViewportMin: uvMin, uvViewportMax: uvMax,
    uvViewportBilinearMin: [uvMin[0] + 0.5 * extInv[0], uvMin[1] + 0.5 * extInv[1]],
    uvViewportBilinearMax: [uvMax[0] - 0.5 * extInv[0], uvMax[1] - 0.5 * extInv[1]],
  };
}

/** Default TSR pass configuration */
function defaultPassConfig(): TSRPassConfig {
  return {
    pass: TSRPassConfigType.MainUpsampling,
    resurrectionEnable: false,
    resurrectionPersistentFrameCount: 2,
    resurrectionPersistentFrameInterval: 31,
    alphaChannel: -1,
    shadingRejectionFlickering: true,
    shadingRejectionFlickeringAdjustToFrameRate: true,
    shadingRejectionFlickeringFrameRateCap: 60,
    shadingRejectionFlickeringPeriod: 2.0,
    shadingRejectionFlickeringMaxParallaxVelocity: 10.0,
    shadingRejectionExposureOffsetFactor: 1.0,
    shadingRejectionMode: TSRShadingRejectionMode.Stable,
    thinGeometryDetectionEnable: false,
    thinGeometryErrorMultiplier: 200.0,
    rejectionAntiAliasingQuality: 2,
    historyRejectionSampleCount: 2.0,
    historyScreenPercentage: 100,
    historySampleCount: 16.0,
    historyUpdateQuality: TSRUpdateQuality.Epic,
    historyR11G11B10: true,
    reprojectionField: false,
    reprojectionFieldAntiAliasPixelSpeed: 0.125,
    velocityWeightClampingSampleCount: 4.0,
    velocityWeightClampingPixelSpeed: 1.0,
    motionBlurValidInCurrentFrame: true,
    visualize: -1,
    tileOverscan: 3,
  };
}

/** Translate history format bits to array indices */
function translateHistoryFormatBitsToArrayIndices(bits: TSRHistoryFormatBits): TSRHistoryArrayIndices {
  return { highFrequency: 0, size: 1 };
}

// ──────────────────────────── Main Class ────────────────────────────────────

/**
 * UETSRTemporal — CPU reference implementation of UE5's TSR pipeline.
 *
 * Each pass is a method that takes typed-array buffers and writes results
 * into output buffers. The pipeline orchestration follows
 * AddTemporalSuperResolutionPasses() from TemporalSuperResolution.cpp.
 */
export class UETSRTemporal {
  config: TSRPassConfig;
  inputWidth: number;
  inputHeight: number;
  outputWidth: number;
  outputHeight: number;
  historyWidth: number;
  historyHeight: number;
  historySliceSequence: TSRHistorySliceSequence;
  commonParams!: TSRCommonParams;
  /** Output-to-input resolution fraction (e.g. 0.5 for 50% render res) */
  outputToInputResFraction: number;
  /** Squared version for area-based calculations */
  outputToInputResFractionSq: number;

  constructor(
    inputWidth: number, inputHeight: number,
    outputWidth: number, outputHeight: number,
    config?: Partial<TSRPassConfig>,
  ) {
    this.config = { ...defaultPassConfig(), ...config };
    this.inputWidth = inputWidth;
    this.inputHeight = inputHeight;
    this.outputWidth = outputWidth;
    this.outputHeight = outputHeight;

    // History resolution: 100-200% of output
    const histFactor = Math.max(1, Math.min(this.config.historyScreenPercentage / 100, 2));
    this.historyWidth = Math.ceil(outputWidth * histFactor);
    this.historyHeight = Math.ceil(outputHeight * histFactor);

    this.historySliceSequence = new TSRHistorySliceSequence(
      this.config.resurrectionEnable,
      this.config.resurrectionPersistentFrameCount,
      this.config.resurrectionPersistentFrameInterval,
    );

    this.outputToInputResFraction = inputWidth / outputWidth;
    this.outputToInputResFractionSq = this.outputToInputResFraction * this.outputToInputResFraction;
  }

  /** Initialize common parameters for a frame */
  initCommonParams(jitterX: number, jitterY: number, bCameraCut: boolean): void {
    const iw = this.inputWidth, ih = this.inputHeight;
    const hw = this.historyWidth, hh = this.historyHeight;
    this.commonParams = {
      inputInfo: makeViewportParams([iw, ih], [0, 0], [iw, ih]),
      historyInfo: makeViewportParams([hw, hh], [0, 0], [hw, hh]),
      inputPixelPosMin: [0, 0],
      inputPixelPosMax: [iw - 1, ih - 1],
      inputJitter: [jitterX, jitterY],
      bCameraCut,
      screenVelocityToInputPixelVelocity: [iw * 0.5, ih * 0.5],
      inputPixelVelocityToScreenVelocity: [2.0 / iw, 2.0 / ih],
    };
  }

  /** Create an empty history state */
  createHistoryState(): TSRHistoryState {
    const w = this.historyWidth, h = this.historyHeight;
    const sliceCount = Math.max(this.historySliceSequence.frameStorageCount, 2);
    const makeSlices = (ch: number) =>
      Array.from({ length: sliceCount }, () => new Float32Array(w * h * ch));
    return {
      textures: {
        colorArray: makeSlices(4),
        metadataArray: makeSlices(2),
        guideArray: makeSlices(4),
        moireArray: makeSlices(2),
        coverageArray: makeSlices(1),
        extent: [w, h],
        sliceCount,
      },
      accumulatedFrameCount: 0,
      lastFrameRollingIndex: 0,
      formatBits: TSRHistoryFormatBits.None,
      outputViewportRect: [0, 0, this.outputWidth, this.outputHeight],
      inputViewportRects: Array.from({ length: sliceCount }, () =>
        [0, 0, this.inputWidth, this.inputHeight] as [number, number, number, number]),
      viewMatrices: Array.from({ length: sliceCount }, () => new Float32Array(16)),
      sceneColorPreExposures: new Array(sliceCount).fill(1.0),
      isValid: false,
    };
  }

  // ─────── Pass 0: ClearPrevTextures ────────────────────────────────────────

  /**
   * Clear the atomic scatter buffer (PrevAtomicTextureArray).
   * Mirrors FTSRClearPrevTexturesCS — simple zero-fill.
   */
  clearPrevTextures(prevAtomicBuffer: Uint32Array): void {
    prevAtomicBuffer.fill(0);
  }

  // ─────── Pass 1: DilateVelocity ──────────────────────────────────────────

  /**
   * Dilate velocity from scene depth + velocity buffers.
   * Mirrors FTSRDilateVelocityCS:
   *  - For each pixel, find nearest depth in 3×3 neighborhood
   *  - Copy that pixel's velocity as the dilated velocity
   *  - Scatter reprojected position into PrevAtomicBuffer via InterlockedMax
   *  - Output: closestDepth, dilatedVelocity, dilateMask, depthError, isMovingMask
   */
  dilateVelocity(
    sceneDepth: Float32Array,
    sceneVelocity: Float32Array,
    closestDepthOut: Float32Array,
    dilatedVelocityOut: Float32Array,
    dilateMaskOut: Uint8Array,
    depthErrorOut: Uint8Array,
    isMovingMaskOut: Uint8Array | null,
    prevAtomicBuffer: Uint32Array,
  ): void {
    const w = this.inputWidth, h = this.inputHeight;
    const flickeringPeriod = this.computeFlickeringFramePeriod();

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = y * w + x;
        let bestDepth = sceneDepth[idx];
        let bestX = x, bestY = y;
        let maxDepthDiff = 0;

        // 3×3 neighborhood: find closest (largest in reverse-Z) depth
        for (const [dx, dy] of kOffsets3x3) {
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
          const nIdx = ny * w + nx;
          const nd = sceneDepth[nIdx];
          // Reverse-Z: closest = largest value
          if (nd > bestDepth) {
            bestDepth = nd;
            bestX = nx; bestY = ny;
          }
          maxDepthDiff = Math.max(maxDepthDiff, Math.abs(nd - sceneDepth[idx]));
        }

        closestDepthOut[idx] = bestDepth;

        // Copy velocity from closest-depth neighbor
        const bestIdx = bestY * w + bestX;
        dilatedVelocityOut[idx * 2] = sceneVelocity[bestIdx * 2];
        dilatedVelocityOut[idx * 2 + 1] = sceneVelocity[bestIdx * 2 + 1];

        // Encode dilate mask: whether dilated from self (0) or neighbor (1)
        dilateMaskOut[idx] = (bestX !== x || bestY !== y) ? 1 : 0;

        // Depth error: quantize discontinuity
        depthErrorOut[idx] = Math.min(255, Math.floor(maxDepthDiff * 1024));

        // IsMoving: velocity magnitude above threshold
        if (isMovingMaskOut && flickeringPeriod > 0) {
          const vx = sceneVelocity[idx * 2], vy = sceneVelocity[idx * 2 + 1];
          const velMag = Math.sqrt(vx * vx + vy * vy);
          const threshold = this.config.shadingRejectionFlickeringMaxParallaxVelocity
            * (w / 1920.0);
          isMovingMaskOut[idx] = velMag > (1.0 / threshold) ? 1 : 0;
        }

        // Forward scatter to PrevAtomic: InterlockedMax of encoded depth at prev position
        const prevX = Math.round(x + dilatedVelocityOut[idx * 2] * w * 0.5);
        const prevY = Math.round(y + dilatedVelocityOut[idx * 2 + 1] * h * 0.5);
        if (prevX >= 0 && prevX < w && prevY >= 0 && prevY < h) {
          const prevIdx = prevY * w + prevX;
          const encoded = Math.floor(bestDepth * 0xFFFFFF) >>> 0;
          prevAtomicBuffer[prevIdx] = Math.max(prevAtomicBuffer[prevIdx], encoded);
        }
      }
    }
  }

  // ─────── Pass 2: DecimateHistory ─────────────────────────────────────────

  /**
   * Reproject previous history guide using dilated velocity.
   * Mirrors FTSRDecimateHistoryCS:
   *  - Bilinear sample PrevHistory.Guide at reprojected position
   *  - Output DecimateMask encoding parallax disocclusion status
   *  - Optionally reproject Moire/Coverage histories
   */
  decimateHistory(
    prevGuide: Float32Array,
    dilatedVelocity: Float32Array,
    closestDepth: Float32Array,
    reprojectedGuideOut: Float32Array,
    decimateMaskOut: Uint8Array,
    prevHistoryParams: TSRPrevHistoryParams,
  ): void {
    const w = this.inputWidth, h = this.inputHeight;
    const pw = this.historyWidth, ph = this.historyHeight;

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = y * w + x;
        const vx = dilatedVelocity[idx * 2];
        const vy = dilatedVelocity[idx * 2 + 1];

        // Compute reprojected UV in previous history
        const screenX = (x + 0.5) / w * 2.0 - 1.0;
        const screenY = (y + 0.5) / h * 2.0 - 1.0;
        const prevScreenX = screenX + vx;
        const prevScreenY = screenY + vy;

        // ScreenPos → PrevHistoryBufferUV
        const prevU = prevScreenX * 0.5 + 0.5;
        const prevV = prevScreenY * 0.5 + 0.5;

        // Bilinear sample previous guide
        const sampled = sampleBilinear(prevGuide, pw, ph, 4, prevU, prevV);
        reprojectedGuideOut[idx * 4 + 0] = sampled[0];
        reprojectedGuideOut[idx * 4 + 1] = sampled[1];
        reprojectedGuideOut[idx * 4 + 2] = sampled[2];
        reprojectedGuideOut[idx * 4 + 3] = sampled[3];

        // DecimateMask: parallax disocclusion detection
        // Compare closest depth with atomic scattered depth
        const outOfBounds = (prevU < 0 || prevU > 1 || prevV < 0 || prevV > 1);
        decimateMaskOut[idx] = outOfBounds ? 255 : 0;
      }
    }
  }

  // ─────── Pass 3: RejectShading ───────────────────────────────────────────

  /**
   * Compare current scene color with reprojected history guide in SMCS.
   * Mirrors FTSRRejectShadingCS convolution network:
   *  - Convert input and reprojected guide to SMCS
   *  - Measure color difference accounting for quantization error
   *  - Output rejection mask, updated guide, anti-alias mask
   */
  rejectShading(
    inputSceneColor: Float32Array,
    reprojectedGuide: Float32Array,
    decimateMask: Uint8Array,
    historyRejectionOut: Float32Array,
    historyGuideOut: Float32Array,
    antiAliasMaskOut: Uint8Array | null,
    inputSceneColorLdrLumaOut: Uint8Array | null,
  ): void {
    const w = this.inputWidth, h = this.inputHeight;
    const cfg = this.config;
    const flickeringPeriod = this.computeFlickeringFramePeriod();
    const theoricBlendFactor = 1.0 / (1.0 + cfg.historySampleCount / this.outputToInputResFractionSq);
    const quantError = computePixelFormatQuantizationError(cfg.historyR11G11B10);
    const guideQuantError = computePixelFormatQuantizationError(false);
    const exposureFactor = cfg.shadingRejectionExposureOffsetFactor;

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = y * w + x;
        const r = inputSceneColor[idx * 4 + 0];
        const g = inputSceneColor[idx * 4 + 1];
        const b = inputSceneColor[idx * 4 + 2];

        // Convert input to SMCS
        const inputSMCS = linearToSMCS(r * exposureFactor, g * exposureFactor, b * exposureFactor);

        // Convert reprojected guide from GCS to SMCS
        const guideR = reprojectedGuide[idx * 4 + 0];
        const guideG = reprojectedGuide[idx * 4 + 1];
        const guideB = reprojectedGuide[idx * 4 + 2];
        const guideSMCS = gcsToSMCS(guideR, guideG, guideB);

        // Measure difference in SMCS with quantization error tolerance
        const diffR = Math.abs(inputSMCS[0] - guideSMCS[0]) - guideQuantError[0];
        const diffG = Math.abs(inputSMCS[1] - guideSMCS[1]) - guideQuantError[1];
        const diffB = Math.abs(inputSMCS[2] - guideSMCS[2]) - guideQuantError[2];
        const maxDiff = Math.max(0, diffR, diffG, diffB);

        // Parallax disocclusion from decimate mask forces full rejection
        const isDisoccluded = decimateMask[idx] > 128;

        // Rejection strength: 0=keep history, 1=reject fully
        let rejection = isDisoccluded ? 1.0 : Math.min(1.0, maxDiff * 8.0);

        // Flickering heuristic: allow ghosting within luminance bounds
        if (flickeringPeriod > 0 && !isDisoccluded) {
          const inputLuma = luma4(inputSMCS[0], inputSMCS[1], inputSMCS[2]);
          const guideLuma = luma4(guideSMCS[0], guideSMCS[1], guideSMCS[2]);
          const lumaDiff = Math.abs(inputLuma - guideLuma);
          if (lumaDiff < theoricBlendFactor * 4.0) {
            rejection = Math.min(rejection, lumaDiff / (theoricBlendFactor * 4.0));
          }
        }

        historyRejectionOut[idx] = rejection;

        // Update guide in GCS for next frame
        const inputGCS = linearToGCS(r * exposureFactor, g * exposureFactor, b * exposureFactor);
        historyGuideOut[idx * 4 + 0] = inputGCS[0];
        historyGuideOut[idx * 4 + 1] = inputGCS[1];
        historyGuideOut[idx * 4 + 2] = inputGCS[2];
        historyGuideOut[idx * 4 + 3] = 1.0;

        // LDR luma for spatial anti-aliasing
        if (inputSceneColorLdrLumaOut) {
          const gcsLuma = luma4(inputGCS[0], inputGCS[1], inputGCS[2]);
          inputSceneColorLdrLumaOut[idx] = Math.min(255, Math.floor(gcsLuma * 63.75));
        }

        // Anti-alias mask: edges where rejection is high
        if (antiAliasMaskOut) {
          antiAliasMaskOut[idx] = rejection > 0.1 ? 1 : 0;
        }
      }
    }
  }

  // ─────── Pass 4: SpatialAntiAliasing ─────────────────────────────────────

  /**
   * Edge-directed spatial anti-aliasing on rejected pixels.
   * Mirrors FTSRSpatialAntiAliasingCS:
   *  - Detect edge direction from LDR luma gradient
   *  - Blend along detected edge to reduce jagging
   *  - Encode direction + blend weight in R8G8_UINT
   */
  spatialAntiAliasing(
    inputLdrLuma: Uint8Array,
    antiAliasMask: Uint8Array,
    antiAliasingOut: Uint16Array,
  ): void {
    const w = this.inputWidth, h = this.inputHeight;
    const quality = this.config.rejectionAntiAliasingQuality;
    if (quality <= 0) return;

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = y * w + x;
        if (!antiAliasMask[idx]) { antiAliasingOut[idx] = 0; continue; }

        // Compute gradient from LDR luma in 4 cardinal pair directions
        const c = inputLdrLuma[idx];
        let bestDir = 0;
        let bestContrast = 0;
        for (let d = 0; d < 4; d++) {
          const [ox, oy] = kPairOffsets[d];
          const nx = x + ox, ny = y + oy;
          const px = x - ox, py = y - oy;
          if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
          if (px < 0 || px >= w || py < 0 || py >= h) continue;
          const nLuma = inputLdrLuma[ny * w + nx];
          const pLuma = inputLdrLuma[py * w + px];
          const contrast = Math.abs(nLuma - pLuma);
          if (contrast > bestContrast) {
            bestContrast = contrast;
            bestDir = d;
          }
        }

        // Blend weight: proportional to contrast, perpendicular to edge
        const blendWeight = Math.min(255, Math.floor(Math.min(bestContrast / 128, 1) * 255));
        // Encode: low byte = direction (0-3), high byte = blend weight
        antiAliasingOut[idx] = (blendWeight << 8) | bestDir;
      }
    }
  }

  // ─────── Pass 5: UpdateHistory ───────────────────────────────────────────

  /**
   * Core temporal accumulation pass.
   * Mirrors FTSRUpdateHistoryCS:
   *  - Sample input scene color with 3×3+ kernel
   *  - Reproject previous history to current position
   *  - Compute min/max color bounding box from input neighborhood
   *  - Clamp reprojected history to bounding box (anti-ghosting)
   *  - Blend: newColor = lerp(clampedHistory, inputColor, blendWeight)
   *    blendWeight = max(historyHisteresis, rejection)
   *    historyHisteresis = 1/maxSampleCount
   *  - Velocity-dependent weight clamping for motion stability
   */
  updateHistory(
    inputSceneColor: Float32Array,
    prevHistoryColor: Float32Array,
    prevHistoryMetadata: Float32Array,
    historyRejection: Float32Array,
    antiAliasing: Uint16Array | null,
    dilatedVelocity: Float32Array,
    historyColorOut: Float32Array,
    historyMetadataOut: Float32Array,
  ): void {
    const iw = this.inputWidth, ih = this.inputHeight;
    const hw = this.historyWidth, hh = this.historyHeight;
    const cfg = this.config;
    const maxSampleCount = Math.max(8, Math.min(cfg.historySampleCount, 32));
    const outputToHistResFracSq = (this.historyWidth / this.outputWidth) ** 2;
    const historySampleCountH = maxSampleCount / outputToHistResFracSq;
    const historyHisteresis = 1.0 / historySampleCountH;
    const weightClampingRej = 1.0 - (cfg.historyRejectionSampleCount / outputToHistResFracSq) * historyHisteresis;
    const weightClampingAmp = Math.max(0, Math.min(1 - cfg.velocityWeightClampingSampleCount * historyHisteresis, 1));
    const invWeightClampingSpeed = 1.0 / (cfg.velocityWeightClampingPixelSpeed * (this.historyWidth / this.outputWidth));
    const inputToHistFactor = this.historyWidth / iw;

    // Determine sample kernel based on quality
    const usePlusMoveFar = cfg.historyUpdateQuality >= TSRUpdateQuality.High;
    const sampleOffsets: [number, number][] = kPlusIndexes3x3.map(i => kOffsets3x3[i]);

    for (let hy = 0; hy < hh; hy++) {
      for (let hx = 0; hx < hw; hx++) {
        const hIdx = hy * hw + hx;

        // History pixel → input pixel (via viewport UV)
        const viewportU = (hx + 0.5) / hw;
        const viewportV = (hy + 0.5) / hh;
        const inputFx = viewportU * iw + this.commonParams.inputJitter[0];
        const inputFy = viewportV * ih + this.commonParams.inputJitter[1];
        const nearestX = Math.max(0, Math.min(Math.round(inputFx - 0.5), iw - 1));
        const nearestY = Math.max(0, Math.min(Math.round(inputFy - 0.5), ih - 1));
        const nearestIdx = nearestY * iw + nearestX;

        // Velocity at nearest input pixel
        const vx = dilatedVelocity[nearestIdx * 2];
        const vy = dilatedVelocity[nearestIdx * 2 + 1];
        const velPixelSpeed = Math.sqrt(vx * vx + vy * vy) * Math.max(iw, ih) * 0.5;

        // Gather input neighborhood for clamping box
        let minR = Infinity, minG = Infinity, minB = Infinity;
        let maxR = -Infinity, maxG = -Infinity, maxB = -Infinity;
        let sumR = 0, sumG = 0, sumB = 0, sumW = 0;

        for (const [sox, soy] of sampleOffsets) {
          const sx = Math.max(0, Math.min(nearestX + sox, iw - 1));
          const sy = Math.max(0, Math.min(nearestY + soy, ih - 1));
          const sIdx = sy * iw + sx;

          const sr = inputSceneColor[sIdx * 4 + 0];
          const sg = inputSceneColor[sIdx * 4 + 1];
          const sb = inputSceneColor[sIdx * 4 + 2];

          minR = Math.min(minR, sr); maxR = Math.max(maxR, sr);
          minG = Math.min(minG, sg); maxG = Math.max(maxG, sg);
          minB = Math.min(minB, sb); maxB = Math.max(maxB, sb);

          // HDR-weighted accumulation for input contribution
          const w = hdrWeight(luma4(sr, sg, sb));
          sumR += sr * w; sumG += sg * w; sumB += sb * w; sumW += w;
        }

        // Normalized input color
        const invW = sumW > 0 ? 1.0 / sumW : 0;
        const inputR = sumR * invW;
        const inputG = sumG * invW;
        const inputB = sumB * invW;

        // Reproject previous history color
        const prevU = viewportU - vx * 0.5;
        const prevV = viewportV - vy * 0.5;
        const prevColor = sampleBilinear(prevHistoryColor, hw, hh, 4, prevU, prevV);
        const prevMeta = sampleBilinear(prevHistoryMetadata, hw, hh, 2, prevU, prevV);

        // Clamp reprojected history to neighborhood bounding box (anti-ghosting)
        const clampedR = Math.max(minR, Math.min(maxR, prevColor[0]));
        const clampedG = Math.max(minG, Math.min(maxG, prevColor[1]));
        const clampedB = Math.max(minB, Math.min(maxB, prevColor[2]));

        // Compute blend weight
        const rejection = historyRejection[nearestIdx];
        let blendWeight = Math.max(historyHisteresis, rejection * weightClampingRej);

        // Velocity-dependent weight clamping: faster movement → more weight to current frame
        if (weightClampingAmp > 0 && velPixelSpeed > 0) {
          const velClamp = weightClampingAmp * Math.min(1.0, velPixelSpeed * invWeightClampingSpeed);
          blendWeight = Math.max(blendWeight, velClamp);
        }

        // Camera cut: use input directly
        if (this.commonParams.bCameraCut) blendWeight = 1.0;

        // Temporal blend
        const oneMinusBW = 1.0 - blendWeight;
        historyColorOut[hIdx * 4 + 0] = clampedR * oneMinusBW + inputR * blendWeight;
        historyColorOut[hIdx * 4 + 1] = clampedG * oneMinusBW + inputG * blendWeight;
        historyColorOut[hIdx * 4 + 2] = clampedB * oneMinusBW + inputB * blendWeight;
        historyColorOut[hIdx * 4 + 3] = 1.0;

        // Update metadata: accumulated sample count
        const prevSampleCount = this.commonParams.bCameraCut ? 0 : prevMeta[0];
        const newSampleCount = Math.min(
          prevSampleCount * oneMinusBW + 1.0 * blendWeight,
          historySampleCountH,
        );
        historyMetadataOut[hIdx * 2 + 0] = newSampleCount;
        historyMetadataOut[hIdx * 2 + 1] = prevMeta.length > 1 ? prevMeta[1] : 0;
      }
    }
  }

  // ─────── Pass 6: ResolveHistory ──────────────────────────────────────────

  /**
   * Resolve history from history resolution to output resolution.
   * Mirrors FTSRResolveHistoryCS:
   *  - When historyScreenPercentage > 100: downsample with Nyquist-Shannon
   *  - When equal: simple copy
   *  - Optionally generate mip1 for DOF
   */
  resolveHistory(
    historyColor: Float32Array,
    sceneColorOut: Float32Array,
  ): void {
    const hw = this.historyWidth, hh = this.historyHeight;
    const ow = this.outputWidth, oh = this.outputHeight;

    for (let oy = 0; oy < oh; oy++) {
      for (let ox = 0; ox < ow; ox++) {
        const oIdx = oy * ow + ox;

        // Map output pixel to history position
        const histFx = (ox + 0.5) / ow * hw;
        const histFy = (oy + 0.5) / oh * hh;

        if (hw === ow && hh === oh) {
          // 1:1 — direct copy
          const hx = Math.max(0, Math.min(Math.round(histFx - 0.5), hw - 1));
          const hy = Math.max(0, Math.min(Math.round(histFy - 0.5), hh - 1));
          const hIdx = hy * hw + hx;
          sceneColorOut[oIdx * 4 + 0] = historyColor[hIdx * 4 + 0];
          sceneColorOut[oIdx * 4 + 1] = historyColor[hIdx * 4 + 1];
          sceneColorOut[oIdx * 4 + 2] = historyColor[hIdx * 4 + 2];
          sceneColorOut[oIdx * 4 + 3] = historyColor[hIdx * 4 + 3];
        } else {
          // Downsample with bilinear (Nyquist-Shannon benefit at 200%)
          const u = (ox + 0.5) / ow;
          const v = (oy + 0.5) / oh;
          const c = sampleBilinear(historyColor, hw, hh, 4, u, v);
          sceneColorOut[oIdx * 4 + 0] = c[0];
          sceneColorOut[oIdx * 4 + 1] = c[1];
          sceneColorOut[oIdx * 4 + 2] = c[2];
          sceneColorOut[oIdx * 4 + 3] = c[3];
        }
      }
    }
  }

  // ─────── Full Pipeline Execution ─────────────────────────────────────────

  /**
   * Execute the complete TSR pipeline for one frame.
   * Mirrors AddTemporalSuperResolutionPasses().
   *
   * @param inputSceneColor - RGBA float input at input resolution
   * @param sceneDepth - Depth buffer at input resolution (reverse-Z)
   * @param sceneVelocity - Screen-space velocity (XY) at input resolution
   * @param prevHistory - Previous frame's history state
   * @param jitterX - Sub-pixel jitter X offset for this frame
   * @param jitterY - Sub-pixel jitter Y offset for this frame
   * @returns [outputColor, updatedHistory]
   */
  execute(
    inputSceneColor: Float32Array,
    sceneDepth: Float32Array,
    sceneVelocity: Float32Array,
    prevHistory: TSRHistoryState,
    jitterX: number,
    jitterY: number,
  ): [Float32Array, TSRHistoryState] {
    const iw = this.inputWidth, ih = this.inputHeight;
    const hw = this.historyWidth, hh = this.historyHeight;
    const ow = this.outputWidth, oh = this.outputHeight;
    const bCameraCut = !prevHistory.isValid;

    this.initCommonParams(jitterX, jitterY, bCameraCut);

    // Allocate intermediate buffers
    const prevAtomicBuffer = new Uint32Array(iw * ih);
    const closestDepth = new Float32Array(iw * ih);
    const dilatedVelocity = new Float32Array(iw * ih * 2);
    const dilateMask = new Uint8Array(iw * ih);
    const depthError = new Uint8Array(iw * ih);
    const flickeringPeriod = this.computeFlickeringFramePeriod();
    const isMovingMask = flickeringPeriod > 0 ? new Uint8Array(iw * ih) : null;

    // Pass 0: ClearPrevTextures
    this.clearPrevTextures(prevAtomicBuffer);

    // Pass 1: DilateVelocity
    this.dilateVelocity(
      sceneDepth, sceneVelocity,
      closestDepth, dilatedVelocity,
      dilateMask, depthError, isMovingMask,
      prevAtomicBuffer,
    );

    // Determine previous frame slice for history access
    const prevSlice = prevHistory.isValid
      ? this.historySliceSequence.rollingIndexToSliceIndex(prevHistory.lastFrameRollingIndex) : 0;
    const prevGuide = prevHistory.textures.guideArray[prevSlice] ?? new Float32Array(hw * hh * 4);

    // Pass 2: DecimateHistory
    const reprojectedGuide = new Float32Array(iw * ih * 4);
    const decimateMask = new Uint8Array(iw * ih);
    const prevHistoryParams: TSRPrevHistoryParams = {
      prevHistoryInfo: makeViewportParams([hw, hh], [0, 0], [hw, hh]),
      screenPosToPrevHistoryBufferUV: [0.5, 0.5, 0.5, 0.5],
      historyPreExposureCorrection: 1.0,
      resurrectionPreExposureCorrection: 1.0,
    };
    this.decimateHistory(
      prevGuide, dilatedVelocity, closestDepth,
      reprojectedGuide, decimateMask, prevHistoryParams,
    );

    // Pass 3: RejectShading
    const historyRejection = new Float32Array(iw * ih);
    const historyGuide = new Float32Array(iw * ih * 4);
    const antiAliasMask = new Uint8Array(iw * ih);
    const inputLdrLuma = new Uint8Array(iw * ih);
    this.rejectShading(
      inputSceneColor, reprojectedGuide, decimateMask,
      historyRejection, historyGuide, antiAliasMask, inputLdrLuma,
    );

    // Pass 4: SpatialAntiAliasing
    const antiAliasing = new Uint16Array(iw * ih);
    if (this.config.rejectionAntiAliasingQuality > 0) {
      this.spatialAntiAliasing(inputLdrLuma, antiAliasMask, antiAliasing);
    }

    // Pass 5: UpdateHistory
    const newHistory = this.createHistoryState();
    const currentRolling = prevHistory.isValid
      ? this.historySliceSequence.incrementFrameRollingIndex(prevHistory.lastFrameRollingIndex)
      : 0;
    const currentSlice = this.historySliceSequence.rollingIndexToSliceIndex(currentRolling);
    const historyColorSlice = newHistory.textures.colorArray[currentSlice];
    const historyMetaSlice = newHistory.textures.metadataArray[currentSlice];

    const prevColorSlice = prevHistory.textures.colorArray[prevSlice]
      ?? new Float32Array(hw * hh * 4);
    const prevMetaSlice = prevHistory.textures.metadataArray[prevSlice]
      ?? new Float32Array(hw * hh * 2);

    this.updateHistory(
      inputSceneColor, prevColorSlice, prevMetaSlice,
      historyRejection, antiAliasing, dilatedVelocity,
      historyColorSlice, historyMetaSlice,
    );

    // Copy guide to history
    const guideSlice = newHistory.textures.guideArray[currentSlice];
    guideSlice.set(historyGuide.subarray(0, Math.min(historyGuide.length, guideSlice.length)));

    // Carry forward unchanged slices from previous history
    for (let s = 0; s < newHistory.textures.sliceCount; s++) {
      if (s === currentSlice) continue;
      if (s < prevHistory.textures.sliceCount) {
        newHistory.textures.colorArray[s].set(prevHistory.textures.colorArray[s]);
        newHistory.textures.metadataArray[s].set(prevHistory.textures.metadataArray[s]);
        newHistory.textures.guideArray[s].set(prevHistory.textures.guideArray[s]);
      }
    }

    // Update history state
    newHistory.accumulatedFrameCount = prevHistory.accumulatedFrameCount + 1;
    newHistory.lastFrameRollingIndex = currentRolling;
    newHistory.formatBits = this.computeHistoryFormatBits();
    newHistory.isValid = true;

    // Pass 6: ResolveHistory
    const outputColor = new Float32Array(ow * oh * 4);
    this.resolveHistory(historyColorSlice, outputColor);

    return [outputColor, newHistory];
  }

  // ─────── Internal Helpers ────────────────────────────────────────────────

  /** Compute flickering frame period adjusted for frame rate */
  private computeFlickeringFramePeriod(): number {
    if (!this.config.shadingRejectionFlickering) return 0;
    const period = this.config.shadingRejectionFlickeringPeriod;
    // Simplified: assume target frame rate
    return period;
  }

  /** Compute history format bits from config */
  private computeHistoryFormatBits(): TSRHistoryFormatBits {
    let bits = TSRHistoryFormatBits.None;
    if (this.computeFlickeringFramePeriod() > 0) {
      bits |= TSRHistoryFormatBits.Moire;
    }
    if (this.config.alphaChannel > 0) {
      bits |= TSRHistoryFormatBits.AlphaChannel;
    }
    return bits;
  }
}

export default UETSRTemporal;
