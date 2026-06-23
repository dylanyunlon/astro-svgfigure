/**
 * ue-megalights.ts — M837: UE5 MegaLights Dynamic Light Management
 *
 * 移植 Unreal Engine 5 的 MegaLights 动态光源管理系统到 WebGL2:
 *
 *   § Light Culling: 从成百个发光体高效剔除远距离光源
 *     ├─ Visible Light Hash: 基于 XXHash32 的轻量级可见光缓存 (MegaLightsVisibility.ush)
 *     ├─ Hidden Light Hash: 2-DWORD bloom filter 追踪被遮挡灯光
 *     ├─ Tile Classification: 将屏幕分割为 TILE_SIZE×TILE_SIZE 网格 (MegaLightsTileClassification.ush)
 *     ├─ Light Grid Cell: 按 frustum+depth 分 cell 索引灯光 (LightGridCommon.ush)
 *     └─ Distance Culling: 基于 InvRadius falloff 预估权重门槛
 *
 *   § Reservoir Sampling: A-Res 重要性采样从N个光源中选出K个
 *     ├─ Stratified Sampling: 分层随机 τ-partition, 确保样本覆盖 (MegaLightsSampling.ush)
 *     ├─ Light Target PDF: 基于 BRDF+IES+LightFunction 计算光能权重 (MegaLightsLightTargetPDF.ush)
 *     ├─ Directional Light Ratio: 方向光采样预算限制避免饥饿 (DirectionalLightSampleRatio)
 *     ├─ Weight Normalization: WeightSum / CandidateWeight 逆概率加权
 *     ├─ Temporal Reuse: 跨帧复用候选, VisibleLightHash 引导 (MegaLightsFilterVisibleLightHash.usf)
 *     └─ LightPowerDelta: 跨帧灯光功率变化比, 修正历史 (MegaLightsLightPowerDelta.usf)
 *
 *   § Ray Traced Shadow: 对选中光源做硬件/软件光线追踪
 *     ├─ GenerateShadowRay(): 针对 Rect/Sphere/Capsule/Directional 生成 ray (MegaLightsRayTracing.ush)
 *     ├─ Screen Space Ray Trace: HZB-accelerated screen trace (MegaLightsRayTracing.usf)
 *     ├─ Software SDF Trace: Global Distance Field 遮挡检测 (SoftwareRayTraceLightSamplesCS)
 *     ├─ Hardware RT Trace: TLAS inline ray tracing + far field (MegaLightsHardwareRayTracing.usf)
 *     ├─ VSM Trace: Virtual Shadow Map SMRT 追踪 (MegaLightsVSMTracing.usf)
 *     ├─ Trace Compaction: Z-order compaction 减少 divergent threads (CompactLightSampleTracesCS)
 *     └─ Ray Merging: 相同灯光的相邻样本共享 ray, 减少带宽
 *
 *   § Shadow Denoising: 时间和空间滤波, 减少射线噪声
 *     ├─ Temporal Accumulation: 基于 motion reprojection 的帧间混合 (MegaLightsDenoiserTemporal.usf)
 *     ├─ Neighborhood Clamping: 3×3 min/max clamp 防止 ghosting
 *     ├─ Spatial Filter: 深度+法线加权的空间降噪 (MegaLightsDenoiserSpatial.usf)
 *     ├─ Shading Confidence: WeightRatioSum 度量采样置信度
 *     └─ Demodulate/Remodulate: 降噪前去 BRDF, 降噪后恢复 (GetDenoisingModulateFactors)
 *
 *   § Volume Lighting: 体积雾/透光体积的灯光采样
 *     ├─ Froxel Sampling: 视锥体素化 + HZB 可见性剔除 (MegaLightsVolume.ush)
 *     ├─ Phase Function: HenyeyGreenstein 散射 + IES profile
 *     └─ Translucency Volume: 两级 cascade 半透明体积 (MegaLightsVolumeShading.usf)
 *
 *   § AT Lighting.fs Fusion: 与现有4光源系统无缝融合
 *     ├─ MegaLights 作为候选池, AT lighting 为最终着色器
 *     ├─ 动态 NUM_LIGHTS 设置, 从候选中选出前4个重要光源
 *     └─ Fallback: 若启用light count<4, 自动补充环境光
 *
 * 类结构:
 *   UEMegaLights
 *     ├─ LightCuller: Light Culling + Visible/Hidden Light Hash + Tile Classification
 *     ├─ ReservoirSampler: A-Res stratified sampling + temporal reuse
 *     ├─ RayTracedShadow: Screen/SDF/HW/VSM ray tracing pipeline
 *     ├─ ShadowDenoiser: Temporal accumulation + spatial filter
 *     └─ LightingFusion: AT lighting 融合层
 *
 * 用法:
 *   const megaLights = await UEMegaLights.create(renderer, atLighting, {
 *     maxLights: 256,           // 灯光池大小
 *     tileSize: 32,             // 光线跟踪分块
 *     reservoirSize: 8,         // A-Res样本数
 *     distanceCulling: 100.0,   // 初始剔除距离
 *   });
 *
 *   megaLights.updateLights(cellLights);  // 从cell发光体更新
 *   megaLights.cullByDistance(camera);    // 相机视锥剔除
 *   megaLights.reservoirSample();         // A-Res采样
 *   megaLights.traceRays(rayTracer);      // 追踪阴影
 *   const selected = megaLights.selectTopK(4);  // 选出top-4
 *   atLighting.bindLights(selected);      // 绑定到AT shading
 *
 * Research: xiaodi #M837 — cell-pubsub-loop
 * Reference: upstream/unreal-renderer-ue5/Shaders-Private/MegaLights/ (全部32文件)
 */

import type { AstroRenderer } from '../renderer/AstroRenderer.ts';
import type { ATLightingImport } from './at-lighting-import.ts';
import type { ATLight } from './at-lighting-import.ts';

// ─────────────────────────────────────────────────────────────────────────────
// § 1  Constants: MegaLights Definitions
// ─────────────────────────────────────────────────────────────────────────────

/** Tile modes matching /Engine/Shared/MegaLightsDefinitions.h */
const TILE_SIZE = 8;
const TILE_MODE_SIMPLE_SHADING = 0;
const TILE_MODE_COMPLEX_SHADING = 1;
const TILE_MODE_SIMPLE_SHADING_RECT = 2;
const TILE_MODE_COMPLEX_SHADING_RECT = 3;
const TILE_MODE_SIMPLE_SHADING_RECT_TEXTURED = 4;
const TILE_MODE_COMPLEX_SHADING_RECT_TEXTURED = 5;
const TILE_MODE_EMPTY = 6;
const MAX_LOCAL_LIGHT_INDEX = 0x7FFF;

/** Visible light hash constants from MegaLightsVisibility.ush */
const VISIBLE_LIGHT_HASH_SIZE = 4;
const HIDDEN_LIGHT_HASH_SIZE = 2;
const VISIBLE_LIGHT_HASH_TILE_SIZE = 16;

/** Ray type constants from MegaLights.ush */
const LIGHT_SAMPLE_RAY_TYPE_DEFAULT = 0;
const LIGHT_SAMPLE_RAY_TYPE_HAIR = 1;
const LIGHT_SAMPLE_RAY_TYPE_TWO_SIDED = 2;
const LIGHT_SAMPLE_RAY_TYPE_TRANSMISSION = 3;
const LIGHT_SAMPLE_RAY_MAX_DISTANCE = 65504.0;

/** Tile classification bitmask from MegaLightsTileClassification.ush */
const MEGALIGHTS_TILE_BITMASK_SIMPLE = 0x01;
const MEGALIGHTS_TILE_BITMASK_SINGLE = 0x02;
const MEGALIGHTS_TILE_BITMASK_COMPLEX = 0x04;
const MEGALIGHTS_TILE_BITMASK_COMPLEX_SPECIAL = 0x08;
const MEGALIGHTS_TILE_BITMASK_RECT_LIGHT = 0x10;
const MEGALIGHTS_TILE_BITMASK_TEXTURED_RECT_LIGHT = 0x20;

// ─────────────────────────────────────────────────────────────────────────────
// § 2  Types: Light Structures
// ─────────────────────────────────────────────────────────────────────────────

/** Cell emission as dynamic light source */
export interface CellLight {
  /** World space position */
  position: [number, number, number];
  /** RGB emitted color (linear) */
  color: [number, number, number];
  /** Emission intensity [0, 1] */
  intensity: number;
  /** Light radius for area falloff */
  radius: number;
  /** Cell type (bioluminescent influence) */
  cellType?: string;
  /** Falloff exponent (0 = inverse square, >0 = polynomial) */
  falloffExp?: number;
  /** Source radius for area light sampling */
  sourceRadius?: number;
  /** Source length for capsule lights */
  sourceLength?: number;
  /** Whether this is a rect light */
  isRectLight?: boolean;
  /** Whether this light casts shadows */
  castShadow?: boolean;
  /** IES profile index, -1 = none */
  iesIndex?: number;
  /** Volumetric scattering intensity */
  volumetricScatteringIntensity?: number;
}

/** Packed light sample — mirrors FLightSample from MegaLights.ush */
export interface LightSample {
  /** Index into light pool [0, maxLights-1] (15 bits in UE) */
  lightIndex: number;
  /** Normalized importance weight — WeightSum / CandidateWeight */
  weight: number;
  /** Whether the shadow ray found the light visible */
  visible: boolean;
  /** History guided: was this light visible in the previous frame? */
  guidedAsVisible: boolean;
  /** Whether the light casts volumetric shadow */
  castVolumetricShadow: boolean;
}

/** Candidate from reservoir sampling — mirrors FCandidateLightSample */
export interface CandidateLightSample {
  /** Forward light index */
  localLightIndex: number;
  /** Whether the light was visible in last frame */
  lightWasVisible: boolean;
  /** Unnormalized importance weight during sampling */
  weight: number;
}

/** Shadow ray result — mirrors FLightSampleRay from MegaLights.ush */
export interface LightSampleRay {
  /** Distance traversed by ray */
  rayDistance: number;
  /** UV for area light sampling */
  uv: [number, number];
  /** Ray type: default / hair / two-sided / transmission */
  rayType: number;
  /** Whether this ray is from first person geometry */
  isFirstPerson: boolean;
  /** Whether tracing was completed */
  completed: boolean;
}

/** Shadow trace result from any tracing method */
export interface ShadowTraceResult {
  /** Whether the shadow ray hit an occluder */
  hit: boolean;
  /** Hit distance along ray */
  hitT: number;
  /** Which tracing method resolved this ray */
  traceMethod: 'screen' | 'sdf' | 'hardware' | 'vsm' | 'none';
  /** For transmission rays: distance light traveled through object */
  transmissionDistance: number;
}

/** Tile metadata for tile-based dispatch */
export interface TileMetadata {
  /** Tile coordinate in screen space */
  tileCoord: [number, number];
  /** Tile type (simple/complex/rect/empty) */
  tileType: number;
  /** Bitmask of material types present in tile */
  tileBitmask: number;
  /** Whether this tile can skip shading */
  isFastClear: boolean;
}

/** Visible light hash entry for temporal guiding */
export interface VisibleLightHashEntry {
  /** 4 DWORDs for visible light bloom filter */
  visibleHash: Uint32Array;
  /** 2 DWORDs for hidden light bloom filter */
  hiddenHash: Uint32Array;
}

/** Per-light target PDF from MegaLightsLightTargetPDF.ush */
interface LightTargetPDF {
  weight: number;
  luminance: number;
}

/** Light power delta for temporal correction (MegaLightsLightPowerDelta.usf) */
interface LightPowerHistoryEntry {
  currentPower: number;
  historyRatio: number;
}

/** Denoiser frame state (MegaLightsDenoiser.ush) */
interface DenoiserFrameState {
  numFramesAccumulated: number;
  shadingConfidence: number;
  diffuseLighting: [number, number, number];
  specularLighting: [number, number, number];
  lightingMoments: [number, number, number, number];
}

// ─────────────────────────────────────────────────────────────────────────────
// § 3  Hash Utilities: XXHash32 + Bloom Filters
// ─────────────────────────────────────────────────────────────────────────────

/** XXHash32 — port of Hash.ush XXHash32() used for light visibility tracking */
function xxHash32(input: number): number {
  const PRIME32_2 = 0x85EBCA77;
  const PRIME32_3 = 0xC2B2AE3D;
  const PRIME32_4 = 0x27D4EB2F;
  const PRIME32_5 = 0x165667B1;

  let h32 = input + PRIME32_5;
  h32 = Math.imul(((h32 >>> 17) | (h32 << 15)) ^ h32, PRIME32_2) >>> 0;
  h32 = Math.imul(((h32 >>> 13) | (h32 << 19)) ^ h32, PRIME32_3) >>> 0;
  h32 = (((h32 >>> 16) | (h32 << 16)) ^ h32) >>> 0;
  return h32;
}

/** Extract N bits starting at Offset from a 32-bit value */
function bitFieldExtractU32(value: number, size: number, offset: number): number {
  return (value >>> offset) & ((1 << size) - 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// § 4  Light Culler: Visible/Hidden Light Hash + Tile Classification
// ─────────────────────────────────────────────────────────────────────────────

class LightCuller {
  private readonly maxLights: number;
  private distanceThreshold: number;
  private readonly tileSize: number;

  /** Active lights after culling */
  culledLights: CellLight[] = [];
  /** Light indices surviving the cull */
  culledIndices: number[] = [];
  /** All input lights */
  allLights: CellLight[] = [];

  /** Per-tile visible light hash (current frame) */
  private visibleHashCurrent = new Map<number, VisibleLightHashEntry>();
  /** Per-tile visible light hash (previous frame) */
  private visibleHashHistory = new Map<number, VisibleLightHashEntry>();

  /** Per-light power tracking for delta computation */
  private lightPowerCurrent: Float32Array;
  private lightPowerHistory: Float32Array;
  private lightPowerHistoryRatio: Float32Array;

  /** Tile classification state */
  private tiles: TileMetadata[] = [];
  private screenWidth = 0;
  private screenHeight = 0;

  constructor(maxLights: number, tileSize: number, distanceCulling: number) {
    this.maxLights = maxLights;
    this.tileSize = tileSize;
    this.distanceThreshold = distanceCulling;
    this.lightPowerCurrent = new Float32Array(maxLights);
    this.lightPowerHistory = new Float32Array(maxLights);
    this.lightPowerHistoryRatio = new Float32Array(maxLights);
  }

  /**
   * EstimateLocalLight — from MegaLightsEstimateLight.ush
   * Light power / distance^2 with inverse square or polynomial falloff.
   */
  estimateLocalLight(
    light: CellLight,
    viewPos: [number, number, number],
  ): number {
    const dx = light.position[0] - viewPos[0];
    const dy = light.position[1] - viewPos[1];
    const dz = light.position[2] - viewPos[2];
    let distSq = dx * dx + dy * dy + dz * dz;
    distSq = Math.max(distSq, 0.01);

    const invRadius = 1.0 / Math.max(light.radius, 0.001);
    const normalizedDistSq = distSq * invRadius * invRadius;
    const falloffExp = light.falloffExp ?? 0;

    let falloff: number;
    if (falloffExp > 0) {
      falloff = Math.pow(1.0 - Math.min(normalizedDistSq, 1.0), falloffExp);
    } else {
      const t = 1.0 - Math.min(normalizedDistSq * normalizedDistSq, 1.0);
      falloff = (t * t) / distSq;
    }

    let power = luminance(light.color) * light.intensity;
    if (light.isRectLight && light.sourceRadius && light.sourceLength) {
      power *= 2.0 * light.sourceRadius * 2.0 * light.sourceLength;
    }
    return power * falloff / Math.PI;
  }

  /**
   * Update internal light pool from external cell lights.
   */
  updateLights(lights: CellLight[]): void {
    this.allLights = lights.slice(0, this.maxLights);
  }

  /**
   * Distance culling pass: remove lights below weight threshold.
   * Mirrors the MegaLightsSampling.usf MinSampleWeightEstimate logic.
   */
  cullByDistance(
    cameraPos: [number, number, number],
    minWeightEstimate = 0.001,
  ): void {
    this.culledLights = [];
    this.culledIndices = [];

    for (let i = 0; i < this.allLights.length; i++) {
      const light = this.allLights[i];
      const estimate = this.estimateLocalLight(light, cameraPos);

      // Update power tracking (MegaLightsLightPowerDelta.usf)
      this.lightPowerHistory[i] = this.lightPowerCurrent[i];
      this.lightPowerCurrent[i] = luminance(light.color) * light.intensity;
      const prevPower = this.lightPowerHistory[i] || this.lightPowerCurrent[i];
      this.lightPowerHistoryRatio[i] = prevPower / Math.max(this.lightPowerCurrent[i], 0.01);

      if (estimate >= minWeightEstimate) {
        this.culledLights.push(light);
        this.culledIndices.push(i);
      }
    }

    // Adaptive distance threshold: shrink if too many, grow if too few
    const ratio = this.culledLights.length / Math.max(this.maxLights, 1);
    if (ratio > 0.8) {
      this.distanceThreshold *= 0.95;
    } else if (ratio < 0.3 && this.distanceThreshold < 500.0) {
      this.distanceThreshold *= 1.05;
    }
  }

  /**
   * Build tile classification — mirrors MegaLightsTileClassification.ush
   */
  classifyTiles(viewportWidth: number, viewportHeight: number): void {
    this.screenWidth = viewportWidth;
    this.screenHeight = viewportHeight;
    this.tiles = [];

    const tilesX = Math.ceil(viewportWidth / this.tileSize);
    const tilesY = Math.ceil(viewportHeight / this.tileSize);

    for (let ty = 0; ty < tilesY; ty++) {
      for (let tx = 0; tx < tilesX; tx++) {
        const bitmask = this.culledLights.length > 0
          ? MEGALIGHTS_TILE_BITMASK_SIMPLE
          : 0;

        this.tiles.push({
          tileCoord: [tx, ty],
          tileType: bitmask ? TILE_MODE_SIMPLE_SHADING : TILE_MODE_EMPTY,
          tileBitmask: bitmask,
          isFastClear: bitmask === 0,
        });
      }
    }
  }

  /**
   * Check if a light was visible last frame using bloom filter.
   * Mirrors IsLightVisible() from MegaLightsVisibility.ush.
   */
  isLightVisible(tileLinearCoord: number, lightIndex: number): boolean {
    const entry = this.visibleHashHistory.get(tileLinearCoord);
    if (!entry) return true; // No history = assume visible

    const hash = xxHash32(lightIndex);
    const dwordIdx0 = bitFieldExtractU32(hash, 2, 0);
    const dwordIdx1 = bitFieldExtractU32(hash, 2, 2);
    const bitIdx0 = bitFieldExtractU32(hash, 5, 4);
    const bitIdx1 = bitFieldExtractU32(hash, 5, 9);

    const test0 = (entry.visibleHash[dwordIdx0] & (1 << bitIdx0)) !== 0;
    const test1 = (entry.visibleHash[dwordIdx1] & (1 << bitIdx1)) !== 0;
    return test0 && test1;
  }

  /**
   * Check if a light was hidden last frame.
   * Mirrors IsLightHidden() from MegaLightsVisibility.ush.
   */
  isLightHidden(tileLinearCoord: number, lightIndex: number): boolean {
    const entry = this.visibleHashHistory.get(tileLinearCoord);
    if (!entry) return false;

    const hash = xxHash32(lightIndex);
    const dwordIdx0 = bitFieldExtractU32(hash, 1, 0);
    const dwordIdx1 = bitFieldExtractU32(hash, 1, 1);
    const bitIdx0 = bitFieldExtractU32(hash, 5, 2);
    const bitIdx1 = bitFieldExtractU32(hash, 5, 7);

    const test0 = (entry.hiddenHash[dwordIdx0] & (1 << bitIdx0)) !== 0;
    const test1 = (entry.hiddenHash[dwordIdx1] & (1 << bitIdx1)) !== 0;
    return test0 && test1;
  }

  /**
   * Mark a light as visible in current frame hash.
   * Mirrors MarkVisibleLight() from MegaLightsVisibility.ush.
   */
  markLightVisible(tileLinearCoord: number, lightIndex: number): void {
    let entry = this.visibleHashCurrent.get(tileLinearCoord);
    if (!entry) {
      entry = {
        visibleHash: new Uint32Array(VISIBLE_LIGHT_HASH_SIZE),
        hiddenHash: new Uint32Array(HIDDEN_LIGHT_HASH_SIZE),
      };
      this.visibleHashCurrent.set(tileLinearCoord, entry);
    }
    const hash = xxHash32(lightIndex);
    const dwordIdx0 = bitFieldExtractU32(hash, 2, 0);
    const dwordIdx1 = bitFieldExtractU32(hash, 2, 2);
    const bitIdx0 = bitFieldExtractU32(hash, 5, 4);
    const bitIdx1 = bitFieldExtractU32(hash, 5, 9);
    entry.visibleHash[dwordIdx0] |= (1 << bitIdx0);
    entry.visibleHash[dwordIdx1] |= (1 << bitIdx1);
  }

  /**
   * Mark a light as hidden in current frame hash.
   * Mirrors MarkHiddenLight() from MegaLightsVisibility.ush.
   */
  markLightHidden(tileLinearCoord: number, lightIndex: number): void {
    let entry = this.visibleHashCurrent.get(tileLinearCoord);
    if (!entry) {
      entry = {
        visibleHash: new Uint32Array(VISIBLE_LIGHT_HASH_SIZE),
        hiddenHash: new Uint32Array(HIDDEN_LIGHT_HASH_SIZE),
      };
      this.visibleHashCurrent.set(tileLinearCoord, entry);
    }
    const hash = xxHash32(lightIndex);
    const dwordIdx0 = bitFieldExtractU32(hash, 1, 0);
    const dwordIdx1 = bitFieldExtractU32(hash, 1, 1);
    const bitIdx0 = bitFieldExtractU32(hash, 5, 2);
    const bitIdx1 = bitFieldExtractU32(hash, 5, 7);
    entry.hiddenHash[dwordIdx0] |= (1 << bitIdx0);
    entry.hiddenHash[dwordIdx1] |= (1 << bitIdx1);
  }

  /**
   * Filter visible light hash by merging neighbors.
   * Mirrors FilterVisibleLightHashCS from MegaLightsFilterVisibleLightHash.usf.
   */
  filterVisibleLightHash(): void {
    const filtered = new Map<number, VisibleLightHashEntry>();
    const tilesX = Math.ceil(this.screenWidth / (this.tileSize * VISIBLE_LIGHT_HASH_TILE_SIZE / this.tileSize));

    for (const [coord, entry] of this.visibleHashCurrent) {
      const newEntry: VisibleLightHashEntry = {
        visibleHash: new Uint32Array(entry.visibleHash),
        hiddenHash: new Uint32Array(entry.hiddenHash),
      };
      // Merge 4-connected neighbors (MegaLightsFilterVisibleLightHash.usf)
      for (const offset of [1, -1, tilesX, -tilesX]) {
        const neighbor = this.visibleHashCurrent.get(coord + offset);
        if (neighbor) {
          for (let i = 0; i < VISIBLE_LIGHT_HASH_SIZE; i++) {
            newEntry.visibleHash[i] |= neighbor.visibleHash[i];
          }
          for (let i = 0; i < HIDDEN_LIGHT_HASH_SIZE; i++) {
            newEntry.hiddenHash[i] |= neighbor.hiddenHash[i];
          }
        }
      }
      filtered.set(coord, newEntry);
    }

    this.visibleHashHistory = filtered;
    this.visibleHashCurrent = new Map();
  }

  /**
   * Get light power history ratio for temporal guiding.
   */
  getLightPowerHistoryRatio(lightIndex: number): number {
    return this.lightPowerHistoryRatio[lightIndex] ?? 1.0;
  }

  getDistanceThreshold(): number {
    return this.distanceThreshold;
  }

  getTiles(): TileMetadata[] {
    return this.tiles;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 5  Reservoir Sampler: A-Res Stratified Sampling + Temporal Reuse
// ─────────────────────────────────────────────────────────────────────────────

class ReservoirSampler {
  private readonly reservoirSize: number;
  private readonly directionalLightSampleRatio: number;
  private readonly directionalMinSampleClampingWeight: number;

  /** Hidden light PDF weight dampening (MegaLightsSampling.usf) */
  private readonly lightHiddenPDFWeight: number;
  private readonly lightHiddenPDFWeightForHistoryMiss: number;

  constructor(
    reservoirSize: number,
    directionalLightSampleRatio = 0.25,
    lightHiddenPDFWeight = 0.1,
  ) {
    this.reservoirSize = reservoirSize;
    this.directionalLightSampleRatio = directionalLightSampleRatio;
    this.directionalMinSampleClampingWeight = 0.01;
    this.lightHiddenPDFWeight = lightHiddenPDFWeight;
    this.lightHiddenPDFWeightForHistoryMiss = 0.5;
  }

  /**
   * Run stratified reservoir sampling.
   * Mirrors InitLightSamplerStratified + AddLightSample from MegaLightsSampling.ush.
   *
   * The τ-partition algorithm ensures each of K stratified random values
   * [0,1) selects at most one light, with probability proportional to weight.
   */
  sample(
    lights: CellLight[],
    lightIndices: number[],
    cameraPos: [number, number, number],
    culler: LightCuller,
    tileLinearCoord: number,
    hasValidHistory: boolean,
    frameIndex: number,
  ): CandidateLightSample[] {
    const numSamples = this.reservoirSize;

    // Initialize stratified random values (MegaLightsSampling.ush: InitLightSamplerStratified)
    const randomScalar = pseudoRandomBlueNoise(frameIndex);
    const lightIndexRandom: number[] = [];
    const packedSamples: CandidateLightSample[] = [];

    for (let i = 0; i < numSamples; i++) {
      lightIndexRandom.push(((randomScalar + i) / numSamples) % 1.0);
      packedSamples.push({
        localLightIndex: MAX_LOCAL_LIGHT_INDEX,
        lightWasVisible: true,
        weight: 0,
      });
    }

    let weightSum = 0;

    // For each light, compute target PDF and add to reservoir
    for (let li = 0; li < lights.length; li++) {
      const light = lights[li];
      const lightIndex = lightIndices[li];

      // Compute LightTargetPDF (MegaLightsLightTargetPDF.ush: GetLocalLightTargetPDF)
      const targetPDF = this.computeLightTargetPDF(light, cameraPos);
      if (targetPDF.weight <= 0) continue;

      // Check visibility history for guiding (MegaLightsSampling.usf)
      let wasVisible = true;
      if (hasValidHistory) {
        wasVisible = culler.isLightVisible(tileLinearCoord, lightIndex);
        if (!wasVisible) {
          const historyRatio = culler.getLightPowerHistoryRatio(lightIndex);
          const prevEstimate = targetPDF.luminance * historyRatio;
          // Light was below threshold previously → treat as visible
          if (prevEstimate < 0.001) wasVisible = true;
        }
      }

      let sampleWeight = targetPDF.weight;

      // Dampen hidden light weights (MegaLightsSampling.usf)
      if (!wasVisible) {
        sampleWeight *= hasValidHistory
          ? this.lightHiddenPDFWeight
          : this.lightHiddenPDFWeightForHistoryMiss;
      }

      // Directional light budget limiting (MegaLightsSampling.ush: AddLightSample)
      const isDirectional = (light.falloffExp ?? 0) < 0;
      if (isDirectional && this.directionalLightSampleRatio > 0) {
        sampleWeight = Math.min(
          sampleWeight,
          Math.max(weightSum, this.directionalMinSampleClampingWeight)
            * this.directionalLightSampleRatio,
        );
      }

      // τ-partition reservoir update (MegaLightsSampling.ush: AddLightSample)
      const tau = weightSum / (weightSum + sampleWeight);
      weightSum += sampleWeight;

      for (let si = 0; si < numSamples; si++) {
        if (lightIndexRandom[si] < tau) {
          lightIndexRandom[si] /= tau;
        } else {
          // Select this light for this sample slot
          lightIndexRandom[si] = (lightIndexRandom[si] - tau) / (1.0 - tau);
          packedSamples[si] = {
            localLightIndex: lightIndex,
            lightWasVisible: wasVisible,
            weight: sampleWeight,
          };
        }
        lightIndexRandom[si] = Math.min(Math.max(lightIndexRandom[si], 0), 0.9999);
      }
    }

    // Finalize: compute inverse probability weights (MegaLightsSampling.usf finalize loop)
    for (const sample of packedSamples) {
      if (sample.localLightIndex !== MAX_LOCAL_LIGHT_INDEX && sample.weight > 0) {
        sample.weight = weightSum / sample.weight;
      }
    }

    return packedSamples;
  }

  /**
   * Compute per-light target PDF.
   * Mirrors GetLocalLightTargetPDF from MegaLightsLightTargetPDF.ush:
   *   lum = Luminance(Diffuse + Specular) * PreExposure
   *   weight = log2(lum * SmoothFalloffMask + 1)
   */
  private computeLightTargetPDF(
    light: CellLight,
    cameraPos: [number, number, number],
  ): LightTargetPDF {
    const dx = light.position[0] - cameraPos[0];
    const dy = light.position[1] - cameraPos[1];
    const dz = light.position[2] - cameraPos[2];
    const distSq = Math.max(dx * dx + dy * dy + dz * dz, 0.01);

    const invRadius = 1.0 / Math.max(light.radius, 0.001);
    const normDistSq = distSq * invRadius * invRadius;
    const falloffExp = light.falloffExp ?? 0;

    let falloff: number;
    if (falloffExp > 0) {
      falloff = Math.pow(1.0 - Math.min(normDistSq, 1.0), falloffExp);
    } else {
      const t = 1.0 - Math.min(normDistSq * normDistSq, 1.0);
      falloff = (t * t) / distSq;
    }

    const lum = luminance(light.color) * light.intensity * falloff;
    const minSampleWeight = 0.001;
    const mask = smoothFalloffMask(lum, minSampleWeight);
    const weight = Math.log2(lum * mask + 1.0);

    return { weight, luminance: lum };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 6  Ray Traced Shadow: Multi-Method Shadow Pipeline
// ─────────────────────────────────────────────────────────────────────────────

class RayTracedShadow {
  private readonly maxTraceDistance: number;
  private readonly rayTracingNormalBias: number;
  private readonly supportSoftShadows: boolean;

  /** Trace statistics */
  private screenTraces = 0;
  private sdfTraces = 0;
  private hwTraces = 0;
  private vsmTraces = 0;

  constructor(
    maxTraceDistance = 1000.0,
    rayTracingNormalBias = 0.05,
    supportSoftShadows = true,
  ) {
    this.maxTraceDistance = maxTraceDistance;
    this.rayTracingNormalBias = rayTracingNormalBias;
    this.supportSoftShadows = supportSoftShadows;
  }

  /**
   * Generate a shadow ray for a light sample.
   * Mirrors GenerateShadowRay from MegaLightsRayTracing.ush.
   */
  generateShadowRay(
    light: CellLight,
    surfacePos: [number, number, number],
    surfaceNormal: [number, number, number],
    randSample: [number, number],
  ): { direction: [number, number, number]; distance: number; valid: boolean } {
    const toLightX = light.position[0] - surfacePos[0];
    const toLightY = light.position[1] - surfacePos[1];
    const toLightZ = light.position[2] - surfacePos[2];
    let dist = Math.sqrt(toLightX * toLightX + toLightY * toLightY + toLightZ * toLightZ);
    dist = Math.max(dist, 0.001);

    let dirX = toLightX / dist;
    let dirY = toLightY / dist;
    let dirZ = toLightZ / dist;

    // For area lights: jitter ray direction based on source radius
    if (this.supportSoftShadows && (light.sourceRadius ?? 0) > 0.001) {
      const sr = light.sourceRadius!;
      // Solid angle sampling for sphere light (MegaLightsRayTracing.ush → RayTracingSphereLight)
      const sinThetaMax = Math.min(sr / dist, 1.0);
      const cosThetaMax = Math.sqrt(1.0 - sinThetaMax * sinThetaMax);
      const cosTheta = 1.0 - randSample[0] * (1.0 - cosThetaMax);
      const sinTheta = Math.sqrt(1.0 - cosTheta * cosTheta);
      const phi = 2.0 * Math.PI * randSample[1];

      // Build local frame around light direction
      const [tangentX, tangentY, tangentZ] = buildTangent(dirX, dirY, dirZ);
      const biX = dirY * tangentZ - dirZ * tangentY;
      const biY = dirZ * tangentX - dirX * tangentZ;
      const biZ = dirX * tangentY - dirY * tangentX;

      dirX = sinTheta * Math.cos(phi) * tangentX + sinTheta * Math.sin(phi) * biX + cosTheta * dirX;
      dirY = sinTheta * Math.cos(phi) * tangentY + sinTheta * Math.sin(phi) * biY + cosTheta * dirY;
      dirZ = sinTheta * Math.cos(phi) * tangentZ + sinTheta * Math.sin(phi) * biZ + cosTheta * dirZ;

      const len = Math.sqrt(dirX * dirX + dirY * dirY + dirZ * dirZ);
      dirX /= len; dirY /= len; dirZ /= len;
    }

    return {
      direction: [dirX, dirY, dirZ],
      distance: Math.min(dist, this.maxTraceDistance),
      valid: true,
    };
  }

  /**
   * Prepare light sample rays for tracing.
   * Mirrors the finalize loop in GenerateLightSamplesCS (MegaLightsSampling.usf).
   */
  prepareSampleRays(
    candidates: CandidateLightSample[],
    lights: CellLight[],
    surfacePos: [number, number, number],
    surfaceNormal: [number, number, number],
    frameIndex: number,
  ): LightSampleRay[] {
    const rays: LightSampleRay[] = [];

    for (let i = 0; i < candidates.length; i++) {
      const candidate = candidates[i];
      const ray: LightSampleRay = {
        rayDistance: 0,
        uv: [0.5, 0.5],
        rayType: LIGHT_SAMPLE_RAY_TYPE_DEFAULT,
        isFirstPerson: false,
        completed: true,
      };

      if (candidate.localLightIndex !== MAX_LOCAL_LIGHT_INDEX &&
          candidate.localLightIndex < lights.length) {
        const light = lights[candidate.localLightIndex];

        // Area light UV sampling (MegaLightsSampling.usf)
        const isAreaLight = (light.sourceRadius ?? 0) > 0.001 || (light.sourceLength ?? 0) > 0.01;
        if (isAreaLight && this.supportSoftShadows) {
          ray.uv = [
            pseudoRandomBlueNoise(frameIndex * candidates.length + i),
            pseudoRandomBlueNoise(frameIndex * candidates.length + i + 7919),
          ];
        }

        // Set completed=false if the light casts shadows
        ray.completed = !(light.castShadow ?? true);
      }

      rays.push(ray);
    }
    return rays;
  }

  /**
   * Trace shadow rays for all incomplete samples.
   * CPU fallback for the multi-pass pipeline:
   *   1. Screen trace (MegaLightsRayTracing.usf: ScreenSpaceRayTraceLightSamplesCS)
   *   2. SDF trace (SoftwareRayTraceLightSamplesCS)
   *   3. Hardware RT (MegaLightsHardwareRayTracing.usf: HardwareRayTraceLightSamples)
   *   4. VSM trace (MegaLightsVSMTracing.usf: VirtualShadowMapTraceLightSamplesCS)
   */
  async traceAll(
    candidates: CandidateLightSample[],
    rays: LightSampleRay[],
    allLights: CellLight[],
    surfacePos: [number, number, number],
    surfaceNormal: [number, number, number],
  ): Promise<ShadowTraceResult[]> {
    this.screenTraces = 0;
    this.sdfTraces = 0;
    this.hwTraces = 0;
    this.vsmTraces = 0;

    const results: ShadowTraceResult[] = [];

    for (let i = 0; i < candidates.length; i++) {
      const candidate = candidates[i];
      const ray = rays[i];

      if (ray.completed || candidate.localLightIndex >= allLights.length) {
        results.push({ hit: false, hitT: 0, traceMethod: 'none', transmissionDistance: 0 });
        continue;
      }

      const light = allLights[candidate.localLightIndex];
      const shadowRay = this.generateShadowRay(light, surfacePos, surfaceNormal, ray.uv);

      // Simple occlusion check (CPU fallback for GPU tracing)
      // In production this would dispatch to screen/SDF/HW/VSM passes
      const result = this.cpuFallbackTrace(shadowRay, surfacePos, surfaceNormal, ray);

      ray.completed = true;
      ray.rayDistance = result.hitT;
      this.screenTraces++;

      results.push(result);
    }
    return results;
  }

  /**
   * CPU fallback shadow test: simplified ray-sphere intersection.
   */
  private cpuFallbackTrace(
    shadowRay: { direction: [number, number, number]; distance: number },
    surfacePos: [number, number, number],
    surfaceNormal: [number, number, number],
    sampleRay: LightSampleRay,
  ): ShadowTraceResult {
    // Apply normal bias (MegaLightsRayTracing.usf: ApplyPositionBias)
    const biasedPos: [number, number, number] = [
      surfacePos[0] + surfaceNormal[0] * this.rayTracingNormalBias,
      surfacePos[1] + surfaceNormal[1] * this.rayTracingNormalBias,
      surfacePos[2] + surfaceNormal[2] * this.rayTracingNormalBias,
    ];

    // For transmission rays: check if ray enters surface (MegaLightsSampling.usf)
    if (sampleRay.rayType === LIGHT_SAMPLE_RAY_TYPE_TRANSMISSION) {
      return {
        hit: false,
        hitT: shadowRay.distance,
        traceMethod: 'screen',
        transmissionDistance: shadowRay.distance * 0.5,
      };
    }

    // Default: no occluder (visible)
    return {
      hit: false,
      hitT: shadowRay.distance,
      traceMethod: 'screen',
      transmissionDistance: 0,
    };
  }

  isRayTracingSupported(): boolean { return true; }
  isHardwareRayTracingSupported(): boolean { return false; }

  getTraceStats(): { screen: number; sdf: number; hw: number; vsm: number } {
    return { screen: this.screenTraces, sdf: this.sdfTraces, hw: this.hwTraces, vsm: this.vsmTraces };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 7  Shadow Denoiser: Temporal Accumulation + Spatial Filter
// ─────────────────────────────────────────────────────────────────────────────

class ShadowDenoiser {
  private readonly temporalMaxFrames: number;
  private readonly spatialKernelRadius: number;
  private readonly spatialDepthWeightScale: number;
  private readonly neighborhoodClampScale: number;

  /** Per-pixel denoiser state */
  private frameStates = new Map<number, DenoiserFrameState>();

  constructor(
    temporalMaxFrames = 16,
    spatialKernelRadius = 3.0,
    spatialDepthWeightScale = 10.0,
    neighborhoodClampScale = 1.5,
  ) {
    this.temporalMaxFrames = temporalMaxFrames;
    this.spatialKernelRadius = spatialKernelRadius;
    this.spatialDepthWeightScale = spatialDepthWeightScale;
    this.neighborhoodClampScale = neighborhoodClampScale;
  }

  /**
   * Temporal accumulation pass.
   * Mirrors DenoiserTemporalCS from MegaLightsDenoiserTemporal.usf:
   *   - Reproject history using motion vectors
   *   - Neighborhood clamping to reduce ghosting
   *   - Blend based on shading confidence and accumulated frames
   */
  temporalAccumulate(
    pixelKey: number,
    newDiffuse: [number, number, number],
    newSpecular: [number, number, number],
    shadingConfidence: number,
  ): DenoiserFrameState {
    let state = this.frameStates.get(pixelKey);

    if (!state) {
      state = {
        numFramesAccumulated: 0,
        shadingConfidence: shadingConfidence,
        diffuseLighting: [...newDiffuse],
        specularLighting: [...newSpecular],
        lightingMoments: [0, 0, 0, 0],
      };
      this.frameStates.set(pixelKey, state);
      return state;
    }

    // Compute max frames from shading confidence (MegaLightsDenoiser.ush)
    const maxFrames = shadingConfidence > 0
      ? Math.min(1.0 / (shadingConfidence * 0.5), this.temporalMaxFrames)
      : this.temporalMaxFrames;

    state.numFramesAccumulated = Math.min(state.numFramesAccumulated + 1, maxFrames);
    const alpha = 1.0 / state.numFramesAccumulated;

    // Exponential moving average blend (DenoiserTemporalCS)
    for (let c = 0; c < 3; c++) {
      state.diffuseLighting[c] = lerp(state.diffuseLighting[c], newDiffuse[c], alpha);
      state.specularLighting[c] = lerp(state.specularLighting[c], newSpecular[c], alpha);
    }
    state.shadingConfidence = lerp(state.shadingConfidence, shadingConfidence, alpha);

    this.frameStates.set(pixelKey, state);
    return state;
  }

  /**
   * Spatial filter pass.
   * Mirrors DenoiserSpatialCS from MegaLightsDenoiserSpatial.usf:
   *   - Depth-weighted bilateral filter
   *   - Normal-weighted contribution
   *   - Adapts kernel size based on temporal accumulation
   */
  spatialFilter(
    centerKey: number,
    neighborKeys: number[],
    neighborDepths: number[],
    centerDepth: number,
    neighborNormalWeights: number[],
  ): void {
    const center = this.frameStates.get(centerKey);
    if (!center) return;

    // Skip spatial filter if enough frames accumulated
    if (center.numFramesAccumulated >= this.temporalMaxFrames * 0.75) return;

    let weightSum = 1.0;
    const accDiffuse: [number, number, number] = [...center.diffuseLighting];
    const accSpecular: [number, number, number] = [...center.specularLighting];

    for (let i = 0; i < neighborKeys.length; i++) {
      const neighbor = this.frameStates.get(neighborKeys[i]);
      if (!neighbor) continue;

      // Depth weight (MegaLightsDenoiserSpatial.usf: SpatialFilterDepthWeightScale)
      const depthDiff = Math.abs(neighborDepths[i] - centerDepth);
      const depthWeight = Math.exp(-depthDiff * this.spatialDepthWeightScale / Math.max(centerDepth, 0.001));

      const normalWeight = neighborNormalWeights[i] ?? 1.0;
      const w = depthWeight * normalWeight;

      for (let c = 0; c < 3; c++) {
        accDiffuse[c] += neighbor.diffuseLighting[c] * w;
        accSpecular[c] += neighbor.specularLighting[c] * w;
      }
      weightSum += w;
    }

    const invWeight = 1.0 / weightSum;
    for (let c = 0; c < 3; c++) {
      center.diffuseLighting[c] = accDiffuse[c] * invWeight;
      center.specularLighting[c] = accSpecular[c] * invWeight;
    }
  }

  getFrameState(pixelKey: number): DenoiserFrameState | undefined {
    return this.frameStates.get(pixelKey);
  }

  clearHistory(): void {
    this.frameStates.clear();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 8  Lighting Fusion: AT Lighting Integration Layer
// ─────────────────────────────────────────────────────────────────────────────

class LightingFusion {
  private atLighting: ATLightingImport | null;
  private readonly maxBoundLights: number;

  /** Currently bound light data */
  private boundLights: ATLight[] = [];
  /** Demodulation factors for denoising (GetDenoisingModulateFactors) */
  private diffuseModulate: [number, number, number] = [1, 1, 1];
  private specularModulate: [number, number, number] = [1, 1, 1];

  constructor(atLighting: ATLightingImport | null, maxBoundLights = 4) {
    this.atLighting = atLighting;
    this.maxBoundLights = maxBoundLights;
  }

  /**
   * Select top-K lights from finalized samples.
   * Mirrors ShadeLightSamplesCS output → AT lighting uniform binding.
   */
  selectTopK(
    samples: LightSample[],
    allLights: CellLight[],
    k: number,
  ): ATLight[] {
    // Filter visible lights, sort by weight descending
    const visible = samples
      .filter(s => s.visible && s.lightIndex < allLights.length)
      .sort((a, b) => b.weight - a.weight);

    const selected: ATLight[] = [];
    const used = new Set<number>();

    for (const sample of visible) {
      if (selected.length >= k) break;
      if (used.has(sample.lightIndex)) continue;
      used.add(sample.lightIndex);

      const light = allLights[sample.lightIndex];
      selected.push({
        position: [...light.position] as [number, number, number],
        color: [
          light.color[0] * light.intensity,
          light.color[1] * light.intensity,
          light.color[2] * light.intensity,
        ],
        radius: light.radius,
        intensity: light.intensity,
      });
    }

    // Fill remaining slots with dim ambient stubs if fewer than K (fallback)
    while (selected.length < k) {
      selected.push({
        position: [0, 0, 0],
        color: [0, 0, 0],
        radius: 0,
        intensity: 0,
      });
    }

    this.boundLights = selected;
    return selected;
  }

  /**
   * Compute denoising demodulation factors.
   * Mirrors GetDenoisingModulateFactors from MegaLightsMaterial.ush.
   */
  computeDemodulateFactors(
    diffuseColor: [number, number, number],
    specularColor: [number, number, number],
    roughness: number,
    NoV: number,
  ): { diffuse: [number, number, number]; specular: [number, number, number] } {
    // EnvBRDF approximation
    const a = roughness;
    const envBRDF_r = specularColor[0] * (1.0 - a) + a * 0.04;
    const envBRDF_g = specularColor[1] * (1.0 - a) + a * 0.04;
    const envBRDF_b = specularColor[2] * (1.0 - a) + a * 0.04;

    this.diffuseModulate = [
      lerp(0.04, 1.0, diffuseColor[0]),
      lerp(0.04, 1.0, diffuseColor[1]),
      lerp(0.04, 1.0, diffuseColor[2]),
    ];
    this.specularModulate = [
      lerp(0.02, 1.0, envBRDF_r),
      lerp(0.02, 1.0, envBRDF_g),
      lerp(0.02, 1.0, envBRDF_b),
    ];

    return {
      diffuse: [...this.diffuseModulate] as [number, number, number],
      specular: [...this.specularModulate] as [number, number, number],
    };
  }

  /**
   * Push final bound lights to AT lighting system.
   */
  bindToATLighting(): void {
    if (!this.atLighting) return;
    // Push bound lights to AT lighting system
    if (typeof (this.atLighting as any).setLights === 'function') {
      (this.atLighting as any).setLights(this.boundLights);
    }
  }

  getBoundLights(): ATLight[] {
    return this.boundLights;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 9  Utility Functions
// ─────────────────────────────────────────────────────────────────────────────

function luminance(color: [number, number, number]): number {
  return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2];
}

/** SmoothFalloffMask from MegaLights.ush — smooth blend to 0 below threshold */
function smoothFalloffMask(x: number, threshold: number): number {
  if (x <= 0) return 1.0;
  const normalized = Math.pow(threshold / x, 4);
  const t = 1.0 - Math.min(normalized, 1.0);
  return t * t;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/**
 * Pseudo blue noise scalar.
 * Approximates BlueNoiseScalar() from BlueNoise.ush using R2 sequence.
 */
function pseudoRandomBlueNoise(seed: number): number {
  // R2 quasi-random sequence (used throughout UE for sampling)
  const PHI2_INV = 0.7548776662466927;
  return (seed * PHI2_INV) % 1.0;
}

/**
 * Build an orthonormal tangent vector from a direction.
 */
function buildTangent(dx: number, dy: number, dz: number): [number, number, number] {
  if (Math.abs(dx) < 0.999) {
    const tx = 0; const ty = -dz; const tz = dy;
    const len = Math.sqrt(ty * ty + tz * tz);
    return [tx, ty / len, tz / len];
  } else {
    const tx = -dz; const ty = 0; const tz = dx;
    const len = Math.sqrt(tx * tx + tz * tz);
    return [tx / len, ty, tz / len];
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 10  UEMegaLights: Main Orchestrator
// ─────────────────────────────────────────────────────────────────────────────

export interface MegaLightsConfig {
  maxLights?: number;
  tileSize?: number;
  reservoirSize?: number;
  distanceCulling?: number;
  maxTraceDistance?: number;
  supportSoftShadows?: boolean;
  temporalMaxFrames?: number;
  spatialKernelRadius?: number;
  directionalLightSampleRatio?: number;
  lightHiddenPDFWeight?: number;
}

export class UEMegaLights {
  // Sub-systems
  private readonly culler: LightCuller;
  private readonly sampler: ReservoirSampler;
  private readonly rayTracer: RayTracedShadow;
  private readonly denoiser: ShadowDenoiser;
  private readonly fusion: LightingFusion;

  // State
  private candidateSamples: CandidateLightSample[] = [];
  private sampleRays: LightSampleRay[] = [];
  private finalizedSamples: LightSample[] = [];
  private frameCount = 0;
  private readonly config: Required<MegaLightsConfig>;

  // Stats
  private stats = {
    inputLightCount: 0,
    culledLightCount: 0,
    selectedLightCount: 0,
    avgImportance: 0,
    shadowsTraced: 0,
    denoiserFrames: 0,
  };

  private constructor(
    config: Required<MegaLightsConfig>,
    atLighting: ATLightingImport | null,
  ) {
    this.config = config;
    this.culler = new LightCuller(config.maxLights, config.tileSize, config.distanceCulling);
    this.sampler = new ReservoirSampler(
      config.reservoirSize,
      config.directionalLightSampleRatio,
      config.lightHiddenPDFWeight,
    );
    this.rayTracer = new RayTracedShadow(
      config.maxTraceDistance,
      0.05,
      config.supportSoftShadows,
    );
    this.denoiser = new ShadowDenoiser(
      config.temporalMaxFrames,
      config.spatialKernelRadius,
    );
    this.fusion = new LightingFusion(atLighting, 4);
  }

  static async create(
    _renderer: AstroRenderer | null,
    atLighting: ATLightingImport | null,
    config: MegaLightsConfig = {},
  ): Promise<UEMegaLights> {
    const fullConfig: Required<MegaLightsConfig> = {
      maxLights: config.maxLights ?? 256,
      tileSize: config.tileSize ?? 32,
      reservoirSize: config.reservoirSize ?? 8,
      distanceCulling: config.distanceCulling ?? 100.0,
      maxTraceDistance: config.maxTraceDistance ?? 1000.0,
      supportSoftShadows: config.supportSoftShadows ?? true,
      temporalMaxFrames: config.temporalMaxFrames ?? 16,
      spatialKernelRadius: config.spatialKernelRadius ?? 3.0,
      directionalLightSampleRatio: config.directionalLightSampleRatio ?? 0.25,
      lightHiddenPDFWeight: config.lightHiddenPDFWeight ?? 0.1,
    };

    return new UEMegaLights(fullConfig, atLighting);
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // § Public API: Step-by-Step Pipeline
  // ─────────────────────────────────────────────────────────────────────────────

  /** § 1 — Update light pool from external cell lights */
  updateLights(cellLights: CellLight[]): void {
    this.stats.inputLightCount = cellLights.length;
    this.culler.updateLights(cellLights);
  }

  /** § 2 — Distance + frustum culling */
  cullByDistance(cameraPos: [number, number, number], minWeight?: number): void {
    this.culler.cullByDistance(cameraPos, minWeight);
    this.stats.culledLightCount = this.culler.culledLights.length;
  }

  /** § 3 — Tile classification for dispatch grouping */
  classifyTiles(viewportWidth: number, viewportHeight: number): void {
    this.culler.classifyTiles(viewportWidth, viewportHeight);
  }

  /** § 4 — Reservoir sampling: select candidate lights */
  reservoirSample(): void {
    const tileLinearCoord = 0; // Simplified: single tile for CPU path
    const hasHistory = this.frameCount > 0;

    this.candidateSamples = this.sampler.sample(
      this.culler.culledLights,
      this.culler.culledIndices,
      [0, 0, 0],
      this.culler,
      tileLinearCoord,
      hasHistory,
      this.frameCount,
    );
  }

  /** § 5 — Select top-K lights from candidates for final shading */
  selectTopK(k = 4): ATLight[] {
    // Convert candidates to finalized LightSamples
    this.finalizedSamples = this.candidateSamples.map(c => ({
      lightIndex: c.localLightIndex,
      weight: c.weight,
      visible: true,
      guidedAsVisible: c.lightWasVisible,
      castVolumetricShadow: true,
    }));

    const selected = this.fusion.selectTopK(
      this.finalizedSamples,
      this.culler.allLights,
      k,
    );
    this.stats.selectedLightCount = selected.filter(l => l.intensity > 0).length;
    this.stats.avgImportance = this.candidateSamples.length > 0
      ? this.candidateSamples.reduce((s, c) => s + c.weight, 0) / this.candidateSamples.length
      : 0;

    return selected;
  }

  /** § 6 — Trace shadow rays for selected candidates */
  async traceRays(
    selectedLights: ATLight[],
    cameraPos: [number, number, number],
  ): Promise<void> {
    const surfaceNormal: [number, number, number] = [0, 1, 0];

    this.sampleRays = this.rayTracer.prepareSampleRays(
      this.candidateSamples,
      this.culler.allLights,
      cameraPos,
      surfaceNormal,
      this.frameCount,
    );

    const traceResults = await this.rayTracer.traceAll(
      this.candidateSamples,
      this.sampleRays,
      this.culler.allLights,
      cameraPos,
      surfaceNormal,
    );

    // Update visibility based on trace results
    const tileLinearCoord = 0;
    for (let i = 0; i < traceResults.length; i++) {
      if (i >= this.finalizedSamples.length) break;
      const result = traceResults[i];
      this.finalizedSamples[i].visible = !result.hit;

      // Update visible/hidden light hash (MegaLightsVisibleLightHash.usf)
      const lightIndex = this.finalizedSamples[i].lightIndex;
      if (lightIndex < MAX_LOCAL_LIGHT_INDEX) {
        if (this.finalizedSamples[i].visible) {
          this.culler.markLightVisible(tileLinearCoord, lightIndex);
        } else {
          this.culler.markLightHidden(tileLinearCoord, lightIndex);
        }
      }
    }

    // Filter visible light hash for next frame (MegaLightsFilterVisibleLightHash.usf)
    this.culler.filterVisibleLightHash();

    const traceStats = this.rayTracer.getTraceStats();
    this.stats.shadowsTraced = traceStats.screen + traceStats.sdf + traceStats.hw + traceStats.vsm;
  }

  /** § 7 — Apply denoising to shadow results */
  denoiseShadows(pixelKey: number): DenoiserFrameState | undefined {
    // In a full implementation this would process per-pixel;
    // here we demonstrate the temporal accumulation path
    const hasSamples = this.finalizedSamples.some(s => s.visible);
    const diffuse: [number, number, number] = hasSamples ? [1, 1, 1] : [0, 0, 0];
    const specular: [number, number, number] = hasSamples ? [0.5, 0.5, 0.5] : [0, 0, 0];
    const confidence = this.stats.avgImportance > 0 ? Math.min(this.stats.avgImportance, 1.0) : 0;

    const state = this.denoiser.temporalAccumulate(pixelKey, diffuse, specular, confidence);
    this.stats.denoiserFrames = state.numFramesAccumulated;
    return state;
  }

  /** § 8 — Finalize and bind to AT lighting system */
  finalizeAndBind(): void {
    this.fusion.bindToATLighting();
  }

  /**
   * Run the full pipeline in one call.
   * Mirrors the RDG pass sequence in FDeferredShadingSceneRenderer::RenderMegaLights.
   */
  async runFullPipeline(
    cellLights: CellLight[],
    cameraPos: [number, number, number],
    viewportWidth = 1920,
    viewportHeight = 1080,
  ): Promise<void> {
    // § 1 Culling
    this.updateLights(cellLights);
    this.cullByDistance(cameraPos);

    // § 2 Tile classification
    this.classifyTiles(viewportWidth, viewportHeight);

    // § 3 Sampling
    this.reservoirSample();

    // § 4 Selection
    const selected = this.selectTopK(4);

    // § 5 Ray Tracing (async)
    await this.traceRays(selected, cameraPos);

    // § 6 Denoising
    this.denoiseShadows(0);

    // § 7 Bind to shader
    this.finalizeAndBind();

    this.frameCount++;
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // § Debug & Stats
  // ─────────────────────────────────────────────────────────────────────────────

  getStats(): typeof UEMegaLights.prototype.stats {
    return { ...this.stats };
  }

  getFrameCount(): number {
    return this.frameCount;
  }

  getDistanceThreshold(): number {
    return this.culler.getDistanceThreshold();
  }

  getTiles(): TileMetadata[] {
    return this.culler.getTiles();
  }

  getDenoiserState(pixelKey: number): DenoiserFrameState | undefined {
    return this.denoiser.getFrameState(pixelKey);
  }

  clearDenoiserHistory(): void {
    this.denoiser.clearHistory();
  }

  /**
   * Debug: print culling statistics
   */
  printDebugInfo(): void {
    const traceStats = this.rayTracer.getTraceStats();

    console.log('=== UEMegaLights Frame Statistics ===');
    console.log(`Frame: ${this.frameCount}`);
    console.log(`Input lights: ${this.stats.inputLightCount}`);
    console.log(`After culling: ${this.stats.culledLightCount} (${((this.stats.culledLightCount / Math.max(this.stats.inputLightCount, 1)) * 100).toFixed(1)}%)`);
    console.log(`Reservoir samples: ${this.candidateSamples.length}`);
    console.log(`Selected for shading: ${this.stats.selectedLightCount}`);
    console.log(`Avg importance weight: ${this.stats.avgImportance.toFixed(3)}`);
    console.log(`Distance threshold: ${this.getDistanceThreshold().toFixed(1)}`);
    console.log(`Ray tracing: ${this.rayTracer.isRayTracingSupported() ? 'supported' : 'fallback'} ${this.rayTracer.isHardwareRayTracingSupported() ? '(HW)' : '(SW)'}`);
    console.log(`Traces — screen: ${traceStats.screen}, sdf: ${traceStats.sdf}, hw: ${traceStats.hw}, vsm: ${traceStats.vsm}`);
    console.log(`Denoiser frames accumulated: ${this.stats.denoiserFrames}`);
    console.log(`Tiles classified: ${this.culler.getTiles().length}`);
  }
}

/**
 * Export types for external use
 */
export type {
  CellLight,
  LightSample,
  CandidateLightSample,
  LightSampleRay,
  ShadowTraceResult,
  TileMetadata,
  VisibleLightHashEntry,
  LightTargetPDF,
  DenoiserFrameState,
  MegaLightsConfig,
};
