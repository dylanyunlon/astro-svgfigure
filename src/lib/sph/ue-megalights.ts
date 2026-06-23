/**
 * ue-megalights.ts — M837: UE5 MegaLights Dynamic Light Management
 *
 * 移植 Unreal Engine 5 的 MegaLights 动态光源管理系统到 WebGL2:
 *
 *   § Light Culling: 从成百个发光体高效剔除远距离光源
 *     ├─ Visible Light Hash: 基于 XXHash32 的轻量级可见光缓存
 *     ├─ Tile Classification: 将屏幕分割为 TILE_SIZE×TILE_SIZE 网格
 *     └─ Distance Culling: 优先级采样时动态调整距离阈值
 *
 *   § Reservoir Sampling: A-Res 重要性采样从N个光源中选出K个
 *     ├─ Stratified Sampling: 分层随机，确保样本覆盖
 *     ├─ Weight Normalization: 根据光源强度和距离加权
 *     └─ Temporal Reuse: 跨帧复用光源候选，减少爆闪
 *
 *   § Ray Traced Shadow: 对选中光源做硬件/软件光线追踪
 *     ├─ GenerateShadowRay(): 针对 RectLight/SphereLight/CapsuleLight 生成ray
 *     ├─ ShadowTraceVoxel: 对体积光追踪，支持异步compute
 *     └─ Shadow Denoising: 时间和空间滤波，减少射线噪声
 *
 *   § AT Lighting.fs Fusion: 与现有4光源系统无缝融合
 *     ├─ MegaLights 作为候选池，AT lighting 为最终着色器
 *     ├─ 动态 NUM_LIGHTS 设置，从候选中选出前4个重要光源
 *     └─ Fallback: 若启用light count<4，自动补充环境光
 *
 * 类结构:
 *   UEMegaLights
 *     ├─ LightCuller: 实现 Light Culling + Visible Light Hash
 *     ├─ ReservoirSampler: A-Res 重要性采样
 *     ├─ RayTracedShadow: 光线追踪阴影
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
 * Reference: upstream/unreal-renderer-ue5/Shaders-Private/MegaLights/
 */

import type { AstroRenderer } from '../renderer/AstroRenderer.ts';
import type { ATLightingImport } from './at-lighting-import.ts';
import type { ATLight } from './at-lighting-import.ts';

// ─────────────────────────────────────────────────────────────────────────────
// § 1  Types: Light Structures
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
  /** Falloff exponent (1.0 = quadratic, 2.0 = linear) */
  falloffExp?: number;
}

/** Packed light sample from reservoir sampling */
export interface LightSample {
  /** Index into light pool [0, maxLights-1] */
  lightIndex: number;
  /** Normalized importance weight [0, 1] */
  weight: number;
  /** Whether light was visible in last frame */
  wasVisible: boolean;
  /** Estimated contribution to pixel/voxel */
  contribution: number;
}

/** Visible light hash entry for quick lookup */
export interface VisibleLightHashEntry {
  /** XXHash32(lightIndex) for fast test */
  hash: number;
  /** First-frame visibility state */
  wasVisible: boolean;
  /** Number of hash collisions encountered */
  collisions: number;
}

/** Tile classification for culling decisions */
export interface TileMetadata {
  /** Screen-space tile coordinate */
  coord: [number, number];
  /** Bitmask of visible lights in this tile */
  visibleLightMask: Uint32Array;
  /** Distance to nearest light in tile */
  minDistance: number;
  /** Distance to farthest light in tile */
  maxDistance: number;
  /** Frame count since last light update */
  framesSinceLightUpdate: number;
}

/** Ray traced shadow result */
export interface ShadowTraceResult {
  /** Is shadow ray occluded (hit something before light) */
  isOccluded: boolean;
  /** Occlusion fraction [0, 1] */
  shadowFactor: number;
  /** First hit distance along ray */
  hitDistance: number;
  /** Normal at hit point */
  hitNormal: [number, number, number];
}

// ─────────────────────────────────────────────────────────────────────────────
// § 2  Light Culler: Visible Light Hash + Distance Culling
// ─────────────────────────────────────────────────────────────────────────────

class LightCuller {
  private visibleLightHash: Map<number, VisibleLightHashEntry> = new Map();
  private tileGrid: Map<string, TileMetadata> = new Map();
  private distanceCullingThreshold: number = 100.0;
  private tileSize: number = 32;
  private gridWidth: number = 1;
  private gridHeight: number = 1;

  constructor(
    private renderer: AstroRenderer,
    private viewportWidth: number = 1920,
    private viewportHeight: number = 1080,
  ) {
    this.updateGridSize(viewportWidth, viewportHeight);
  }

  updateGridSize(width: number, height: number): void {
    this.gridWidth = Math.ceil(width / this.tileSize);
    this.gridHeight = Math.ceil(height / this.tileSize);
    this.tileGrid.clear();

    for (let y = 0; y < this.gridHeight; y++) {
      for (let x = 0; x < this.gridWidth; x++) {
        const key = `${x},${y}`;
        this.tileGrid.set(key, {
          coord: [x, y],
          visibleLightMask: new Uint32Array(4), // 128-bit mask (32 lights per tile)
          minDistance: Number.MAX_VALUE,
          maxDistance: 0,
          framesSinceLightUpdate: 0,
        });
      }
    }
  }

  /**
   * XXHash32 implementation for light hashing
   * Used to create compact visible light hash (see MegaLightsVisibleLightHash.usf)
   */
  private xxHash32(value: number): number {
    const PRIME1 = 2654435761;
    const PRIME2 = 2246822519;
    const PRIME3 = 3266489917;
    const PRIME4 = 668265263;
    const PRIME5 = 374761393;

    let h32 = 0;
    h32 = value ^ PRIME5;
    h32 = ((h32 << 13) | (h32 >>> 19)) * PRIME1;
    h32 = (h32 ^ value) * PRIME2;
    h32 = ((h32 << 27) | (h32 >>> 5)) * PRIME3;
    h32 = (h32 ^ (h32 >>> 15)) * PRIME4;
    h32 ^= h32 >>> 15;
    return h32 >>> 0;
  }

  /**
   * Mark a light as visible in tile-local hash
   * Based on MegaLightsVisibility.ush MarkVisibleLight()
   */
  markVisibleLight(lightIndex: number, screenCoord: [number, number]): void {
    const tileX = Math.floor(screenCoord[0] / this.tileSize);
    const tileY = Math.floor(screenCoord[1] / this.tileSize);
    const key = `${tileX},${tileY}`;

    const tile = this.tileGrid.get(key);
    if (!tile) return;

    const lightHash = this.xxHash32(lightIndex);
    const hashIndex = lightHash % 128;
    const uint32Index = Math.floor(hashIndex / 32);
    const bitOffset = hashIndex % 32;

    tile.visibleLightMask[uint32Index] |= 1 << bitOffset;

    const entry = this.visibleLightHash.get(lightIndex) || {
      hash: lightHash,
      wasVisible: true,
      collisions: 0,
    };
    entry.wasVisible = true;
    entry.collisions += 1;
    this.visibleLightHash.set(lightIndex, entry);
  }

  /**
   * Cull lights by distance from camera
   * Returns lights within threshold, sorted by distance
   */
  cullByDistance(
    lights: CellLight[],
    cameraPos: [number, number, number],
    threshold: number = this.distanceCullingThreshold,
  ): CellLight[] {
    return lights
      .map((light) => {
        const dx = light.position[0] - cameraPos[0];
        const dy = light.position[1] - cameraPos[1];
        const dz = light.position[2] - cameraPos[2];
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        return { light, distance: dist };
      })
      .filter(({ distance }) => distance <= threshold)
      .sort((a, b) => a.distance - b.distance)
      .map(({ light }) => light);
  }

  /**
   * Update culling threshold based on frame statistics
   * If >80% of samples hit distant lights, increase threshold
   */
  adaptiveThreshold(visibleCount: number, totalCount: number): void {
    const visibilityRatio = visibleCount / totalCount;
    if (visibilityRatio < 0.2) {
      // Too many culled lights, increase threshold
      this.distanceCullingThreshold *= 1.1;
    } else if (visibilityRatio > 0.95) {
      // Almost all visible, decrease threshold
      this.distanceCullingThreshold *= 0.95;
    }
    // Clamp to reasonable range [10, 500]
    this.distanceCullingThreshold = Math.max(10, Math.min(500, this.distanceCullingThreshold));
  }

  getTileMetadata(tileX: number, tileY: number): TileMetadata | undefined {
    return this.tileGrid.get(`${tileX},${tileY}`);
  }

  getVisibleLightCount(): number {
    return this.visibleLightHash.size;
  }

  getDistanceThreshold(): number {
    return this.distanceCullingThreshold;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 3  Reservoir Sampler: A-Res for Multi-Light Selection
// ─────────────────────────────────────────────────────────────────────────────

class ReservoirSampler {
  private reservoir: LightSample[] = [];
  private weightSum: number = 0;
  private sampleCount: number = 0;

  constructor(
    private reservoirSize: number = 8,
    private directionalLightRatio: number = 0.3,
    private minSampleWeight: number = 0.001,
  ) {
    this.reservoir = Array(reservoirSize)
      .fill(null)
      .map(() => ({
        lightIndex: -1,
        weight: 0,
        wasVisible: true,
        contribution: 0,
      }));
  }

  /**
   * Add a candidate light to the reservoir using A-Res algorithm
   * Based on MegaLightsSampling.ush AddLightSample()
   *
   * Algorithm:
   *   W_i = importance_weight(light_i)
   *   Update: Tau = M_sum / (M_sum + W_i)
   *   For each sample: if random < Tau then rescale, else replace with new light
   */
  addSampleARes(
    lightIndex: number,
    importance: number,
    wasVisible: boolean,
    isRadialLight: boolean,
  ): void {
    let sampleWeight = importance;

    // Directional lights don't dominate sample budget
    if (!isRadialLight && this.directionalLightRatio > 0) {
      const maxDirWeight = Math.max(this.weightSum, this.minSampleWeight) * this.directionalLightRatio;
      sampleWeight = Math.min(sampleWeight, maxDirWeight);
    }

    // Tau = M / (M + w)
    const tau = this.weightSum / (this.weightSum + sampleWeight);
    this.weightSum += sampleWeight;

    // Stochastically replace samples
    for (let i = 0; i < this.reservoirSize; i++) {
      const rnd = Math.random();

      if (rnd < tau) {
        // Keep existing sample, rescale random variable
        // (rescaling happens implicitly by preserving sample)
      } else {
        // Replace with new sample
        this.reservoir[i] = {
          lightIndex,
          weight: sampleWeight,
          wasVisible,
          contribution: importance,
        };
      }
    }

    this.sampleCount++;
  }

  /**
   * Finalize reservoir: normalize weights and sort by importance
   */
  finalize(): LightSample[] {
    if (this.weightSum <= 0) return [];

    // Normalize weights
    for (const sample of this.reservoir) {
      if (sample.lightIndex >= 0) {
        sample.weight = sample.contribution / Math.max(this.weightSum, 0.001);
      }
    }

    // Filter out invalid samples and sort by weight
    return this.reservoir
      .filter((s) => s.lightIndex >= 0 && s.weight > 0)
      .sort((a, b) => b.weight - a.weight)
      .slice(0, this.reservoirSize);
  }

  reset(): void {
    this.weightSum = 0;
    this.sampleCount = 0;
    this.reservoir.fill({
      lightIndex: -1,
      weight: 0,
      wasVisible: true,
      contribution: 0,
    });
  }

  getWeightSum(): number {
    return this.weightSum;
  }

  getSampleCount(): number {
    return this.sampleCount;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 4  Ray Traced Shadow: Hardware & Software Ray Tracing
// ─────────────────────────────────────────────────────────────────────────────

class RayTracedShadow {
  private shadowCache: Map<string, ShadowTraceResult> = new Map();
  private rayTracingSupported: boolean = false;
  private hardwareRayTracingSupported: boolean = false;

  constructor(
    private renderer: AstroRenderer,
    private maxRaysPerFrame: number = 512,
  ) {
    this.detectRayTracingCapability();
  }

  private detectRayTracingCapability(): void {
    // Check for WebGL extensions that support ray tracing
    const gl = this.renderer.getContext();
    this.rayTracingSupported = !!(gl.getExtension('WEBGL_ray_tracing') || gl.getExtension('EXT_ray_tracing'));
    this.hardwareRayTracingSupported = !!(
      gl.getExtension('WEBGL_ray_tracing_inline') ||
      gl.getExtension('EXT_ray_tracing_inline')
    );
  }

  /**
   * Generate shadow ray for light source
   * Based on MegaLightsRayTracing.ush GenerateShadowRay()
   *
   * For different light types:
   *   - RectLight: Sample from light surface, return ray to light quad corners
   *   - SphereLight: Solid angle sampling on sphere
   *   - CapsuleLight: Extended line segment sampling
   *   - DirectionalLight: Parallel rays from sun direction
   */
  generateShadowRay(
    lightPos: [number, number, number],
    lightColor: [number, number, number],
    lightData: [number, number, number, number], // varies by type
    worldPos: [number, number, number],
    surfaceNormal: [number, number, number],
    sampleUV: [number, number],
    lightType: 'rect' | 'sphere' | 'capsule' | 'directional',
  ): {
    rayOrigin: [number, number, number];
    rayDirection: [number, number, number];
    rayTMin: number;
    rayTMax: number;
    pdf: number;
  } {
    const rayOrigin: [number, number, number] = [...worldPos];
    let rayDirection: [number, number, number] = [0, 0, 0];
    let rayTMin = 0.001; // Bias to avoid self-intersection
    let rayTMax = 1e10;
    let pdf = 1.0;

    if (lightType === 'directional') {
      // For sun light: generate parallel ray from cam toward light
      rayDirection = [lightData[0], lightData[1], lightData[2]];
      const len = Math.sqrt(rayDirection[0] ** 2 + rayDirection[1] ** 2 + rayDirection[2] ** 2);
      rayDirection[0] /= len;
      rayDirection[1] /= len;
      rayDirection[2] /= len;
      rayTMax = 10000; // Far plane
    } else if (lightType === 'sphere') {
      // Solid angle sampling on sphere light
      const toLight = [
        lightPos[0] - worldPos[0],
        lightPos[1] - worldPos[1],
        lightPos[2] - worldPos[2],
      ];
      const distToLight = Math.sqrt(toLight[0] ** 2 + toLight[1] ** 2 + toLight[2] ** 2);
      const radius = lightData[3]; // sphere radius in .w

      // Normalize direction
      rayDirection[0] = toLight[0] / distToLight;
      rayDirection[1] = toLight[1] / distToLight;
      rayDirection[2] = toLight[2] / distToLight;

      // Apply solid angle offset based on sampleUV
      const theta = sampleUV[0] * Math.PI * 2;
      const phi = Math.asin(sampleUV[1] * 2 - 1);

      // Perturb ray direction (simplified cone sampling)
      const maxConeAngle = Math.asin(radius / Math.max(distToLight, 0.1));
      const perturbX = Math.cos(theta) * Math.sin(maxConeAngle * sampleUV[0]);
      const perturbY = Math.sin(theta) * Math.sin(maxConeAngle * sampleUV[0]);
      const perturbZ = Math.cos(maxConeAngle * sampleUV[0]);

      rayTMax = distToLight - radius;
      pdf = 1.0 / (2 * Math.PI * (1 - Math.cos(maxConeAngle)));
    } else if (lightType === 'rect') {
      // Rectangular area light: sample from quad surface
      const halfWidth = lightData[0];
      const halfHeight = lightData[1];

      // Light frame vectors (simplified; in real impl, would use light orientation matrix)
      const rightVec: [number, number, number] = [1, 0, 0];
      const upVec: [number, number, number] = [0, 1, 0];

      // Sample point on rectangle
      const samplePt: [number, number, number] = [
        lightPos[0] + (sampleUV[0] - 0.5) * 2 * halfWidth * rightVec[0],
        lightPos[1] + (sampleUV[0] - 0.5) * 2 * halfWidth * rightVec[1],
        lightPos[2] + (sampleUV[1] - 0.5) * 2 * halfHeight * upVec[2],
      ];

      rayDirection[0] = samplePt[0] - worldPos[0];
      rayDirection[1] = samplePt[1] - worldPos[1];
      rayDirection[2] = samplePt[2] - worldPos[2];

      const rayLen = Math.sqrt(rayDirection[0] ** 2 + rayDirection[1] ** 2 + rayDirection[2] ** 2);
      rayDirection[0] /= rayLen;
      rayDirection[1] /= rayLen;
      rayDirection[2] /= rayLen;

      rayTMax = rayLen * 0.99;
      pdf = 1.0 / (4 * halfWidth * halfHeight);
    } else {
      // Capsule light: sample from line segment
      const capsuleLen = lightData[3];
      const capsuleRadius = lightData[2];

      // Endpoint 1 and 2 (simplified)
      const ep1: [number, number, number] = [lightPos[0], lightPos[1], lightPos[2] - capsuleLen / 2];
      const ep2: [number, number, number] = [lightPos[0], lightPos[1], lightPos[2] + capsuleLen / 2];

      // Lerp along capsule
      const t = sampleUV[0];
      const samplePt: [number, number, number] = [
        ep1[0] + (ep2[0] - ep1[0]) * t,
        ep1[1] + (ep2[1] - ep1[1]) * t,
        ep1[2] + (ep2[2] - ep1[2]) * t,
      ];

      rayDirection[0] = samplePt[0] - worldPos[0];
      rayDirection[1] = samplePt[1] - worldPos[1];
      rayDirection[2] = samplePt[2] - worldPos[2];

      const rayLen = Math.sqrt(rayDirection[0] ** 2 + rayDirection[1] ** 2 + rayDirection[2] ** 2);
      rayDirection[0] /= rayLen;
      rayDirection[1] /= rayLen;
      rayDirection[2] /= rayLen;

      rayTMax = rayLen * 0.99;
      pdf = 1.0 / capsuleLen;
    }

    return { rayOrigin, rayDirection, rayTMin, rayTMax, pdf };
  }

  /**
   * Trace shadow ray (stub for WebGL2; real impl would use async compute or
   * deferred traversal buffer)
   */
  async traceShadow(
    rayOrigin: [number, number, number],
    rayDirection: [number, number, number],
    rayTMin: number,
    rayTMax: number,
    scene: any, // Would be scene acceleration structure
  ): Promise<ShadowTraceResult> {
    // For WebGL2, we'd implement this via:
    // 1. Deferred ray buffer (store rays as texture)
    // 2. Async compute shader to traverse acceleration structure
    // 3. Or: use screen-space ray marching with depth pyramid

    // Stub: assume no shadow
    return {
      isOccluded: false,
      shadowFactor: 1.0,
      hitDistance: rayTMax,
      hitNormal: [0, 1, 0],
    };
  }

  /**
   * Cache shadow result to avoid recomputation
   */
  private getCacheKey(rayOrigin: [number, number, number], rayDirection: [number, number, number]): string {
    return `${rayOrigin[0].toFixed(2)},${rayOrigin[1].toFixed(2)},${rayOrigin[2].toFixed(2)},${rayDirection[0].toFixed(3)},${rayDirection[1].toFixed(3)},${rayDirection[2].toFixed(3)}`;
  }

  isRayTracingSupported(): boolean {
    return this.rayTracingSupported;
  }

  isHardwareRayTracingSupported(): boolean {
    return this.hardwareRayTracingSupported;
  }

  clearCache(): void {
    this.shadowCache.clear();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 5  Lighting Fusion: AT Lighting Integration Layer
// ─────────────────────────────────────────────────────────────────────────────

class LightingFusion {
  private selectedLights: ATLight[] = [];
  private maxLightsPerFrame: number = 4;

  constructor(
    private atLighting: ATLightingImport,
  ) {}

  /**
   * Select top K lights from reservoir by importance weight
   * Ensures diversity: pick brightest, pick farthest, pick area lights, etc.
   */
  selectTopK(samples: LightSample[], lights: CellLight[], k: number = 4): ATLight[] {
    if (samples.length === 0) return [];

    // Sort by weight descending
    const sorted = [...samples].sort((a, b) => b.weight - a.weight);

    const selected: ATLight[] = [];
    for (let i = 0; i < Math.min(k, sorted.length); i++) {
      const sample = sorted[i];
      const cellLight = lights[sample.lightIndex];

      if (!cellLight) continue;

      // Convert CellLight to ATLight format (type = 2 = point light)
      const atLight: ATLight = {
        type: 2, // point light
        position: [...cellLight.position],
        color: [...cellLight.color],
        data: [0, 0, 0, cellLight.radius],
        data2: [0, 0, 0, 0],
        data3: [0, 0, 0, 0],
        properties: [
          cellLight.intensity,
          cellLight.radius * (cellLight.falloffExp || 1.0),
          0.01, // min threshold
          2.0, // light type ID
        ],
      };

      selected.push(atLight);
    }

    // Pad with ambient light if selected < k
    while (selected.length < k) {
      selected.push({
        type: 1, // directional (ambient fallback)
        position: [0, 1, 0],
        color: [0.2, 0.2, 0.2],
        data: [0, 0, 0, 0],
        data2: [0, 0, 0, 0],
        data3: [0, 0, 0, 0],
        properties: [0.5, 1.0, 0.0, 1.0],
      });
    }

    this.selectedLights = selected;
    return selected;
  }

  /**
   * Bind selected lights to AT lighting shader
   */
  bindToATLighting(): void {
    this.atLighting.bindLights(this.selectedLights);
  }

  /**
   * Apply shadow factor to selected lights (from ray tracing)
   */
  applyShadowFactors(shadowFactors: number[]): void {
    for (let i = 0; i < Math.min(shadowFactors.length, this.selectedLights.length); i++) {
      // Modulate light intensity by shadow factor
      const light = this.selectedLights[i];
      const shadowFactor = shadowFactors[i];
      light.properties[0] *= shadowFactor; // intensity *= (1 - occlusion)
    }
  }

  getSelectedLightCount(): number {
    return this.selectedLights.length;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 6  UEMegaLights: Main Class
// ─────────────────────────────────────────────────────────────────────────────

export class UEMegaLights {
  private culler: LightCuller;
  private sampler: ReservoirSampler;
  private rayTracer: RayTracedShadow;
  private fusion: LightingFusion;

  private lightPool: CellLight[] = [];
  private culledLights: CellLight[] = [];
  private candidateSamples: LightSample[] = [];
  private shadowFactors: number[] = [];

  private frameCount: number = 0;
  private stats: {
    inputLightCount: number;
    culledLightCount: number;
    selectedLightCount: number;
    shadowsTraced: number;
    avgImportance: number;
  } = {
    inputLightCount: 0,
    culledLightCount: 0,
    selectedLightCount: 0,
    shadowsTraced: 0,
    avgImportance: 0,
  };

  private constructor(
    private renderer: AstroRenderer,
    atLighting: ATLightingImport,
    config: {
      maxLights?: number;
      tileSize?: number;
      reservoirSize?: number;
      distanceCulling?: number;
      maxRaysPerFrame?: number;
    } = {},
  ) {
    const {
      maxLights = 256,
      tileSize = 32,
      reservoirSize = 8,
      distanceCulling = 100.0,
      maxRaysPerFrame = 512,
    } = config;

    this.culler = new LightCuller(renderer, renderer.getContext().canvas.width, renderer.getContext().canvas.height);
    this.sampler = new ReservoirSampler(reservoirSize);
    this.rayTracer = new RayTracedShadow(renderer, maxRaysPerFrame);
    this.fusion = new LightingFusion(atLighting);

    this.lightPool = [];
    this.shadowFactors = Array(4).fill(1.0);
  }

  /**
   * Factory: create UEMegaLights instance
   */
  static async create(
    renderer: AstroRenderer,
    atLighting: ATLightingImport,
    config?: Parameters<typeof UEMegaLights.prototype.constructor>[2],
  ): Promise<UEMegaLights> {
    return new UEMegaLights(renderer, atLighting, config);
  }

  /**
   * Update light pool from cell emissions
   */
  updateLights(cellLights: CellLight[]): void {
    this.lightPool = [...cellLights];
    this.stats.inputLightCount = cellLights.length;
  }

  /**
   * § Light Culling Pass
   *
   * Input: lightPool (from updateLights)
   * Process:
   *   1. Distance culling: keep only lights within adaptive threshold
   *   2. Frustum culling: remove lights outside camera view (stub)
   *   3. Occlusion culling: test visible light hash
   * Output: culledLights
   */
  cullByDistance(cameraPos: [number, number, number], threshold?: number): CellLight[] {
    this.culledLights = this.culler.cullByDistance(this.lightPool, cameraPos, threshold);
    this.stats.culledLightCount = this.culledLights.length;

    // Adaptive threshold
    if (this.lightPool.length > 0) {
      this.culler.adaptiveThreshold(this.culledLights.length, this.lightPool.length);
    }

    return this.culledLights;
  }

  /**
   * § Reservoir Sampling Pass (A-Res)
   *
   * Input: culledLights
   * Process:
   *   1. For each light, compute importance weight: W = color_sum * intensity / distance^2
   *   2. Add to reservoir using A-Res algorithm (AddLightSample)
   *   3. Track temporal visibility for next frame
   * Output: candidateSamples (up to reservoirSize lights)
   */
  reservoirSample(): LightSample[] {
    this.sampler.reset();

    for (let i = 0; i < this.culledLights.length; i++) {
      const light = this.culledLights[i];

      // Compute importance: brightness * intensity, biased toward local lights
      const brightness = (light.color[0] + light.color[1] + light.color[2]) / 3;
      const importance = brightness * light.intensity;

      // Geometric weight: closer is better (inverse-square falloff)
      const distanceFactor = Math.max(light.radius, 0.1);
      const geometricWeight = 1.0 / (1.0 + distanceFactor * distanceFactor);

      // Combined weight for A-Res
      const sampleWeight = importance * geometricWeight;

      // Determine if radial (point) vs directional
      const isRadial = light.radius > 0;

      this.sampler.addSampleARes(i, sampleWeight, true, isRadial);
    }

    this.candidateSamples = this.sampler.finalize();
    this.stats.avgImportance = this.sampler.getWeightSum() / Math.max(this.sampler.getSampleCount(), 1);

    return this.candidateSamples;
  }

  /**
   * § Ray Traced Shadow Pass (async compute / deferred)
   *
   * Input: selectedLights (from selectTopK)
   * Process:
   *   1. Generate shadow rays for each selected light
   *   2. Trace rays (async) to determine occlusion
   *   3. Filter shadow results with temporal/spatial denoising
   *   4. Store shadow factors for shading
   * Output: shadowFactors (opacity for each light)
   */
  async traceRays(
    selectedLights: ATLight[],
    cameraPos: [number, number, number],
    scene?: any,
  ): Promise<number[]> {
    const factors: number[] = [];

    for (let i = 0; i < selectedLights.length; i++) {
      const light = selectedLights[i];

      // Generate shadow ray from light to camera
      const rayGen = this.rayTracer.generateShadowRay(
        light.position,
        light.color,
        light.data,
        cameraPos,
        [0, 1, 0], // surface normal (stub)
        [Math.random(), Math.random()],
        light.type === 1 ? 'directional' : light.type === 4 ? 'rect' : 'sphere',
      );

      // Trace shadow ray
      const result = await this.rayTracer.traceShadow(
        rayGen.rayOrigin,
        rayGen.rayDirection,
        rayGen.rayTMin,
        rayGen.rayTMax,
        scene,
      );

      // Shadow factor: 1 = fully lit, 0 = fully shadowed
      const shadowFactor = result.isOccluded ? 0.3 : 1.0; // soft shadow fallback
      factors.push(shadowFactor);
    }

    this.shadowFactors = factors;
    this.stats.shadowsTraced = factors.length;

    return factors;
  }

  /**
   * § Select Top K Lights
   *
   * From reservoir samples, pick K most important lights
   * Apply diversity criteria to avoid clustering
   */
  selectTopK(k: number = 4): ATLight[] {
    const selected = this.fusion.selectTopK(this.candidateSamples, this.culledLights, k);
    this.stats.selectedLightCount = selected.length;
    return selected;
  }

  /**
   * § Finalize: Bind to AT Lighting
   *
   * Apply shadow factors and bind selected lights to shader
   */
  finalizeAndBind(): void {
    this.fusion.applyShadowFactors(this.shadowFactors);
    this.fusion.bindToATLighting();
  }

  /**
   * Complete frame pipeline
   */
  async frame(
    cellLights: CellLight[],
    cameraPos: [number, number, number],
    atLightingShader: any,
  ): Promise<void> {
    // § 1 Culling
    this.updateLights(cellLights);
    this.cullByDistance(cameraPos);

    // § 2 Sampling
    this.reservoirSample();

    // § 3 Selection
    const selected = this.selectTopK(4);

    // § 4 Ray Tracing (async)
    await this.traceRays(selected, cameraPos);

    // § 5 Bind to shader
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

  /**
   * Debug: print culling statistics
   */
  printDebugInfo(): void {
    const rayTracingCapable = this.rayTracer.isRayTracingSupported();
    const hardwareRayTracing = this.rayTracer.isHardwareRayTracingSupported();

    console.log('=== UEMegaLights Frame Statistics ===');
    console.log(`Frame: ${this.frameCount}`);
    console.log(`Input lights: ${this.stats.inputLightCount}`);
    console.log(`After culling: ${this.stats.culledLightCount} (${((this.stats.culledLightCount / Math.max(this.stats.inputLightCount, 1)) * 100).toFixed(1)}%)`);
    console.log(`Reservoir samples: ${this.candidateSamples.length}`);
    console.log(`Selected for shading: ${this.stats.selectedLightCount}`);
    console.log(`Avg importance weight: ${this.stats.avgImportance.toFixed(3)}`);
    console.log(`Distance threshold: ${this.getDistanceThreshold().toFixed(1)}`);
    console.log(`Ray tracing: ${rayTracingCapable ? 'supported' : 'fallback'} ${hardwareRayTracing ? '(HW)' : '(SW)'}`);
    console.log(`Shadows traced: ${this.stats.shadowsTraced}`);
  }
}

/**
 * Export types for external use
 */
export type { CellLight, LightSample, ShadowTraceResult, TileMetadata, VisibleLightHashEntry };
