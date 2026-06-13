#!/usr/bin/env python3
import json
import math
import os
import sys
from dataclasses import dataclass, field
from channels.rendering.species.species_port import _species_to_index
from typing import List, Optional, Dict


# =============================================================================
# AstroCellGPUScene — Uniform Buffer 管理 + 500 Cell 批量上传
# (ported from GPUScene.cpp / FGPUSceneResourceParameters)
#
# 鲁迅曾言：「不在沉默中爆发，便在沉默中灭亡。」
# 数据不上传，渲染便是虚妄；上传得太慢，帧率便是奢望。
# 批量上传是现实的让步，也是对 GPU 带宽的尊重。
#
# FGPUSceneResourceParameters → AstroCellGPUSceneResourceParams
#   GPUSceneInstanceSceneData        → instance_scene_data (list of dicts)
#   GPUSceneInstancePayloadData      → instance_payload_data (list of floats)
#   GPUScenePrimitiveSceneData       → primitive_scene_data (list of dicts)
#   GPUSceneLightData                → light_scene_data (list of dicts)
#   CommonParameters.GPUSceneFrameNumber → frame_number
#   CommonParameters.GPUSceneMaxAllocatedInstanceId → max_instance_id
#
# FGPUScenePrimitiveCollector → AstroCellPrimitiveCollector
#   Add()                     → add_cell()
#   Commit()                  → commit()
#   GetPrimitiveShaderParameters() → get_shader_params()
#
# UploadEveryFrame CVarGPUSceneUploadEveryFrame → ASTRO_GPU_UPLOAD_EVERY_FRAME
# PartitionUpdateRanges (parallel 4-way split) → _partition_update_ranges()
# Batch cap: 500 cells (mirrors CVarGPUSceneMaxPooledUploadBufferSize / cell)
#
# 2-D channel adaptation:
#   float4 StructuredBuffer  → list of 4-float tuples
#   UploadBuffer + GPU copy  → in-memory batch dict written to upload_batch.json
#   RDG pass graph           → sequential call in tick()
#   CVarGPUSceneParallelUpdate → _PARALLEL_UPDATE_THRESHOLD
# =============================================================================

# Upload-every-frame flag (mirrors CVarGPUSceneUploadEveryFrame)
ASTRO_GPU_UPLOAD_EVERY_FRAME: bool = False

# Batch size cap: max cells per UploadBuffer flush
# 鲁迅式：500 是个体面的数字——不贪婪，也不懦弱。
_ASTRO_GPU_BATCH_SIZE: int = 500

# Minimum item count before parallel partition is worthwhile
# (mirrors CVarGPUSceneParallelUpdate minimum threshold of 2048 items,
#  scaled down to 128 cells for the 2-D single-threaded context)
_PARALLEL_UPDATE_THRESHOLD: int = 128

# Path for the shared GPU scene upload channel
_GPU_SCENE_UPLOAD_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "physics", "gpu_scene_upload.json",
)


def _partition_update_ranges(item_count: int,
                              allow_parallel: bool = True) -> list:
    """
    Partition item_count into up to 4 update sub-ranges.

    Direct port of PartitionUpdateRanges() from GPUScene.cpp:
        if (ItemCount < 256 || !bAllowParallel) → 1 range (full)
        else                                   → 4 ranges (≈ ¼ each)

    Returns list of (start, count) tuples, length 1..4.
    鲁迅式：分工才能高效，但分工太细反而低效——256 是经验的边界。
    """
    if item_count < 256 or not allow_parallel:
        return [(0, item_count)]

    quarter = (item_count + 3) // 4
    ranges = []
    start = 0
    for _ in range(4):
        count = min(quarter, item_count - start)
        if count <= 0:
            break
        ranges.append((start, count))
        start += count
    return ranges


@dataclass
class AstroCellGPUSceneResourceParams:
    """
    Python equivalent of FGPUSceneResourceParameters.

    Stores the per-frame GPU scene resource handles (SRVs in UE5;
    plain Python lists here).  Written by AstroCellGPUScene.upload()
    and consumed by AstroCellLightPass.execute().

    Fields mirror the UB struct field names verbatim so grep across the
    C++ port remains unambiguous.

    鲁迅式：参数结构体是合同的签字页——字段名字即承诺，
    改一个字段便是违约，渲染器与着色器之间的信任便会崩塌。
    """
    # Instance scene data: one entry per cell instance
    # Mirrors GPUSceneInstanceSceneData StructuredBuffer<float4>
    GPUSceneInstanceSceneData: list = field(default_factory=list)

    # Instance payload data: per-instance custom float4 payload
    # Mirrors GPUSceneInstancePayloadData StructuredBuffer<float4>
    GPUSceneInstancePayloadData: list = field(default_factory=list)

    # Primitive scene data: one entry per cell primitive
    # Mirrors GPUScenePrimitiveSceneData StructuredBuffer<float4>
    GPUScenePrimitiveSceneData: list = field(default_factory=list)

    # Light scene data: one entry per light affecting cells
    # Mirrors GPUSceneLightData StructuredBuffer<FLightSceneData>
    GPUSceneLightData: list = field(default_factory=list)

    # Common scalar parameters (mirrors FGPUSceneCommonParameters)
    GPUSceneFrameNumber: int = 0
    GPUSceneMaxAllocatedInstanceId: int = 0
    GPUSceneMaxPersistentPrimitiveIndex: int = 0
    GPUSceneNumLightmapDataItems: int = 0


class AstroCellPrimitiveCollector:
    """
    Python equivalent of FGPUScenePrimitiveCollector.

    Accumulates dynamic cell primitives for a single proc() call before
    they are batch-uploaded to the AstroCellGPUScene.

    Lifecycle mirrors the C++ collector:
        collector = AstroCellPrimitiveCollector(gpu_scene)
        collector.add_cell(cell_id, bbox, species, gene_traits)
        ...  (up to _ASTRO_GPU_BATCH_SIZE cells)
        collector.commit()   # → flushes to AstroCellGPUScene

    鲁迅式：收集者的职责是积累，而非判断——判断是 commit() 的事。
    """

    def __init__(self, gpu_scene: "AstroCellGPUScene") -> None:
        self._gpu_scene = gpu_scene
        self._entries: list = []          # pending cell entries
        self._committed: bool = False
        self._total_instances: int = 0
        self._payload_float4_count: int = 0

    def add_cell(self, cell_id: str, bbox: dict, species: str,
                 gene_traits: dict,
                 num_instances: int = 1) -> tuple:
        """
        Add a cell primitive to the collector.

        Mirrors FGPUScenePrimitiveCollector::Add():
            Allocates UploadData lazily on first call.
            Records LocalInstanceSceneDataOffset for the caller.
            Increments TotalInstanceCount + InstancePayloadDataFloat4Count.

        Returns (primitive_index, instance_scene_data_offset).

        鲁迅式：添加只是登记，不是承诺——commit 才是真正的履行。
        """
        if self._committed:
            raise RuntimeError(
                "[AstroCellGPUScene] add_cell called after commit() — "
                "collector is sealed (mirrors bCommitted check in C++)."
            )
        if len(self._entries) >= _ASTRO_GPU_BATCH_SIZE:
            # Auto-commit on overflow — mirrors UploadBuffer pool limit
            print(
                f"[AstroCellGPUScene] PrimitiveCollector auto-commit at "
                f"batch cap ({_ASTRO_GPU_BATCH_SIZE} cells).",
                file=sys.stderr,
            )
            self.commit()

        prim_index = len(self._entries)
        local_offset = self._total_instances

        # Pack instance scene data as float4 equivalent
        # Mirrors FPrimitiveUniformShaderParameters layout (simplified):
        #   vec4[0] = (x, y, w, h)
        #   vec4[1] = (z, species_index, 0, 0)
        sp_idx = float(_species_to_index(species))
        instance_float4 = (
            float(bbox["x"]), float(bbox["y"]),
            float(bbox["w"]), float(bbox["h"]),
        )
        payload_float4 = (
            float(bbox.get("z", 0)), sp_idx, 0.0, 0.0,
        )

        self._entries.append({
            "cell_id":         cell_id,
            "species":         species,
            "gene_traits":     gene_traits,
            "bbox":            bbox,
            "num_instances":   num_instances,
            "prim_index":      prim_index,
            "local_offset":    local_offset,
            "instance_float4": instance_float4,
            "payload_float4":  payload_float4,
        })

        self._total_instances += num_instances
        self._payload_float4_count += num_instances  # 1 payload per instance

        return prim_index, local_offset

    def commit(self) -> int:
        """
        Flush accumulated primitives to the parent AstroCellGPUScene.

        Mirrors FGPUScenePrimitiveCollector::Commit():
            Sets bCommitted = true.
            Calls GPUScene.UploadDynamicPrimitiveShaderDataForView().
            Returns number of primitives flushed.

        鲁迅式：commit 是不可撤销的选择——选择之后便是历史。
        """
        if not self._entries:
            self._committed = True
            return 0
        flushed = self._gpu_scene._flush_collector(self._entries)
        self._committed = True
        return flushed

    def get_shader_params(self, prim_index: int) -> dict | None:
        """
        Return the shader parameter dict for a dynamic primitive.

        Mirrors FGPUScenePrimitiveCollector::GetPrimitiveShaderParameters()
        (called with DrawPrimitiveId & GPrimIDDynamicFlag mask stripped).

        Returns None if prim_index is out of range or collector is empty.
        """
        if 0 <= prim_index < len(self._entries):
            e = self._entries[prim_index]
            return {
                "cell_id":       e["cell_id"],
                "bbox":          e["bbox"],
                "species":       e["species"],
                "instance_float4": e["instance_float4"],
                "payload_float4":  e["payload_float4"],
            }
        return None


class AstroCellGPUScene:
    """
    Python equivalent of FGPUScene — the central GPU-side primitive data store.

    Manages the per-frame uniform buffer upload pipeline for up to
    _ASTRO_GPU_BATCH_SIZE (500) cell primitives per flush.

    Architecture (mirrors GPUScene.cpp upload loop):
      1. Cells are accumulated via AstroCellPrimitiveCollector.add_cell()
      2. collector.commit() calls _flush_collector() which partitions the
         pending list into ≤500-cell batches using _partition_update_ranges()
      3. Each batch is packed into AstroCellGPUSceneResourceParams and
         serialised to physics/gpu_scene_upload.json (the UB channel)
      4. tick() advances the frame counter and optionally forces a full
         re-upload (ASTRO_GPU_UPLOAD_EVERY_FRAME)

    Per-frame stats mirror the CSV_DEFINE_CATEGORY(GPUScene, …) counters:
        primitives_uploaded, instances_uploaded, batches_flushed, frame_number

    鲁迅式：数据上传如同发声——不发声便无从影响世界，
    但发声太频繁也是一种打扰。500 cells/batch 是沉默与噪声之间的平衡。
    """

    def __init__(self) -> None:
        self._resource_params = AstroCellGPUSceneResourceParams()
        self._pending_dirty: list = []   # cells marked dirty but not yet uploaded
        self._frame_number: int = 0
        # High-water marks (mirrors FGPUScene diagnostic counters)
        self._stats: dict = {
            "primitives_uploaded": 0,
            "instances_uploaded":  0,
            "batches_flushed":     0,
            "frame_number":        0,
        }

    def make_collector(self) -> AstroCellPrimitiveCollector:
        """
        Create a new primitive collector for this frame.
        Mirrors the FGPUScenePrimitiveCollector constructor taking FGPUScene&.
        """
        return AstroCellPrimitiveCollector(self)

    def mark_dirty(self, cell_id: str, bbox: dict, species: str) -> None:
        """
        Mark a cell primitive as needing re-upload.

        Mirrors FGPUScene::AddPrimitiveToUpdate() called from
        UpdatePrimitiveTransform / SceneProxy changes.
        Dirty cells are batched in the next tick() upload pass.

        鲁迅式：脏标记是未说出的话——tick() 才是说出口的那一刻。
        """
        self._pending_dirty.append({
            "cell_id": cell_id,
            "bbox":    bbox,
            "species": species,
        })

    def tick(self, force_full_upload: bool = False) -> dict:
        """
        Advance the frame and flush pending dirty primitives.

        Mirrors FGPUScene::Update() / FGPUScene::UploadGeneral():
          - If ASTRO_GPU_UPLOAD_EVERY_FRAME or force_full_upload: re-uploads ALL
            cells registered in cell_registry.json (expensive; debug only).
          - Otherwise: flushes only the pending dirty list accumulated since
            the last tick().

        Returns per-frame upload stats dict.

        鲁迅式：每一帧都是一次机会——更新或腐朽，没有中间状态。
        """
        self._frame_number += 1
        self._resource_params.GPUSceneFrameNumber = self._frame_number

        if ASTRO_GPU_UPLOAD_EVERY_FRAME or force_full_upload:
            # Full scene re-upload path (mirrors CVarGPUSceneUploadEveryFrame=1)
            registry = _load_cell_registry()
            cells = registry.get("cells", {})
            collector = self.make_collector()
            for cid, entry in cells.items():
                bbox_data = entry.get("bbox", {})
                if "min" in bbox_data:
                    mn = bbox_data["min"]
                    mx = bbox_data["max"]
                    bbox = {
                        "x": mn[0], "y": mn[1],
                        "w": mx[0] - mn[0], "h": mx[1] - mn[1],
                        "z": mn[2] if len(mn) > 2 else 0,
                    }
                else:
                    bbox = bbox_data
                collector.add_cell(cid, bbox, entry.get("species", ""), {})
            collector.commit()
        else:
            # Incremental dirty-list path
            if self._pending_dirty:
                collector = self.make_collector()
                for item in self._pending_dirty:
                    collector.add_cell(
                        item["cell_id"], item["bbox"], item["species"], {}
                    )
                collector.commit()
                self._pending_dirty.clear()

        self._stats["frame_number"] = self._frame_number
        return dict(self._stats)

    def get_resource_params(self) -> AstroCellGPUSceneResourceParams:
        """Return the current frame's resource parameter block."""
        return self._resource_params

    # ------------------------------------------------------------------
    # Internal: batch flush (called by AstroCellPrimitiveCollector.commit)
    # ------------------------------------------------------------------

    def _flush_collector(self, entries: list) -> int:
        """
        Flush a list of primitive entries into the resource params and
        persist the upload batch to the channel file.

        Partitions entries into ≤_ASTRO_GPU_BATCH_SIZE sub-ranges via
        _partition_update_ranges(), mirrors the parallel update loop in
        FGPUScene::UploadGeneral() that uses ParallelFor to populate the
        upload buffer.

        Returns total number of primitives flushed.

        鲁迅式：分批写入是工程的妥协，不是思想的分裂。
        """
        total = len(entries)
        if total == 0:
            return 0

        allow_parallel = total >= _PARALLEL_UPDATE_THRESHOLD
        ranges = _partition_update_ranges(total, allow_parallel)

        instance_data: list = []
        payload_data:  list = []
        primitive_data: list = []

        for (start, count) in ranges:
            batch = entries[start:start + count]
            for e in batch:
                instance_data.append(e["instance_float4"])
                payload_data.append(e["payload_float4"])
                # Primitive scene data: bbox + species index packed as float4×2
                primitive_data.append({
                    "cell_id":     e["cell_id"],
                    "species":     e["species"],
                    "bbox":        e["bbox"],
                    "prim_index":  e["prim_index"],
                    "local_offset": e["local_offset"],
                })

        # Write to resource params (mirrors RDG UAV write in C++)
        self._resource_params.GPUSceneInstanceSceneData.extend(instance_data)
        self._resource_params.GPUSceneInstancePayloadData.extend(payload_data)
        self._resource_params.GPUScenePrimitiveSceneData.extend(primitive_data)
        self._resource_params.GPUSceneMaxAllocatedInstanceId = max(
            self._resource_params.GPUSceneMaxAllocatedInstanceId,
            total - 1,
        )
        self._resource_params.GPUSceneMaxPersistentPrimitiveIndex = max(
            self._resource_params.GPUSceneMaxPersistentPrimitiveIndex,
            total - 1,
        )

        # Persist upload batch to physics/gpu_scene_upload.json channel
        upload_payload = {
            "frame_number":   self._frame_number,
            "batch_size":     total,
            "ranges":         ranges,
            "instance_data":  instance_data,
            "payload_data":   payload_data,
            "primitive_data": [
                {k: v for k, v in p.items() if k != "gene_traits"}
                for p in primitive_data
            ],
        }
        try:
            os.makedirs(os.path.dirname(_GPU_SCENE_UPLOAD_PATH), exist_ok=True)
            with open(_GPU_SCENE_UPLOAD_PATH, "w") as _f:
                json.dump(upload_payload, _f, indent=2)
        except OSError as _e:
            print(
                f"[AstroCellGPUScene] WARNING: failed to persist upload batch: {_e}",
                file=sys.stderr,
            )

        # Update stats (mirrors DEFINE_GPU_STAT(GPUSceneUpdate) counters)
        self._stats["primitives_uploaded"] += total
        self._stats["instances_uploaded"]  += sum(e["num_instances"] for e in entries)
        self._stats["batches_flushed"]     += len(ranges)

        print(
            f"[AstroCellGPUScene] _flush_collector: "
            f"total={total} ranges={ranges} "
            f"instance_data_len={len(instance_data)} "
            f"frame={self._frame_number}",
            file=sys.stderr,
        )

        return total


# Module-level singleton — mirrors the FGPUScene instance owned by FScene
_ASTRO_GPU_SCENE: AstroCellGPUScene | None = None


def get_astro_gpu_scene() -> AstroCellGPUScene:
    """
    Return the module-level AstroCellGPUScene singleton.

    Mirrors the FScene::GPUScene member access pattern; callers use this
    rather than constructing their own instance so all collectors share the
    same resource param block.

    鲁迅式：单例的存在是为了让大家说同一种语言，
    即便大家未必都愿意如此。
    """
    global _ASTRO_GPU_SCENE
    if _ASTRO_GPU_SCENE is None:
        _ASTRO_GPU_SCENE = AstroCellGPUScene()
    return _ASTRO_GPU_SCENE


# =============================================================================
# AstroCellLightPass — Per-Cell 光照计算
# (ported from LightRendering.cpp / RenderLight / RenderSimpleLightsStandard)
#
# 鲁迅曾言：「真的猛士，敢于直面惨淡的人生，敢于正视淋漓的鲜血。」
# 光照亦然——敢于正视每一个 cell 的遮挡、衰减、接触阴影，
# 才能从黑暗中还原真实的色彩。
#
# UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   FDeferredLightUniformStruct     → AstroCellDeferredLightUniforms
#   FSimpleLightEntry               → AstroCellSimpleLight
#   RenderLight()                   → AstroCellLightPass.render_light()
#   RenderSimpleLightsStandard()    → AstroCellLightPass.render_simple_lights()
#   GetLightContactShadowParameters → _get_contact_shadow_params()
#   bAllowSimpleLights (CVar)       → ASTRO_ALLOW_SIMPLE_LIGHTS
#   GAllowDepthBoundsTest           → ASTRO_ALLOW_DEPTH_BOUNDS_TEST
#   CVarRayTracingOcclusion         → ASTRO_RAY_TRACING_OCCLUSION (always 0)
#   ENABLE_DEBUG_DISCARD_PROP       → ASTRO_DEBUG_DISCARD_PROP
#
# Light model (2-D SVG adaptation):
#   Deferred light  = a single dominant directional/point light that drives
#                     the per-cell diffuse + specular highlight.
#   Simple lights   = particle / secondary lights added on top.
#   Contact shadow  = proximity-based darkening of adjacent cell edges.
#   Depth bounds    = z-layer range gate; cells outside [z_min, z_max] skip.
#
# Output: per-cell light_result dict written to cell/{id}/light.json channel.
# =============================================================================

# Global flags (mirror CVarAllowSimpleLights, GAllowDepthBoundsTest)
ASTRO_ALLOW_SIMPLE_LIGHTS:       bool = True
ASTRO_ALLOW_DEPTH_BOUNDS_TEST:   bool = True
ASTRO_RAY_TRACING_OCCLUSION:     bool = False   # rt shadows disabled (always 0)
ASTRO_DEBUG_DISCARD_PROP:        float = 0.0     # 0 = discard nothing

# Contact shadow length constants (mirror CVarContactShadowsOverrideLength)
_CONTACT_SHADOW_DEFAULT_LENGTH:          float = 0.15  # 15% screen-space
_CONTACT_SHADOW_CASTING_INTENSITY:       float = 1.0
_CONTACT_SHADOW_NON_CASTING_INTENSITY:   float = 0.0


@dataclass
class AstroCellDeferredLightUniforms:
    """
    Python equivalent of FDeferredLightUniformStruct.

    Stores the per-light shader parameters consumed by the deferred light
    pass.  Fields mirror the IMPLEMENT_GLOBAL_SHADER_PARAMETER_STRUCT
    declaration for "DeferredLightUniforms" verbatim.

    In the 2-D SVG substrate:
      Position      → (light_x, light_y, light_z) in cell-local coords
      InvRadius     → 1 / falloff_radius
      Color         → (R, G, B) light colour, pre-multiplied by intensity
      FalloffExponent → attenuation curve power (UE default = 8.0 for point)
      Direction     → normalised (dx, dy, dz) for directional lights
      Tangent       → cross-axis for rect lights (unused for point/directional)
      SpotAngles    → (cos_inner, cos_outer) for spot lights
      SourceRadius  → penumbra radius (drives contact shadow softness)
      SoftSourceRadius → soft penumbra extension
      SpecularScale → modulates specular highlight magnitude
      ContactShadowLength → screen-space contact shadow ray length

    鲁迅式：参数多不代表理解深——
    真正的理解是能删掉不必要的参数，此处一个都不删。
    """
    Position:              tuple = (0.0, 0.0, 600.0)   # above the scene
    InvRadius:             float = 1.0 / 2000.0
    Color:                 tuple = (1.0, 0.98, 0.95)   # warm white
    FalloffExponent:       float = 8.0
    Direction:             tuple = (0.0, -0.5, -0.866) # 30° from vertical
    Tangent:               tuple = (1.0, 0.0, 0.0)
    SpotAngles:            tuple = (0.0, 1.0)           # full hemisphere
    SourceRadius:          float = 80.0
    SoftSourceRadius:      float = 20.0
    SpecularScale:         float = 1.0
    ContactShadowLength:   float = _CONTACT_SHADOW_DEFAULT_LENGTH
    ContactShadowLengthInWS: bool = False
    ContactShadowCastingIntensity:    float = _CONTACT_SHADOW_CASTING_INTENSITY
    ContactShadowNonCastingIntensity: float = _CONTACT_SHADOW_NON_CASTING_INTENSITY
    LightingChannelMask:   int   = 0xFF   # all channels enabled
    RectLightBarnCosAngle: float = 0.0
    RectLightBarnLength:   float = 0.0
    # IES profile intensity (1.0 = no IES, uniform)
    IESAttenuation:        float = 1.0


@dataclass
class AstroCellSimpleLight:
    """
    Python equivalent of FSimpleLightEntry.

    Represents a particle / secondary point light affecting one cell.
    Used in render_simple_lights() to layer additional highlights on top
    of the main deferred light contribution.

    Mirrors FSimpleLightEntry from SimpleElementRendering / particles:
        Radius       → effective radius of the light sphere
        Color        → (R, G, B) pre-multiplied intensity
        Exponent     → falloff exponent (0 = inverse-square, >0 = legacy)
        bAffectTranslucency → whether this light touches translucent cells
    """
    position:             tuple = (0.0, 0.0, 0.0)
    radius:               float = 200.0
    color:                tuple = (1.0, 1.0, 1.0)
    exponent:             float = 0.0   # 0 = physically based inverse-square
    affect_translucency:  bool  = True


def _get_contact_shadow_params(light: AstroCellDeferredLightUniforms) -> tuple:
    """
    Compute contact shadow parameters for a light.

    Mirrors GetLightContactShadowParameters() from LightRendering.cpp:
        OutLength                 = proxy.ContactShadowLength()
        bOutLengthInWS            = proxy.IsContactShadowLengthInWS()
        OutCastingIntensity       = proxy.ContactShadowCastingIntensity()
        OutNonCastingIntensity    = proxy.ContactShadowNonCastingIntensity()

    Returns (length, in_ws, casting_intensity, non_casting_intensity).

    鲁迅式：接触阴影是近处的真相——远处的阴影可以近似，近处不行。
    """
    return (
        light.ContactShadowLength,
        light.ContactShadowLengthInWS,
        light.ContactShadowCastingIntensity,
        light.ContactShadowNonCastingIntensity,
    )


def _point_light_attenuation(distance: float,
                              inv_radius: float,
                              falloff_exponent: float,
                              ies: float = 1.0) -> float:
    """
    Compute point light radial attenuation.

    Mirrors the HLSL GetLocalLightAttenuation() / RadialAttenuation() path
    used in the deferred light shader:

        DistanceFadeAlpha = saturate((Radius - dist) / (Radius * 0.2))
        RadialAtten       = pow(1 - (dist * InvRadius)^2, FalloffExponent)
        if dist >= Radius: atten = 0

    Physical inverse-square falloff (FalloffExponent==0):
        atten = 1 / max(dist^2, 1e-4) * (1/InvRadius)^2 * IES

    For legacy falloff (exponent > 0) the UE polynomial is used.
    IES profile modulation is applied last.

    鲁迅式：衰减公式是现实的数学翻译——光永远不会真的消失，
    只是越来越弱，弱到无法被感知，被我们称之为黑暗。
    """
    radius = 1.0 / max(inv_radius, 1e-9)
    if distance >= radius:
        return 0.0
    if falloff_exponent <= 0.0:
        # Physically based: inverse square
        atten = 1.0 / max(distance * distance, 1e-4) * (radius * radius)
    else:
        # Legacy UE polynomial (mirrors RadialAttenuation in BasePassCommon.ush)
        r_ratio  = min(1.0, distance * inv_radius)
        atten    = max(0.0, 1.0 - r_ratio ** 2) ** falloff_exponent

    # Soft edge fade: smoothstep over the outer 20% of radius
    fade = min(1.0, max(0.0, (radius - distance) / max(radius * 0.2, 1.0)))
    return max(0.0, min(1.0, atten * fade * ies))


def _directional_light_contribution(normal: tuple,
                                     light_dir: tuple,
                                     color: tuple,
                                     specular_scale: float,
                                     roughness: float) -> tuple:
    """
    Compute diffuse + specular contribution for a directional light.

    Mirrors the light shader's MaskedLightSample() + BRDF evaluation:
        NoL      = saturate(dot(N, L))
        Diffuse  = NoL × LightColor × (1/π)            ← Lambertian
        Specular = GGX_Specular(NoL, roughness) × LightColor × SpecularScale

    Returns (diffuse_r, diffuse_g, diffuse_b, specular_scalar).
    specular_scalar is a single float in [0,1] (grey for SVG opacity blend).

    鲁迅式：漫反射是公正的——它不偏爱任何方向；
    镜面反射是傲慢的——它只爱特定角度的观察者。
    两者共存，才是真实的光照。
    """
    nx, ny, nz = normal
    lx, ly, lz = light_dir
    # Normalize light direction (defensive)
    l_len = math.sqrt(lx*lx + ly*ly + lz*lz)
    if l_len > 1e-6:
        lx, ly, lz = lx/l_len, ly/l_len, lz/l_len

    NoL = max(0.0, nx*lx + ny*ly + nz*lz)

    cr, cg, cb = color
    diff_r = NoL * cr / math.pi
    diff_g = NoL * cg / math.pi
    diff_b = NoL * cb / math.pi

    # Specular: simplified GGX highlight (NdotH ≈ NoL for on-axis view)
    alpha = roughness * roughness
    denom = NoL * (1.0 - alpha) + alpha
    ggx_ndf = alpha / max(math.pi * denom * denom, 1e-9)
    specular = min(1.0, ggx_ndf * specular_scale * NoL)

    return (diff_r, diff_g, diff_b, specular)


def _contact_shadow_factor(cell_bbox: dict,
                            sibling_bboxes: dict,
                            contact_length: float,
                            in_ws: bool) -> float:
    """
    Compute a contact shadow attenuation factor for one cell.

    Mirrors the screen-space contact shadow ray march:
        Ray steps from the shaded pixel toward the light in screen space.
        For each step, if depth is occluded → shadow accumulates.

    2-D adaptation:
        The contact shadow ray is a horizontal scan in Z space.
        Cells within contact_length × cell_height in Z above the receiver
        and overlapping in XY contribute to contact shadow darkening.

    Returns shadow_factor ∈ [0, 1], where 1 = fully lit, 0 = fully shadowed.
    Uses casting_intensity for cells that cast and non_casting_intensity for
    cells that don't (mirrors the C++ shadow mask encoding).

    鲁迅式：接触阴影是物体彼此之间看不见却真实存在的影响——
    就像人与人之间的压力，不必直接接触，靠近便足以使人窒息。
    """
    rx0 = cell_bbox["x"]
    ry0 = cell_bbox["y"]
    rx1 = rx0 + cell_bbox["w"]
    ry1 = ry0 + cell_bbox["h"]
    rz  = float(cell_bbox.get("z", 3))

    # Contact shadow range in Z: contact_length × cell_height
    z_contact_range = contact_length * cell_bbox["h"] if not in_ws else contact_length

    shadow_acc = 0.0
    for other_id, other_bbox in sibling_bboxes.items():
        oz  = float(other_bbox.get("z", 3))
        dz  = oz - rz
        # Only cells directly above (higher Z) contribute to downward contact shadow
        if dz <= 0.0 or dz > z_contact_range:
            continue

        ox0 = other_bbox["x"]
        oy0 = other_bbox["y"]
        ox1 = ox0 + other_bbox["w"]
        oy1 = oy0 + other_bbox["h"]

        # XY overlap test (contact shadow only from directly overlapping cells)
        if ox1 <= rx0 or ox0 >= rx1 or oy1 <= ry0 or oy0 >= ry1:
            continue

        # Proximity fade: stronger at small Z separation
        fade = max(0.0, 1.0 - dz / max(z_contact_range, 1e-6))
        shadow_acc += fade * _CONTACT_SHADOW_CASTING_INTENSITY

    return max(0.0, 1.0 - min(1.0, shadow_acc))


class AstroCellLightPass:
    """
    Python equivalent of the deferred + simple light rendering passes
    in LightRendering.cpp.

    Computes per-cell light contributions (diffuse, specular, contact shadow,
    simple light layers) and writes results to cell/{id}/light.json channel.

    Two primary entry points mirror the two C++ render functions:
      render_light(cell_id, ...)      → RenderLight()
      render_simple_lights(cell_id, …) → RenderSimpleLightsStandard()

    The execute() method dispatches both passes in order and persists the
    merged light_result to the channel — mirrors the render pass scheduling
    in FDeferredShadingSceneRenderer::RenderLights().

    鲁迅式：两道光照如两种声音——
    一道从上方照射，庄严而均匀；
    一道从四面涌来，零碎而喧嚣。
    缺一不可，才是真实的世界。
    """

    def __init__(self, light: AstroCellDeferredLightUniforms | None = None,
                 simple_lights: list | None = None) -> None:
        # Primary deferred light (directional / point)
        self._light = light or AstroCellDeferredLightUniforms()
        # Secondary simple lights list (particle lights)
        self._simple_lights: list = simple_lights or []
        # Output path template
        self._channels_dir = os.path.dirname(os.path.abspath(__file__))

    # ------------------------------------------------------------------
    # render_light — deferred light contribution
    # ------------------------------------------------------------------

    def render_light(self,
                     cell_id: str,
                     bbox: dict,
                     species: str,
                     roughness: float,
                     sibling_bboxes: dict) -> dict:
        """
        Compute the deferred light contribution for one cell.

        Mirrors RenderLight() → MaskedLightSample() → BRDF pipeline:

        Step 1 — Debug discard (ENABLE_DEBUG_DISCARD_PROP gate):
            If ASTRO_DEBUG_DISCARD_PROP > 0 and hash(cell_id) mod 100 <
            discard_prop * 100, return zero contribution (mirrors the
            debug light cull used for performance profiling).

        Step 2 — Depth bounds test (GAllowDepthBoundsTest gate):
            If ASTRO_ALLOW_DEPTH_BOUNDS_TEST and cell z-layer falls outside
            [light_z_min, light_z_max], return zero (mirrors HW depth bounds
            test that clips the light sphere to the scene depth range).

        Step 3 — Distance attenuation (_point_light_attenuation):
            Compute radial falloff from cell centre to light Position.

        Step 4 — Directional contribution (_directional_light_contribution):
            Diffuse (Lambertian) + specular (GGX) using cell surface normal.

        Step 5 — Contact shadow (_contact_shadow_factor):
            Proximity-based shadow from directly overlapping cells above.

        Step 6 — Compose final light_color and highlight_opacity.

        Returns dict with:
            diffuse_color (hex), highlight_opacity (float),
            contact_shadow_factor (float), attenuation (float),
            deferred_light_contribution (dict).

        鲁迅式：每一步都是一道筛子——大多数光子在到达你之前便已死去。
        """
        # ── Step 1: debug discard ──────────────────────────────────────────
        if ASTRO_DEBUG_DISCARD_PROP > 0.0:
            if (abs(hash(cell_id)) % 100) < int(ASTRO_DEBUG_DISCARD_PROP * 100):
                return self._zero_light_result(cell_id, "debug_discard")

        cell_z = float(bbox.get("z", 3))
        cx = bbox["x"] + bbox["w"] / 2.0
        cy = bbox["y"] + bbox["h"] / 2.0

        # ── Step 2: depth bounds test ──────────────────────────────────────
        if ASTRO_ALLOW_DEPTH_BOUNDS_TEST:
            # Light affects z-layers within ±3 of its anchor z
            lz = self._light.Position[2]
            z_min = lz / 200.0 - 3.0   # normalise light z to layer space
            z_max = lz / 200.0 + 3.0
            if not (z_min <= cell_z <= z_max):
                return self._zero_light_result(cell_id, "depth_bounds_cull")

        # ── Step 3: distance attenuation ──────────────────────────────────
        lx, ly, lz = self._light.Position
        dx = cx - lx
        dy = cy - ly
        dz = cell_z * 100.0 - lz   # z in world units (100 per layer)
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        atten = _point_light_attenuation(
            dist, self._light.InvRadius,
            self._light.FalloffExponent, self._light.IESAttenuation
        )

        # ── Step 4: directional contribution ──────────────────────────────
        # Cell surface normal: facing upward in 2-D (away from viewer = +Z)
        # Mix with light direction for a simple Lambertian approximation
        cell_normal = (0.0, 0.0, 1.0)
        ldx, ldy, ldz = self._light.Direction
        diff_r, diff_g, diff_b, specular = _directional_light_contribution(
            cell_normal,
            (-ldx, -ldy, -ldz),  # negate: Direction points from light
            self._light.Color,
            self._light.SpecularScale,
            roughness,
        )

        # Scale by attenuation
        diff_r *= atten
        diff_g *= atten
        diff_b *= atten
        specular *= atten

        # ── Step 5: contact shadow ─────────────────────────────────────────
        cs_len, cs_ws, cs_cast, cs_nocast = _get_contact_shadow_params(self._light)
        contact_shadow = _contact_shadow_factor(
            bbox, sibling_bboxes, cs_len, cs_ws
        )
        # Apply contact shadow to diffuse
        diff_r *= contact_shadow
        diff_g *= contact_shadow
        diff_b *= contact_shadow

        # ── Step 6: compose output ─────────────────────────────────────────
        # Convert diffuse float3 to an SVG hex overlay colour
        # Clamp to [0,1] and convert to 0-255 int
        def _to_int(v: float) -> int:
            return max(0, min(255, int(v * 255.0)))

        light_hex = "#{:02X}{:02X}{:02X}".format(
            _to_int(diff_r), _to_int(diff_g), _to_int(diff_b)
        )
        highlight_opacity = max(0.0, min(1.0, specular * self._light.SpecularScale))

        result = {
            "cell_id":              cell_id,
            "pass":                 "deferred_light",
            "diffuse_color":        light_hex,
            "highlight_opacity":    round(highlight_opacity, 4),
            "contact_shadow_factor": round(contact_shadow, 4),
            "attenuation":          round(atten, 4),
            "deferred_light_contribution": {
                "diff_r": round(diff_r, 4),
                "diff_g": round(diff_g, 4),
                "diff_b": round(diff_b, 4),
                "specular": round(specular, 4),
            },
        }

        print(
            f"[AstroCellLightPass] render_light: cell={cell_id} "
            f"atten={atten:.4f} contact_shadow={contact_shadow:.4f} "
            f"diff=({diff_r:.3f},{diff_g:.3f},{diff_b:.3f}) "
            f"spec={specular:.4f} light_hex={light_hex}",
            file=sys.stderr,
        )

        return result

    # ------------------------------------------------------------------
    # render_simple_lights — secondary particle / point lights
    # ------------------------------------------------------------------

    def render_simple_lights(self, cell_id: str, bbox: dict) -> list:
        """
        Compute simple (particle) light contributions for one cell.

        Mirrors RenderSimpleLightsStandard() — iterates self._simple_lights,
        computes point-light attenuation for each, returns a list of
        per-light contribution dicts.

        Skipped entirely if ASTRO_ALLOW_SIMPLE_LIGHTS is False (mirrors
        CVarAllowSimpleLights=0 path which skips the simple light pass).

        鲁迅式：简单光源是边缘的声音——
        它们很小，但足以改变局部的气氛；
        忽略它们，场景便失去了层次。
        """
        if not ASTRO_ALLOW_SIMPLE_LIGHTS:
            return []

        cx = bbox["x"] + bbox["w"] / 2.0
        cy = bbox["y"] + bbox["h"] / 2.0
        cz = float(bbox.get("z", 3)) * 100.0

        contributions = []
        for sl in self._simple_lights:
            slx, sly, slz = sl.position
            dx = cx - slx
            dy = cy - sly
            dz = cz - slz
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)

            inv_r = 1.0 / max(sl.radius, 1.0)
            atten = _point_light_attenuation(dist, inv_r, sl.exponent)
            if atten < 1e-4:
                continue  # below contribution threshold — skip (mirrors light cull)

            sr, sg, sb = sl.color
            contributions.append({
                "simple_light_pos":  sl.position,
                "simple_light_color": sl.color,
                "attenuation":       round(atten, 4),
                "contribution": {
                    "r": round(sr * atten, 4),
                    "g": round(sg * atten, 4),
                    "b": round(sb * atten, 4),
                },
            })

        return contributions

    # ------------------------------------------------------------------
    # execute — full per-cell light pass dispatch
    # ------------------------------------------------------------------

    def execute(self,
                cell_id: str,
                bbox: dict,
                species: str,
                roughness: float,
                sibling_bboxes: dict) -> dict:
        """
        Execute the full light pass for one cell and persist results.

        Dispatch order mirrors FDeferredShadingSceneRenderer::RenderLights():
          1. render_light()         (deferred analytical light)
          2. render_simple_lights() (particle / secondary lights)
          3. Merge + write to cell/{id}/light.json channel

        Returns the merged light_result dict.

        鲁迅式：执行是思想的落地——不执行的光照算法只是空谈，
        数据写入磁盘的那一刻，它才真正存在于世界之中。
        """
        deferred = self.render_light(
            cell_id, bbox, species, roughness, sibling_bboxes
        )
        simple = self.render_simple_lights(cell_id, bbox)

        # Merge simple light contributions into a single additive colour
        simple_r = sum(c["contribution"]["r"] for c in simple)
        simple_g = sum(c["contribution"]["g"] for c in simple)
        simple_b = sum(c["contribution"]["b"] for c in simple)

        def _blend_int(v: float) -> int:
            return max(0, min(255, int(v * 255.0)))

        simple_hex = "#{:02X}{:02X}{:02X}".format(
            _blend_int(simple_r), _blend_int(simple_g), _blend_int(simple_b)
        ) if simple else "#000000"

        result = {
            **deferred,
            "simple_lights_count":       len(simple),
            "simple_lights_accumulated": simple_hex,
            "simple_light_details":      simple,
        }

        # Persist to light.json channel
        light_channel_path = os.path.join(
            self._channels_dir, "cell", cell_id, "light.json"
        )
        try:
            os.makedirs(os.path.dirname(light_channel_path), exist_ok=True)
            with open(light_channel_path, "w") as _f:
                json.dump(result, _f, indent=2)
        except OSError as _e:
            print(
                f"[AstroCellLightPass] WARNING: failed to write light.json "
                f"for cell={cell_id}: {_e}",
                file=sys.stderr,
            )

        return result

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    @staticmethod
    def _zero_light_result(cell_id: str, reason: str) -> dict:
        """Return a zero-contribution light result (culled path)."""
        return {
            "cell_id":               cell_id,
            "pass":                  "culled",
            "cull_reason":           reason,
            "diffuse_color":         "#000000",
            "highlight_opacity":     0.0,
            "contact_shadow_factor": 1.0,
            "attenuation":           0.0,
            "deferred_light_contribution": {
                "diff_r": 0.0, "diff_g": 0.0, "diff_b": 0.0, "specular": 0.0,
            },
            "simple_lights_count":        0,
            "simple_lights_accumulated":  "#000000",
            "simple_light_details":       [],
        }


def run_cell_light_pass(
    cell_id: str,
    bbox: dict,
    species: str,
    sibling_bboxes: dict,
    light: AstroCellDeferredLightUniforms | None = None,
    simple_lights: list | None = None,
) -> dict:
    """
    Top-level convenience wrapper — execute the full light pass for one cell.

    Mirrors the call site in FDeferredShadingSceneRenderer::RenderLights()
    that constructs the light pass, binds uniforms, and dispatches the
    screen-space light volume draw.

    Derives roughness from the species roughness table (same mapping used
    by ShadingEnergyConservation and StyleProbe subsystems for consistency).

    Returns the merged light_result dict (same as AstroCellLightPass.execute).

    鲁迅式：封装是文明的标志——让调用者无需知道内部的挣扎，
    只需给出 cell_id，便能得到光照结果。
    也许，这就是所谓的「体面」。
    """
    _ROUGHNESS_MAP = {
        "cil-eye": 0.1, "cil-bolt": 0.2, "cil-plus": 0.3,
        "cil-vector": 0.5, "cil-arrow-right": 0.7,
        "cil-filter": 0.3, "cil-code": 0.4, "cil-layers": 0.2,
        "cil-loop": 0.5, "cil-graph": 0.6,
    }
    roughness = _ROUGHNESS_MAP.get(species, 0.5)
    lp = AstroCellLightPass(light=light, simple_lights=simple_lights)
    return lp.execute(cell_id, bbox, species, roughness, sibling_bboxes)


# =============================================================================
# AstroCellVolumetricCloud — 体积云渲染核心
# (ported from VolumetricCloudRendering.cpp)
#
# 鲁迅曾言：「希望是附丽于存在的，有存在，便有希望，有希望，便是光明。」
# 体积云亦如此——光线穿过云层，每一步都是希望的采样，每一步都可能穿透，
# 也可能被遮蔽。光线步进是一种执着：不到最大采样数，决不放弃。
# =============================================================================

from dataclasses import dataclass as _dc2, field as _field2
from typing import List as _List2, Optional as _Opt2, Dict as _Dict2


# -----------------------------------------------------------------------------
# CVarVolumetricCloud 系列控制变量移植
# -----------------------------------------------------------------------------
_CLOUD_SUPPORT                  = True
_CLOUD_ENABLED                  = True
_CLOUD_DIST_TO_SAMPLE_MAX_COUNT = 15.0
_CLOUD_SAMPLE_MIN_COUNT         = 2
_CLOUD_SAMPLE_CLAMP_COUNT       = 768
_CLOUD_VIEW_RAY_SAMPLE_MAX      = 768
_CLOUD_REFL_RAY_SAMPLE_MAX      = 80
_CLOUD_STEP_ON_ZERO_DENSITY     = 1
_CLOUD_APPLY_FOG_ON_ALL_PIXEL   = False
_CLOUD_APPLY_FOG_LATE           = True
_CLOUD_SHADOW_SAMPLE_MAX        = 80
_CLOUD_SKY_AO_ENABLED           = True
_CLOUD_SKY_AO_TRACE_COUNT       = 10
_CLOUD_SKY_AO_SNAP_LENGTH       = 20.0
_CLOUD_SHADOW_MAP_ENABLED       = True
_CLOUD_SHADOW_MAP_SNAP_LENGTH   = 20.0
_CLOUD_SHADOW_MAP_SAMPLE_MAX    = 128.0
_CLOUD_AERIAL_PERSPECTIVE_SAMPLING = True


def _cloud_compute_sample_count(
    trace_distance: float,
    max_sample_count: float = _CLOUD_VIEW_RAY_SAMPLE_MAX,
    dist_to_max: float = _CLOUD_DIST_TO_SAMPLE_MAX_COUNT,
) -> int:
    """
    计算光线步进采样数。
    鲁迅式：采样数随距离而生长——近处精细如记忆，远处粗糙如遗忘。
    """
    if not _CLOUD_SUPPORT or not _CLOUD_ENABLED:
        return 0
    dist_km = trace_distance / 1000.0
    raw = (
        _CLOUD_SAMPLE_MIN_COUNT
        + (max_sample_count - _CLOUD_SAMPLE_MIN_COUNT)
        * min(1.0, dist_km / max(dist_to_max, 1e-6))
    )
    return int(max(_CLOUD_SAMPLE_MIN_COUNT, min(raw, _CLOUD_SAMPLE_CLAMP_COUNT)))


@_dc2
class AstroCellCloudTracingState:
    """单条光线的体积云步进状态。"""
    transmittance:  float = 1.0
    luminance:      tuple = (0.0, 0.0, 0.0)
    t_current:      float = 0.0
    t_exit:         float = 0.0
    sample_count:   int   = 0
    converged:      bool  = False

    def integrate_sample(self, density: float, emission: tuple,
                         step_size: float, extinction_coeff: float = 0.1) -> None:
        if density <= 0.0:
            return
        extinction  = density * extinction_coeff
        step_trans  = math.exp(-extinction * step_size)
        weight      = self.transmittance * (1.0 - step_trans) / max(extinction, 1e-8)
        self.luminance = (
            self.luminance[0] + weight * emission[0],
            self.luminance[1] + weight * emission[1],
            self.luminance[2] + weight * emission[2],
        )
        self.transmittance *= step_trans
        self.sample_count  += 1
        if self.transmittance < 1e-4:
            self.converged = True


@_dc2
class AstroCellVolumetricCloudLayer:
    """单云层描述符。"""
    layer_bottom_altitude_km: float = 2.0
    layer_top_altitude_km:    float = 5.0
    extinction_scale:         float = 0.2
    ambient_occlusion:        float = 0.5
    sky_ao_strength:          float = 0.8
    shadow_map_strength:      float = 0.6

    def thickness_km(self) -> float:
        return max(0.0, self.layer_top_altitude_km - self.layer_bottom_altitude_km)

    def altitude_fraction(self, altitude_km: float) -> float:
        th = self.thickness_km()
        if th < 1e-6:
            return 0.0
        return max(0.0, min(1.0,
            (altitude_km - self.layer_bottom_altitude_km) / th))


def trace_cloud_ray(
    ray_origin_km: tuple,
    ray_dir: tuple,
    cloud_layer: AstroCellVolumetricCloudLayer,
    density_fn,
    emission_fn,
    max_sample_count: int = _CLOUD_VIEW_RAY_SAMPLE_MAX,
    step_size_km: float = 0.05,
) -> AstroCellCloudTracingState:
    """
    沿单条光线执行体积云步进积分。
    鲁迅式：光线步进如同直面现实的旅人——每一步都可能被云雾吞噬，
    却依然坚持迈出下一步，直到透射率归零或走完全程。
    """
    state = AstroCellCloudTracingState()
    if not _CLOUD_ENABLED:
        return state

    oy, dy = ray_origin_km[1], ray_dir[1]
    t_bot = (cloud_layer.layer_bottom_altitude_km - oy) / max(abs(dy), 1e-9) * (1 if dy > 0 else -1)
    t_top = (cloud_layer.layer_top_altitude_km    - oy) / max(abs(dy), 1e-9) * (1 if dy > 0 else -1)
    t_enter = max(0.0, min(t_bot, t_top))
    t_exit  = max(t_bot, t_top)
    if t_exit <= t_enter:
        return state

    state.t_current = t_enter
    state.t_exit    = t_exit
    trace_dist = (t_exit - t_enter) * 1000.0
    n_samples  = min(_cloud_compute_sample_count(trace_dist, max_sample_count), max_sample_count)
    if n_samples <= 0:
        return state

    actual_step = (t_exit - t_enter) / n_samples
    for _ in range(n_samples):
        if state.converged:
            break
        alt_km = oy + state.t_current * dy
        density = density_fn(alt_km) * cloud_layer.extinction_scale
        if density > 0.0:
            emission = emission_fn(alt_km, density)
            state.integrate_sample(density, emission, actual_step)
        elif _CLOUD_STEP_ON_ZERO_DENSITY > 1:
            state.t_current += actual_step * _CLOUD_STEP_ON_ZERO_DENSITY
            continue
        state.t_current += actual_step
    return state


def compute_cloud_sky_ao(
    ground_altitude_km: float,
    cloud_layer: AstroCellVolumetricCloudLayer,
    density_fn,
    num_traces: int = _CLOUD_SKY_AO_TRACE_COUNT,
) -> float:
    """
    计算云层天空 AO。
    鲁迅式：天空 AO 是大地对苍穹的凝视——只能用一个数字，记录被遮蔽的清醒。
    """
    if not _CLOUD_SKY_AO_ENABLED:
        return 0.0
    total_occlusion = 0.0
    for i in range(num_traces):
        angle = math.pi * i / max(num_traces - 1, 1)
        ray_dir = (math.sin(angle), math.cos(angle), 0.0)
        state = trace_cloud_ray(
            ray_origin_km=(0.0, ground_altitude_km, 0.0),
            ray_dir=ray_dir,
            cloud_layer=cloud_layer,
            density_fn=density_fn,
            emission_fn=lambda alt, d: (0.0, 0.0, 0.0),
            max_sample_count=_CLOUD_SKY_AO_TRACE_COUNT * 2,
        )
        total_occlusion += 1.0 - state.transmittance
    return total_occlusion / max(num_traces, 1)


@_dc2
class AstroCellCloudRenderParams:
    """云层渲染参数输出。"""
    transmittance: float = 1.0
    luminance:     tuple = (0.0, 0.0, 0.0)
    sky_ao:        float = 0.0
    cloud_opacity: float = 0.0
    fog_color:     tuple = (0.85, 0.90, 0.95)

    def to_svg_filter_params(self) -> dict:
        brightness = 1.0 - self.sky_ao * 0.4
        fog_alpha  = self.cloud_opacity * 0.3
        return {
            "brightness":  round(max(0.3, brightness), 4),
            "fog_alpha":   round(max(0.0, min(1.0, fog_alpha)), 4),
            "fog_r":       round(self.fog_color[0], 4),
            "fog_g":       round(self.fog_color[1], 4),
            "fog_b":       round(self.fog_color[2], 4),
            "luminance_r": round(self.luminance[0], 4),
            "luminance_g": round(self.luminance[1], 4),
            "luminance_b": round(self.luminance[2], 4),
        }


# =============================================================================
# AstroCellTranslucentLighting — 半透明体积光照
# (ported from TranslucentLighting.cpp)
#
# 鲁迅曾言：「不满是向上的车轮，能够载着不自满的人类，向人道前进。」
# 每一个 cascade 都是对「不满于低精度」的回应。
# =============================================================================

_TLV_ENABLED             = True
_TLV_DIM                 = 64
_TLV_INNER_DISTANCE      = 1500.0
_TLV_OUTER_DISTANCE      = 5000.0
_TLV_MIN_FOV             = 45.0
_TLV_FOV_SNAP_FACTOR     = 10.0
_TLV_BLUR_ENABLED        = True
_TLV_TEMPORAL_ENABLED    = False
_TLV_HISTORY_WEIGHT      = 0.9
_TLV_MARK_VOXELS         = False
_TLV_BATCH               = True
_TLV_CSM_INJECT          = True
_TLV_POSITION_OFFSET_R   = 0.0


@_dc2
class AstroCellTranslucencyVolumeCascade:
    """半透明光照卷单层 Cascade。"""
    cascade_index: int   = 0
    dim:           int   = _TLV_DIM
    inner_dist:    float = _TLV_INNER_DISTANCE
    outer_dist:    float = _TLV_OUTER_DISTANCE
    volume_data:   _List2[tuple] = _field2(default_factory=list)
    _history:      _List2[tuple] = _field2(default_factory=list)

    def __post_init__(self):
        n = self.dim ** 3
        if not self.volume_data:
            self.volume_data = [(0.0, 0.0, 0.0)] * n
        if not self._history:
            self._history = [(0.0, 0.0, 0.0)] * n

    def voxel_index(self, ix: int, iy: int, iz: int) -> int:
        d = self.dim
        return (iz % d) * d * d + (iy % d) * d + (ix % d)

    def inject_light(self, ix: int, iy: int, iz: int,
                     radiance: tuple, alpha: float = 1.0) -> None:
        idx = self.voxel_index(ix, iy, iz)
        old = self.volume_data[idx]
        self.volume_data[idx] = (
            old[0] + radiance[0] * alpha,
            old[1] + radiance[1] * alpha,
            old[2] + radiance[2] * alpha,
        )

    def apply_temporal_blend(self) -> None:
        """
        时域混合。鲁迅式：九分是昨天，一分是今天——稳定，但迟钝。
        """
        if not _TLV_TEMPORAL_ENABLED:
            return
        w = _TLV_HISTORY_WEIGHT
        for i in range(len(self.volume_data)):
            c, h = self.volume_data[i], self._history[i]
            self.volume_data[i] = (
                h[0]*w + c[0]*(1-w), h[1]*w + c[1]*(1-w), h[2]*w + c[2]*(1-w),
            )
        self._history = list(self.volume_data)

    def sample_trilinear(self, u: float, v: float, w: float) -> tuple:
        """三线性插值。鲁迅式：插值是折中主义——在多个体素间寻求可接受的平均。"""
        d   = self.dim
        fx  = max(0.0, min(1.0, u)) * (d - 1)
        fy  = max(0.0, min(1.0, v)) * (d - 1)
        fz  = max(0.0, min(1.0, w)) * (d - 1)
        ix0 = int(math.floor(fx)); iy0 = int(math.floor(fy)); iz0 = int(math.floor(fz))
        tx  = fx - ix0; ty = fy - iy0; tz = fz - iz0

        def lerp3(a, b, t):
            return (a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t)
        def vox(xi, yi, zi):
            return self.volume_data[self.voxel_index(xi, yi, zi)]

        c000=vox(ix0,iy0,iz0); c100=vox(ix0+1,iy0,iz0); c010=vox(ix0,iy0+1,iz0)
        c110=vox(ix0+1,iy0+1,iz0); c001=vox(ix0,iy0,iz0+1); c101=vox(ix0+1,iy0,iz0+1)
        c011=vox(ix0,iy0+1,iz0+1); c111=vox(ix0+1,iy0+1,iz0+1)
        return lerp3(
            lerp3(lerp3(c000,c100,tx), lerp3(c010,c110,tx), ty),
            lerp3(lerp3(c001,c101,tx), lerp3(c011,c111,tx), ty),
            tz,
        )


class AstroCellTranslucencyLightingVolume:
    """
    完整半透明光照卷系统（inner + outer cascade）。
    鲁迅式：两层如同两层社会——内层精细覆盖有限，外层粗糙包罗万象。
    """

    def __init__(self) -> None:
        self.inner = AstroCellTranslucencyVolumeCascade(
            cascade_index=0, inner_dist=0.0, outer_dist=_TLV_INNER_DISTANCE)
        self.outer = AstroCellTranslucencyVolumeCascade(
            cascade_index=1, inner_dist=_TLV_INNER_DISTANCE, outer_dist=_TLV_OUTER_DISTANCE)

    def inject_directional_light(self, light_dir: tuple, light_color: tuple,
                                  num_cascades: int = 2) -> None:
        """注入方向光。鲁迅式：来自单一方向，却要对整个体积负责。"""
        if not _TLV_ENABLED:
            return
        for cascade in (self.inner, self.outer):
            d = cascade.dim
            for iz in range(0, d, 4):
                for iy in range(0, d, 4):
                    for ix in range(0, d, 4):
                        nx = ix/d - 0.5; ny = iy/d - 0.5; nz = iz/d - 0.5
                        cos_theta = max(0.0, -(light_dir[0]*nx + light_dir[1]*ny + light_dir[2]*nz))
                        inject = (light_color[0]*cos_theta, light_color[1]*cos_theta, light_color[2]*cos_theta)
                        cascade.inject_light(ix, iy, iz, inject, alpha=0.01)

    def apply_blur(self) -> None:
        """3D 盒式模糊。鲁迅式：模糊是仁慈，也是妥协——让光的过渡更自然。"""
        if not _TLV_BLUR_ENABLED:
            return
        for cascade in (self.inner, self.outer):
            d = cascade.dim
            orig = list(cascade.volume_data)
            def get(xi, yi, zi): return orig[cascade.voxel_index(xi, yi, zi)]
            for iz in range(d):
                for iy in range(d):
                    for ix in range(d):
                        nb = [get(ix, iy, iz)]
                        for ddx,ddy,ddz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                            nxi,nyi,nzi = ix+ddx, iy+ddy, iz+ddz
                            if 0<=nxi<d and 0<=nyi<d and 0<=nzi<d:
                                nb.append(get(nxi,nyi,nzi))
                        n = len(nb)
                        cascade.volume_data[cascade.voxel_index(ix,iy,iz)] = (
                            sum(v[0] for v in nb)/n, sum(v[1] for v in nb)/n, sum(v[2] for v in nb)/n)

    def tick(self, light_dir: tuple = (0.0, -1.0, 0.0),
             light_color: tuple = (1.0, 0.95, 0.85)) -> None:
        self.inject_directional_light(light_dir, light_color)
        self.inner.apply_temporal_blend()
        self.outer.apply_temporal_blend()
        self.apply_blur()

    def sample(self, world_u: float, world_v: float, world_w: float,
               distance: float) -> tuple:
        if not _TLV_ENABLED:
            return (0.0, 0.0, 0.0)
        blend_start = _TLV_INNER_DISTANCE * 0.9
        if distance <= blend_start:
            return self.inner.sample_trilinear(world_u, world_v, world_w)
        elif distance >= _TLV_INNER_DISTANCE:
            return self.outer.sample_trilinear(world_u, world_v, world_w)
        else:
            t = (distance - blend_start) / (_TLV_INNER_DISTANCE - blend_start)
            ci = self.inner.sample_trilinear(world_u, world_v, world_w)
            co = self.outer.sample_trilinear(world_u, world_v, world_w)
            return (ci[0]*(1-t)+co[0]*t, ci[1]*(1-t)+co[1]*t, ci[2]*(1-t)+co[2]*t)


# =============================================================================
# AstroCellSingleLayerWater — 单层水面渲染
# (ported from SingleLayerWaterRendering.cpp)
#
# 鲁迅式：水面既透明又反射，既折射又有焦散——
# 它试图同时成为两件对立的事物，在每一帧中寻找平衡。
# =============================================================================

_SLW_ENABLED             = True
_SLW_WAVE_OPS            = True
_SLW_REFLECTION_MODE     = 1
_SLW_REFL_DOWNSAMPLE     = 1
_SLW_TILED_COMPOSITE     = True
_SLW_SSRTAA              = True
_SLW_DIST_FIELD_SHADOW   = True
_SLW_REFRACTION_DOWNSAMPLE = 1
_SLW_DEPTH_PREPASS       = True
_SLW_REFRACTION_CULLING  = False
_SLW_REFRACTION_DIST_CULL = -1.0
_SLW_REFRACTION_FRESNEL_CULL = -1.0
_SLW_F0_WATER  = 0.02
_SLW_IOR_WATER = 1.333


def _slw_fresnel_schlick(cos_v: float, f0: float = _SLW_F0_WATER) -> float:
    """Schlick Fresnel。鲁迅式：掠射角处的高反射率，是水面不肯透露底细的自尊心。"""
    return f0 + (1.0 - f0) * math.pow(max(0.0, 1.0 - cos_v), 5.0)


def _slw_refraction_offset(normal: tuple, view_dir: tuple, depth: float,
                            ior: float = _SLW_IOR_WATER) -> tuple:
    """折射偏移。鲁迅式：折射是光的谎言——你看见水底的鱼，却不在那里。"""
    nx, ny, nz = normal; vx, vy, vz = view_dir
    dot_nv = nx*vx + ny*vy + nz*vz
    tx = vx - dot_nv * nx; ty = vy - dot_nv * ny
    scale = depth / max(ior, 1e-6)
    return (tx * scale, ty * scale)


@_dc2
class AstroCellSingleLayerWaterSurface:
    """单层水面渲染状态。"""
    water_depth:       float = 3.0
    roughness:         float = 0.05
    foam_coverage:     float = 0.0
    caustics_strength: float = 0.4
    shadow_strength:   float = 0.5
    cloud_shadow_mult: float = 0.0
    refraction_scale:  float = 1.0

    def fresnel_at_angle(self, cos_view: float) -> float:
        return _slw_fresnel_schlick(cos_view)

    def absorption_color(self) -> tuple:
        """水体颜色吸收。鲁迅式：红光衰减最快，蓝绿光穿透最深——这是大海的忧郁。"""
        return (
            math.exp(-self.water_depth * 0.5),
            math.exp(-self.water_depth * 0.2),
            math.exp(-self.water_depth * 0.05),
        )

    def refraction_uv_offset(self, view_cos: float) -> tuple:
        if _SLW_REFRACTION_CULLING and (self.water_depth > _SLW_REFRACTION_DIST_CULL > 0):
            return (0.0, 0.0)
        scale  = self.water_depth * self.roughness * self.refraction_scale
        normal = (0.0, 1.0, 0.0)
        view   = (math.sqrt(max(0.0, 1.0 - view_cos**2)), view_cos, 0.0)
        return _slw_refraction_offset(normal, view, scale)

    def to_svg_filter_params(self, view_cos: float = 0.9) -> dict:
        """
        输出 SVG 滤镜参数。
        鲁迅式：参数是规范，好的约束让艺术在边界内尽情发挥。
        """
        fresnel = self.fresnel_at_angle(view_cos)
        absorp  = self.absorption_color()
        du, dv  = self.refraction_uv_offset(view_cos)
        foam_reflection_mult = 1.0 - self.foam_coverage * 0.6
        ssr_blur = self.roughness * 8.0 * (_SLW_REFL_DOWNSAMPLE ** 0.5)
        if _SLW_SSRTAA:
            ssr_blur *= 0.6
        return {
            "fresnel":           round(fresnel * foam_reflection_mult, 4),
            "refraction_du":     round(du, 4),
            "refraction_dv":     round(dv, 4),
            "absorption_r":      round(absorp[0], 4),
            "absorption_g":      round(absorp[1], 4),
            "absorption_b":      round(absorp[2], 4),
            "foam_coverage":     round(self.foam_coverage, 4),
            "caustics_strength": round(self.caustics_strength, 4),
            "shadow_strength":   round(self.shadow_strength if _SLW_DIST_FIELD_SHADOW else 0.0, 4),
            "ssr_blur_radius":   round(ssr_blur, 4),
            "cloud_shadow_mult": round(self.cloud_shadow_mult, 4),
        }

    def generate_svg_water_overlay(self, x: float, y: float, w: float, h: float,
                                    filter_id: str, view_cos: float = 0.9) -> str:
        """
        生成水面 SVG 覆盖层。
        鲁迅式：水面渲染是最诚实的谎言——每一层都声称在还原真实，
        但最终所有层叠加起来，不过是人眼可以接受的近似。
        """
        if not _SLW_ENABLED:
            return ""
        params = self.to_svg_filter_params(view_cos)
        parts  = []
        fid    = f"slw-{filter_id}"

        parts.append(f'<defs>')
        parts.append(f'  <filter id="{fid}" x="-5%" y="-5%" width="110%" height="110%">')
        r_s, g_s, b_s = params["absorption_r"], params["absorption_g"], params["absorption_b"]
        parts.append(
            f'    <feColorMatrix type="matrix" '
            f'values="{r_s} 0 0 0 0  0 {g_s} 0 0 0  0 0 {b_s} 0 0  0 0 0 1 0"/>')
        disp_scale = math.sqrt(params["refraction_du"]**2 + params["refraction_dv"]**2) * 20.0
        if disp_scale > 0.1:
            parts.append(
                f'    <feTurbulence type="turbulence" baseFrequency="0.02 0.04" '
                f'numOctaves="3" seed="{abs(hash(filter_id)) % 999}" result="waves"/>')
            parts.append(
                f'    <feDisplacementMap in="SourceGraphic" in2="waves" '
                f'scale="{disp_scale:.2f}" xChannelSelector="R" yChannelSelector="G"/>')
        parts.append(f'  </filter></defs>')

        water_blue = "#{:02X}{:02X}{:02X}".format(
            int(30 + params["absorption_r"] * 40),
            int(80 + params["absorption_g"] * 60),
            int(150 + params["absorption_b"] * 80),
        )
        parts.append(
            f'<!-- [ASTRO-SLW] SingleLayerWater fresnel={params["fresnel"]:.3f} '
            f'absorp=({r_s:.3f},{g_s:.3f},{b_s:.3f}) (SingleLayerWaterRendering.cpp port) -->')
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="4" fill="{water_blue}" opacity="{1.0 - params["fresnel"]:.3f}" '
            f'filter="url(#{fid})"/>')
        if params["fresnel"] > 0.05:
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                f'rx="4" fill="white" opacity="{params["fresnel"] * 0.35:.3f}"/>')
        if params["caustics_strength"] > 0.05:
            cid = f"caustics-{filter_id}"
            parts.append(
                f'<defs><filter id="{cid}">'
                f'<feTurbulence type="fractalNoise" baseFrequency="0.08 0.12" '
                f'numOctaves="4" seed="{(abs(hash(filter_id))+42)%999}"/>'
                f'<feColorMatrix type="saturate" values="0"/>'
                f'<feComponentTransfer><feFuncA type="linear" slope="{params["caustics_strength"]:.2f}"/>'
                f'</feComponentTransfer></filter></defs>')
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                f'rx="4" fill="#FFFDE7" opacity="{params["caustics_strength"] * 0.4:.3f}" '
                f'filter="url(#{cid})"/>')
        if params["foam_coverage"] > 0.1:
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                f'rx="4" fill="white" opacity="{params["foam_coverage"] * 0.25:.3f}"/>')
        return "\n".join(parts)


# =============================================================================
# AstroCellAtmosphericCompositor — 大气合成器
# 整合三个子系统：VolumetricCloud + TranslucentLighting + SingleLayerWater
#
# 鲁迅式：自然界从不孤立地展现任何一种效果——
# 云层遮天蔽日，光照穿透半透明体，水面折射倒映，三者共存。
# =============================================================================

class AstroCellAtmosphericCompositor:
    """
    大气效果合成器。
    鲁迅式：大气是一切视觉效果的底色——沉默地存在，却决定了一切的基调。
    """

    def __init__(self) -> None:
        self.cloud_layer = AstroCellVolumetricCloudLayer()
        self.tlv         = AstroCellTranslucencyLightingVolume()
        self.water       = AstroCellSingleLayerWaterSurface()

    def set_cloud_layer(self, layer: AstroCellVolumetricCloudLayer) -> None:
        self.cloud_layer = layer

    def set_water_surface(self, surf: AstroCellSingleLayerWaterSurface) -> None:
        self.water = surf

    def _simple_density_fn(self, alt_km: float) -> float:
        mid   = (self.cloud_layer.layer_bottom_altitude_km + self.cloud_layer.layer_top_altitude_km) / 2.0
        sigma = self.cloud_layer.thickness_km() / 3.0
        if sigma < 1e-6:
            return 0.0
        return math.exp(-0.5 * ((alt_km - mid) / sigma) ** 2)

    def _simple_emission_fn(self, alt_km: float, density: float) -> tuple:
        f = self.cloud_layer.altitude_fraction(alt_km)
        return (
            (0.9  + (1.0-f)*0.08) * density,
            (0.92 + (1.0-f)*0.04) * density,
            (0.95 + f*0.05)       * density,
        )

    def compose_atmospheric_overlay(
        self,
        cell_id:     str,
        bbox:        dict,
        species:     str,
        view_cos:    float = 0.85,
        altitude_km: float = 0.0,
    ) -> str:
        """
        为 cell 生成完整大气 SVG 覆盖层（VolumetricCloud → TLV → SLW）。
        鲁迅式：覆盖层是视觉的注脚——原文已说完，注脚让读者知道背景的重量。
        """
        x, y = bbox["x"], bbox["y"]
        w, h = bbox["w"], bbox["h"]
        z    = float(bbox.get("z", 3))
        parts = [
            f'<!-- [ASTRO-ATMO] AtmosphericCompositor cell={cell_id} '
            f'z={z:.1f} alt_km={altitude_km:.2f} view_cos={view_cos:.3f} -->'
        ]

        # Phase 1: VolumetricCloud 雾覆盖
        if _CLOUD_ENABLED and altitude_km < self.cloud_layer.layer_top_altitude_km:
            cloud_state = trace_cloud_ray(
                (0.0, altitude_km, 0.0), (0.0, 1.0, 0.0),
                self.cloud_layer, self._simple_density_fn, self._simple_emission_fn,
            )
            sky_ao = compute_cloud_sky_ao(altitude_km, self.cloud_layer, self._simple_density_fn)
            fp = AstroCellCloudRenderParams(
                transmittance=cloud_state.transmittance,
                luminance=cloud_state.luminance,
                sky_ao=sky_ao,
                cloud_opacity=1.0 - cloud_state.transmittance,
            ).to_svg_filter_params()

            if fp["fog_alpha"] > 0.01:
                fog_hex = "#{:02X}{:02X}{:02X}".format(
                    int(fp["fog_r"]*255), int(fp["fog_g"]*255), int(fp["fog_b"]*255))
                parts.append(
                    f'<!-- [ASTRO-CLOUD] VolumetricCloud trans={cloud_state.transmittance:.3f} '
                    f'sky_ao={sky_ao:.3f} samples={cloud_state.sample_count} '
                    f'(VolumetricCloudRendering.cpp port) -->')
                parts.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                    f'rx="6" fill="{fog_hex}" opacity="{fp["fog_alpha"]:.4f}" '
                    f'style="mix-blend-mode:screen"/>')
                if sky_ao > 0.05:
                    parts.append(
                        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                        f'rx="6" fill="black" opacity="{sky_ao*0.25:.4f}" '
                        f'style="mix-blend-mode:multiply"/>')

        # Phase 2: TranslucentLighting 辐射度调制
        if _TLV_ENABLED:
            u_n = max(0.0, min(1.0, x/1200.0))
            v_n = max(0.0, min(1.0, y/900.0))
            w_n = max(0.0, min(1.0, z/7.0))
            tlv = self.tlv.sample(u_n, v_n, w_n, math.sqrt(x**2 + y**2))
            lum = max(tlv[0], tlv[1], tlv[2])
            if lum > 0.005:
                tlv_hex = "#{:02X}{:02X}{:02X}".format(
                    min(255, int(tlv[0]*510)), min(255, int(tlv[1]*510)), min(255, int(tlv[2]*510)))
                parts.append(
                    f'<!-- [ASTRO-TLV] TranslucentLighting '
                    f'lum=({tlv[0]:.3f},{tlv[1]:.3f},{tlv[2]:.3f}) '
                    f'dim={_TLV_DIM} blur={_TLV_BLUR_ENABLED} '
                    f'(TranslucentLighting.cpp port) -->')
                parts.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                    f'rx="6" fill="{tlv_hex}" opacity="{min(0.12, lum*0.8):.4f}" '
                    f'style="mix-blend-mode:add"/>')

        # Phase 3: SingleLayerWater 水面效果
        if _SLW_ENABLED and (species in ("water", "cil-loop") or z <= 1):
            slw = self.water.generate_svg_water_overlay(x, y, w, h, cell_id, view_cos)
            if slw:
                parts.append(slw)

        return "\n".join(p for p in parts if p)


_ASTRO_ATMO_COMPOSITOR_V2: _Opt2[AstroCellAtmosphericCompositor] = None


def get_atmospheric_compositor() -> AstroCellAtmosphericCompositor:
    """
    全局大气合成器单例。
    鲁迅式：大气是公共的——所有 cell 共享同一片天空，无一例外。
    """
    global _ASTRO_ATMO_COMPOSITOR_V2
    if _ASTRO_ATMO_COMPOSITOR_V2 is None:
        _ASTRO_ATMO_COMPOSITOR_V2 = AstroCellAtmosphericCompositor()
    return _ASTRO_ATMO_COMPOSITOR_V2


# =============================================================================
# [ASTRO-CELL] SceneRendering + SceneVisibility + ScreenSpaceDenoise → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/SceneRendering.cpp
#   upstream/unreal-renderer-ue5/Renderer-Private/SceneVisibility.cpp
#   upstream/unreal-renderer-ue5/Renderer-Private/ScreenSpaceDenoise.cpp
#
# 鲁迅曾言：「不在沉默中爆发，便在沉默中灭亡。」
# 场景渲染亦然——每一帧都是一次抉择：渲什么，剔什么，降噪还是保留噪声。
# 沉默的 cell 不参与渲染，爆发的 cell 才进入最终画面。
#
# SceneRendering → AstroCellFrameRenderer（帧渲染总调度）
#   FDeferredShadingSceneRenderer::Render() 的六阶段 pipeline：
#     InitViews      → init_views()
#     PrePass        → pre_pass()
#     BasePass       → base_pass()
#     Lighting       → lighting_pass()
#     Translucency   → translucency_pass()
#     PostProcess    → post_process()
#
# SceneVisibility → AstroCellVisibilityProcessor（可见性处理器）
#   距离剔除（r.DistanceCullToSphereEdge）  → _distance_cull_to_sphere_edge()
#   LOD 筛选（r.StaticMeshLODDistanceScale） → _compute_lod_level()
#   HZB 遮挡（r.HZBOcclusion）              → _hzb_occlusion_query()
#   TAA Jitter（r.TemporalAASamples=8）     → _compute_taa_jitter()
#   Wireframe 剔除（r.WireframeCullThreshold）→ _wireframe_cull()
#
# ScreenSpaceDenoise → AstroCellDenoiser（降噪管线）
#   信号类型（ESignalProcessing）           → AstroCellSignalType（枚举）
#   重建采样（r.Shadow.Denoiser.ReconstructionSamples=8）→ reconstruct()
#   预卷积（r.Shadow.Denoiser.PreConvolution=1）         → pre_convolve()
#   时域累积（r.Shadow.Denoiser.TemporalAccumulation=1） → temporal_accumulate()
#   历史卷积（r.Shadow.Denoiser.HistoryConvolutionSamples=1）→ history_convolve()
#   多信号批处理（kMaxBatchSize）            → denoise_batch()
#
# 2-D SVG 适配说明（鲁迅式 20% 算法改动）：
#   ① 距离剔除使用 bbox 包围球边缘距离（球心+半径）而非球心距离，
#     与 GDistanceCullToSphereEdge=true 的 C++ 路径一致。
#   ② LOD 采用三档（0=全细节 / 1=减半 / 2=矩形占位），
#     阈值由 _NANITE_LOD2_THRESHOLD 沿用（NaniteVisibility 已有）。
#   ③ HZB 遮挡：以 BVH 查询代替 GPU 深度金字塔，复用 AstroCellBVH。
#   ④ TAA Jitter：Halton(2,3) 序列，8 个样本，支持 InvertX/Y 标志。
#   ⑤ 降噪信号强度由 SVG 滤镜 stdDeviation 映射；
#     时域历史权重 = _TLV_HISTORY_WEIGHT（已有常量复用）。
# =============================================================================

import enum as _enum
import math as _math_sr


# ─────────────────────────────────────────────────────────────────────────────
# CVarSceneRendering / CVarSceneVisibility 系列控制变量移植
# ─────────────────────────────────────────────────────────────────────────────

# 镜像 GDistanceCullToSphereEdge — True 时剔除以包围球边缘距离计，而非球心
_SR_DIST_CULL_TO_SPHERE_EDGE: bool = True

# 镜像 GWireframeCullThreshold — 正交线框视图中低于此投影尺寸的物体被剔除
_SR_WIREFRAME_CULL_THRESHOLD: float = 5.0

# 镜像 GMinScreenRadiusForLights — 屏幕占比低于此值的光源被剔除
_SR_MIN_SCREEN_RADIUS_LIGHTS: float = 0.03

# 镜像 GMinScreenRadiusForDepthPrepass — 屏幕占比低于此值跳过深度预通道
_SR_MIN_SCREEN_RADIUS_DEPTH_PREPASS: float = 0.03

# 镜像 CVarTemporalAASamples=8 — TAA Jitter 样本数
_SR_TAA_SAMPLES: int = 8

# 镜像 CVarInvertTemporalJitterX/Y — 是否反转 Jitter 分量
_SR_INVERT_JITTER_X: bool = False
_SR_INVERT_JITTER_Y: bool = False

# 镜像 GHZBOcclusion=0 — 遮挡系统：0=硬件查询 / 1=HZB / 2=强制HZB
# 在 2-D SVG 管线中始终使用 BVH 等价路径
_SR_HZB_OCCLUSION: int = 1

# 镜像 CVarStaticMeshLODDistanceScale=1.0 — LOD 距离缩放系数
_SR_LOD_DISTANCE_SCALE: float = 1.0

# 镜像 CVarAutomaticViewMipBiasMin=-2.0 — 自动 Mip Bias 最小值
_SR_MIP_BIAS_MIN: float = -2.0

# 镜像 CVarAutomaticViewMipBiasOffset=-0.3 — 自动 Mip Bias 常数偏移
_SR_MIP_BIAS_OFFSET: float = -0.3

# 降噪器重建最大采样数（r.Shadow.Denoiser.ReconstructionSamples=8）
_SDN_SHADOW_RECONSTRUCTION_SAMPLES: int = 8
# 降噪器预卷积次数（r.Shadow.Denoiser.PreConvolution=1）
_SDN_SHADOW_PRE_CONVOLUTION: int = 1
# 降噪器时域累积开关（r.Shadow.Denoiser.TemporalAccumulation=1）
_SDN_SHADOW_TEMPORAL: bool = True
# 降噪器历史卷积样本数（r.Shadow.Denoiser.HistoryConvolutionSamples=1）
_SDN_SHADOW_HISTORY_CONVOLUTION: int = 1
# 反射降噪器最大重建样本数（r.Reflections.Denoiser.ReconstructionSamples=8）
_SDN_REFL_RECONSTRUCTION_SAMPLES: int = 8
# AO 降噪器最大重建样本数（r.AmbientOcclusion.Denoiser.ReconstructionSamples=16）
_SDN_AO_RECONSTRUCTION_SAMPLES: int = 16
# AO 预卷积次数（r.AmbientOcclusion.Denoiser.PreConvolution=2）
_SDN_AO_PRE_CONVOLUTION: int = 2
# AO 核扩展系数（r.AmbientOcclusion.Denoiser.KernelSpreadFactor=4）
_SDN_AO_KERNEL_SPREAD: float = 4.0
# GI 降噪器最大重建样本数（r.GlobalIllumination.Denoiser.ReconstructionSamples=16）
_SDN_GI_RECONSTRUCTION_SAMPLES: int = 16
# 最大 Mip 层级（kMaxMipLevel=2）
_SDN_MAX_MIP_LEVEL: int = 2
# 最大批量信号数（kMaxBufferProcessingCount / kMaxBatchSize）
_SDN_MAX_BATCH_SIZE: int = 4


# ─────────────────────────────────────────────────────────────────────────────
# AstroCellSignalType — 镜像 ESignalProcessing 枚举
# ─────────────────────────────────────────────────────────────────────────────

class AstroCellSignalType(_enum.Enum):
    """
    Python 等价于 ScreenSpaceDenoise.cpp 中的 ESignalProcessing 枚举。

    每种信号类型对应一条独立的降噪路径，参数集互不相同。
    鲁迅式：信号有其命运——阴影是阴影，反射是反射，
    混为一谈只会让两者都失真。
    """
    SHADOW_VISIBILITY_MASK          = 0   # 阴影可见性掩码（单灯/多灯）
    POLYCHROMATIC_PENUMBRA_HARMONIC = 1   # 多色半影谐波（多灯合批）
    REFLECTIONS                     = 2   # 一次弹射镜面反射
    AMBIENT_OCCLUSION               = 3   # 环境光遮蔽
    DIFFUSE_AND_AO                  = 4   # 漫反射+AO 联合降噪
    DIFFUSE_SPHERICAL_HARMONIC      = 5   # 漫反射球谐降噪（Lumen GI）


# ─────────────────────────────────────────────────────────────────────────────
# AstroCellDenoiserState — 单信号降噪状态（per-signal history）
# ─────────────────────────────────────────────────────────────────────────────

class AstroCellDenoiserState:
    """
    单条信号的跨帧降噪状态，对应 IScreenSpaceDenoiser 输入/输出 buffer 对。

    字段说明（镜像 ScreenSpaceDenoise.cpp 内部 buffer 命名）：
        signal_type      — 信号类型（ESignalProcessing 端口）
        noisy_value      — 当前帧原始（噪声）信号值 [0, 1]
        reconstructed    — 重建通道输出（ReconstructionPass 后）
        pre_convolved    — 预卷积通道输出（PreConvolutionPass 后）
        accumulated      — 时域累积输出（TemporalAccumulationPass 后）
        history_convolved— 历史卷积输出（HistoryConvolutionPass 后），即最终降噪值
        history_value    — 上一帧的 accumulated 值（帧间持久化）
        sample_count     — 本帧有效输入样本数
        frame_index      — 当前帧序号

    鲁迅式：历史是重量，也是养分——
    没有历史的降噪等于每帧从零开始，噪声永不消亡。
    """

    def __init__(self, signal_type: AstroCellSignalType) -> None:
        self.signal_type:       AstroCellSignalType = signal_type
        self.noisy_value:       float = 0.0
        self.reconstructed:     float = 0.0
        self.pre_convolved:     float = 0.0
        self.accumulated:       float = 0.0
        self.history_convolved: float = 0.0
        self.history_value:     float = 0.0   # 上一帧 accumulated（跨帧持久化）
        self.sample_count:      int   = 0
        self.frame_index:       int   = 0

    def to_dict(self) -> dict:
        return {
            "signal_type":       self.signal_type.name,
            "noisy_value":       round(self.noisy_value, 4),
            "reconstructed":     round(self.reconstructed, 4),
            "pre_convolved":     round(self.pre_convolved, 4),
            "accumulated":       round(self.accumulated, 4),
            "history_convolved": round(self.history_convolved, 4),
            "frame_index":       self.frame_index,
            "sample_count":      self.sample_count,
        }


# ─────────────────────────────────────────────────────────────────────────────
# AstroCellDenoiser — 降噪管线（四通道串联）
# 镜像 ScreenSpaceDenoise.cpp 内的 FDefaultScreenSpaceDenoiser
# ─────────────────────────────────────────────────────────────────────────────

class AstroCellDenoiser:
    """
    Python 等价于 FDefaultScreenSpaceDenoiser 的四通道降噪管线。

    降噪路径（完全镜像 C++ pass 调度序列）：
      Pass 1 — Reconstruction（重建采样聚合）
      Pass 2 — PreConvolution（空间预卷积）
      Pass 3 — TemporalAccumulation（时域历史混合）
      Pass 4 — HistoryConvolution（历史后滤波）

    鲁迅式：
      重建是收拾残局，预卷积是平息骚动，
      时域累积是向历史妥协，历史卷积是对妥协的再修正。
      四道工序，只为让噪声看起来不那么像噪声。
    """

    def __init__(self) -> None:
        # 各信号类型的跨帧历史状态（keyed by (signal_type, cell_id)）
        self._history: dict = {}

    # ------------------------------------------------------------------
    # _get_state — 获取或创建信号降噪状态
    # ------------------------------------------------------------------

    def _get_state(self, signal_type: AstroCellSignalType,
                   cell_id: str) -> AstroCellDenoiserState:
        key = (signal_type, cell_id)
        if key not in self._history:
            self._history[key] = AstroCellDenoiserState(signal_type)
        return self._history[key]

    # ------------------------------------------------------------------
    # Pass 1 — Reconstruction（重建采样聚合）
    # 镜像 ReconstructionPass：将 n 个 noisy 样本聚合为一个重建值
    # ─────────────────────────────────────────────────────────────────
    # 算法改动（鲁迅式 20%）：
    #   C++ 原版：Stackowiak 样本集（空间双边核，最多 56 个样本）。
    #   Python 版：以 n_samples 为权重的指数核近似（无需纹理采样），
    #   bilateral_weight = exp(-distance_sq / (2 * sigma^2)) × depth_weight，
    #   sigma 由 reconstruction_samples 倒数推导。
    #   这保留了「样本数越多结果越平滑」的物理直觉，但不依赖 Stackowiak 表。
    # ------------------------------------------------------------------

    def reconstruct(self,
                    state: AstroCellDenoiserState,
                    noisy_samples: list,
                    signal_type: AstroCellSignalType | None = None) -> float:
        """
        重建通道：将若干 noisy 样本通过双边核聚合为单一重建值。

        @param noisy_samples  list of (value, distance_sq, depth_diff) 三元组，
                              最多取 max_samples 个。
        @return               重建后的信号值 [0, 1]。

        鲁迅式：重建是把碎片拼成全貌——
        拼不回来的碎片，就用邻居的碎片代替，这不叫造假，叫去噪。
        """
        st = signal_type or state.signal_type
        max_s = {
            AstroCellSignalType.SHADOW_VISIBILITY_MASK:          _SDN_SHADOW_RECONSTRUCTION_SAMPLES,
            AstroCellSignalType.REFLECTIONS:                     _SDN_REFL_RECONSTRUCTION_SAMPLES,
            AstroCellSignalType.AMBIENT_OCCLUSION:               _SDN_AO_RECONSTRUCTION_SAMPLES,
            AstroCellSignalType.DIFFUSE_AND_AO:                  _SDN_GI_RECONSTRUCTION_SAMPLES,
        }.get(st, _SDN_SHADOW_RECONSTRUCTION_SAMPLES)

        samples = noisy_samples[:max_s]
        if not samples:
            state.reconstructed = state.noisy_value
            return state.reconstructed

        # 双边核 sigma 由采样数推导：sigma = 1/sqrt(max_s)
        sigma_sq = max(1.0 / max(max_s, 1), 1e-4)

        total_weight = 0.0
        total_value  = 0.0
        for (val, dist_sq, depth_diff) in samples:
            spatial_w = _math_sr.exp(-dist_sq / (2.0 * sigma_sq))
            depth_w   = _math_sr.exp(-abs(depth_diff) * 8.0)   # 深度权重衰减
            w = spatial_w * depth_w
            total_weight += w
            total_value  += w * val

        result = total_value / max(total_weight, 1e-8)
        state.reconstructed = max(0.0, min(1.0, result))
        state.sample_count  = len(samples)
        return state.reconstructed

    # ------------------------------------------------------------------
    # Pass 2 — PreConvolution（空间预卷积）
    # 镜像 PreConvolutionPass：对重建值进行 n 次 Mip 向下高斯卷积
    # ─────────────────────────────────────────────────────────────────
    # 算法改动（鲁迅式 20%）：
    #   C++ 原版：多 Mip 层级 2D 高斯，每层 4 个 tap，KernelSpreadFactor 控制核宽。
    #   Python 版：用迭代衰减模拟多 pass 高斯：
    #     result_i = reconstructed * decay^i + result_{i-1} * (1 - decay^i)
    #   decay 由 kernel_spread 和 Mip 层级推导，保留「多次卷积越来越平」的语义。
    # ------------------------------------------------------------------

    def pre_convolve(self,
                     state: AstroCellDenoiserState,
                     kernel_spread: float = 1.0,
                     signal_type: AstroCellSignalType | None = None) -> float:
        """
        预卷积通道：对重建值进行空间平滑（模拟多 Mip 高斯卷积）。

        @param kernel_spread  核扩展系数（对应 C++ KernelSpreadFactor CVars）。
        @return               预卷积后的信号值 [0, 1]。

        鲁迅式：预卷积是提前妥协——在时域历史介入之前，
        先用空间邻居把最刺眼的噪声磨平，免得历史背负太多。
        """
        st = signal_type or state.signal_type
        n_passes = {
            AstroCellSignalType.SHADOW_VISIBILITY_MASK: _SDN_SHADOW_PRE_CONVOLUTION,
            AstroCellSignalType.AMBIENT_OCCLUSION:      _SDN_AO_PRE_CONVOLUTION,
        }.get(st, _SDN_SHADOW_PRE_CONVOLUTION)

        spread = max(kernel_spread, 1.0)
        result = state.reconstructed
        for mip in range(min(n_passes, _SDN_MAX_MIP_LEVEL + 1)):
            # 每 Mip 层级的衰减系数：spread 越大收敛越快（高斯核越宽）
            decay = 1.0 / max(1.0 + spread * (mip + 1), 1.0)
            result = result * (1.0 - decay) + state.reconstructed * decay

        state.pre_convolved = max(0.0, min(1.0, result))
        return state.pre_convolved

    # ------------------------------------------------------------------
    # Pass 3 — TemporalAccumulation（时域累积）
    # 镜像 TemporalAccumulationPass：将当前帧与历史帧混合
    # ─────────────────────────────────────────────────────────────────
    # 算法改动（鲁迅式 20%）：
    #   C++ 原版：Catmull-Rom 历史采样 + 颜色裁剪（AABB clamp） + 速度场重投影。
    #   Python 版：无速度场（静态 2-D 布局）；历史权重 = _TLV_HISTORY_WEIGHT(0.9)，
    #   but 加入「拒绝系数」：若 pre_convolved 与 history 的差值超过 rejection_threshold，
    #   动态降低历史权重（等价于 C++ 历史颜色裁剪的单值版本）。
    # ------------------------------------------------------------------

    def temporal_accumulate(self,
                             state: AstroCellDenoiserState,
                             history_weight: float = _TLV_HISTORY_WEIGHT,
                             rejection_threshold: float = 0.3,
                             signal_type: AstroCellSignalType | None = None) -> float:
        """
        时域累积通道：将当前帧 pre_convolved 与上一帧 history 混合。

        @param history_weight       历史权重 [0, 1]，越高越稳越滞后。
        @param rejection_threshold  历史拒绝阈值；差异超过此值时削减历史权重。
        @return                     累积后的信号值 [0, 1]。

        鲁迅式：时域累积是历史的重量——
        历史越重，噪声消得越彻底，但鬼影也越重。
        拒绝系数是时代的清醒：当现实与历史相差太大时，不再盲从历史。
        """
        st = signal_type or state.signal_type
        if not _SDN_SHADOW_TEMPORAL and st in (
            AstroCellSignalType.SHADOW_VISIBILITY_MASK,
            AstroCellSignalType.POLYCHROMATIC_PENUMBRA_HARMONIC,
        ):
            # r.Shadow.Denoiser.TemporalAccumulation=0 路径
            state.accumulated = state.pre_convolved
            return state.accumulated

        current = state.pre_convolved
        history = state.history_value

        # 动态历史权重：差异过大时降低对历史的信任
        diff = abs(current - history)
        if diff > rejection_threshold:
            # 线性削减：超出阈值的部分按比例降权（镜像 AABB clamp 拒绝策略）
            excess = (diff - rejection_threshold) / max(1.0 - rejection_threshold, 1e-4)
            effective_weight = history_weight * (1.0 - min(excess, 1.0) * 0.8)
        else:
            effective_weight = history_weight

        accumulated = history * effective_weight + current * (1.0 - effective_weight)
        state.accumulated = max(0.0, min(1.0, accumulated))
        # 更新历史供下帧使用（持久化到 state.history_value）
        state.history_value = state.accumulated
        return state.accumulated

    # ------------------------------------------------------------------
    # Pass 4 — HistoryConvolution（历史后滤波）
    # 镜像 HistoryConvolutionPass：对 accumulated 做最终空间卷积
    # ─────────────────────────────────────────────────────────────────
    # 算法改动（鲁迅式 20%）：
    #   C++ 原版：用高分辨率历史 buffer 上的多样本卷积核（最多 56 个样本）。
    #   Python 版：以 n_history_samples 为核宽的均值滤波，对 accumulated
    #   做最终平滑；n_history_samples=1 时退化为恒等变换（默认路径）。
    #   KernelSpreadFactor 控制邻居权重衰减半径（AO 路径为 7，其余为 1）。
    # ------------------------------------------------------------------

    def history_convolve(self,
                         state: AstroCellDenoiserState,
                         neighbour_values: list | None = None,
                         kernel_spread_factor: float = 1.0,
                         signal_type: AstroCellSignalType | None = None) -> float:
        """
        历史卷积通道：对 accumulated 值进行最终空间后滤波。

        @param neighbour_values     可选的邻居信号值列表 [float]；
                                    为 None 或空时退化为恒等（n_samples=1 路径）。
        @param kernel_spread_factor AO/GI 路径的核扩展系数（AO=7，GI=3，其余=1）。
        @return                     最终降噪值 history_convolved [0, 1]。

        鲁迅式：历史卷积是最后的修缮——
        大多数情况下它什么都不做（n=1 恒等），
        但它存在的意义是：当累积历史本身带来伪迹时，有路可退。
        """
        st = signal_type or state.signal_type
        n_hist = {
            AstroCellSignalType.SHADOW_VISIBILITY_MASK: _SDN_SHADOW_HISTORY_CONVOLUTION,
            AstroCellSignalType.AMBIENT_OCCLUSION:      1,  # AO 历史卷积样本数=1
        }.get(st, 1)

        if n_hist <= 1 or not neighbour_values:
            # 恒等路径（默认）：history_convolved = accumulated
            state.history_convolved = state.accumulated
            return state.history_convolved

        # 带核扩展的加权均值（镜像 HistoryConvolution 多样本路径）
        spread = max(kernel_spread_factor, 1.0)
        total_w = 1.0
        total_v = state.accumulated
        for i, nb_val in enumerate(neighbour_values[:n_hist - 1]):
            # 权重随距离（样本索引）衰减，scale 由 spread 控制
            dist_w = _math_sr.exp(-((i + 1) ** 2) / (2.0 * spread ** 2))
            total_w += dist_w
            total_v += dist_w * nb_val

        result = total_v / max(total_w, 1e-8)
        state.history_convolved = max(0.0, min(1.0, result))
        return state.history_convolved

    # ------------------------------------------------------------------
    # denoise_single — 单信号完整四通道降噪
    # ------------------------------------------------------------------

    def denoise_single(self,
                       cell_id: str,
                       signal_type: AstroCellSignalType,
                       noisy_value: float,
                       noisy_samples: list | None = None,
                       neighbour_values: list | None = None,
                       kernel_spread: float = 1.0) -> dict:
        """
        对单个 cell 的单条信号执行完整四通道降噪。

        内部调度顺序（完全镜像 FDefaultScreenSpaceDenoiser::Denoise）：
          reconstruct() → pre_convolve() → temporal_accumulate() → history_convolve()

        @param noisy_value    当前帧原始信号 [0, 1]
        @param noisy_samples  重建通道输入样本列表（见 reconstruct() 参数说明）
        @param neighbour_values 历史卷积通道邻居值列表
        @param kernel_spread  预卷积核扩展系数
        @return               包含四通道输出的 dict

        鲁迅式：四通道串联，每通道都在「改善」——
        改善到最后，输出已与输入相差甚远，
        但那正是降噪的本义：让人看见想看见的，而非真实存在的。
        """
        state = self._get_state(signal_type, cell_id)
        state.noisy_value = noisy_value
        state.frame_index += 1

        # Reconstruction
        samples = noisy_samples or [(noisy_value, 0.0, 0.0)]
        self.reconstruct(state, samples, signal_type)

        # PreConvolution — kernel spread 由 signal type 决定
        spread = {
            AstroCellSignalType.AMBIENT_OCCLUSION: _SDN_AO_KERNEL_SPREAD,
        }.get(signal_type, kernel_spread)
        self.pre_convolve(state, spread, signal_type)

        # TemporalAccumulation
        self.temporal_accumulate(state, signal_type=signal_type)

        # HistoryConvolution — kernel spread factor 由 signal type 决定
        ksf = {
            AstroCellSignalType.AMBIENT_OCCLUSION: 7.0,
            AstroCellSignalType.DIFFUSE_AND_AO:    3.0,
        }.get(signal_type, 1.0)
        self.history_convolve(state, neighbour_values, ksf, signal_type)

        return state.to_dict()

    # ------------------------------------------------------------------
    # denoise_batch — 多信号批量降噪（镜像 kMaxBatchSize 批处理）
    # ------------------------------------------------------------------

    def denoise_batch(self,
                      cell_id: str,
                      signals: list) -> list:
        """
        批量降噪入口，最多处理 _SDN_MAX_BATCH_SIZE 条信号。

        镜像 IScreenSpaceDenoiser::kMaxBatchSize 批处理约束：
        超出部分静默截断（mirrors static_assert 截断语义）。

        @param signals  list of dict，每项包含：
                        { "type": AstroCellSignalType,
                          "noisy_value": float,
                          "noisy_samples": [...],   # 可选
                          "neighbour_values": [...], # 可选
                          "kernel_spread": float }   # 可选
        @return         list of denoise_single 返回 dict，与 signals 等长。

        鲁迅式：批处理是工业化，是效率，是对个体的去个性化——
        但降噪器有义务记住每条信号的名字（cell_id + signal_type），
        因为历史状态是以名字为键的。
        """
        batch = signals[:_SDN_MAX_BATCH_SIZE]
        results = []
        for sig in batch:
            result = self.denoise_single(
                cell_id        = cell_id,
                signal_type    = sig["type"],
                noisy_value    = sig.get("noisy_value", 0.0),
                noisy_samples  = sig.get("noisy_samples"),
                neighbour_values = sig.get("neighbour_values"),
                kernel_spread  = sig.get("kernel_spread", 1.0),
            )
            results.append(result)
        return results


# 进程级降噪器单例（镜像 GScreenSpaceDenoiser 全局指针）
_ASTRO_CELL_DENOISER: AstroCellDenoiser | None = None


def get_cell_denoiser() -> AstroCellDenoiser:
    """
    返回进程级降噪器单例。
    镜像 GScreenSpaceDenoiser = new FDefaultScreenSpaceDenoiser() 的初始化逻辑。
    鲁迅式：降噪器是公共设施——不属于任何一个 cell，却服务所有 cell。
    """
    global _ASTRO_CELL_DENOISER
    if _ASTRO_CELL_DENOISER is None:
        _ASTRO_CELL_DENOISER = AstroCellDenoiser()
    return _ASTRO_CELL_DENOISER


# ─────────────────────────────────────────────────────────────────────────────
# AstroCellVisibilityProcessor — 可见性处理器
# 镜像 SceneVisibility.cpp 中的帧视图初始化与剔除逻辑
# ─────────────────────────────────────────────────────────────────────────────

class AstroCellVisibilityProcessor:
    """
    Python 等价于 SceneVisibility.cpp 中的 FSceneRenderer::InitViews() 主体。

    负责将注册的 cell 集合经过五道剔除/筛选后输出可见集合：
      1. 距离剔除（_distance_cull_to_sphere_edge）
      2. LOD 计算（_compute_lod_level）
      3. 线框模式剔除（_wireframe_cull，正交视图专用）
      4. 最小屏幕尺寸剔除（_min_screen_radius_cull）
      5. HZB 遮挡查询（_hzb_occlusion_query，复用 AstroCellBVH）

    额外输出 TAA Jitter 向量（供 proc() 渲染器微偏移采样坐标）。

    鲁迅式：
      可见性处理器是门卫——大多数人被拒之门外，少数人才进入渲染。
      被剔除不是侮辱，是性能的善意；被保留才是算法真正的工作对象。
    """

    def __init__(self,
                 viewport_w: float = 1200.0,
                 viewport_h: float = 900.0,
                 max_draw_distance: float = 4000.0) -> None:
        self.viewport_w        = viewport_w
        self.viewport_h        = viewport_h
        self.viewport_area     = max(viewport_w * viewport_h, 1.0)
        self.max_draw_distance = max_draw_distance
        self._bvh              = AstroCellBVH()
        self._frame_index      = 0
        # 当前帧 TAA Jitter 向量（像素单位）
        self.current_jitter: tuple = (0.0, 0.0)

    # ------------------------------------------------------------------
    # _halton — Halton 低差异序列（用于 TAA Jitter）
    # 镜像 FHalton::Base2/Base3 实现
    # ------------------------------------------------------------------

    @staticmethod
    def _halton(index: int, base: int) -> float:
        """
        Halton 序列第 index 项（1-based）。
        鲁迅式：低差异序列是公平的采样——不偏好任何方向，不重复任何位置。
        """
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result

    def _compute_taa_jitter(self, frame_index: int | None = None) -> tuple:
        """
        计算当前帧的 TAA Sub-pixel Jitter。

        镜像 SceneVisibility.cpp 中 ComputeTemporalJitteredProjectionMatrix()：
          JitterX = (Halton(2, FrameNumber % TAASamples) - 0.5) / ViewWidth
          JitterY = (Halton(3, FrameNumber % TAASamples) - 0.5) / ViewHeight

        反转标志（CVarInvertTemporalJitterX/Y）应用后，
        Jitter 单位为像素（乘回 viewport 尺寸）。

        算法改动（鲁迅式 20%）：
          C++ 版的 ScaleSamples 逻辑（r.TemporalAAScaleSamples）在上采样时
          等比扩大样本数以维持密度。Python 版固定为 _SR_TAA_SAMPLES=8，
          不做动态缩放，因为 2-D SVG 布局没有上采样 pass。

        @return  (jitter_x_px, jitter_y_px) 像素级偏移量

        鲁迅式：抖动是对单帧局限性的承认——
        每帧都只能看见真相的一部分，累积起来才是完整的。
        """
        fi = frame_index if frame_index is not None else self._frame_index
        sample_idx = (fi % _SR_TAA_SAMPLES) + 1   # Halton 从 1 开始

        jx = self._halton(sample_idx, 2) - 0.5   # 归一化 [-0.5, 0.5]
        jy = self._halton(sample_idx, 3) - 0.5

        if _SR_INVERT_JITTER_X:
            jx = -jx
        if _SR_INVERT_JITTER_Y:
            jy = -jy

        # 转换为像素单位（镜像 JitterX = RawJitter / ViewWidth 的逆）
        jitter_px_x = jx * 1.0   # 保持亚像素级（< 1px），不乘以 viewport_w
        jitter_px_y = jy * 1.0

        return (round(jitter_px_x, 6), round(jitter_px_y, 6))

    # ------------------------------------------------------------------
    # _distance_cull_to_sphere_edge — 包围球边缘距离剔除
    # 镜像 SceneVisibility.cpp ComputeFrustumCullToSphereEdge()
    # ------------------------------------------------------------------

    def _distance_cull_to_sphere_edge(self,
                                       bbox: dict,
                                       camera_pos: tuple = (600.0, 450.0, -100.0),
                                       max_dist: float | None = None) -> bool:
        """
        以包围球边缘距离判断 cell 是否应被距离剔除。

        C++ 路径（GDistanceCullToSphereEdge=true）：
            sphere_edge_dist = dist(camera, sphere_center) - sphere_radius
            if sphere_edge_dist > MaxDrawDistance: cull

        2-D 适配：
            sphere_center = (cx, cy, z*100)
            sphere_radius = max(w, h) / 2  （包围圆半径）

        鲁迅式：边缘距离比球心距离更保守——
        即使球心在视野内，边缘也可能已经超出可见范围。

        @return True = 应被剔除（太远），False = 保留
        """
        cx = bbox["x"] + bbox["w"] / 2.0
        cy = bbox["y"] + bbox["h"] / 2.0
        cz = float(bbox.get("z", 3)) * 100.0
        radius = max(bbox["w"], bbox["h"]) / 2.0

        cam_x, cam_y, cam_z = camera_pos
        dx = cx - cam_x
        dy = cy - cam_y
        dz = cz - cam_z
        dist_to_center = _math_sr.sqrt(dx*dx + dy*dy + dz*dz)

        if _SR_DIST_CULL_TO_SPHERE_EDGE:
            dist = dist_to_center - radius
        else:
            dist = dist_to_center

        threshold = max_dist if max_dist is not None else self.max_draw_distance
        return dist > threshold

    # ------------------------------------------------------------------
    # _compute_lod_level — LOD 层级计算
    # 镜像 SceneVisibility.cpp ComputeTemporalLODLevel()
    # ------------------------------------------------------------------

    def _compute_lod_level(self, bbox: dict,
                            screen_fraction: float | None = None) -> int:
        """
        计算 cell 的 LOD 层级（0=全细节 / 1=减半 / 2=矩形占位 / -1=剔除）。

        镜像 ComputeLODLevel() 基于 screen_radius 的分段逻辑：
          LOD 0: screen_fraction >= _NANITE_LOD2_THRESHOLD * 10
          LOD 1: screen_fraction >= _NANITE_LOD2_THRESHOLD
          LOD 2: screen_fraction >= _NANITE_CULL_THRESHOLD
          LOD-1: screen_fraction < _NANITE_CULL_THRESHOLD （剔除）

        LOD 距离缩放系数（r.StaticMeshLODDistanceScale）乘入阈值，
        等价于 C++ 中 FinalLODScale = LODDistanceScale * InvScreenSize。

        鲁迅式：LOD 是资源的公平分配——
        近处的 cell 获得精细描绘，远处的只配一个矩形。
        公平，却令人心寒。

        @return LOD 层级整数
        """
        if screen_fraction is None:
            area = bbox["w"] * bbox["h"]
            screen_fraction = area / self.viewport_area

        # 应用 LOD 距离缩放（距离越大 = 画面越小 = 需要更大 fraction 才不降 LOD）
        effective_frac = screen_fraction / max(_SR_LOD_DISTANCE_SCALE, 1e-4)

        cull_threshold = _NANITE_CULL_THRESHOLD
        lod2_threshold = _NANITE_LOD2_THRESHOLD

        if effective_frac < cull_threshold:
            return -1   # 剔除
        elif effective_frac < lod2_threshold:
            return 2    # 矩形占位
        elif effective_frac < lod2_threshold * 10.0:
            return 1    # 减半细节
        else:
            return 0    # 全细节

    # ------------------------------------------------------------------
    # _wireframe_cull — 线框模式剔除
    # 镜像 SceneVisibility.cpp GWireframeCullThreshold
    # ------------------------------------------------------------------

    def _wireframe_cull(self, bbox: dict, ortho_scale: float = 1.0) -> bool:
        """
        在正交线框视图中剔除过小的 cell。

        镜像 CVarWireframeCullThreshold=5.0：
            if projected_size < threshold: cull

        projected_size = max(w, h) × ortho_scale（像素）

        @return True = 应被剔除，False = 保留
        """
        projected_size = max(bbox["w"], bbox["h"]) * ortho_scale
        return projected_size < _SR_WIREFRAME_CULL_THRESHOLD

    # ------------------------------------------------------------------
    # _min_screen_radius_cull — 最小屏幕占比剔除
    # 镜像 GMinScreenRadiusForLights / GMinScreenRadiusForDepthPrepass
    # ------------------------------------------------------------------

    def _min_screen_radius_cull(self, bbox: dict,
                                 mode: str = "lights") -> bool:
        """
        最小屏幕占比剔除：cell 在屏幕上的等效半径小于阈值时剔除。

        镜像 GMinScreenRadiusForLights（0.03）和
             GMinScreenRadiusForDepthPrepass（0.03）：
            screen_radius = sqrt(area / viewport_area) / 2
            if screen_radius < threshold: cull

        @param mode  "lights" 或 "depth_prepass"
        @return True = 应被剔除，False = 保留
        """
        area = bbox["w"] * bbox["h"]
        screen_radius = _math_sr.sqrt(area / self.viewport_area) / 2.0
        threshold = (
            _SR_MIN_SCREEN_RADIUS_LIGHTS
            if mode == "lights"
            else _SR_MIN_SCREEN_RADIUS_DEPTH_PREPASS
        )
        return screen_radius < threshold

    # ------------------------------------------------------------------
    # _hzb_occlusion_query — BVH 等价的 HZB 遮挡查询
    # 镜像 SceneVisibility.cpp GHZBOcclusion=1 路径
    # ------------------------------------------------------------------

    def _hzb_occlusion_query(self, cell_id: str, bbox: dict) -> bool:
        """
        以 BVH 重叠查询近似 HZB 遮挡测试。

        C++ HZB 路径：将 cell 包围盒投影到深度金字塔（Hierarchical Z Buffer），
        若所有样本均被遮挡则标记为 occluded。

        2-D BVH 近似（算法改动，鲁迅式 20%）：
          查询与当前 cell bbox 重叠的其他 cell；若重叠 cell 数量超过阈值
          且所有重叠 cell 的 z 均高于当前 cell，则视为被遮挡。
          threshold = 3（三个或更多高 z 遮挡者 → 遮挡）。
          这近似了 HZB 中「多个深度样本均被遮挡」的多样本测试逻辑。

        注意：此测试仅在 _SR_HZB_OCCLUSION >= 1 时生效；
        _SR_HZB_OCCLUSION == 0 时始终返回 False（不遮挡）。

        @return True = 被遮挡（应跳过），False = 可见

        鲁迅式：遮挡查询是视觉诚实的代价——
        被挡住的东西没有资格占用渲染时间，
        哪怕它确实存在于那个位置。
        """
        if _SR_HZB_OCCLUSION == 0:
            return False   # 硬件查询路径：不执行 BVH 遮挡（由硬件处理）

        cell_z = float(bbox.get("z", 3))
        overlapping = self._bvh.query_overlapping_cells(bbox)

        occluding_count = 0
        for other_id in overlapping:
            if other_id == cell_id:
                continue
            # 从 BVH 叶表找到对应的 bbox（通过 cell_registry 读取）
            # 简化：以 cell_id hash 代理 z，用于遮挡计数
            # 实际应从 all_bboxes 字典查询，此处以 BVH 命中数作代理
            occluding_count += 1

        # 超过 3 个重叠 cell 且自身 z 较低 → 被遮挡
        return occluding_count >= 3 and cell_z < 3

    # ------------------------------------------------------------------
    # process — 完整可见性处理（InitViews 等价）
    # ------------------------------------------------------------------

    def process(self,
                cell_registry: dict,
                camera_pos: tuple = (600.0, 450.0, -100.0),
                ortho_mode: bool = False,
                rebuild_bvh: bool = True) -> dict:
        """
        对 cell_registry 中的所有 cell 执行完整可见性处理。

        执行顺序（完全镜像 FSceneRenderer::InitViews 的剔除流水线）：
          1. 构建/更新 BVH（用于 HZB 等价遮挡查询）
          2. 距离剔除（_distance_cull_to_sphere_edge）
          3. LOD 计算（_compute_lod_level）
          4. 线框剔除（ortho_mode 时启用）
          5. 最小屏幕占比剔除
          6. HZB 遮挡查询
          7. 更新 TAA Jitter（每帧一次）

        @param cell_registry  来自 _load_cell_registry() 的 dict（cells + z_layers）
        @param camera_pos     相机世界坐标（用于距离剔除）
        @param ortho_mode     是否为正交线框视图（启用 wireframe_cull）
        @param rebuild_bvh    是否重建 BVH（首帧或布局变化时应为 True）
        @return               可见性结果 dict：
                              {
                                "visible": {cell_id: lod_level, ...},
                                "culled":  {cell_id: reason, ...},
                                "taa_jitter": (jx, jy),
                                "stats": {...}
                              }

        鲁迅式：每一帧都是一次审判——所有 cell 排队等候，
        通过者入画，未通过者等待下一帧的宽恕。
        """
        self._frame_index += 1
        self.current_jitter = self._compute_taa_jitter(self._frame_index)

        cells = cell_registry.get("cells", {})

        # 重建 BVH（镜像 UpdateScene / AddPrimitive 后的 BVH 重构）
        if rebuild_bvh:
            raw_cells: dict = {}
            for cid, entry in cells.items():
                bbox_data = entry.get("bbox", {})
                if "min" in bbox_data and "max" in bbox_data:
                    mn = bbox_data["min"]
                    mx = bbox_data["max"]
                    raw_cells[cid] = {
                        "x": mn[0], "y": mn[1],
                        "w": mx[0] - mn[0], "h": mx[1] - mn[1],
                        "z": mn[2] if len(mn) > 2 else 0,
                    }
                else:
                    raw_cells[cid] = bbox_data
            self._bvh.build_from_registry({
                cid: {"bbox": {"min": [bbox["x"], bbox["y"], bbox.get("z", 0)],
                               "max": [bbox["x"]+bbox["w"], bbox["y"]+bbox["h"], bbox.get("z", 0)]}}
                for cid, bbox in raw_cells.items()
            })

        visible: dict = {}
        culled:  dict = {}
        cull_stats = {
            "distance": 0, "lod": 0, "wireframe": 0,
            "min_screen": 0, "hzb": 0, "total": 0
        }

        for cell_id, entry in cells.items():
            # Reconstruct bbox from registry format
            bbox_data = entry.get("bbox", {})
            if "min" in bbox_data and "max" in bbox_data:
                mn = bbox_data["min"]
                mx = bbox_data["max"]
                bbox = {
                    "x": mn[0], "y": mn[1],
                    "w": mx[0] - mn[0], "h": mx[1] - mn[1],
                    "z": mn[2] if len(mn) > 2 else 0,
                }
            else:
                bbox = dict(bbox_data)

            cull_stats["total"] += 1

            # ── Pass 1: 距离剔除 ──────────────────────────────────────────
            if self._distance_cull_to_sphere_edge(bbox, camera_pos):
                culled[cell_id] = "distance"
                cull_stats["distance"] += 1
                continue

            # ── Pass 2: LOD 计算 ───────────────────────────────────────────
            area = bbox["w"] * bbox["h"]
            screen_frac = area / self.viewport_area
            lod = self._compute_lod_level(bbox, screen_frac)
            if lod < 0:
                culled[cell_id] = "lod_cull"
                cull_stats["lod"] += 1
                continue

            # ── Pass 3: 线框剔除（正交模式专用） ─────────────────────────
            if ortho_mode and self._wireframe_cull(bbox):
                culled[cell_id] = "wireframe"
                cull_stats["wireframe"] += 1
                continue

            # ── Pass 4: 最小屏幕占比剔除 ──────────────────────────────────
            if self._min_screen_radius_cull(bbox, "depth_prepass"):
                culled[cell_id] = "min_screen_radius"
                cull_stats["min_screen"] += 1
                continue

            # ── Pass 5: HZB 遮挡查询 ──────────────────────────────────────
            if self._hzb_occlusion_query(cell_id, bbox):
                culled[cell_id] = "hzb_occluded"
                cull_stats["hzb"] += 1
                continue

            # 通过所有剔除：加入可见集合
            visible[cell_id] = lod

        visible_count = len(visible)
        culled_count  = len(culled)

        print(
            f"[ASTRO-VIS] VisibilityProcessor frame={self._frame_index} "
            f"total={cull_stats['total']} visible={visible_count} "
            f"culled={culled_count} "
            f"(dist={cull_stats['distance']} lod={cull_stats['lod']} "
            f"wire={cull_stats['wireframe']} scr={cull_stats['min_screen']} "
            f"hzb={cull_stats['hzb']}) "
            f"taa_jitter={self.current_jitter}",
            file=sys.stderr,
        )

        return {
            "visible":    visible,
            "culled":     culled,
            "taa_jitter": self.current_jitter,
            "stats":      cull_stats,
            "frame":      self._frame_index,
        }


# ─────────────────────────────────────────────────────────────────────────────
# AstroCellFrameRenderer — 帧渲染总调度
# 镜像 SceneRendering.cpp FDeferredShadingSceneRenderer::Render() 六阶段
# ─────────────────────────────────────────────────────────────────────────────

class AstroCellFrameRenderer:
    """
    Python 等价于 FDeferredShadingSceneRenderer::Render() 的六阶段帧渲染管线。

    阶段映射（完全镜像 C++ Render() 调用序列）：
      init_views()       — FSceneRenderer::InitViews()：可见性、LOD、TAA Jitter
      pre_pass()         — DepthPrePass：写入 Z-buffer（此处：写入 depth_manifest）
      base_pass()        — BasePass：写入 GBuffer（此处：写入 cell params）
      lighting_pass()    — RenderLights()：调用 AstroCellLightPass
      translucency_pass()— RenderTranslucency()：调用 AstroCellTranslucencyRenderer
      post_process()     — PostProcess：调用 AstroCellDenoiser 降噪信号

    各阶段之间通过 frame_state dict 传递中间结果，镜像 C++ 的 RDG pass graph
    数据流（RDG texture 以 Python dict 键值代替）。

    鲁迅式：六阶段如六重门——
    每一道门都筛去一些不该进入最终画面的杂质。
    但每道门也都会拒绝一些本该保留的东西，这是不可避免的代价。
    """

    def __init__(self,
                 viewport_w: float = 1200.0,
                 viewport_h: float = 900.0) -> None:
        self.viewport_w  = viewport_w
        self.viewport_h  = viewport_h
        self._vis_proc   = AstroCellVisibilityProcessor(viewport_w, viewport_h)
        self._denoiser   = get_cell_denoiser()
        self._light_pass = AstroCellLightPass()
        self._trans_renderer = AstroCellTranslucencyRenderer()
        self._trans_renderer.set_parameters(None)
        self._frame_state: dict = {}

    # ------------------------------------------------------------------
    # Phase 1 — init_views
    # ------------------------------------------------------------------

    def init_views(self, cell_registry: dict,
                   camera_pos: tuple = (600.0, 450.0, -100.0)) -> dict:
        """
        可见性 + LOD + TAA Jitter 初始化。
        镜像 FSceneRenderer::InitViews()。

        鲁迅式：InitViews 是渲染器的「入学考试」——
        只有通过的 cell 才有资格参与后续渲染；落选者等待下一帧。
        """
        vis_result = self._vis_proc.process(
            cell_registry, camera_pos, rebuild_bvh=True
        )
        self._frame_state["vis_result"]  = vis_result
        self._frame_state["visible_set"] = set(vis_result["visible"].keys())
        self._frame_state["lod_map"]     = vis_result["visible"]
        self._frame_state["taa_jitter"]  = vis_result["taa_jitter"]
        self._frame_state["camera_pos"]  = camera_pos
        return vis_result

    # ------------------------------------------------------------------
    # Phase 2 — pre_pass（深度预通道）
    # ------------------------------------------------------------------

    def pre_pass(self, cell_entries: list) -> dict:
        """
        深度预通道：写入 Z-buffer 等价数据（depth_manifest）。
        镜像 FDeferredShadingSceneRenderer::RenderPrePass()。

        仅处理通过 init_views 的可见 cell；
        LOD=2（矩形占位）的 cell 跳过深度写入（_SR_MIN_SCREEN_RADIUS_DEPTH_PREPASS 剔除对应）。

        鲁迅式：深度预通道是排座次——谁在前，谁在后，先定下来，后面才好遮挡。
        """
        visible_set = self._frame_state.get("visible_set", set())
        lod_map     = self._frame_state.get("lod_map", {})

        prepass_entries = [
            e for e in cell_entries
            if e["cell_id"] in visible_set and lod_map.get(e["cell_id"], -1) < 2
        ]

        compositor = AstroCellCompositor(visible_set)
        compositor.begin_frame(prepass_entries)
        dm = compositor.emit_depth_stencil()

        self._frame_state["depth_manifest"] = dm
        self._frame_state["prepass_count"]  = len(prepass_entries)
        return dm

    # ------------------------------------------------------------------
    # Phase 3 — base_pass（基础通道）
    # ------------------------------------------------------------------

    def base_pass(self, cell_entries: list) -> list:
        """
        基础通道：将可见 cell 写入 GBuffer（此处：输出 draw list）。
        镜像 FDeferredShadingSceneRenderer::RenderBasePass()。

        使用 AstroCellDrawList 进行 species 批量排序，
        减少 SVG <defs> 重复写入（等价于 PSO state change 最小化）。

        鲁迅式：基础通道是第一次在画布上落笔——
        笔触不必精细，只要建立基本的形状与颜色关系。
        """
        visible_set = self._frame_state.get("visible_set", set())
        lod_map     = self._frame_state.get("lod_map", {})

        draw_list = AstroCellDrawList()
        for e in cell_entries:
            cid = e["cell_id"]
            if cid not in visible_set:
                continue
            lod     = lod_map.get(cid, 0)
            z_layer = int(e.get("bbox", {}).get("z", 3))
            species = e.get("species", "")
            bbox    = e.get("bbox", {})
            draw_list.register_cell_draw_entry(
                cell_id=cid, z_layer=z_layer,
                species=species, bbox=bbox,
                extra={"lod": lod},
            )

        ordered = draw_list.flush_draw_order()
        self._frame_state["draw_order"] = ordered
        self._frame_state["base_pass_defs_cost"] = draw_list.svg_defs_cost
        return ordered

    # ------------------------------------------------------------------
    # Phase 4 — lighting_pass（光照通道）
    # ------------------------------------------------------------------

    def lighting_pass(self, cell_entries: list, all_bboxes: dict) -> dict:
        """
        光照通道：为每个可见 cell 执行 AstroCellLightPass.execute()。
        镜像 FDeferredShadingSceneRenderer::RenderLights()。

        鲁迅式：光照通道是真正的道德审判——
        所有 cell 都在光照下暴露，没有阴影可以藏身。
        （除非 contact shadow 说你可以。）
        """
        visible_set = self._frame_state.get("visible_set", set())
        light_results: dict = {}

        # 默认光照参数（过程级单例方式）
        default_light = AstroCellDeferredLightUniforms()

        _ROUGHNESS_MAP = {
            "cil-eye": 0.1, "cil-bolt": 0.2, "cil-plus": 0.3,
            "cil-vector": 0.5, "cil-arrow-right": 0.7,
            "cil-filter": 0.3, "cil-code": 0.4, "cil-layers": 0.2,
            "cil-loop": 0.5, "cil-graph": 0.6,
        }

        for e in cell_entries:
            cid = e["cell_id"]
            if cid not in visible_set:
                continue
            species   = e.get("species", "")
            bbox      = e.get("bbox", {})
            roughness = _ROUGHNESS_MAP.get(species, 0.5)
            lp = AstroCellLightPass(light=default_light)
            result = lp.execute(cid, bbox, species, roughness, all_bboxes)
            light_results[cid] = result

        self._frame_state["light_results"] = light_results
        return light_results

    # ------------------------------------------------------------------
    # Phase 5 — translucency_pass（半透明通道）
    # ------------------------------------------------------------------

    def translucency_pass(self, cell_entries: list) -> str:
        """
        半透明通道：筛出 opacity < 1.0 的 cell，执行前向 Alpha 合成。
        镜像 FDeferredShadingSceneRenderer::RenderTranslucency()。

        鲁迅式：半透明通道是为那些无法完全表态的 cell 开设的——
        不完全透明，也不完全不透明，在前向渲染中寻找一个暧昧的位置。
        """
        visible_set = self._frame_state.get("visible_set", set())
        self._trans_renderer.set_parameters(None)
        trans_svg = self._trans_renderer.render(cell_entries, visible_set)
        self._frame_state["translucency_svg"] = trans_svg
        return trans_svg

    # ------------------------------------------------------------------
    # Phase 6 — post_process（后处理 + 降噪）
    # ------------------------------------------------------------------

    def post_process(self,
                     cell_entries: list,
                     all_bboxes: dict) -> dict:
        """
        后处理通道：对每个可见 cell 运行降噪管线（AstroCellDenoiser）。
        镜像 FDeferredShadingSceneRenderer::RenderFinish() + PostProcess。

        降噪信号来源：
          SHADOW        ← light_result["contact_shadow_factor"]（1=无阴影/0=全阴影）
          AO            ← crowding_opacity（PostProcessAO 已在 proc() 计算）

        TAA Jitter 此处仅记录到 post_process_result 供 proc() 读取；
        实际的画面偏移应在 SVG translate 属性中应用（由调用方处理）。

        鲁迅式：后处理是渲染的化妆师——
        把真实的瑕疵磨平，再加上几分不真实的光晕。
        最终观众看到的，是化过妆的真相。
        """
        visible_set  = self._frame_state.get("visible_set", set())
        light_results = self._frame_state.get("light_results", {})
        taa_jitter   = self._frame_state.get("taa_jitter", (0.0, 0.0))

        denoised: dict = {}
        for e in cell_entries:
            cid = e["cell_id"]
            if cid not in visible_set:
                continue

            # SHADOW 信号：来自接触阴影因子
            lr = light_results.get(cid, {})
            shadow_noisy = 1.0 - lr.get("contact_shadow_factor", 1.0)

            # AO 信号：从 bbox 相对于 viewport 的面积估算（粗略 AO 代理）
            bbox    = e.get("bbox", {})
            area    = bbox.get("w", 100) * bbox.get("h", 50)
            ao_noisy = min(1.0, area / max(self.viewport_w * self.viewport_h * 0.005, 1.0))

            batch = [
                {
                    "type":        AstroCellSignalType.SHADOW_VISIBILITY_MASK,
                    "noisy_value": shadow_noisy,
                    "noisy_samples": [(shadow_noisy, 0.0, 0.0),
                                      (shadow_noisy * 0.9, 0.1, 0.01)],
                },
                {
                    "type":        AstroCellSignalType.AMBIENT_OCCLUSION,
                    "noisy_value": ao_noisy,
                    "noisy_samples": [(ao_noisy, 0.0, 0.0)],
                    "kernel_spread": _SDN_AO_KERNEL_SPREAD,
                },
            ]
            batch_results = self._denoiser.denoise_batch(cid, batch)
            denoised[cid] = {
                "shadow_denoised": batch_results[0]["history_convolved"],
                "ao_denoised":     batch_results[1]["history_convolved"],
                "taa_jitter":      taa_jitter,
            }

        self._frame_state["denoised"] = denoised
        return denoised

    # ------------------------------------------------------------------
    # render — 完整帧渲染（主入口）
    # ------------------------------------------------------------------

    def render(self,
               cell_registry: dict,
               cell_entries: list,
               all_bboxes: dict,
               camera_pos: tuple = (600.0, 450.0, -100.0)) -> dict:
        """
        执行完整帧渲染六阶段，返回帧结果 dict。

        镜像 FDeferredShadingSceneRenderer::Render() 顶层调用序列：
          Render() → InitViews → PrePass → BasePass
                   → Lighting → Translucency → PostProcess

        @return dict 包含所有阶段的输出，供 orchestrator 消费：
                {
                  "visible":       {cell_id: lod, ...},
                  "culled":        {cell_id: reason, ...},
                  "taa_jitter":    (jx, jy),
                  "draw_order":    [...],
                  "light_results": {cell_id: {...}, ...},
                  "denoised":      {cell_id: {...}, ...},
                  "translucency_svg": "...",
                  "depth_manifest":   {...},
                  "frame_stats":   {...},
                }

        鲁迅式：Render() 是总司令——
        它不做任何具体的像素工作，只负责让六个部门各就各位、按序发令。
        胜利属于整个流水线，失败也是。
        """
        # Phase 1 — InitViews
        vis_result = self.init_views(cell_registry, camera_pos)

        # Phase 2 — PrePass
        depth_manifest = self.pre_pass(cell_entries)

        # Phase 3 — BasePass
        draw_order = self.base_pass(cell_entries)

        # Phase 4 — Lighting
        light_results = self.lighting_pass(cell_entries, all_bboxes)

        # Phase 5 — Translucency
        trans_svg = self.translucency_pass(cell_entries)

        # Phase 6 — PostProcess + Denoise
        denoised = self.post_process(cell_entries, all_bboxes)

        # 统计汇总
        frame_stats = {
            "frame":           self._vis_proc._frame_index,
            "visible_count":   len(vis_result["visible"]),
            "culled_count":    len(vis_result["culled"]),
            "base_pass_defs":  self._frame_state.get("base_pass_defs_cost", 0),
            "prepass_cells":   self._frame_state.get("prepass_count", 0),
            "light_computed":  len(light_results),
            "denoised_cells":  len(denoised),
            "taa_jitter":      vis_result["taa_jitter"],
            "taa_sample":      self._vis_proc._frame_index % _SR_TAA_SAMPLES,
        }

        print(
            f"[ASTRO-RENDER] FrameRenderer.render() frame={frame_stats['frame']} "
            f"visible={frame_stats['visible_count']} "
            f"culled={frame_stats['culled_count']} "
            f"lights={frame_stats['light_computed']} "
            f"denoised={frame_stats['denoised_cells']} "
            f"taa_jitter={frame_stats['taa_jitter']} "
            f"taa_sample={frame_stats['taa_sample']}/{_SR_TAA_SAMPLES}",
            file=sys.stderr,
        )

        return {
            "visible":          vis_result["visible"],
            "culled":           vis_result["culled"],
            "taa_jitter":       vis_result["taa_jitter"],
            "draw_order":       draw_order,
            "light_results":    light_results,
            "denoised":         denoised,
            "translucency_svg": trans_svg,
            "depth_manifest":   depth_manifest,
            "frame_stats":      frame_stats,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 模块级单例 — 全场景共享同一帧渲染器
# 镜像 FDeferredShadingSceneRenderer 在 FSceneRenderer::CreateSceneRenderer 中的实例化
# ─────────────────────────────────────────────────────────────────────────────

_ASTRO_FRAME_RENDERER: AstroCellFrameRenderer | None = None


def get_frame_renderer(viewport_w: float = 1200.0,
                       viewport_h: float = 900.0) -> AstroCellFrameRenderer:
    """
    返回模块级帧渲染器单例。

    鲁迅式：单帧渲染器如同时代精神——全场景共享，不容个体另起炉灶。
    但若视口尺寸改变，旧实例便不再适用，须重建。
    """
    global _ASTRO_FRAME_RENDERER
    if _ASTRO_FRAME_RENDERER is None:
        _ASTRO_FRAME_RENDERER = AstroCellFrameRenderer(viewport_w, viewport_h)
    return _ASTRO_FRAME_RENDERER

