import json
import math
import os
import sys

from channels.cell_component import _load_cell_registry

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

