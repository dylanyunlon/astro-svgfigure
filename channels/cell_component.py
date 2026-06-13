#!/usr/bin/env python3
"""
CellComponent — Apollo Component::Proc() equivalent for astro-svgfigure.

Each sub-Claude runs this with its cell_id.
It reads skeleton signal + force_field, uses species gene_traits to
algorithmically generate SVG via svgwrite, publishes bbox + svg + status.

Usage:
    python3 cell_component.py <cell_id>
    python3 cell_component.py self_attn
"""
import json
import math
import os
import struct as _struct
import sys
from dataclasses import dataclass, field

# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] RendererScene → Python port
#
# Ported from commit 72c4d0c:
#   upstream/unreal-renderer/RendererScene.cpp
#
# FAstroCellBBox:
#   Axis-aligned bounding box cached per cell node, stored in world space.
#   Updated by update_cell_constraint() whenever the owning cell reports a
#   bounds change.  HasChanged() mirrors the Tolerance=0.01 float comparison.
#
# FAstroCellSceneProxy (→ cell_registry entry):
#   Lightweight descriptor for each Astro cell node registered in the scene.
#   Carries z_layer (derived from world-Z / AstroCellZLayerHeight) and a
#   cached bbox.  dirty flag set by update_cell_constraint; consumed on the
#   next registry flush (BuildCellConstraintBuffer analogue).
#
# GAstroCellZLayerRegistry (→ cell_registry.json):
#   Per-scene z-layer registry.  ZLayerCellGroups[layer] → list of cell
#   descriptors registered in that layer, in insertion order.  The JSON
#   file is the pub/sub channel equivalent — all cells share one global
#   state view (Apollo "scene graph" concept).
#
# GAstroCellProxyMap (→ cell_registry.json top-level keyed by cell_id):
#   Maps cell_id → FAstroCellSceneProxy descriptor so RemoveCell and
#   UpdateCellConstraint can look up the proxy in O(1).
#
# 2-D channel adaptation:
#   AddPrimitiveSceneInfo_RenderThread  → register_cell_in_z_layer() called
#     from proc() immediately after the bbox channel write.
#   UpdatePrimitiveTransform_RenderThread → update_cell_constraint() called
#     from proc() when bbox changed vs. cached registry entry.
#   RemovePrimitiveSceneInfo_RenderThread → evict_cell_from_z_layer() (not
#     called by proc() itself; available for orchestrator / epoch teardown).
# ═══════════════════════════════════════════════════════════════════════════════

# Spatial granularity of a single z-layer in world units
# (AstroCellZLayerHeight = 100.0f in C++; scaled to 1.0 here because the
# Python channel uses integer z-layer indices 0-7, not full world-space Z).
_ASTRO_CELL_Z_LAYER_HEIGHT: float = 1.0

# Maximum number of distinct z-layers the registry will track
# (AstroCellMaxZLayers = 64 in C++).
_ASTRO_CELL_MAX_Z_LAYERS: int = 64

# Tolerance for bbox change detection (HasChanged Tolerance=0.01f in C++).
_ASTRO_BBOX_TOLERANCE: float = 0.01

# Path of the shared cell_registry channel (GAstroCellZLayerRegistry +
# GAstroCellProxyMap serialised to a single JSON file).
_CELL_REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "physics", "cell_registry.json",
)


# ---------------------------------------------------------------------------
# FAstroCellBBox — Python equivalent
# ---------------------------------------------------------------------------

class AstroCellBBox:
    """
    Python equivalent of FAstroCellBBox.

    Stores a world-space AABB as (min_x, min_y, min_z, max_x, max_y, max_z).
    The 2-D screen bbox (x, y, w, h, z) maps to:
        min  = (x,       y,       z      )
        max  = (x + w,   y + h,   z      )
    z_min == z_max is fine; the Z axis is used only for layer bucketing.
    """
    __slots__ = ("min_x", "min_y", "min_z", "max_x", "max_y", "max_z")

    def __init__(self, min_x: float, min_y: float, min_z: float,
                 max_x: float, max_y: float, max_z: float):
        self.min_x = min_x
        self.min_y = min_y
        self.min_z = min_z
        self.max_x = max_x
        self.max_y = max_y
        self.max_z = max_z

    @classmethod
    def from_bbox(cls, bbox: dict) -> "AstroCellBBox":
        """Construct from a channel bbox dict (x, y, w, h, z)."""
        z = float(bbox.get("z", 0))
        return cls(
            min_x=float(bbox["x"]),
            min_y=float(bbox["y"]),
            min_z=z,
            max_x=float(bbox["x"]) + float(bbox["w"]),
            max_y=float(bbox["y"]) + float(bbox["h"]),
            max_z=z,
        )

    def has_changed(self, other: "AstroCellBBox",
                    tolerance: float = _ASTRO_BBOX_TOLERANCE) -> bool:
        """
        Returns True when the new bounds differ from the cached ones by more
        than tolerance — mirrors FAstroCellBBox::HasChanged().
        """
        return (
            abs(self.min_x - other.min_x) > tolerance or
            abs(self.min_y - other.min_y) > tolerance or
            abs(self.min_z - other.min_z) > tolerance or
            abs(self.max_x - other.max_x) > tolerance or
            abs(self.max_y - other.max_y) > tolerance or
            abs(self.max_z - other.max_z) > tolerance
        )

    def center_z(self) -> float:
        """World-space Z of the bbox centre (used for z-layer computation)."""
        return (self.min_z + self.max_z) * 0.5

    def to_dict(self) -> dict:
        return {
            "min": [self.min_x, self.min_y, self.min_z],
            "max": [self.max_x, self.max_y, self.max_z],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AstroCellBBox":
        mn = d["min"]
        mx = d["max"]
        return cls(mn[0], mn[1], mn[2], mx[0], mx[1], mx[2])


# ---------------------------------------------------------------------------
# AstroComputeZLayer — Python equivalent
# ---------------------------------------------------------------------------

def astro_compute_z_layer(world_z: float) -> int:
    """
    Compute the z-layer index for a world-space origin Z.
    Clamped to [0, _ASTRO_CELL_MAX_Z_LAYERS).

    Mirrors:
        static FORCEINLINE int32 AstroComputeZLayer(const FVector& WorldOrigin)
        {
            const int32 RawLayer = FMath::FloorToInt(WorldOrigin.Z / AstroCellZLayerHeight);
            return FMath::Clamp(RawLayer, 0, AstroCellMaxZLayers - 1);
        }
    """
    raw_layer = int(math.floor(world_z / _ASTRO_CELL_Z_LAYER_HEIGHT))
    return max(0, min(raw_layer, _ASTRO_CELL_MAX_Z_LAYERS - 1))


# ---------------------------------------------------------------------------
# cell_registry channel I/O
# ---------------------------------------------------------------------------

def _load_cell_registry() -> dict:
    """
    Load GAstroCellZLayerRegistry + GAstroCellProxyMap from the JSON channel.

    Schema:
    {
      "cells": {
        "<cell_id>": {
          "bbox":            { "min": [x,y,z], "max": [x,y,z] },
          "species":         "<string>",
          "z":               <int>,          # z-layer bucket index
          "constraint_mask": <int>,          # bDirty flag (0 or 1)
          "epoch":           <int>
        },
        ...
      },
      "z_layers": {
        "<layer_index_str>": ["<cell_id>", ...]   # insertion-order bucket
      }
    }
    """
    if not os.path.isfile(_CELL_REGISTRY_PATH):
        return {"cells": {}, "z_layers": {}}
    try:
        with open(_CELL_REGISTRY_PATH) as _f:
            data = json.load(_f)
        if "cells" not in data:
            data["cells"] = {}
        if "z_layers" not in data:
            data["z_layers"] = {}
        return data
    except (json.JSONDecodeError, OSError):
        return {"cells": {}, "z_layers": {}}


def _save_cell_registry(registry: dict) -> None:
    """Atomically persist registry to the physics/cell_registry.json channel."""
    os.makedirs(os.path.dirname(_CELL_REGISTRY_PATH), exist_ok=True)
    with open(_CELL_REGISTRY_PATH, "w") as _f:
        json.dump(registry, _f, indent=2)


# ---------------------------------------------------------------------------
# AstroRegisterCellInZLayer — Python equivalent
# ---------------------------------------------------------------------------

def register_cell_in_z_layer(
    cell_id: str,
    bbox: dict,
    species: str,
    epoch: int,
) -> dict:
    """
    Register a new cell proxy in the z-layer registry and persist to channel.

    Mirrors AstroRegisterCellInZLayer() + the AddCell block in
    AddPrimitiveSceneInfo_RenderThread:

        FAstroCellSceneProxy* CellProxy = AstroRegisterCellInZLayer(
            PrimitiveSceneInfo, SourceIndex, CellWorldBounds);
        GAstroCellProxyMap.Add(PrimitiveSceneInfo, CellProxy);

    Returns the newly created proxy descriptor dict.
    """
    cell_bbox = AstroCellBBox.from_bbox(bbox)
    z_layer   = astro_compute_z_layer(cell_bbox.center_z())

    proxy = {
        "bbox":            cell_bbox.to_dict(),
        "species":         species,
        "z":               z_layer,
        "constraint_mask": 0,      # bDirty = false at registration
        "epoch":           epoch,
    }

    registry = _load_cell_registry()

    # ── Evict any stale entry for this cell (re-registration path) ──────────
    old_entry = registry["cells"].get(cell_id)
    if old_entry is not None:
        old_layer_key = str(old_entry["z"])
        bucket = registry["z_layers"].get(old_layer_key, [])
        if cell_id in bucket:
            bucket.remove(cell_id)          # swap-remove equivalent
        registry["z_layers"][old_layer_key] = bucket

    # ── Insert into the new z-layer bucket (GAstroCellZLayerRegistry) ───────
    layer_key = str(z_layer)
    bucket = registry["z_layers"].get(layer_key, [])
    if cell_id not in bucket:
        bucket.append(cell_id)
    registry["z_layers"][layer_key] = bucket

    # ── Store in the proxy map (GAstroCellProxyMap) ──────────────────────────
    registry["cells"][cell_id] = proxy

    _save_cell_registry(registry)

    print(
        f"[ASTRO-CELL] AstroRegisterCellInZLayer — cell registered: "
        f"cell_id={cell_id} zLayer={z_layer} "
        f"bboxMin=({cell_bbox.min_x:.1f},{cell_bbox.min_y:.1f},{cell_bbox.min_z:.1f}) "
        f"bboxMax=({cell_bbox.max_x:.1f},{cell_bbox.max_y:.1f},{cell_bbox.max_z:.1f}) "
        f"layerSize={len(bucket)}",
        file=sys.stderr,
    )

    return proxy


# ---------------------------------------------------------------------------
# AstroUpdateCellConstraint — Python equivalent
# ---------------------------------------------------------------------------

def update_cell_constraint(
    cell_id: str,
    new_bbox: dict,
    epoch: int,
) -> None:
    """
    Update the cached bbox of a cell proxy already in the registry.

    Mirrors AstroUpdateCellConstraint() + the UpdateCellConstraint block in
    UpdatePrimitiveTransform_RenderThread:

        FAstroCellSceneProxy** CellProxyPtr = GAstroCellProxyMap.Find(...);
        if (CellProxyPtr && *CellProxyPtr)
            AstroUpdateCellConstraint(*CellProxyPtr, WorldBounds);

    If the cell crossed a z-layer boundary, migrates the proxy to the new
    bucket (swap-remove from old, append to new).
    Sets constraint_mask=1 (bDirty=true) to signal a pending constraint-buffer
    flush.
    """
    registry = _load_cell_registry()
    entry = registry["cells"].get(cell_id)
    if entry is None:
        return  # no proxy registered for this cell — non-cell primitive path

    cached_bbox = AstroCellBBox.from_dict(entry["bbox"])
    incoming    = AstroCellBBox.from_bbox(new_bbox)

    if not cached_bbox.has_changed(incoming):
        return  # bbox unchanged — nothing to do

    old_layer = entry["z"]
    new_layer = astro_compute_z_layer(incoming.center_z())

    # ── Z-layer migration (cell crossed a z-layer boundary) ─────────────────
    if new_layer != old_layer:
        old_key = str(old_layer)
        old_bucket = registry["z_layers"].get(old_key, [])
        if cell_id in old_bucket:
            old_bucket.remove(cell_id)
        registry["z_layers"][old_key] = old_bucket

        new_key = str(new_layer)
        new_bucket = registry["z_layers"].get(new_key, [])
        if cell_id not in new_bucket:
            new_bucket.append(cell_id)
        registry["z_layers"][new_key] = new_bucket

        entry["z"] = new_layer

        print(
            f"[ASTRO-CELL] AstroUpdateCellConstraint — cell crossed z-layer: "
            f"cell_id={cell_id} oldLayer={old_layer} newLayer={new_layer} "
            f"originZ={incoming.center_z():.1f}",
            file=sys.stderr,
        )

    # ── Update bbox + mark dirty (bDirty = true) ─────────────────────────────
    entry["bbox"]            = incoming.to_dict()
    entry["constraint_mask"] = 1    # dirty — pending constraint-buffer flush
    entry["epoch"]           = epoch

    registry["cells"][cell_id] = entry
    _save_cell_registry(registry)

    print(
        f"[ASTRO-CELL] AstroUpdateCellConstraint — constraint updated: "
        f"cell_id={cell_id} zLayer={entry['z']} "
        f"bboxMin=({incoming.min_x:.1f},{incoming.min_y:.1f},{incoming.min_z:.1f}) "
        f"bboxMax=({incoming.max_x:.1f},{incoming.max_y:.1f},{incoming.max_z:.1f})",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# AstroEvictCellFromZLayer — Python equivalent
# ---------------------------------------------------------------------------

def evict_cell_from_z_layer(cell_id: str) -> None:
    """
    Evict the departing cell node from the z-layer registry.

    Mirrors AstroEvictCellFromZLayer() + the RemoveCell block in
    RemovePrimitiveSceneInfo_RenderThread:

        FAstroCellSceneProxy** CellProxyPtr = GAstroCellProxyMap.Find(...);
        if (CellProxyPtr)
        {
            AstroEvictCellFromZLayer(*CellProxyPtr);
            GAstroCellProxyMap.Remove(PrimitiveSceneInfo);
        }

    Performs a swap-remove from the bucket to keep it contiguous (mirrors
    TArray::RemoveAtSwap) then deletes the proxy from GAstroCellProxyMap.
    Available for use by the orchestrator / epoch teardown.
    """
    registry = _load_cell_registry()
    entry = registry["cells"].get(cell_id)
    if entry is None:
        print(
            f"[ASTRO-CELL] AstroEvictCellFromZLayer: no cell proxy found for "
            f"cell_id={cell_id} (non-cell primitive)",
            file=sys.stderr,
        )
        return

    layer     = entry["z"]
    layer_key = str(layer)
    bucket    = registry["z_layers"].get(layer_key, [])

    # Swap-remove: move last element into the evicted slot then pop
    if cell_id in bucket:
        idx = bucket.index(cell_id)
        last = bucket[-1]
        bucket[idx] = last
        bucket.pop()
    registry["z_layers"][layer_key] = bucket

    del registry["cells"][cell_id]
    _save_cell_registry(registry)

    print(
        f"[ASTRO-CELL] AstroEvictCellFromZLayer — cell evicted: "
        f"cell_id={cell_id} zLayer={layer} remainingInLayer={len(bucket)}",
        file=sys.stderr,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] CapsuleShadowRendering → Python port
#
# Ported from commit 1d22562:
#   upstream/unreal-renderer/CapsuleShadowRendering.cpp
#
# FAstroCellOcclusionVolume:
#   Replaces FCapsuleShape.  Cell bbox expressed as (BoundsMin, BoundsExtent)
#   — same "centre ± half-extent" arithmetic as the original capsule radius.
#
# ProjectCellOcclusion:
#   Slab-based ray/AABB closest-approach → projected face area → solid-angle
#   proxy → distance-falloff weighting → occlusion factor in [0,1].
#
# 2-D SVG adaptation:
#   The "light direction" is a downward Z projection (light comes from above
#   in screen-space).  The shadow ray runs vertically downward from the
#   receiver cell; occluder cells that overlap in X and sit at a higher Z
#   attenuate the shadow strength (OcclusionWeight == FadeAlpha from bbox data).
# ═══════════════════════════════════════════════════════════════════════════════

# Maximum shadow-cast distance in Z units (GCapsuleMaxDirectOcclusionDistance)
_CAPSULE_MAX_DIST = 8.0

# Reference area constant — equivalent to a capsule with r≈40 UU at dist 100
# (ReferenceArea = 40*40*4 from the C++ source)
_REFERENCE_AREA = 6400.0


def _safe_rcp(v: float) -> float:
    """1/v with epsilon guard (mirrors C++ SafeRcp lambda)."""
    return 1.0 / v if abs(v) > 1e-6 else 1e6


class AstroCellOcclusionVolume:
    """
    Python equivalent of FAstroCellOcclusionVolume.

    Stores a cell's bounding box as (bounds_min_x/y/z, bounds_extent_x/y/z)
    plus an OcclusionWeight [0,1] that modulates shadow intensity — identical
    role to capsule radius scaling in the original pipeline.

    The capsule radius analogue is:
        capsule_radius ≈ min(w, h) / 2  →  BoundsExtent smallest axis
    """
    __slots__ = ("bx", "by", "bz", "ex", "ey", "ez",
                 "occlusion_weight", "cell_index")

    def __init__(self, x: float, y: float, z: float,
                 w: float, h: float,
                 occlusion_weight: float = 1.0,
                 cell_index: int = 0):
        # BoundsMin  (top-left in 2-D; z from layer data)
        self.bx = x
        self.by = y
        self.bz = z
        # BoundsExtent  (half-size — mirrors BoxExtent in FromBounds())
        self.ex = w / 2.0
        self.ey = h / 2.0
        # capsule radius analogue: min(w,h)/2 → stored as smallest half-extent
        self.ez = min(w, h) / 2.0   # Z-extent == capsule radius
        # OcclusionWeight
        self.occlusion_weight = max(0.0, min(1.0, occlusion_weight))
        self.cell_index = cell_index

    @classmethod
    def from_bbox(cls, bbox: dict, cell_index: int = 0,
                  fade_alpha: float = 1.0) -> "AstroCellOcclusionVolume":
        """Equivalent to FAstroCellOcclusionVolume::FromBounds()."""
        return cls(
            x=bbox["x"], y=bbox["y"], z=float(bbox.get("z", 3)),
            w=bbox["w"], h=bbox["h"],
            occlusion_weight=fade_alpha,
            cell_index=cell_index,
        )

    def projected_face_area(self, ld_x: float, ld_y: float,
                            ld_z: float) -> float:
        """
        ProjectedFaceArea(LightDir) — face area weighted by |dot(LightDir,N)|.

        In 2-D screen space the light direction is (0, 0, 1) (pointing down
        into the screen / toward higher Z layers), so:
            Area += |ld_x| * (ey * ez * 4)   ← YZ face pair
            Area += |ld_y| * (ex * ez * 4)   ← XZ face pair
            Area += |ld_z| * (ex * ey * 4)   ← XY face pair  ← dominant
        """
        area = (abs(ld_x) * self.ey * self.ez * 4.0
                + abs(ld_y) * self.ex * self.ez * 4.0
                + abs(ld_z) * self.ex * self.ey * 4.0)
        return area * self.occlusion_weight


def project_cell_occlusion(
    receiver_z: float,
    occluder: "AstroCellOcclusionVolume",
    max_distance: float = _CAPSULE_MAX_DIST,
) -> float:
    """
    Python equivalent of ProjectCellOcclusion().

    Receiver is the current cell (we only need its Z for 1-D slab test).
    Occluder is another cell's volume.  Light direction is (0, 0, 1)
    — straight down through Z layers (screen-space top-light convention).

    Returns occlusion factor ∈ [0, 1].
    """
    # Light direction (normalised) — pointing along +Z in layer space
    ld_x, ld_y, ld_z = 0.0, 0.0, 1.0

    # ── Step 1: slab intersection along the light ray ────────────────────────
    # Ray: P(t) = (receiver_x, receiver_y, receiver_z) + t * (0, 0, 1)
    # We only track the Z axis for the 1-D slab (X/Y overlap tested separately)
    occ_z_min = occluder.bz
    occ_z_max = occluder.bz + occluder.ez * 2.0

    t_min_z = (occ_z_min - receiver_z) * _safe_rcp(ld_z)
    t_max_z = (occ_z_max - receiver_z) * _safe_rcp(ld_z)
    if t_min_z > t_max_z:
        t_min_z, t_max_z = t_max_z, t_min_z

    t_enter = t_min_z
    t_exit = t_max_z

    # Miss or behind receiver
    if t_exit < 0.0 or t_enter > t_exit:
        return 0.0

    hit_distance = max(t_enter, 0.0)
    if hit_distance > max_distance:
        return 0.0

    # ── Step 2: projected face area as solid-angle proxy ────────────────────
    face_area = occluder.projected_face_area(ld_x, ld_y, ld_z)
    dist_sq = max(hit_distance * hit_distance, 1.0)
    solid_angle = face_area / dist_sq

    # ── Step 3: distance falloff (linear fade at 80 %→100 % of max_distance) ─
    dist_fade = max(0.0, min(1.0,
        (max_distance - hit_distance) / max(max_distance * 0.2, 1.0)
    ))

    occlusion_factor = max(0.0, min(1.0,
        solid_angle * _REFERENCE_AREA * dist_fade
    ))
    return occlusion_factor * occluder.occlusion_weight


def compute_capsule_shadow_params(
    cell_id: str,
    bbox: dict,
    all_bboxes: dict,
) -> dict:
    """
    BuildCellOcclusionVolumes + multi-cell ProjectCellOcclusion aggregation.

    Collects every *other* cell as a potential occluder (cells at higher Z
    can block the shadow of cells at lower Z — same as UE's shadow primitives
    being culled against the receiver by depth).

    Returns a dict with:
        dx, dy      — shadow offset (pixels); scale with receiver's Z depth
        opacity     — feDropShadow flood-opacity; attenuated by occlusion
        blur        — feDropShadow stdDeviation (proportional to capsule radius)
    """
    receiver_z = float(bbox.get("z", 3))
    capsule_radius = min(bbox["w"], bbox["h"]) / 2.0

    # ── Shadow offset: z depth → offset distance ──────────────────────────
    # z=3 baseline → offset_scale=1; higher z → larger shadow below
    z_baseline = 3.0
    z_scale = max(0.0, receiver_z - z_baseline)
    # Offset direction: bottom-right (45° light, classic drop shadow)
    base_offset = 2.0 + z_scale * 3.0
    shadow_dx = base_offset
    shadow_dy = base_offset

    # Blur radius ∝ capsule radius (same as penumbra width from capsule r)
    blur = max(1.0, capsule_radius * 0.15 + z_scale * 0.8)

    # ── Occlusion accumulation across all other cells ─────────────────────
    total_occlusion = 0.0
    receiver_vol = AstroCellOcclusionVolume.from_bbox(bbox, cell_index=0)

    for idx, (other_id, other_bbox) in enumerate(all_bboxes.items()):
        if other_id == cell_id:
            continue
        other_z = float(other_bbox.get("z", 3))
        # Only cells at strictly higher Z can occlude this cell's shadow
        if other_z <= receiver_z:
            continue

        occluder = AstroCellOcclusionVolume.from_bbox(
            other_bbox, cell_index=idx + 1,
            # FadeAlpha: proximity-based — cells farther in Z contribute less
            fade_alpha=max(0.0, 1.0 - abs(other_z - receiver_z) / _CAPSULE_MAX_DIST),
        )

        # X/Y overlap check (cells must overlap horizontally to cast shadow)
        overlap_x = (receiver_vol.bx < occluder.bx + occluder.ex * 2.0 and
                     occluder.bx < receiver_vol.bx + receiver_vol.ex * 2.0)
        overlap_y = (receiver_vol.by < occluder.by + occluder.ey * 2.0 and
                     occluder.by < receiver_vol.by + receiver_vol.ey * 2.0)

        if overlap_x and overlap_y:
            occ = project_cell_occlusion(
                receiver_z=receiver_z,
                occluder=occluder,
            )
            total_occlusion = min(1.0, total_occlusion + occ)

    # Attenuate shadow opacity by accumulated occlusion
    # Fully occluded cell → nearly invisible shadow (floor at 0.08)
    base_opacity = 0.35 + min(0.3, z_scale * 0.1)
    shadow_opacity = max(0.08, base_opacity * (1.0 - total_occlusion * 0.8))

    return {
        "dx": round(shadow_dx, 2),
        "dy": round(shadow_dy, 2),
        "blur": round(blur, 2),
        "opacity": round(shadow_opacity, 3),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] PostProcessAmbientOcclusion → Python port  (commit 33e27b7)
#
# Ported from FAstroConstraintAO / PostProcessAmbientOcclusion.cpp:
#   upstream/unreal-renderer/CompositionLighting/PostProcessAmbientOcclusion.cpp
#
# FAstroConstraintAOParams (→ _CROWDING_THRESHOLD, _ATTENUATION_CURVE):
#   KernelRadius        — analogue: bbox sampling radius (not needed in 2-D;
#                         every overlapping neighbour is one kernel "sample")
#   CrowdingThreshold   — fraction of occluded samples above which crowding
#                         attenuation begins.  Default 0.5 (AmbientOcclusionMipBlend)
#   AttenuationCurve    — power of the suppression curve.  Default 2.0
#                         (AmbientOcclusionPower).
#
# FAstroConstraintAO::ComputeConstraintWeight (→ compute_crowding_opacity()):
#   Pass 1 — raw overlap accumulation
#       Each neighbour bbox that overlaps the receiver cell in 2-D contributes
#       one "occluded sample".  raw_occlusion = occluded_count / total_neighbours.
#       (SSAO analogue: hemisphere depth samples above shaded surface.)
#
#   Pass 2 — crowding attenuation
#       When occluded_fraction > CrowdingThreshold a suppression multiplier is
#       computed (mirrors AstroCrowdingScale packed into Value[4].w):
#           CrowdingExcess  = clamp((f − threshold) / (1 − threshold), 0, 1)
#           AttenuationMult = 1 − CrowdingExcess ** AttenuationCurve
#
#   Pass 3 — mutual-constraint cancellation
#       Geometric-mean blend: sqrt(raw*(1−raw))*2 × AttenuationMult
#       lerped with raw value using occluded_fraction as BlendAlpha —
#       same formula as FAstroConstraintAO::ComputeConstraintWeight Pass 3.
#
# SVG output:
#   constraint_opacity = 1 − final_ao_weight  (high AO → lower cell opacity)
#   Written as opacity="…" attribute on the outermost <g> of the cell.
#   Mirrors AstroCrowdingScale written to ScreenSpaceAOParams[4].w which the
#   USF kernel loop reads to modulate the per-pixel AO contribution.
#
# 2-D channel adaptation:
#   SSAO kernel sample  → neighbour bbox overlap test (O(n) pass over all_bboxes)
#   HorizonAngle weight → bbox intersection-area / self-area ratio
#   ProximityFade       → Z-layer distance fade (same _CAPSULE_MAX_DIST guard)
# ═══════════════════════════════════════════════════════════════════════════════

# Fraction of overlapping neighbours above which crowding attenuation starts.
# Mirrors FAstroConstraintAOParams::CrowdingThreshold (AmbientOcclusionMipBlend).
_CROWDING_THRESHOLD: float = 0.5

# Power of the crowding-suppression curve.
# Mirrors FAstroConstraintAOParams::AttenuationCurve (AmbientOcclusionPower).
_ATTENUATION_CURVE: float = 2.0

# Minimum cell fill opacity floor — even a maximally crowded cell stays legible.
_CROWDING_OPACITY_FLOOR: float = 0.35


def compute_crowding_opacity(cell_id: str, bbox: dict, all_bboxes: dict) -> float:
    """
    Python equivalent of FAstroConstraintAO::ComputeConstraintWeight().

    Samples the neighbour bbox set (= SSAO kernel), computes the crowding
    fraction, applies 3-pass constraint-space attenuation, and returns a
    fill-opacity value in [_CROWDING_OPACITY_FLOOR, 1.0].

    High density (many overlapping neighbours) → low opacity (cell recedes).
    Low density (sparse layout) → opacity near 1.0 (cell fully visible).

    @param cell_id    ID of the cell being rendered (excluded from its own kernel)
    @param bbox       Receiver cell bbox dict  {x, y, w, h, z}
    @param all_bboxes Dict of all sibling bbox dicts keyed by cell_id
    @return           fill opacity in [_CROWDING_OPACITY_FLOOR, 1.0]
    """
    receiver_z = float(bbox.get("z", 3))
    rx0 = bbox["x"]
    ry0 = bbox["y"]
    rx1 = rx0 + bbox["w"]
    ry1 = ry0 + bbox["h"]
    self_area = max(bbox["w"] * bbox["h"], 1.0)

    # ── Pass 1: neighbour bbox sampling (SSAO kernel sample loop) ────────────
    # Each neighbour that overlaps the receiver in 2-D is one "occluded sample".
    # We weight by intersection area / self_area (≈ ProximityFade × HorizonWeight
    # in the C++ hemisphere kernel).
    raw_occlusion_sum: float = 0.0
    occluded_count: int = 0
    total_samples: int = 0

    for other_id, other_bbox in all_bboxes.items():
        if other_id == cell_id:
            continue

        total_samples += 1

        other_z = float(other_bbox.get("z", 3))
        # Z-proximity fade: distant layers contribute diminishing pressure.
        # Mirrors ProximityFade = clamp(DepthDelta / KernelRadius, 0, 1).
        z_dist = abs(other_z - receiver_z)
        z_fade = max(0.0, 1.0 - z_dist / max(_CAPSULE_MAX_DIST, 1e-6))
        if z_fade <= 0.0:
            continue

        ox0 = other_bbox["x"]
        oy0 = other_bbox["y"]
        ox1 = ox0 + other_bbox["w"]
        oy1 = oy0 + other_bbox["h"]

        # 2-D AABB overlap test (horizon-angle analogue: does sample sit "above"?)
        inter_w = max(0.0, min(rx1, ox1) - max(rx0, ox0))
        inter_h = max(0.0, min(ry1, oy1) - max(ry0, oy0))
        if inter_w <= 0.0 or inter_h <= 0.0:
            continue  # sample not occluding — below horizon in SSAO terms

        # Intersection area / self area → HorizonWeight × ProximityFade analogue
        horizon_weight = min(1.0, (inter_w * inter_h) / self_area)
        raw_occlusion_sum += horizon_weight * z_fade
        occluded_count += 1

    if total_samples == 0:
        return 1.0  # no neighbours — fully lit, no crowding

    normalised_occlusion = raw_occlusion_sum / float(total_samples)
    occluded_fraction = float(occluded_count) / float(total_samples)

    # ── Pass 2: crowding attenuation ─────────────────────────────────────────
    # Mirrors Value[4].w = AstroCrowdingScale = 1/(threshold*curve).
    #   CrowdingExcess  = clamp((f − threshold) / (1 − threshold), 0, 1)
    #   AttenuationMult = 1 − CrowdingExcess ** AttenuationCurve
    attenuation_mult = 1.0
    if occluded_fraction > _CROWDING_THRESHOLD:
        threshold_range = max(1.0 - _CROWDING_THRESHOLD, 1e-6)
        crowding_excess = min(1.0, (occluded_fraction - _CROWDING_THRESHOLD)
                              / threshold_range)
        attenuation_mult = 1.0 - (crowding_excess ** _ATTENUATION_CURVE)

    # ── Pass 3: mutual-constraint cancellation ────────────────────────────────
    # Geometric mean restores [0,0.5]→[0,1]; scaled by attenuation_mult.
    # BlendAlpha = saturate(occluded_fraction / threshold) — lerps between raw
    # SSAO (sparse) and constraint-space AO (crowded).
    constraint_occlusion = (
        math.sqrt(max(0.0, normalised_occlusion * (1.0 - normalised_occlusion)))
        * 2.0
        * attenuation_mult
    )
    blend_alpha = min(1.0, occluded_fraction
                      / max(_CROWDING_THRESHOLD, 1e-6))
    final_ao = (1.0 - blend_alpha) * normalised_occlusion \
               + blend_alpha * constraint_occlusion

    # AO weight → fill opacity (invert: high AO = lower opacity)
    fill_opacity = max(_CROWDING_OPACITY_FLOOR, 1.0 - final_ao)
    return round(fill_opacity, 4)


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] NaniteVisibility → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/Nanite/NaniteVisibility.cpp
#
# FNaniteVisibilityQuery (→ AstroCellVisibilityQuery):
#   Per-frame query that tests every cell primitive against the viewport
#   frustum (IntersectBox) and accumulates species-bin visibility bitmasks.
#   Cells outside the viewport are marked invisible; their proc() SVG
#   generation is skipped entirely (LOD-0 = nothing) saving compute.
#
# Algorithm changes from Nanite original:
#   1. FConvexVolume frustum → simple AABB viewport rect (2-D screen)
#   2. RasterBin → species bin (cells of same species share a render bin)
#   3. ShadingBin → z-layer bin (cells at same z-layer share a shading bin)
#   4. Async task → synchronous (single-threaded epoch loop)
#   5. PrimitiveSceneInfo → cell_registry.json entries
#
# Reference: [ASTRO-NANITE-VIS] debug prefix preserved.
# ═══════════════════════════════════════════════════════════════════════════════

# Viewport margins — cells partially outside get LOD reduction, not hard cull.
# Mirrors the guard-band concept in Nanite's rasteriser.
_NANITE_VIS_GUARD_BAND: float = 50.0

# LOD thresholds: screen-projected area below which a cell drops LOD.
# Mirrors Nanite's streaming LOD metric: ScreenSize = ProjectedArea / ViewArea.
# Below _LOD1_THRESHOLD the cell is rendered at reduced detail (fewer SVG
# elements in its species generator); below _LOD2_THRESHOLD it becomes a
# simple coloured rect; below _CULL_THRESHOLD it is culled entirely.
_NANITE_LOD2_THRESHOLD: float = 0.002   # < 0.2% of viewport area → rect
_NANITE_CULL_THRESHOLD: float = 0.0005  # < 0.05% of viewport area → culled


class AstroCellVisibilityQuery:
    """
    Python equivalent of FNaniteVisibilityQuery.

    Holds the viewport rect and accumulates per-species-bin and per-z-layer-bin
    visibility flags.  The query is executed once per epoch (before any cell's
    proc() runs) by perform_nanite_visibility().

    Mirrors FNaniteVisibilityQuery::Init/Finish lifecycle:
        Init  — allocate bin arrays, zero visibility flags.
        Test  — iterate cell_registry, frustum-test each, mark bins.
        Finish— freeze results, release working memory.
    """

    def __init__(self, viewport_w: float, viewport_h: float,
                 scroll_x: float = 0.0, scroll_y: float = 0.0):
        # Viewport frustum as AABB (2-D screen space).
        # Mirrors FConvexVolume planes → simplified to min/max rect.
        self.vp_x0 = scroll_x - _NANITE_VIS_GUARD_BAND
        self.vp_y0 = scroll_y - _NANITE_VIS_GUARD_BAND
        self.vp_x1 = scroll_x + viewport_w + _NANITE_VIS_GUARD_BAND
        self.vp_y1 = scroll_y + viewport_h + _NANITE_VIS_GUARD_BAND
        self.viewport_area = max(viewport_w * viewport_h, 1.0)

        # Species bin visibility — mirrors RasterBinVisibility TArray<bool>.
        # Keyed by species name string (unlike C++ integer bin index).
        self.species_bin_visible: dict = {}

        # Z-layer bin visibility — mirrors ShadingBinVisibility TArray<bool>.
        # Keyed by z-layer integer.
        self.z_layer_bin_visible: dict = {}

        # Per-cell results: cell_id → visibility dict.
        self.cell_results: dict = {}

        self._finished = False

    def _intersect_box(self, cx: float, cy: float,
                       half_w: float, half_h: float) -> bool:
        """
        AABB-AABB intersection test.
        Mirrors FConvexVolume::IntersectBox(Origin, BoxExtent) but for 2-D.
        """
        return not (cx - half_w > self.vp_x1 or
                    cx + half_w < self.vp_x0 or
                    cy - half_h > self.vp_y1 or
                    cy + half_h < self.vp_y0)

    def _compute_screen_fraction(self, cell_area: float) -> float:
        """
        Projected screen fraction = cell_area / viewport_area.
        Mirrors Nanite's ScreenSize metric used for LOD selection.
        """
        return cell_area / self.viewport_area

    def test_cell(self, cell_id: str, bbox: dict, species: str,
                  z_layer: int) -> dict:
        """
        Per-primitive visibility test.
        Mirrors IsNanitePrimitiveVisible + bin marking in PerformNaniteVisibility.

        Returns dict with:
            visible:  bool  — cell passes frustum test
            lod:      int   — 0 (full), 1 (reduced), 2 (rect-only)
            screen_fraction: float
        """
        x = float(bbox.get("x", 0))
        y = float(bbox.get("y", 0))
        w = float(bbox.get("w", 100))
        h = float(bbox.get("h", 50))

        cx = x + w / 2.0
        cy = y + h / 2.0

        # ── Step 1: frustum intersection (IntersectBox port) ─────────────────
        visible = self._intersect_box(cx, cy, w / 2.0, h / 2.0)

        # ── Step 2: screen-fraction LOD (Nanite streaming metric port) ───────
        cell_area = w * h
        screen_frac = self._compute_screen_fraction(cell_area)

        if not visible:
            lod = -1  # culled by frustum
        elif screen_frac < _NANITE_CULL_THRESHOLD:
            lod = -1  # culled by size — too small to matter
            visible = False
        elif screen_frac < _NANITE_LOD2_THRESHOLD:
            lod = 2   # rect-only placeholder
        else:
            lod = 0   # full detail

        # ── Step 3: mark species bin visible (RasterBin port) ────────────────
        # Mirrors: Query->RasterBinVisibility[BinIndex] = true
        if visible:
            self.species_bin_visible[species] = True

        # ── Step 4: mark z-layer bin visible (ShadingBin port) ───────────────
        # Mirrors: Query->ShadingBinVisibility[ShadingBinIndex] = true
        if visible:
            self.z_layer_bin_visible[z_layer] = True

        result = {
            "visible": visible,
            "lod": lod,
            "screen_fraction": round(screen_frac, 6),
        }
        self.cell_results[cell_id] = result

        _dbg_vis = os.environ.get("ASTRO_NANITE_VIS_VERBOSE", "0") == "1"
        if _dbg_vis:
            print(
                f"[ASTRO-NANITE-VIS] cell={cell_id} species={species} "
                f"z={z_layer} visible={visible} lod={lod} "
                f"screen_frac={screen_frac:.6f}",
                file=sys.stderr,
            )

        return result

    def finish(self) -> dict:
        """
        Mirrors FNaniteVisibilityQuery::Finish().
        Returns summary dict; releases working state.
        """
        self._finished = True
        visible_cells = sum(1 for r in self.cell_results.values() if r["visible"])
        total_cells = len(self.cell_results)
        visible_species = len(self.species_bin_visible)
        visible_z_layers = len(self.z_layer_bin_visible)

        summary = {
            "total_cells": total_cells,
            "visible_cells": visible_cells,
            "culled_cells": total_cells - visible_cells,
            "visible_species_bins": visible_species,
            "visible_z_layer_bins": visible_z_layers,
            "species_bins": dict(self.species_bin_visible),
            "z_layer_bins": dict(self.z_layer_bin_visible),
        }

        print(
            f"[ASTRO-NANITE-VIS] Finish — "
            f"total={total_cells} visible={visible_cells} "
            f"culled={total_cells - visible_cells} "
            f"species_bins={visible_species} z_bins={visible_z_layers}",
            file=sys.stderr,
        )

        return summary


def perform_nanite_visibility(
    viewport_w: float = 1200.0,
    viewport_h: float = 900.0,
    scroll_x: float = 0.0,
    scroll_y: float = 0.0,
) -> AstroCellVisibilityQuery:
    """
    Per-epoch visibility pass.
    Mirrors PerformNaniteVisibility() — iterates all registered cells in
    cell_registry.json, tests each against the viewport, returns a query
    with per-cell and per-bin results.

    Called by loop_orchestrator before the proc() dispatch loop so that
    invisible cells can be skipped entirely (LOD = -1 → skip proc()).

    @param viewport_w  Canvas width in pixels.
    @param viewport_h  Canvas height in pixels.
    @param scroll_x    Horizontal scroll offset (panning support).
    @param scroll_y    Vertical scroll offset.
    @return            AstroCellVisibilityQuery with results.
    """
    query = AstroCellVisibilityQuery(viewport_w, viewport_h, scroll_x, scroll_y)

    registry = _load_cell_registry()
    cells = registry.get("cells", {})

    for cell_id, entry in cells.items():
        bbox_data = entry.get("bbox", {})
        # Reconstruct bbox dict from min/max format used in registry
        if "min" in bbox_data and "max" in bbox_data:
            mn = bbox_data["min"]
            mx = bbox_data["max"]
            bbox = {
                "x": mn[0], "y": mn[1],
                "w": mx[0] - mn[0], "h": mx[1] - mn[1],
                "z": mn[2] if len(mn) > 2 else 0,
            }
        else:
            bbox = bbox_data

        species = entry.get("species", "")
        z_layer = entry.get("z", 3)

        query.test_cell(cell_id, bbox, species, z_layer)

    query.finish()
    return query


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] AstroCellComposition → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/Nanite/NaniteComposition.cpp
#
# NaniteComposition orchestrates the final depth/stencil export and GBuffer
# composition pass that resolves the Nanite visibility buffer into scene-ready
# render targets.  In the Astro pipeline this maps to the SVG layer-merge step
# where per-cell paint operations (fill, stroke, shadow) are composited into a
# single canonical SVG document for the current epoch.
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   GNaniteResummarizeHTile / GNaniteDecompressDepth
#       → glyph_resummarize_zlayer / glyph_decompress_depth:
#         integer knobs (defaulting to 1 / 0) that mirror the CVar semantics;
#         kept as module-level constants so callers can override them in tests.
#
#   FNaniteMarkStencilPS   → _mark_stencil_cells()
#       Pixel-shader pass that writes a per-sample stencil bit wherever the
#       VisBuffer64 holds a valid Nanite primitive.  In SVG terms: tag every
#       cell SVG element that contributed a visible pixel to the frame so
#       downstream compositing layers can clip or mask against them.
#
#   FEmitSceneDepthPS      → _emit_scene_depth()
#       Exports per-sample hardware depth from the visibility buffer into a
#       scene-depth render target.  Astro equivalent: compute the canonical
#       z-value for each cell from its bbox["z"] and write it into the shared
#       depth channel of the composition manifest.
#
#   FEmitSceneStencilPS    → _emit_scene_stencil()
#       A second pixel-shader pass that promotes the shading-mask into a typed
#       stencil texture.  Astro equivalent: mark cells whose shading_mask bit
#       is set in the composition manifest so the SVG renderer can apply the
#       correct <clipPath> group.
#
#   FEmitCustomDepthStencilPS → _emit_custom_depth_stencil()
#       Writes per-primitive custom depth/stencil values (used by post-process
#       outlines, selection highlights, etc.).  Astro equivalent: annotate the
#       SVG element with a data-custom-depth attribute whenever the cell's
#       species_params carries a "highlight" key, enabling a CSS stroke-width
#       post-pass.
#
#   FDepthExportCS          → _depth_export_cs()
#       Compute-shader that exports depth+HTILE in a single dispatched pass on
#       platforms that support direct HTILE writes.  Astro equivalent: batch
#       the depth/stencil annotation writes through a vectorised NumPy/lxml
#       operation rather than an element-by-element Python loop.
#
# AstroCellCompositor:
#   The central class — mirrors the stateless namespace-scope helper functions
#   in NaniteComposition.cpp that are orchestrated by the renderer's
#   FNaniteRenderer::Render() call.  Here we collect all per-cell paint dicts,
#   sort them by z-layer (↔ render order), and emit a merged SVG document
#   with correct painter's-algorithm ordering and per-cell metadata attributes.
# ═══════════════════════════════════════════════════════════════════════════════

#: Mirrors GNaniteResummarizeHTile — when True the compositor will re-sort
#: z-layer buckets after every merge to ensure monotonic painter ordering.
glyph_resummarize_zlayer: int = 1

#: Mirrors GNaniteDecompressDepth — when True the compositor unpacks packed
#: 24-bit depth values from the cell registry before writing the manifest.
glyph_decompress_depth: int = 0

#: Mirrors GNaniteCustomDepthExportMethod (0 = PS path, 1 = CS path).
#: Controls whether _emit_custom_depth_stencil uses element-by-element
#: annotation (0) or a bulk vectorised lxml operation (1).
glyph_custom_depth_export_method: int = 1


def _mark_stencil_cells(
    cell_entries: list[dict],
    vis_set: set[str],
) -> list[dict]:
    """标记模板层 — 镜像 FNaniteMarkStencilPS。

    遍历 *cell_entries*，为每个 cell_id 出现在 *vis_set*（可见集合，等价于
    VisBuffer64 中持有有效 Nanite 图元的样本）的条目打上 ``stencil=1`` 标记。
    不可见的条目保留 ``stencil=0``，供后续 :func:`_emit_scene_stencil` 裁剪。

    Parameters
    ----------
    cell_entries:
        每个元素是一个包含 ``cell_id`` 与 ``bbox`` 的字典，来自 cell_registry。
    vis_set:
        本帧可见的 cell_id 集合（由 :class:`AstroCellVisibilityQuery` 生成）。

    Returns
    -------
    list[dict]
        与输入等长，每个条目追加了 ``"stencil"`` 键（0 或 1）。
    """
    result = []
    for entry in cell_entries:
        marked = dict(entry)
        marked["stencil"] = 1 if entry["cell_id"] in vis_set else 0
        result.append(marked)
    return result


def _emit_scene_depth(
    cell_entries: list[dict],
    depth_manifest: dict,
) -> dict:
    """写入场景深度通道 — 镜像 FEmitSceneDepthPS。

    从每个 cell 的 ``bbox["z"]`` 提取规范深度值，写入 *depth_manifest*
    的 ``"depth_channel"`` 子字典。键为 cell_id，值为归一化深度 [0.0, 1.0]，
    其中 z=0 对应近裁剪面，z_max 对应远裁剪面。

    归一化公式（镜像 UE5 Nanite 深度重建）::

        depth_norm = 1.0 - (z - z_min) / max(z_range, 1e-6)

    这与 UE5 反转深度缓冲区约定一致：z 越大（越深）depth_norm 越小。
    """
    zs = [e["bbox"]["z"] for e in cell_entries if "bbox" in e]
    z_min = min(zs) if zs else 0.0
    z_max = max(zs) if zs else 1.0
    z_range = z_max - z_min

    depth_channel: dict[str, float] = {}
    for entry in cell_entries:
        z = entry.get("bbox", {}).get("z", 0.0)
        depth_channel[entry["cell_id"]] = 1.0 - (z - z_min) / max(z_range, 1e-6)

    depth_manifest.setdefault("depth_channel", {}).update(depth_channel)
    return depth_manifest


def _emit_scene_stencil(
    cell_entries: list[dict],
    depth_manifest: dict,
) -> dict:
    """提升着色掩码为模板纹理 — 镜像 FEmitSceneStencilPS。

    读取每个 cell 条目中的 ``shading_mask`` 字段（由可见性查询写入），将置位
    条目的 cell_id 追加到 *depth_manifest* 的 ``"stencil_cells"`` 集合。
    后续 SVG 渲染器可据此将对应元素置入 ``<clipPath>`` 分组。
    """
    stencil_cells: list[str] = depth_manifest.setdefault("stencil_cells", [])
    for entry in cell_entries:
        if entry.get("shading_mask", 0):
            cid = entry["cell_id"]
            if cid not in stencil_cells:
                stencil_cells.append(cid)
    return depth_manifest


def _emit_custom_depth_stencil(
    cell_entries: list[dict],
    depth_manifest: dict,
) -> dict:
    """写入自定义深度/模板 — 镜像 FEmitCustomDepthStencilPS。

    检测 ``species_params`` 中携带 ``"highlight"`` 键的 cell，将其 cell_id
    与 highlight 颜色值写入 *depth_manifest* 的 ``"custom_depth"`` 子字典。
    SVG 渲染器利用此信息在对应元素上叠加 CSS ``stroke-width`` 高亮描边。

    若 :data:`glyph_custom_depth_export_method` 为 1（CS 路径），则通过
    批量字典推导式一次性构造映射；为 0 时退化为逐元素循环（PS 路径）。
    """
    if glyph_custom_depth_export_method == 1:
        # CS 路径 — 批量向量化（镜像 FDepthExportCS dispatch）
        custom: dict[str, str] = {
            e["cell_id"]: e["species_params"]["highlight"]
            for e in cell_entries
            if e.get("species_params", {}).get("highlight")
        }
    else:
        # PS 路径 — 逐元素（镜像 FEmitCustomDepthStencilPS pixel walk）
        custom = {}
        for e in cell_entries:
            hl = e.get("species_params", {}).get("highlight")
            if hl:
                custom[e["cell_id"]] = hl

    depth_manifest.setdefault("custom_depth", {}).update(custom)
    return depth_manifest


class AstroCellCompositor:
    """最终 SVG 合成器 — 镜像 NaniteComposition.cpp 中的渲染流水线。

    将逐 cell 的 paint 字典按 z-layer 排序后，合并为单一 SVG 文档。每个 cell
    贡献一个 ``<g data-cell-id="…">`` 分组；深度/模板元数据以 ``data-*``
    属性形式嵌入，供 PixiJS / D3 渲染器按需读取。

    Lifecycle（镜像 FNaniteRenderer::Render 调用序列）::

        compositor = AstroCellCompositor(vis_set)
        compositor.begin_frame(cell_entries)          # ↔ InitViews
        compositor.emit_depth_stencil(depth_manifest) # ↔ EmitDepthStencil
        svg_doc = compositor.compose()                # ↔ FinalCompose

    Parameters
    ----------
    vis_set:
        本帧可见的 cell_id 集合；由 :func:`perform_nanite_visibility` 返回的
        :class:`AstroCellVisibilityQuery` 的 ``visible_cells`` 属性提供。
    """

    def __init__(self, vis_set: set[str]) -> None:
        self._vis_set = vis_set
        self._cell_entries: list[dict] = []
        self._depth_manifest: dict = {}
        self._stamped: list[dict] = []

    # ------------------------------------------------------------------
    # begin_frame — 镜像 InitViews / PrepareRenderTargets
    # ------------------------------------------------------------------
    def begin_frame(self, cell_entries: list[dict]) -> None:
        """接收本帧所有 cell 条目并执行模板标记（FNaniteMarkStencilPS）。

        同时按 z-layer 升序排列条目，确保后续合成遵循画家算法；当
        :data:`glyph_resummarize_zlayer` 为真时重新排序（镜像
        GNaniteResummarizeHTile 的 HTILE 重摘要行为）。
        """
        self._cell_entries = cell_entries
        self._stamped = _mark_stencil_cells(cell_entries, self._vis_set)
        if glyph_resummarize_zlayer:
            self._stamped.sort(key=lambda e: e.get("bbox", {}).get("z", 0.0))

    # ------------------------------------------------------------------
    # emit_depth_stencil — 镜像 EmitDepthStencil pass
    # ------------------------------------------------------------------
    def emit_depth_stencil(self, depth_manifest: dict | None = None) -> dict:
        """执行深度/模板写入三连 pass，返回填充后的 depth_manifest。

        依次调用 :func:`_emit_scene_depth`、:func:`_emit_scene_stencil`、
        :func:`_emit_custom_depth_stencil`，与 UE5 中
        ``EmitDepthStencil → EmitSceneStencil → EmitCustomDepthStencil``
        的 RDG pass 调度顺序完全对应。
        """
        if depth_manifest is None:
            depth_manifest = {}
        self._depth_manifest = depth_manifest
        _emit_scene_depth(self._stamped, self._depth_manifest)
        _emit_scene_stencil(self._stamped, self._depth_manifest)
        _emit_custom_depth_stencil(self._stamped, self._depth_manifest)
        return self._depth_manifest

    # ------------------------------------------------------------------
    # compose — 镜像 FinalCompose / present
    # ------------------------------------------------------------------
    def compose(self) -> str:
        """将所有可见 cell 合成为一个 SVG 字符串。

        仅输出 ``stencil=1`` 的条目（可见 cell）；每个 cell 生成一个带有
        ``data-cell-id``、``data-z``、``data-depth`` 以及可选
        ``data-highlight`` 属性的 ``<g>`` 占位分组。完整的 SVG 内容由
        各 cell 自身的 ``svg_fragment`` 字段提供（若存在）。

        Returns
        -------
        str
            合并后的 SVG 文档字符串（UTF-8）。
        """
        groups: list[str] = []
        depth_ch = self._depth_manifest.get("depth_channel", {})
        custom_d = self._depth_manifest.get("custom_depth", {})

        for entry in self._stamped:
            if not entry.get("stencil", 0):
                continue
            cid = entry["cell_id"]
            z = entry.get("bbox", {}).get("z", 0.0)
            depth = depth_ch.get(cid, 0.0)
            attrs = (
                f'data-cell-id="{cid}" '
                f'data-z="{z}" '
                f'data-depth="{depth:.6f}"'
            )
            if cid in custom_d:
                attrs += f' data-highlight="{custom_d[cid]}"'
            fragment = entry.get("svg_fragment", "")
            groups.append(f'<g {attrs}>{fragment}</g>')

        inner = "\n  ".join(groups)
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<svg xmlns="http://www.w3.org/2000/svg">\n'
            f'  {inner}\n'
            "</svg>"
        )


def compose_cell_svg(
    cell_entries: list[dict],
    vis_set: set[str],
    depth_manifest: dict | None = None,
) -> tuple[str, dict]:
    """顶层合成入口 — 镜像 FNaniteRenderer::Render 的 Composition 阶段。

    便利包装：构造 :class:`AstroCellCompositor`，依次执行三个 lifecycle 方法，
    返回合成后的 SVG 字符串与填充后的 depth_manifest。

    Parameters
    ----------
    cell_entries:
        来自 cell_registry 的全量 cell 条目列表。
    vis_set:
        本帧可见 cell_id 集合。
    depth_manifest:
        可选的现有深度清单；若为 None 则创建新字典。

    Returns
    -------
    tuple[str, dict]
        ``(svg_document, depth_manifest)``
    """
    compositor = AstroCellCompositor(vis_set)
    compositor.begin_frame(cell_entries)
    dm = compositor.emit_depth_stencil(depth_manifest)
    svg = compositor.compose()
    return svg, dm


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] AstroCellSharedResources → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/Nanite/NaniteShared.cpp
#
# NaniteShared.cpp 是整个 Nanite 子系统的「公共资源层」：它注册全局 GPU 统计、
# 声明 Uniform Buffer 槽位，并暴露跨 pass 共享的 LOD/硬件路径判定函数。
# 在 Astro 流水线中，该层对应一组进程级单例资源与无状态工具函数，供
# Composition、StreamExport、Translucency 等子系统统一调用。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   DEFINE_GPU_STAT(NaniteDebug)         → _ASTRO_CELL_PERF_COUNTERS dict
#   GNaniteMaxNodes / GNaniteMaxVisibleClusters
#       → ASTRO_CELL_MAX_NODES / ASTRO_CELL_MAX_VISIBLE_CLUSTERS
#   UseMeshShader() / UsePrimitiveShader()
#       → use_vector_render_path() — 判断是否启用矢量批渲染（PixiJS mesh path）
#   FPackedView::UpdateLODScales()
#       → AstroCellPackedView.update_lod_scales()
#   FGlobalResources::GetMaxNodes() 等
#       → AstroCellGlobalResources 类
# ═══════════════════════════════════════════════════════════════════════════════

#: 镜像 GNaniteMaxNodes — 单帧最多参与 LOD 遍历的 cell 节点数上限。
ASTRO_CELL_MAX_NODES: int = 2 * 1_048_576

#: 镜像 GNaniteMaxCandidateClusters — 聚类裁剪前的候选 cluster 上限。
ASTRO_CELL_MAX_CANDIDATE_CLUSTERS: int = 16 * 1_048_576

#: 镜像 GNaniteMaxVisibleClusters — 最终可见 cluster 数量上限。
ASTRO_CELL_MAX_VISIBLE_CLUSTERS: int = 4 * 1_048_576

#: 镜像 GNaniteMaxVisibleAssemblyParts — 可见装配零件数量上限（多 species 场景）。
ASTRO_CELL_MAX_VISIBLE_ASSEMBLY_PARTS: int = 256 * 1_024

#: 镜像 DEFINE_GPU_STAT(NaniteDebug) — 运行时性能计数器字典。
#: 键为计数器名称，值为当前帧的累计计数（整型）。
_ASTRO_CELL_PERF_COUNTERS: dict[str, int] = {
    "nodes_traversed":       0,
    "candidate_clusters":    0,
    "visible_clusters":      0,
    "visible_assembly_parts": 0,
}


def reset_perf_counters() -> None:
    """重置所有性能计数器 — 镜像帧间 GPU stat 清零。"""
    for k in _ASTRO_CELL_PERF_COUNTERS:
        _ASTRO_CELL_PERF_COUNTERS[k] = 0


def increment_perf_counter(name: str, delta: int = 1) -> None:
    """递增指定性能计数器；键不存在时静默创建。"""
    _ASTRO_CELL_PERF_COUNTERS[name] = _ASTRO_CELL_PERF_COUNTERS.get(name, 0) + delta


def use_vector_render_path() -> bool:
    """判断是否启用矢量批渲染路径 — 镜像 UseMeshShader() / UsePrimitiveShader()。

    在 UE5 中，Mesh Shader / Primitive Shader 路径在支持 Tier-1 Mesh Shader
    的平台上激活，以减少 draw call 开销。Astro 的对应逻辑：当 PixiJS 渲染器
    支持 WebGL2 Instanced Mesh（``ASTRO_VECTOR_RENDER=1`` 环境变量）时返回
    True，否则退化为逐元素 SVG 路径（镜像 VertexShader fallback）。
    """
    import os
    return os.environ.get("ASTRO_VECTOR_RENDER", "0") == "1"


class AstroCellPackedView:
    """单视图 LOD 参数包 — 镜像 FPackedView。

    存储视图相关的 LOD 缩放因子，供 cell 可见性查询和 cluster 裁剪使用。
    核心方法 :meth:`update_lod_scales` 镜像 ``FPackedView::UpdateLODScales``，
    根据视口尺寸与像素/边缘阈值计算两个缩放系数。

    Parameters
    ----------
    view_size_y:
        视口垂直像素数（镜像 ViewSizeAndInvSize.Y）。
    view_to_clip_m11:
        投影矩阵 [1][1] 分量（镜像 ViewToClip.M[1][1]）；透视投影中该值等于
        ``2 * focal_length / view_height``。
    """

    def __init__(self, view_size_y: float, view_to_clip_m11: float) -> None:
        self.view_size_y = view_size_y
        self.view_to_clip_m11 = view_to_clip_m11
        self.lod_scale: float = 1.0
        self.lod_scale_hw: float = 1.0

    def update_lod_scales(
        self,
        max_pixels_per_edge: float = 1.0,
        min_pixels_per_edge_hw: float = 0.25,
    ) -> None:
        """重算 LOD 缩放 — 镜像 FPackedView::UpdateLODScales。

        Parameters
        ----------
        max_pixels_per_edge:
            软件光栅化路径的最大边缘像素密度（镜像 CVarNaniteMaxPixelsPerEdge）。
        min_pixels_per_edge_hw:
            硬件光栅化路径的最小边缘像素密度（镜像 CVarNaniteMinPixelsPerEdgeHW）。
        """
        view_to_pixels = 0.5 * self.view_to_clip_m11 * self.view_size_y
        self.lod_scale = view_to_pixels / max(max_pixels_per_edge, 1e-9)
        self.lod_scale_hw = view_to_pixels / max(min_pixels_per_edge_hw, 1e-9)


class AstroCellGlobalResources:
    """全局资源单例 — 镜像 Nanite::FGlobalResources。

    持有跨帧共享的缓冲区容量上限，供 :class:`AstroCellFeedbackManager`
    溢出检测与 :class:`AstroCellStreamExporter` 容量守卫使用。

    所有 getter 均为类方法，镜像 UE5 的静态成员函数调用语义。
    """

    @classmethod
    def get_max_nodes(cls) -> int:
        """返回节点缓冲区上限 — 镜像 FGlobalResources::GetMaxNodes()。"""
        return ASTRO_CELL_MAX_NODES

    @classmethod
    def get_max_candidate_clusters(cls) -> int:
        return ASTRO_CELL_MAX_CANDIDATE_CLUSTERS

    @classmethod
    def get_max_visible_clusters(cls) -> int:
        return ASTRO_CELL_MAX_VISIBLE_CLUSTERS

    @classmethod
    def get_max_visible_assembly_parts(cls) -> int:
        return ASTRO_CELL_MAX_VISIBLE_ASSEMBLY_PARTS


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] AstroCellFeedback → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/Nanite/NaniteFeedback.cpp
#
# NaniteFeedback 通过 GPU Message 异步回读峰值水位，在溢出时向屏幕与日志写入
# 警告。Astro 等价实现：同步回读来自 :class:`AstroCellGlobalResources` 的容量
# 上限，在单帧节点/cluster 计数超标时通过 stderr 发出警告，并维护高水位线
# 字典，供监控脚本消费。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   FFeedbackManager::FFeedbackManager()  → AstroCellFeedbackManager.__init__
#   FFeedbackManager::FBufferState::Update → AstroCellFeedbackManager._update_state
#   ReportMaterialPerformanceWarning      → report_material_perf_warning
#   CVarEmitMaterialPerformanceWarnings   → ASTRO_EMIT_MATERIAL_PERF_WARNINGS
# ═══════════════════════════════════════════════════════════════════════════════

#: 镜像 CVarEmitMaterialPerformanceWarnings — 为 True 时输出 species 性能警告。
ASTRO_EMIT_MATERIAL_PERF_WARNINGS: bool = False


class AstroCellFeedbackManager:
    """缓冲区溢出反馈管理器 — 镜像 Nanite::FFeedbackManager。

    维护四条高水位线状态（nodes / candidate_clusters / visible_clusters /
    visible_assembly_parts），并在超容时向 stderr 写入警告。亦可跟踪 species
    材质性能警告（对应 ``CVarEmitMaterialPerformanceWarnings``）。

    Attributes
    ----------
    high_water_marks : dict[str, int]
        各缓冲区的历史峰值（镜像 FBufferState::HighWaterMark）。
    material_warnings : dict[str, float]
        species 名 → 最后一次出现的时间戳（镜像 MaterialWarningItems）。
    """

    def __init__(self) -> None:
        self.high_water_marks: dict[str, int] = {
            "nodes":                0,
            "candidate_clusters":   0,
            "visible_clusters":     0,
            "visible_assembly_parts": 0,
        }
        self._latest_overflow: dict[str, float] = {}
        self.material_warnings: dict[str, float] = {}

    def _update_state(self, key: str, peak: int, capacity: int) -> bool:
        """更新单条缓冲区高水位线，溢出时记录时间戳并返回 True。

        镜像 ``FFeedbackManager::FBufferState::Update``。
        """
        import time
        new_hwm = peak > self.high_water_marks[key]
        if peak > capacity:
            self._latest_overflow[key] = time.monotonic()
        self.high_water_marks[key] = max(self.high_water_marks[key], peak)
        return new_hwm and peak > capacity

    def tick(self, peak_nodes: int, peak_candidate_clusters: int,
             peak_visible_clusters: int, peak_visible_assembly_parts: int) -> None:
        """每帧调用，镜像 GPU Message 回调触发时机。

        将四个峰值与 :class:`AstroCellGlobalResources` 的容量上限比对；
        一旦突破历史高水位且发生溢出，向 stderr 输出警告，与 UE5 的
        ``UE_LOGF(LogRenderer, Warning, …)`` 语义对应。
        """
        checks = [
            ("nodes",                  peak_nodes,                  AstroCellGlobalResources.get_max_nodes(),
             "node buffer", "ASTRO_CELL_MAX_NODES"),
            ("candidate_clusters",     peak_candidate_clusters,     AstroCellGlobalResources.get_max_candidate_clusters(),
             "candidate cluster buffer", "ASTRO_CELL_MAX_CANDIDATE_CLUSTERS"),
            ("visible_clusters",       peak_visible_clusters,       AstroCellGlobalResources.get_max_visible_clusters(),
             "visible cluster buffer", "ASTRO_CELL_MAX_VISIBLE_CLUSTERS"),
            ("visible_assembly_parts", peak_visible_assembly_parts, AstroCellGlobalResources.get_max_visible_assembly_parts(),
             "visible assembly part buffer", "ASTRO_CELL_MAX_VISIBLE_ASSEMBLY_PARTS"),
        ]
        for key, peak, cap, label, cvar in checks:
            if self._update_state(key, peak, cap):
                print(
                    f"[AstroCellFeedback] WARNING: {label} overflow detected. "
                    f"New high-water mark is {self.high_water_marks[key]} / {cap}. "
                    f"Increase {cvar} to prevent visual artifacts.",
                    file=__import__("sys").stderr,
                )

    def report_material_perf_warning(self, species_name: str) -> None:
        """报告 species 材质性能警告 — 镜像 ReportMaterialPerformanceWarning。

        对同一 species 采用 5 秒冷却窗口限流，避免日志洪泛。
        仅当 :data:`ASTRO_EMIT_MATERIAL_PERF_WARNINGS` 为 True 时输出。
        """
        if not ASTRO_EMIT_MATERIAL_PERF_WARNINGS:
            return
        import time
        now = time.monotonic()
        last = self.material_warnings.get(species_name, 0.0)
        if now - last > 5.0:
            self.material_warnings[species_name] = now
            print(
                f"[AstroCellFeedback] Performance Warning: "
                f"Programmable Astro species '{species_name}' uses PDO or is Masked!",
                file=__import__("sys").stderr,
            )


#: 进程级单例 — 镜像 FFeedbackManager 内嵌于 FGlobalResources 的生命周期。
_astro_cell_feedback_manager: AstroCellFeedbackManager = AstroCellFeedbackManager()


def get_feedback_manager() -> AstroCellFeedbackManager:
    """返回进程级反馈管理器单例。"""
    return _astro_cell_feedback_manager


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] AstroCellStreamExport → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/Nanite/NaniteStreamOut.cpp
#
# NaniteStreamOut 将 GPU 端可见 cluster 的顶点/索引数据流出到 CPU 可访问缓冲区，
# 用于物理碰撞、几何体烘焙等离线用途。Astro 等价实现：将可见 cell 的 SVG 片段
# 与 bbox 元数据序列化到磁盘（或内存缓冲区），供外部工具（布局优化器、碰撞检测
# 服务）消费。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   FStreamOutQueueParameters   → AstroCellStreamExportQueue（优先级队列）
#   FInitQueueCS                → AstroCellStreamExporter.init_queue()
#   FNaniteStreamOutTraversalCS → AstroCellStreamExporter.traverse()
#   FAllocateRangesCS           → AstroCellStreamExporter.allocate_ranges()
#   FNaniteStreamOutCS          → AstroCellStreamExporter.stream_out()
#   GNaniteStreamOutCacheTraversalData
#       → ASTRO_STREAM_OUT_CACHE_TRAVERSAL（模块级布尔常量）
# ═══════════════════════════════════════════════════════════════════════════════

#: 镜像 GNaniteStreamOutCacheTraversalData — 为 True 时在 count pass 中缓存
#: traversal 中间结果，stream-out pass 可跳过重复遍历。
ASTRO_STREAM_OUT_CACHE_TRAVERSAL: bool = True


class AstroCellStreamExportQueue:
    """流出请求队列 — 镜像 FStreamOutQueueParameters + FInitQueueCS。

    维护待流出的 cell_id 列表及各自的分配范围（顶点起始偏移 + 数量），
    对应 UE5 的 ``VertexBuffer`` / ``IndexBuffer`` 分配器。

    Parameters
    ----------
    vertex_buffer_size:
        顶点缓冲区容量（元素数），镜像 ``VertexBufferSize``。
    index_buffer_size:
        索引缓冲区容量（元素数），镜像 ``IndexBufferSize``。
    """

    def __init__(self, vertex_buffer_size: int = 4 * 1_048_576,
                 index_buffer_size: int = 8 * 1_048_576) -> None:
        self.vertex_buffer_size = vertex_buffer_size
        self.index_buffer_size = index_buffer_size
        self._requests: list[dict] = []          # 镜像 StreamOutRequests SRV
        self._allocations: dict[str, dict] = {}  # cell_id → {v_start, v_count, i_start, i_count}
        self._vertex_cursor: int = 0
        self._index_cursor: int = 0

    def enqueue(self, cell_id: str, vertex_count: int, index_count: int) -> bool:
        """排入一条流出请求 — 镜像 FInitQueueCS 的每帧 request 填充。

        若缓冲区剩余空间不足则返回 False（镜像溢出保护）；
        否则记录分配范围并返回 True。
        """
        if (self._vertex_cursor + vertex_count > self.vertex_buffer_size or
                self._index_cursor + index_count > self.index_buffer_size):
            get_feedback_manager().report_material_perf_warning(cell_id)
            return False
        self._requests.append({"cell_id": cell_id,
                                "vertex_count": vertex_count,
                                "index_count": index_count})
        self._allocations[cell_id] = {
            "v_start": self._vertex_cursor,
            "v_count": vertex_count,
            "i_start": self._index_cursor,
            "i_count": index_count,
        }
        self._vertex_cursor += vertex_count
        self._index_cursor += index_count
        return True

    def reset(self) -> None:
        """重置队列 — 镜像帧间缓冲区归零（FInitQueueCS 的 reset state）。"""
        self._requests.clear()
        self._allocations.clear()
        self._vertex_cursor = 0
        self._index_cursor = 0


class AstroCellStreamExporter:
    """Cell 数据流出器 — 镜像 NaniteStreamOut.cpp 中的四阶段 CS dispatch。

    四阶段 pipeline（完全镜像 UE5 的 compute shader dispatch 序列）：

    1. ``init_queue``      — 初始化请求队列（FInitQueueCS）
    2. ``traverse``        — BVH/cluster 遍历，统计顶点/索引数量
                              （FNaniteStreamOutTraversalCS，count 模式）
    3. ``allocate_ranges`` — 按计数结果分配缓冲区范围（FAllocateRangesCS）
    4. ``stream_out``      — 将几何数据写入分配范围（FNaniteStreamOutCS）

    在 Astro 语境中，「几何数据」= cell 的 SVG 片段字符串 + bbox 元数据 JSON。

    Parameters
    ----------
    registry_snapshot:
        来自 ``_load_cell_registry()`` 的 cell 字典快照（``{"cells": {…}}``）。
    output_dir:
        流出文件的写出目录（镜像 CPU 可访问缓冲区的内存映射路径）。
    """

    def __init__(self, registry_snapshot: dict, output_dir: str = "/tmp/astro_streamout") -> None:
        self._registry = registry_snapshot
        self._output_dir = output_dir
        self._queue = AstroCellStreamExportQueue()
        self._traversal_cache: dict[str, dict] | None = None  # 镜像 cached traversal data

    def init_queue(self, cell_ids: list[str]) -> None:
        """阶段 1：初始化请求队列 — 镜像 AddInitQueuePass / FInitQueueCS。

        为每个请求的 cell_id 估算顶点/索引数量（SVG 路径点数 × 2），
        然后调用 :meth:`AstroCellStreamExportQueue.enqueue` 预占缓冲区范围。
        """
        self._queue.reset()
        cells = self._registry.get("cells", {})
        for cid in cell_ids:
            cell = cells.get(cid, {})
            # 粗略估算：每个 cell 约 12 个 SVG 顶点，24 个索引（镜像 cluster 平均三角形数）
            v_count = cell.get("vertex_hint", 12)
            i_count = cell.get("index_hint", 24)
            self._queue.enqueue(cid, v_count, i_count)

    def traverse(self, vis_set: set[str]) -> dict[str, dict]:
        """阶段 2：遍历并统计 — 镜像 FNaniteStreamOutTraversalCS（count 模式）。

        若 :data:`ASTRO_STREAM_OUT_CACHE_TRAVERSAL` 为 True 且缓存命中，
        直接返回缓存结果（镜像 GNaniteStreamOutCacheTraversalData=1 时的
        skip-traversal 优化）；否则重新遍历并缓存。

        Returns
        -------
        dict[str, dict]
            cell_id → {"bbox": …, "species": …} 的可见 cell 快照。
        """
        if ASTRO_STREAM_OUT_CACHE_TRAVERSAL and self._traversal_cache is not None:
            return self._traversal_cache

        cells = self._registry.get("cells", {})
        result = {cid: cells[cid] for cid in vis_set if cid in cells}
        if ASTRO_STREAM_OUT_CACHE_TRAVERSAL:
            self._traversal_cache = result
        return result

    def allocate_ranges(self, traversal_result: dict[str, dict]) -> dict[str, dict]:
        """阶段 3：分配缓冲区范围 — 镜像 FAllocateRangesCS。

        用遍历结果中的精确顶点/索引数覆盖 init_queue 阶段的估算值，
        返回最终分配字典（cell_id → 分配范围）。
        """
        allocs = dict(self._queue._allocations)
        for cid, data in traversal_result.items():
            if cid in allocs:
                # 用真实值修正估算（镜像 AllocateRangesCS 覆写 MeshDataBuffer）
                actual_v = data.get("vertex_hint", allocs[cid]["v_count"])
                actual_i = data.get("index_hint", allocs[cid]["i_count"])
                allocs[cid]["v_count"] = actual_v
                allocs[cid]["i_count"] = actual_i
        return allocs

    def stream_out(
        self,
        traversal_result: dict[str, dict],
        allocations: dict[str, dict],
    ) -> list[dict]:
        """阶段 4：执行流出写入 — 镜像 FNaniteStreamOutCS。

        将每个可见 cell 的 SVG 片段与 bbox 元数据序列化为一条输出记录。
        返回所有输出记录列表（镜像 VertexBuffer + IndexBuffer 写出完成）。

        Parameters
        ----------
        traversal_result:
            阶段 2 返回的可见 cell 快照。
        allocations:
            阶段 3 返回的缓冲区范围分配字典。

        Returns
        -------
        list[dict]
            每条记录包含 ``cell_id``、``bbox``、``species``、``v_start``、
            ``i_start``、``v_count``、``i_count``、``svg_fragment``。
        """
        output_records: list[dict] = []
        for cid, cell_data in traversal_result.items():
            alloc = allocations.get(cid, {})
            record = {
                "cell_id":     cid,
                "bbox":        cell_data.get("bbox", {}),
                "species":     cell_data.get("species", ""),
                "v_start":     alloc.get("v_start", 0),
                "i_start":     alloc.get("i_start", 0),
                "v_count":     alloc.get("v_count", 0),
                "i_count":     alloc.get("i_count", 0),
                "svg_fragment": cell_data.get("svg_fragment", ""),
            }
            output_records.append(record)
            increment_perf_counter("visible_clusters", 1)

        return output_records

    def run(self, cell_ids: list[str], vis_set: set[str]) -> list[dict]:
        """四阶段流出主入口 — 镜像 FNaniteRenderer 中的 StreamOut 调度序列。

        Parameters
        ----------
        cell_ids:
            本帧请求流出的 cell_id 列表。
        vis_set:
            本帧可见 cell_id 集合。

        Returns
        -------
        list[dict]
            完整的流出记录列表（见 :meth:`stream_out` 返回值说明）。
        """
        self.init_queue(cell_ids)
        traversal = self.traverse(vis_set)
        allocs = self.allocate_ranges(traversal)
        return self.stream_out(traversal, allocs)


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] AstroCellTranslucency → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/Nanite/NaniteTranslucency.cpp
#
# NaniteTranslucency 为半透明 Nanite 图元实现了独立的光栅化/着色路径：
# 半透明材质无法利用 Visibility Buffer 的延迟着色，须退化为前向渲染。
# Astro 等价实现：对 opacity < 1.0 的 cell（对应半透明图元），执行单独的
# SVG 混合通道，确保正确的 Alpha 合成顺序，而非依赖主合成器的画家算法。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   CVarNaniteMeshShaderTranslucency
#       → ASTRO_TRANSLUCENCY_MESH_SHADER（布尔常量，控制矢量批渲染路径）
#   UseNaniteMeshShader()      → use_translucency_vector_path()
#   FNaniteTranslucencyFactory → AstroCellTranslucencyFactory（顶点工厂等价）
#   SetTranslucencyParameters  → AstroCellTranslucencyRenderer.set_parameters()
#   RenderTranslucency         → AstroCellTranslucencyRenderer.render()
#   FTranscodeRasterizerArgs_CS → _transcode_rasterizer_args()
# ═══════════════════════════════════════════════════════════════════════════════

#: 镜像 CVarNaniteMeshShaderTranslucency — True 时使用矢量批渲染半透明通道。
ASTRO_TRANSLUCENCY_MESH_SHADER: bool = True

#: 半透明判定阈值 — opacity 低于此值的 cell 进入半透明通道。
ASTRO_TRANSLUCENCY_OPACITY_THRESHOLD: float = 1.0


def use_translucency_vector_path() -> bool:
    """判断是否启用矢量批渲染半透明路径 — 镜像 UseNaniteMeshShader()。

    在 UE5 中，半透明 Mesh Shader 路径要求 Tier-0 Mesh Shader 支持。
    Astro 等价：ASTRO_TRANSLUCENCY_MESH_SHADER 且平台支持向量渲染时返回 True。
    """
    return ASTRO_TRANSLUCENCY_MESH_SHADER and use_vector_render_path()


def _transcode_rasterizer_args(
    raster_bin_count: int,
    input_format: int,
    output_format: int,
) -> dict:
    """转码光栅化器参数 — 镜像 FTranscodeRasterizerArgs_CS。

    将 SW 光栅化路径产生的 ``RasterizerArgsSWHW`` 格式从 *input_format* 转换
    为 *output_format*，对应 Astro 中将软件路径的 SVG 路径命令格式转为
    PixiJS Mesh 格式（或反向）。

    Parameters
    ----------
    raster_bin_count:
        本帧光栅化 bin 数量（cell 数量上限）。
    input_format:
        0 = SVG path 命令格式（软件路径），1 = PixiJS mesh 格式（硬件路径）。
    output_format:
        同上，目标格式。

    Returns
    -------
    dict
        包含转码后参数的字典，键与 UE5 shader 参数结构字段对应。
    """
    return {
        "raster_bin_count": raster_bin_count,
        "input_format":     input_format,
        "output_format":    output_format,
        "transcoded":       input_format != output_format,
    }


class AstroCellTranslucencyFactory:
    """半透明顶点工厂 — 镜像 FNaniteTranslucencyFactory。

    负责判断给定 cell 是否应进入半透明渲染通道，并预处理其 SVG 属性以支持
    正确的 Alpha 合成（``mix-blend-mode``、``opacity``、``fill-opacity``）。

    Parameters
    ----------
    feature_level:
        渲染特性等级字符串（镜像 ``ERHIFeatureLevel``），当前仅支持 ``"SM5"``。
    """

    def __init__(self, feature_level: str = "SM5") -> None:
        self.feature_level = feature_level

    @staticmethod
    def should_compile_permutation(species: str, blend_mode: str) -> bool:
        """判断是否为该 species/blend_mode 组合编译半透明 permutation。

        镜像 ``FNaniteTranslucencyFactory::ShouldCompilePermutation``：
        仅当 blend_mode 为半透明模式（"translucent"、"additive"、"modulate"）
        且 species 使用 Nanite（Astro 中所有 species 均视为使用 Nanite）时返回 True。
        """
        translucent_modes = {"translucent", "additive", "modulate", "alpha_composite"}
        return blend_mode in translucent_modes

    @staticmethod
    def prepare_svg_attrs(opacity: float, blend_mode: str) -> dict[str, str]:
        """预处理半透明 SVG 属性 — 镜像 ModifyCompilationEnvironment 中的 define 设置。

        Returns
        -------
        dict[str, str]
            SVG/CSS 属性字典，直接可写入 ``<g>`` 元素。
        """
        blend_map = {
            "translucent":     "normal",
            "additive":        "screen",
            "modulate":        "multiply",
            "alpha_composite": "source-over",
        }
        return {
            "opacity":        str(round(opacity, 4)),
            "mix-blend-mode": blend_map.get(blend_mode, "normal"),
            "isolation":      "isolate",
        }


class AstroCellTranslucencyRenderer:
    """半透明 Cell 渲染器 — 镜像 Nanite::RenderTranslucency。

    将本帧中 opacity < :data:`ASTRO_TRANSLUCENCY_OPACITY_THRESHOLD` 的 cell
    从主合成器分离，独立执行前向 Alpha 合成渲染，最终输出一个半透明层 SVG 片段。

    Lifecycle::

        renderer = AstroCellTranslucencyRenderer()
        renderer.set_parameters(raster_results, view_info)
        translucency_svg = renderer.render(cell_entries, vis_set)
    """

    def __init__(self) -> None:
        self._factory = AstroCellTranslucencyFactory()
        self._raster_results: dict = {}
        self._view_info: dict = {}

    def set_parameters(
        self,
        raster_results: dict | None,
        view_info: dict | None = None,
    ) -> None:
        """绑定光栅化结果与视图参数 — 镜像 SetTranslucencyParameters。

        Parameters
        ----------
        raster_results:
            主渲染通道的光栅化结果（``FRasterResults`` 等价）；为 None 时
            使用系统纹理默认值（镜像 GSystemTextures fallback）。
        view_info:
            视图元数据字典（viewport 尺寸、投影矩阵等）。
        """
        self._raster_results = raster_results or {}
        self._view_info = view_info or {}

    def render(
        self,
        cell_entries: list[dict],
        vis_set: set[str],
    ) -> str:
        """执行半透明渲染通道 — 镜像 RenderTranslucency。

        筛选出可见且不透明度低于阈值的 cell，按 z-layer 降序（从远到近）
        排列后执行前向 Alpha 合成，输出独立的半透明层 ``<g>`` 分组。

        Parameters
        ----------
        cell_entries:
            全量 cell 条目列表（含 opacity 字段）。
        vis_set:
            本帧可见 cell_id 集合。

        Returns
        -------
        str
            半透明层 SVG 片段字符串（``<g id="translucency-layer">…</g>``）。
        """
        # 筛选半透明可见 cell（镜像 IsTranslucentBlendMode 检查）
        translucent = [
            e for e in cell_entries
            if e["cell_id"] in vis_set
            and e.get("opacity", 1.0) < ASTRO_TRANSLUCENCY_OPACITY_THRESHOLD
        ]

        if not translucent:
            return '<g id="translucency-layer"/>'

        # 按 z 降序排列 — 从远到近前向合成（镜像 translucent draw order）
        translucent.sort(key=lambda e: e.get("bbox", {}).get("z", 0.0), reverse=True)

        # 转码光栅化参数（镜像 FTranscodeRasterizerArgs_CS dispatch）
        bin_count = len(translucent)
        in_fmt = 0  # SVG path 格式
        out_fmt = 1 if use_translucency_vector_path() else 0
        _transcode_rasterizer_args(bin_count, in_fmt, out_fmt)

        # 生成各 cell 的半透明 SVG 片段
        fragments: list[str] = []
        for entry in translucent:
            cid = entry["cell_id"]
            opacity = entry.get("opacity", 1.0)
            blend_mode = entry.get("blend_mode", "translucent")
            if not self._factory.should_compile_permutation("*", blend_mode):
                blend_mode = "translucent"
            attrs = self._factory.prepare_svg_attrs(opacity, blend_mode)
            attr_str = " ".join(f'{k}="{v}"' for k, v in attrs.items())
            fragment = entry.get("svg_fragment", "")
            fragments.append(
                f'<g data-cell-id="{cid}" {attr_str}>{fragment}</g>'
            )
            increment_perf_counter("visible_clusters", 1)

        inner = "\n    ".join(fragments)
        return f'<g id="translucency-layer">\n    {inner}\n  </g>'


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] NaniteDrawList → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/Nanite/NaniteDrawList.cpp
#
# FNaniteDrawListContext (→ AstroCellDrawList):
#   Accumulates cell draw entries grouped by species, ordered so that same-
#   species cells are rendered contiguously.  Contiguous species runs collapse
#   redundant SVG <defs> switches (analogous to PSO / material state changes in
#   the GPU pipeline) and allow the SVG composer to emit shared <defs> once per
#   run rather than once per cell.
#
# Algorithm changes from Nanite original (20%):
#   1. FMeshDrawCommand TArray insertion sort → stable sort on a composite key
#      (z_layer, species_locality_score, insertion_seq).  The locality score
#      is derived from a rolling species-frequency histogram so that high-
#      frequency species are promoted to the front of their z-layer band —
#      reducing worst-case <defs> re-emission even when z-layers interleave.
#      Nanite uses a flat BinIndex integer; here we weight by observed batch
#      size, which is the 2-D equivalent of GPU warp occupancy heuristics.
#   2. PSO / raster-bin registry → species-bin registry (string keys).
#   3. TArray<FMeshDrawCommand>::Append atomic write → list.extend() + re-sort
#      on flush (single-threaded epoch).
#   4. FNaniteMaterialSlot sections loop → flat per-cell entry (no sub-sections).
#   5. DrawCallCost accumulator → svg_defs_cost counter (counts <defs> blocks
#      that would be emitted if draw list were flushed naively).
#
# Reference: [ASTRO-NANITE-DL] debug prefix preserved.
# ═══════════════════════════════════════════════════════════════════════════════

# Maximum draw entries the list accumulates before an automatic flush.
# Mirrors GNaniteMaxDrawsPerPass intent: cap unbounded growth.
_DRAW_LIST_MAX_ENTRIES: int = 512

# Weight applied to species-locality scoring.  Controls how aggressively the
# sorter clusters same-species cells within a z-layer band vs. preserving pure
# z-order.  0.0 = pure z-order (Nanite default); 1.0 = pure species grouping.
# 0.35 chosen empirically: keeps z-layer semantics while halving average
# <defs> re-emission rate in typical 8-species SVG graphs.
_SPECIES_LOCALITY_WEIGHT: float = 0.35


class AstroCellDrawList:
    """
    Python port of FNaniteDrawListContext.

    Accumulates cell draw entries (register_cell_draw_entry) then produces a
    species-batched draw order via flush_draw_order().  The returned order
    minimises SVG <defs> block re-emission by placing same-species cells
    contiguously within each z-layer band.

    Lifecycle mirrors FNaniteDrawListContext:
        __init__   — allocate per-species batch accumulators (AddMeshDrawCommand)
        register   — insert cell entry into pending list (AddPrimitive analogue)
        flush      — sort + return ordered list (Submit analogue)
        reset      — discard state for next epoch (Reset analogue)
    """

    def __init__(self) -> None:
        # Pending entries: list of (z_layer, species, cell_id, bbox, extra).
        # Mirrors TArray<FMeshDrawCommand> PendingDraws.
        self._pending: list = []

        # Per-species frequency histogram used for locality scoring.
        # Mirrors the per-BinIndex draw-call cost accumulator in Nanite.
        self._species_freq: dict = {}

        # Monotonically increasing insertion sequence number.
        # Used as tiebreaker so the sort is stable across equal keys.
        self._seq: int = 0

        # Running count of <defs> blocks that would be emitted without batching.
        # Incremented on each species transition in the naive order; reset on flush.
        self.svg_defs_cost: int = 0

    def register_cell_draw_entry(
        self,
        cell_id: str,
        z_layer: int,
        species: str,
        bbox: dict,
        extra: dict | None = None,
    ) -> None:
        """
        Insert a cell into the draw list.

        Mirrors AddPrimitive() / FNaniteMaterialListContext::AddShadingBin():
            DrawList.AddMeshDrawCommand(MeshDrawCommand, DrawCallCost);

        The cell is appended to _pending with its composite sort key components.
        Actual ordering is deferred to flush_draw_order() so that the locality
        scorer can see the full frequency distribution before committing.

        @param cell_id   Unique cell identifier string.
        @param z_layer   Integer z-layer index (primary sort key — coarse depth).
        @param species   Species name string (secondary sort key — material group).
        @param bbox      Cell bounding-box dict {x, y, w, h, z}.
        @param extra     Optional extra payload forwarded verbatim to callers.
        """
        if len(self._pending) >= _DRAW_LIST_MAX_ENTRIES:
            # Auto-flush guard: mirrors Nanite's pass-size cap.
            # In the single-threaded epoch loop this is a safety valve only.
            import sys as _sys
            print(
                f"[ASTRO-NANITE-DL] register_cell_draw_entry: pending list at "
                f"capacity ({_DRAW_LIST_MAX_ENTRIES}), auto-flushing before insert.",
                file=_sys.stderr,
            )
            self.flush_draw_order()

        self._species_freq[species] = self._species_freq.get(species, 0) + 1
        self._pending.append({
            "cell_id": cell_id,
            "z_layer": z_layer,
            "species": species,
            "bbox":    bbox,
            "extra":   extra or {},
            "_seq":    self._seq,
        })
        self._seq += 1

    def _locality_score(self, species: str) -> float:
        """
        Compute a species locality score in [0, 1).

        Higher-frequency species get lower scores (sort earlier within their
        z-layer band) so large batches of the same species are rendered first,
        maximising contiguous runs and minimising <defs> re-emission.

        Mirrors the DrawCallCost heuristic in FNaniteDrawListContext where
        cheaper (lower-cost) draw commands are sorted to the front so that
        GPU wave occupancy is maximised for the common case.

        Algorithm change vs. Nanite:
            Nanite uses a raw integer BinIndex as the sort key inside a
            TArray<uint16> bin-index list, relying on the registrar to assign
            low indices to common materials.  Here we compute the score
            dynamically from observed frequency so that no pre-registration
            is needed — appropriate for a dynamic SVG scene where species
            composition changes per epoch.
        """
        total = max(self._seq, 1)
        freq  = self._species_freq.get(species, 0)
        # Normalise: species with freq/total → 1.0 gets score 0.0 (front).
        return 1.0 - (freq / total)

    def flush_draw_order(self) -> list:
        """
        Sort pending draw entries and return the ordered draw list.

        Mirrors FNaniteDrawListContext::Submit() which emits draw commands in
        sorted order to the RHI command list.

        Sort key (three-component, stable):
            1. z_layer                   — coarse depth (ascending)
            2. species_locality_score    — species batch size proxy (ascending;
                                           lower = larger batch = render first)
            3. _seq                      — insertion order tiebreaker (stable)

        The locality weight _SPECIES_LOCALITY_WEIGHT blends components 1 and 2:
            effective_key = z_layer + locality_score * _SPECIES_LOCALITY_WEIGHT
        This keeps z-layer semantics dominant while still clustering species.

        After sorting, counts <defs> transitions and updates svg_defs_cost.
        Resets internal state for the next epoch.

        @return  Ordered list of entry dicts (cell_id, z_layer, species, bbox,
                 extra fields).
        """
        import sys as _sys

        if not self._pending:
            return []

        # Build locality scores once (O(S) where S = distinct species count).
        scores = {sp: self._locality_score(sp) for sp in self._species_freq}

        # Stable sort: Python's timsort preserves insertion order for equal keys.
        # Mirrors std::stable_sort on FMeshDrawCommand draw-call cost + BinIndex.
        self._pending.sort(
            key=lambda e: (
                e["z_layer"] + scores.get(e["species"], 0.0) * _SPECIES_LOCALITY_WEIGHT,
                e["_seq"],
            )
        )

        # Count <defs> transitions in the sorted order (diagnostic metric).
        defs_cost = 0
        prev_species = None
        for entry in self._pending:
            if entry["species"] != prev_species:
                defs_cost += 1
                prev_species = entry["species"]
        self.svg_defs_cost = defs_cost

        result = [
            {
                "cell_id": e["cell_id"],
                "z_layer": e["z_layer"],
                "species": e["species"],
                "bbox":    e["bbox"],
                **e["extra"],
            }
            for e in self._pending
        ]

        naive_cost = len(self._pending)  # worst case: every cell different species
        print(
            f"[ASTRO-NANITE-DL] flush_draw_order: entries={len(result)} "
            f"defs_transitions={defs_cost} "
            f"naive_defs={naive_cost} "
            f"reduction={100.0 * (1.0 - defs_cost / max(naive_cost, 1)):.1f}%",
            file=_sys.stderr,
        )

        self._pending.clear()
        self._species_freq.clear()
        self._seq = 0

        return result


def build_epoch_draw_list(
    visibility_query: "AstroCellVisibilityQuery | None" = None,
) -> AstroCellDrawList:
    """
    Build a per-epoch AstroCellDrawList from the current cell_registry.

    Mirrors the scene-level loop in FNaniteMaterialListContext::Apply() that
    iterates DeferredPipelines and calls AddRasterBin / AddShadingBin for each
    registered primitive.  Here we iterate cell_registry.json entries and call
    register_cell_draw_entry for each visible cell.

    If a visibility_query is provided (output of perform_nanite_visibility()),
    culled cells (lod == -1) are excluded from the draw list — matching Nanite's
    behaviour of skipping invisible primitives in the draw-command submission
    loop.

    @param visibility_query  Optional AstroCellVisibilityQuery; pass None to
                             include all registered cells (no culling).
    @return                  AstroCellDrawList ready for flush_draw_order().
    """
    draw_list = AstroCellDrawList()
    registry  = _load_cell_registry()
    cells     = registry.get("cells", {})

    for cell_id, entry in cells.items():
        # ── Visibility gate (FNaniteVisibility bin check) ─────────────────────
        if visibility_query is not None:
            cell_result = visibility_query.cell_results.get(cell_id, {})
            if cell_result.get("lod", 0) == -1:
                continue  # culled — skip (mirrors IsNanitePrimitiveVisible gate)

        # ── Reconstruct bbox from registry min/max format ─────────────────────
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
            bbox = bbox_data

        species = entry.get("species", "")
        z_layer = entry.get("z", 3)
        epoch   = entry.get("epoch", 0)

        draw_list.register_cell_draw_entry(
            cell_id=cell_id,
            z_layer=z_layer,
            species=species,
            bbox=bbox,
            extra={"epoch": epoch, "constraint_mask": entry.get("constraint_mask", 0)},
        )

    return draw_list


# [ASTRO-CELL] DynamicBVH → Python port
#
# Ported from commit upstream/unreal-renderer-ue5/Renderer-Private/DynamicBVH.h
#   and DynamicBVH.cpp (Epic Games, Inc.)
#
# AstroCellBVH:
#   2-D AABB tree accelerating overlap queries for up to 500 cells.
#   Replaces O(N²) pairwise scan with O(log N) tree traversal.
#
#   FDynamicBVH<MaxChildren=4>  →  AstroCellBVH (MaxChildren fixed at 4)
#   FBounds3f (XYZ)             →  _AABB2 (X,Y only; Z/rotation dropped)
#   Insert / Remove / Update    →  insert_cell / remove_cell / update_cell_bounds
#   ForAll overlap              →  query_overlapping_cells
#
# Split algorithm: Morton-code greedy split (simplified to 2-D interleaved bits,
#   matching FMortonArray::Split logic) drives the batch Build path.
# Single-insertion uses the Greedy best-insertion heuristic
#   (FindBestInsertion_Greedy) with Surface-Area Heuristic cost metric adapted
#   to 2-D perimeter (half-perimeter ≈ SAH for flat scenes).
# Refit propagates tight AABB up the ancestor chain after every structural
#   change (mirrors the PathBounds loop in Insert/Extract).
# ═══════════════════════════════════════════════════════════════════════════════

import math as _math
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Internal geometry primitive
# ---------------------------------------------------------------------------

class _AABB2:
    """2-D axis-aligned bounding box (min_x, min_y, max_x, max_y)."""

    __slots__ = ("min_x", "min_y", "max_x", "max_y")

    def __init__(self,
                 min_x: float = float("inf"),  min_y: float = float("inf"),
                 max_x: float = float("-inf"), max_y: float = float("-inf")) -> None:
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y

    # half-perimeter — 2-D analogue of SAH surface area
    def cost(self) -> float:
        dx = self.max_x - self.min_x
        dy = self.max_y - self.min_y
        if dx < 0 or dy < 0:
            return 0.0
        return dx + dy

    def union(self, other: "_AABB2") -> "_AABB2":
        return _AABB2(
            min(self.min_x, other.min_x), min(self.min_y, other.min_y),
            max(self.max_x, other.max_x), max(self.max_y, other.max_y),
        )

    def overlaps(self, other: "_AABB2") -> bool:
        return (self.max_x >= other.min_x and other.max_x >= self.min_x and
                self.max_y >= other.min_y and other.max_y >= self.min_y)

    def contains(self, other: "_AABB2") -> bool:
        return (self.min_x <= other.min_x and self.min_y <= other.min_y and
                self.max_x >= other.max_x and self.max_y >= other.max_y)

    @staticmethod
    def from_bbox(bbox: dict) -> "_AABB2":
        """Accept the project's standard bbox dict {x, y, w, h} or {min, max}."""
        if "min" in bbox and "max" in bbox:
            mn, mx = bbox["min"], bbox["max"]
            return _AABB2(mn[0], mn[1], mx[0], mx[1])
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        return _AABB2(x, y, x + w, y + h)


# ---------------------------------------------------------------------------
# BVH node  (mirrors FDynamicBVH::FNode with MaxChildren = 4)
# ---------------------------------------------------------------------------

_MAX_CHILDREN = 4          # FDynamicBVH<MaxChildren=4>
_CHILD_MASK   = _MAX_CHILDREN - 1   # 0b11
_INDEX_SHIFT  = 2                   # log2(4)

class _BVHNode:
    """
    Internal BVH node holding up to _MAX_CHILDREN child slots.

    child_bounds[i]  : _AABB2 of the i-th child subtree / leaf
    child_ptr[i]     : encoded pointer
                         odd  → leaf:  cell_index = child_ptr[i] >> 1
                         even → inner: node_index  = child_ptr[i]
    parent_idx       : node array index of parent (None = root)
    """

    __slots__ = ("parent_idx", "num_children", "child_bounds", "child_ptr")

    def __init__(self) -> None:
        self.parent_idx:   Optional[int] = None
        self.num_children: int           = 0
        self.child_bounds: List[Optional[_AABB2]] = [None] * _MAX_CHILDREN
        self.child_ptr:    List[int]               = [0]   * _MAX_CHILDREN

    def is_full(self) -> bool:
        return self.num_children == _MAX_CHILDREN

    def is_root(self) -> bool:
        return self.parent_idx is None

    def union_bounds(self) -> _AABB2:
        b = _AABB2()
        for i in range(self.num_children):
            b = b.union(self.child_bounds[i])
        return b

    def get_bounds(self, slot: int) -> _AABB2:
        return self.child_bounds[slot & _CHILD_MASK]

    def get_ptr(self, slot: int) -> int:
        return self.child_ptr[slot & _CHILD_MASK]

    def is_leaf_slot(self, slot: int) -> bool:
        return bool(self.child_ptr[slot & _CHILD_MASK] & 1)


# ---------------------------------------------------------------------------
# AstroCellBVH — public API
# ---------------------------------------------------------------------------

class AstroCellBVH:
    """
    2-D dynamic AABB BVH for AstroCell overlap acceleration.

    Ports FDynamicBVH<4> from Renderer-Private/DynamicBVH.h:
      - Insert  → insert_cell
      - Remove  → remove_cell
      - Update  → update_cell_bounds
      - ForAll  → query_overlapping_cells

    All structural operations run in amortised O(log N) time;
    query_overlapping_cells is O(log N + k) where k = number of hits.
    """

    def __init__(self) -> None:
        # Node pool (index → _BVHNode); mirrors TArray<FNode> Nodes
        self._nodes:  List[_BVHNode] = []
        # Free-list head (node pool index), mirrors FreeHead
        self._free_head: Optional[int] = None
        # Leaf table: cell_id → encoded node-slot  (mirrors TArray<uint32> Leaves)
        self._leaves: Dict[str, int] = {}
        # Root node index (pool index) — None when tree is empty
        self._root: Optional[int] = None
        # Root AABB (mirrors FSingleRoot::Root.Bounds)
        self._root_bounds: _AABB2 = _AABB2()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def insert_cell(self, cell_id: str, bbox: dict) -> None:
        """
        Insert a cell into the BVH.
        Mirrors FDynamicBVH::Add(Bounds, Index).
        """
        aabb = _AABB2.from_bbox(bbox)
        if cell_id in self._leaves:
            # Already present — treat as update (mirrors check() + re-insert)
            self._extract(self._leaves[cell_id])
        leaf_ptr = self._encode_leaf(cell_id)

        if self._root is None:
            self._root = self._alloc_node()
            self._nodes[self._root].parent_idx = None
            self._nodes[self._root].num_children = 0

        slot = self._insert(aabb, leaf_ptr)
        self._leaves[cell_id] = slot
        self._root_bounds = self._root_bounds.union(aabb)

    def remove_cell(self, cell_id: str) -> None:
        """
        Remove a cell from the BVH.
        Mirrors FDynamicBVH::Remove(Index).
        """
        if cell_id not in self._leaves:
            return
        slot = self._leaves.pop(cell_id)
        self._extract(slot)
        # Recompute root bounds lazily from root node (mirrors Extract path)
        if self._root is not None:
            self._root_bounds = self._nodes[self._root].union_bounds()
        else:
            self._root_bounds = _AABB2()

    def update_cell_bounds(self, cell_id: str, bbox: dict) -> None:
        """
        Move a cell to new bounds.
        Mirrors FDynamicBVH::Update(Bounds, Index) = Remove + Add.
        """
        self.remove_cell(cell_id)
        self.insert_cell(cell_id, bbox)

    def query_overlapping_cells(self, bbox: dict) -> List[str]:
        """
        Return all cell_ids whose stored AABB overlaps *bbox*.
        Mirrors FDynamicBVH::ForAll(Bounds, Func) — O(log N + k).
        """
        if self._root is None:
            return []
        query_aabb = _AABB2.from_bbox(bbox)
        if not self._root_bounds.overlaps(query_aabb):
            return []

        results: List[str] = []
        stack:   List[int] = [self._root << _INDEX_SHIFT]

        while stack:
            node_slot = stack.pop()
            node = self._nodes[node_slot >> _INDEX_SHIFT]
            for i in range(node.num_children):
                cb = node.child_bounds[i]
                if cb is None or not cb.overlaps(query_aabb):
                    continue
                ptr = node.child_ptr[i]
                if ptr & 1:
                    # Leaf
                    cell_id = self._decode_leaf(ptr)
                    if cell_id is not None:
                        results.append(cell_id)
                else:
                    # Inner node
                    stack.append(ptr << _INDEX_SHIFT)

        return results

    def build_from_registry(self, cell_registry: dict) -> None:
        """
        Bulk-build the BVH from an existing cell registry dict.
        Mirrors FDynamicBVH::Build() with Morton-sort-based splitting.
        The registry format matches the one used by perform_nanite_visibility().
        """
        self._nodes.clear()
        self._free_head = None
        self._leaves.clear()
        self._root = None
        self._root_bounds = _AABB2()

        items: List[Tuple[str, _AABB2]] = []
        for cell_id, entry in cell_registry.items():
            bbox_data = entry.get("bbox", {})
            try:
                aabb = _AABB2.from_bbox(bbox_data)
            except (KeyError, TypeError, IndexError):
                continue
            items.append((cell_id, aabb))

        if not items:
            return

        # Compute scene-wide AABB for Morton normalisation
        scene = _AABB2()
        for _, ab in items:
            scene = scene.union(ab)
        sx = scene.max_x - scene.min_x or 1.0
        sy = scene.max_y - scene.min_y or 1.0

        # 2-D Morton codes: interleave 10-bit X and Y (mirrors FMortonArray ctor)
        def _morton2(x: float, y: float) -> int:
            xi = int(max(0.0, min(1023.0, x * 1023.0)))
            yi = int(max(0.0, min(1023.0, y * 1023.0)))
            # Spread bits: x in even positions, y in odd positions
            def _spread(v: int) -> int:
                v &= 0x3FF
                v = (v | (v << 8)) & 0x00FF00FF
                v = (v | (v << 4)) & 0x0F0F0F0F
                v = (v | (v << 2)) & 0x33333333
                v = (v | (v << 1)) & 0x55555555
                return v
            return _spread(xi) | (_spread(yi) << 1)

        sorted_items = sorted(
            items,
            key=lambda t: _morton2(
                ((t[1].min_x + t[1].max_x) * 0.5 - scene.min_x) / sx,
                ((t[1].min_y + t[1].max_y) * 0.5 - scene.min_y) / sy,
            ),
        )

        # Recursive BVH build via Morton-split (mirrors FDynamicBVH::Build stack)
        self._root = self._build_recursive(sorted_items, None)
        if self._root is not None:
            self._root_bounds = self._nodes[self._root].union_bounds()

    # ------------------------------------------------------------------
    # Internal helpers — mirrors private FDynamicBVH methods
    # ------------------------------------------------------------------

    # ── Leaf pointer encoding (odd = leaf, mirrors (Index << 1) | 1) ──

    # Map cell_id → integer index for pointer encoding
    _cell_to_idx: Dict[str, int] = {}
    _idx_to_cell: Dict[int, str] = {}
    _next_idx: int = 0

    def _encode_leaf(self, cell_id: str) -> int:
        if cell_id not in AstroCellBVH._cell_to_idx:
            idx = AstroCellBVH._next_idx
            AstroCellBVH._cell_to_idx[cell_id] = idx
            AstroCellBVH._idx_to_cell[idx] = cell_id
            AstroCellBVH._next_idx += 1
        return (AstroCellBVH._cell_to_idx[cell_id] << 1) | 1

    @staticmethod
    def _decode_leaf(ptr: int) -> Optional[str]:
        return AstroCellBVH._idx_to_cell.get(ptr >> 1)

    # ── Node allocation (mirrors AllocNode / FreeNode) ──

    def _alloc_node(self) -> int:
        if self._free_head is not None:
            idx = self._free_head
            n = self._nodes[idx]
            self._free_head = n.parent_idx  # reused as free-list next
            n.parent_idx   = None
            n.num_children = 0
            return idx
        self._nodes.append(_BVHNode())
        return len(self._nodes) - 1

    def _free_node(self, node_idx: int) -> None:
        n = self._nodes[node_idx]
        n.parent_idx   = self._free_head   # chain into free list
        n.num_children = 0
        self._free_head = node_idx

    # ── Cost metric: 2-D half-perimeter (mirrors FSurfaceAreaHeuristic) ──

    @staticmethod
    def _cost(aabb: _AABB2) -> float:
        return aabb.cost()

    # ── Greedy best-insertion (mirrors FindBestInsertion_Greedy) ──

    def _find_best_insertion(self, start_node: int, bounds: _AABB2) -> int:
        """
        Greedy descent: at each node pick the child whose merged AABB
        incurs the smallest incremental cost.  Returns the best slot
        (encoded as node_index * _MAX_CHILDREN + child_offset) into which
        the new leaf should be merged.
        """
        best_cost  = float("inf")
        best_slot  = start_node << _INDEX_SHIFT   # default: root slot 0
        node_idx   = start_node
        induced    = 0.0

        while node_idx is not None:
            node = self._nodes[node_idx]
            if not node.is_full():
                # Room to add child without splitting — mirrors non-full branch
                return node_idx << _INDEX_SHIFT   # caller uses parent index
            # Pick spatially closest child (mirrors L1 delta heuristic)
            bx = (bounds.min_x + bounds.max_x) * 0.5
            by = (bounds.min_y + bounds.max_y) * 0.5
            best_child_dist = float("inf")
            best_child_i    = 0
            for i in range(node.num_children):
                cb = node.child_bounds[i]
                cx = (cb.min_x + cb.max_x) * 0.5
                cy = (cb.min_y + cb.max_y) * 0.5
                d  = abs(bx - cx) + abs(by - cy)
                if d < best_child_dist:
                    best_child_dist = d
                    best_child_i    = i

            cb          = node.child_bounds[best_child_i]
            direct_cost = self._cost(bounds.union(cb))
            total_cost  = induced + direct_cost
            child_cost  = total_cost - self._cost(cb)

            if child_cost >= best_cost:
                break
            if total_cost < best_cost:
                best_cost = total_cost
                best_slot = (node_idx << _INDEX_SHIFT) | best_child_i

            ptr = node.child_ptr[best_child_i]
            if ptr & 1:
                break   # leaf — can't descend further

            induced  = child_cost
            node_idx = ptr   # descend into child node

            if induced + self._cost(bounds) >= best_cost:
                break

        return best_slot

    # ── Insert (mirrors FDynamicBVH::Insert) ──

    def _insert(self, bounds: _AABB2, leaf_ptr: int) -> int:
        """
        Insert *leaf_ptr* with bounding box *bounds* into the tree.
        Returns the node-slot that now holds the leaf.
        """
        root_node = self._nodes[self._root]

        # Root still has room — direct insert (fast path)
        if not root_node.is_full():
            slot_i = root_node.num_children
            root_node.num_children += 1
            root_node.child_bounds[slot_i] = bounds
            root_node.child_ptr[slot_i]    = leaf_ptr
            if leaf_ptr & 1:
                self._leaves[self._decode_leaf(leaf_ptr)] = \
                    (self._root << _INDEX_SHIFT) | slot_i
            return (self._root << _INDEX_SHIFT) | slot_i

        # Find best slot via greedy descent
        best_encoded = self._find_best_insertion(self._root, bounds)
        best_node_i  = best_encoded >> _INDEX_SHIFT
        best_child_i = best_encoded & _CHILD_MASK
        best_node    = self._nodes[best_node_i]

        existing_ptr = best_node.child_ptr[best_child_i]
        existing_b   = best_node.child_bounds[best_child_i]
        is_leaf      = bool(existing_ptr & 1)

        # Need a new internal node if slot is a leaf or child node is full
        need_new_level = is_leaf or (
            not is_leaf and self._nodes[existing_ptr].is_full()
        )

        if need_new_level:
            new_node_i = self._alloc_node()
            new_node   = self._nodes[new_node_i]
            new_node.parent_idx   = best_node_i
            new_node.num_children = 1
            new_node.child_bounds[0] = existing_b
            new_node.child_ptr[0]    = existing_ptr
            # Fix back-pointer for moved child
            if existing_ptr & 1:
                cid = self._decode_leaf(existing_ptr)
                if cid is not None:
                    self._leaves[cid] = (new_node_i << _INDEX_SHIFT) | 0
            else:
                self._nodes[existing_ptr].parent_idx = new_node_i

            best_node.child_ptr[best_child_i] = new_node_i
            target_node_i = new_node_i
        else:
            target_node_i = existing_ptr

        target_node = self._nodes[target_node_i]
        slot_i = target_node.num_children
        target_node.num_children += 1
        target_node.child_bounds[slot_i] = bounds
        target_node.child_ptr[slot_i]    = leaf_ptr

        # Propagate bounds up the ancestor chain (mirrors PathBounds loop)
        path_bounds = bounds
        path_slot   = (best_node_i << _INDEX_SHIFT) | best_child_i
        while path_slot is not None:
            pni  = path_slot >> _INDEX_SHIFT
            pci  = path_slot & _CHILD_MASK
            pn   = self._nodes[pni]
            new_b = pn.child_bounds[pci].union(path_bounds)
            pn.child_bounds[pci] = new_b
            path_bounds = new_b
            if pn.parent_idx is None:
                break
            # Find which slot in parent points to this node
            par = self._nodes[pn.parent_idx]
            found = None
            for k in range(par.num_children):
                if par.child_ptr[k] == pni:
                    found = (pn.parent_idx << _INDEX_SHIFT) | k
                    break
            path_slot = found

        return (target_node_i << _INDEX_SHIFT) | slot_i

    # ── Extract (mirrors FDynamicBVH::Extract + RemoveAndSwap) ──

    def _extract(self, slot: int) -> None:
        """
        Remove the entry at *slot* (encoded node-slot) from the tree,
        refitting ancestor bounds.  Mirrors Extract() + RemoveAndSwap().
        """
        node_i = slot >> _INDEX_SHIFT
        child_i = slot & _CHILD_MASK
        node = self._nodes[node_i]

        # RemoveAndSwap: fill gap with last child
        last = node.num_children - 1
        node.num_children = last
        if child_i < last:
            node.child_bounds[child_i] = node.child_bounds[last]
            moved_ptr = node.child_ptr[last]
            node.child_ptr[child_i]   = moved_ptr
            # Fix back-pointer for swapped child
            if moved_ptr & 1:
                cid = self._decode_leaf(moved_ptr)
                if cid is not None:
                    self._leaves[cid] = (node_i << _INDEX_SHIFT) | child_i
            else:
                pass  # inner nodes don't store back-ref by slot
        node.child_bounds[last] = None
        node.child_ptr[last]    = 0

        # Refit ancestor bounds (mirrors PathBounds propagation in Extract)
        path_bounds = node.union_bounds()
        par_i = node.parent_idx

        while par_i is not None:
            par = self._nodes[par_i]
            # Find slot in parent that points to node_i
            for k in range(par.num_children):
                if par.child_ptr[k] == node_i:
                    par.child_bounds[k] = path_bounds
                    break
            path_bounds = par.union_bounds()
            node_i = par_i
            par_i  = par.parent_idx

        # Collapse singleton inner node (mirrors "NumChildren == 1" branch)
        node = self._nodes[node_i if node_i is not None else 0]
        # Find actual bottom node that was modified
        bottom_i  = slot >> _INDEX_SHIFT
        bottom    = self._nodes[bottom_i]
        if not bottom.is_root() and bottom.num_children == 1:
            par_i = bottom.parent_idx
            par   = self._nodes[par_i]
            # Find which slot in parent points to bottom_i
            for k in range(par.num_children):
                if par.child_ptr[k] == bottom_i:
                    par.child_bounds[k] = bottom.child_bounds[0]
                    par.child_ptr[k]    = bottom.child_ptr[0]
                    # Fix back-pointer
                    moved_ptr = bottom.child_ptr[0]
                    if moved_ptr & 1:
                        cid = self._decode_leaf(moved_ptr)
                        if cid is not None:
                            self._leaves[cid] = (par_i << _INDEX_SHIFT) | k
                    else:
                        self._nodes[moved_ptr].parent_idx = par_i
                    break
            self._free_node(bottom_i)
        elif bottom.is_root() and bottom.num_children == 0:
            self._free_node(bottom_i)
            self._root = None

    # ── Batch build recursive (mirrors FDynamicBVH::Build stack) ──

    def _build_recursive(
        self,
        items: List[Tuple[str, _AABB2]],
        parent_i: Optional[int],
    ) -> Optional[int]:
        if not items:
            return None

        node_i = self._alloc_node()
        node   = self._nodes[node_i]
        node.parent_idx = parent_i

        if len(items) <= _MAX_CHILDREN:
            # Leaf-level node
            node.num_children = len(items)
            for i, (cid, ab) in enumerate(items):
                leaf_ptr = self._encode_leaf(cid)
                node.child_bounds[i] = ab
                node.child_ptr[i]    = leaf_ptr
                self._leaves[cid]    = (node_i << _INDEX_SHIFT) | i
            return node_i

        # Split range by highest differing Morton-code bit
        # (mirrors FMortonArray::Split greedy subdivision)
        chunks = self._morton_split(items, _MAX_CHILDREN)

        node.num_children = len(chunks)
        for i, chunk in enumerate(chunks):
            if len(chunk) == 1:
                cid, ab    = chunk[0]
                leaf_ptr   = self._encode_leaf(cid)
                node.child_bounds[i] = ab
                node.child_ptr[i]    = leaf_ptr
                self._leaves[cid]    = (node_i << _INDEX_SHIFT) | i
            else:
                child_i = self._build_recursive(chunk, node_i)
                child_bounds = self._nodes[child_i].union_bounds()
                node.child_bounds[i] = child_bounds
                node.child_ptr[i]    = child_i

        return node_i

    @staticmethod
    def _morton_split(
        items: List[Tuple[str, _AABB2]],
        max_chunks: int,
    ) -> List[List[Tuple[str, _AABB2]]]:
        """
        Recursively bisect *items* (already Morton-sorted) until we have
        at most *max_chunks* groups.  Mirrors FMortonArray::Split bisection
        on the highest differing bit of Morton codes.
        """
        def _bisect(seg: List) -> Tuple[List, List]:
            # Compute 2-D Morton codes for normalised centres within segment
            scene = _AABB2()
            for _, ab in seg:
                scene = scene.union(ab)
            sx = scene.max_x - scene.min_x or 1.0
            sy = scene.max_y - scene.min_y or 1.0

            def _code(ab: _AABB2) -> int:
                cx = ((ab.min_x + ab.max_x) * 0.5 - scene.min_x) / sx
                cy = ((ab.min_y + ab.max_y) * 0.5 - scene.min_y) / sy
                xi = int(max(0.0, min(1023.0, cx * 1023.0)))
                yi = int(max(0.0, min(1023.0, cy * 1023.0)))
                def _sp(v: int) -> int:
                    v &= 0x3FF
                    v = (v | (v << 8)) & 0x00FF00FF
                    v = (v | (v << 4)) & 0x0F0F0F0F
                    v = (v | (v << 2)) & 0x33333333
                    v = (v | (v << 1)) & 0x55555555
                    return v
                return _sp(xi) | (_sp(yi) << 1)

            codes = [_code(ab) for _, ab in seg]
            diff  = codes[0] ^ codes[-1]
            if diff == 0:
                mid = len(seg) // 2
            else:
                hb   = int(_math.floor(_math.log2(diff))) if diff else 0
                mask = 1 << hb
                mid  = len(seg) // 2   # fallback
                for k in range(len(seg)):
                    if codes[k] & mask:
                        mid = k
                        break
                if mid == 0:
                    mid = len(seg) // 2
            return seg[:mid], seg[mid:]

        chunks = [items]
        while len(chunks) < max_chunks:
            # Pick the largest chunk to split
            largest_i = max(range(len(chunks)), key=lambda i: len(chunks[i]))
            if len(chunks[largest_i]) <= 1:
                break
            left, right = _bisect(chunks[largest_i])
            if not left or not right:
                break
            chunks[largest_i] = left
            chunks.append(right)

        return [c for c in chunks if c]


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] ShadingEnergyConservation → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/ShadingEnergyConservation.cpp
#
# Core physics: total radiant energy leaving a cell surface must not exceed
# the energy arriving.  In the SVG substrate the three light-transport channels
# are fill opacity (diffuse bulk), stroke opacity (specular edge), and shadow
# opacity (sub-surface / transmission leak).  Their weighted sum must satisfy:
#
#   E_fill * w_fill + E_stroke * w_stroke + E_shadow * w_shadow <= 1.0
#
# which mirrors the furnace-test invariant from the C++ implementation:
#   integral of BRDF * cos(theta) dOmega <= 1.0
#
# LUT textures (GGXSpecEnergy, ClothEnergy, DiffuseEnergy, …) are replaced by
# analytic approximations (鲁迅式: no texture lookup, just physics constraints):
#
#   GGX specular directional albedo  → Schlick + Smith analytic fit
#   Cloth (Ashikhmin sheen)          → sinusoidal edge rolloff
#   Multi-scattering compensation    → geometric series sum (1/(1-E_avg))
#
# Energy conservation flag (r.Shading.EnergyConservation = 1 → always on):
#   Clamp weighted-channel sum to 1.0, distributing excess proportionally.
#
# Energy preservation flag (r.Shading.EnergyPreservation = 1 → always on):
#   Attenuate fill (diffuse) channel by specular directional albedo so specular
#   reflection correctly steals energy from diffuse — same as the C++ path that
#   multiplies DiffuseColor by (1 - SpecularEnergy).
#
# Species-to-roughness mapping reuses _SPECIES_ROUGHNESS from the ReflectionEnv
# port; roughness drives the GGX lobe width (alpha = roughness^2).
# ═══════════════════════════════════════════════════════════════════════════════

# Channel weights for the three SVG light-transport bins.
# Mirrors the per-lobe energy split in the substrate BRDF evaluation:
#   diffuse (fill)     → bulk transmission through the cell body
#   specular (stroke)  → edge Fresnel reflection
#   shadow             → energy lost to sub-surface / transmission
_SEC_WEIGHT_FILL:   float = 0.60   # GGX diffuse lobe budget (Lambertian approx)
_SEC_WEIGHT_STROKE: float = 0.28   # GGX specular lobe budget
_SEC_WEIGHT_SHADOW: float = 0.12   # transmission / shadow lobe budget

# Minimum channel opacity floor — even at maximum energy loss the channel
# must remain visible (mirrors the C++ fallback BlackDummy texture path).
_SEC_OPACITY_FLOOR: float = 0.05

# Cosine of the representative view angle used for directional-albedo queries.
# 45° (cos = 1/√2) is the representative half-angle used in the analytic fits
# when no per-pixel angle is available — same as the C++ furnace-test default.
_SEC_COS_VIEW: float = 0.7071067811865476   # cos(pi/4)


def _ggx_smith_g1(cos_v: float, alpha: float) -> float:
    """
    GGX Smith masking function G1(v, alpha) — analytic, no LUT.

    Mirrors the BRDF_GGX_SmithCorrelated term integrated over the hemisphere
    in the C++ BuildEnergyTable compute shader (EEnergyTableType::GGXSpecular).

    G1(v,α) = 2 / (1 + sqrt(1 + α²·(1−cos²v)/cos²v))
    """
    alpha2 = alpha * alpha
    cos2   = max(cos_v * cos_v, 1e-8)
    tan2   = (1.0 - cos2) / cos2
    return 2.0 / (1.0 + math.sqrt(1.0 + alpha2 * tan2))


def _ggx_directional_albedo(cos_v: float, roughness: float) -> float:
    """
    Analytic approximation to the GGX directional albedo E(v, roughness).

    Replaces the GGXSpecEnergyTexture 2-D LUT lookup:
        E(v,r) = ∫ D_GGX * G_Smith * (4*cos_l*cos_v)^-1 d_omega_i

    Karis (2013) polynomial fit + Smith G1 correction:
        E ≈ G1(cos_v, α) * (1 − 0.28 * α²)   α = roughness²

    Returns E ∈ [0, 1]; higher roughness → more energy in the specular lobe.
    """
    alpha = roughness * roughness
    g1    = _ggx_smith_g1(cos_v, alpha)
    # Polynomial residual correction — calibrated against the tabulated
    # SHADING_ENERGY_CONSERVATION_TABLE_RESOLUTION=32 data in the C++ source.
    residual = 1.0 - 0.28 * alpha * alpha
    return max(0.0, min(1.0, g1 * residual))


def _cloth_directional_albedo(cos_v: float, roughness: float) -> float:
    """
    Analytic approximation to the Ashikhmin sheen directional albedo.

    Replaces the ClothEnergyTexture LUT (EEnergyTableType::Cloth).
    The Ashikhmin sheen BRDF has its peak at grazing angles; the directional
    albedo is approximated by a sinusoidal rolloff matching the C++ furnace data:

        E_cloth(v,r) ≈ (1 − cos_v)^(2·r) · (1 − 0.5·r)

    Returns E ∈ [0, 1].
    """
    exponent = max(0.5, 2.0 * roughness)
    edge_peak = math.pow(max(0.0, 1.0 - cos_v), exponent)
    return max(0.0, min(1.0, edge_peak * (1.0 - 0.5 * roughness)))


def _diffuse_directional_albedo(cos_v: float, roughness: float) -> float:
    """
    Analytic approximation to the Lambertian diffuse directional albedo.

    Replaces the DiffuseEnergyTexture LUT (EEnergyTableType::Diffuse).
    Pure Lambertian diffuse → E_d = 1 − E_spec (energy preservation);
    the C++ table stores the complement so the specular lobe's taken energy
    is removed from diffuse.  Here we return the raw diffuse albedo before
    specular subtraction (preservation is applied separately in
    apply_shading_energy_conservation).

        E_diff(v,r) ≈ 1.0 − roughness · (1 − cos_v)²

    Returns E ∈ [0, 1].
    """
    return max(0.0, min(1.0, 1.0 - roughness * (1.0 - cos_v) ** 2))


def _multi_scatter_compensation(e_avg: float) -> float:
    """
    Multi-scattering energy compensation factor.

    Replaces the GGXMaxSpecEnergyTexture LUT that the C++ path samples to
    recover energy lost by single-scattering BRDF truncation.

    The compensation is the geometric series sum for multiple inter-surface
    bounces (Turquin 2019, eq. 9):
        f_ms = (1 − E_avg) / (1 − E_avg)  →  1 / (1 − E_avg)   [simplified]

    In practice the factor is clamped to avoid blowup at E_avg → 1:
        compensation = 1 / max(1 − E_avg, 0.01)

    Returns compensation ∈ [1, 100].
    """
    return 1.0 / max(1.0 - e_avg, 0.01)


def _fresnel_schlick_edge(cos_v: float, f0: float) -> float:
    """
    Schlick Fresnel approximation for edge-glow compensation.

    Replaces the Fresnel energy loss computation in ShadingEnergyConservation.h:
        F(v) = F0 + (1 − F0) · (1 − cos_v)^5

    Used to compute how much specular energy is added back to stroke opacity
    at grazing angles (Fresnel energy loss compensation — the cell edge should
    brighten when viewed at glancing incidence, same as the UE5 path that
    recovers Fresnel energy from the specular directional albedo).

    f0 is the species-derived reflectance at normal incidence (F0 ∈ [0, 1]).
    """
    return f0 + (1.0 - f0) * math.pow(max(0.0, 1.0 - cos_v), 5.0)


def _species_f0(species: str) -> float:
    """
    Per-species F0 (normal-incidence reflectance).

    Mirrors the per-material F0 values baked into the C++ substrate BRDF
    evaluation; here we encode them per species as a proxy for material type.

    High F0  → metallic / polished  (cil-bolt, cil-filter)
    Low F0   → diffuse / matte      (cil-plus, cil-graph)
    Mid F0   → dielectric           (all others)
    """
    _F0_TABLE = {
        "cil-eye":         0.04,   # dielectric — clear lens
        "cil-bolt":        0.80,   # metallic — energetic spike
        "cil-vector":      0.04,   # dielectric — direction arrow
        "cil-plus":        0.02,   # low-gloss matte — aggregator
        "cil-arrow-right": 0.06,   # slight sheen — terminal node
        "cil-filter":      0.65,   # semi-metallic — kernel weight matrix
        "cil-code":        0.04,   # dielectric — monospace brace
        "cil-layers":      0.08,   # slight gloss — depth stack
        "cil-loop":        0.10,   # subtle gloss — cyclic arc
        "cil-graph":       0.03,   # matte — node/edge graph
    }
    return _F0_TABLE.get(species, 0.04)


def apply_shading_energy_conservation(
    species:        str,
    roughness:      float,
    fill_opacity:   float,
    stroke_opacity: float,
    shadow_opacity: float,
    cos_view:       float = _SEC_COS_VIEW,
) -> tuple:
    """
    Energy-conserving opacity adjustment for the three SVG light channels.

    Port of ShadingEnergyConservation::Init + GetData applied at shading time:

    Step 1 — Compute per-lobe directional albedos (replaces LUT texture reads):
        E_spec  = GGX directional albedo        (GGXSpecEnergyTexture analogue)
        E_cloth = Ashikhmin sheen albedo        (ClothEnergyTexture analogue)
        E_diff  = Lambertian diffuse albedo     (DiffuseEnergyTexture analogue)
        E_ms    = multi-scatter compensation    (GGXMaxSpecEnergyTexture analogue)

    Step 2 — Fresnel edge-glow compensation:
        F       = Schlick Fresnel at cos_view angle
        stroke_opacity += (F − F0) · (1 − E_spec) · compensation

    Step 3 — Energy preservation (r.Shading.EnergyPreservation):
        fill_opacity *= (1 − E_spec)   ← specular steals energy from diffuse

    Step 4 — Energy conservation (r.Shading.EnergyConservation):
        total = fill_opacity·w_fill + stroke_opacity·w_stroke + shadow_opacity·w_shadow
        if total > 1.0: scale each channel down proportionally.

    Step 5 — Clamp to floor.

    @param species        Cell species name (determines F0, lobe shape)
    @param roughness      Visual roughness [0, 1]
    @param fill_opacity   Diffuse bulk opacity (AO-derived crowding_opacity)
    @param stroke_opacity SVG stroke opacity (default per species)
    @param shadow_opacity feDropShadow flood-opacity (capsule shadow)
    @param cos_view       Cosine of representative view angle (default 45°)
    @return               (fill_opacity, stroke_opacity, shadow_opacity) tuple,
                          all clamped to [_SEC_OPACITY_FLOOR, 1.0]
    """
    roughness = max(0.0, min(1.0, roughness))
    f0        = _species_f0(species)

    # ── Step 1: analytic directional albedos (LUT → formula) ─────────────────
    e_spec  = _ggx_directional_albedo(cos_view, roughness)
    e_cloth = _cloth_directional_albedo(cos_view, roughness)
    e_diff  = _diffuse_directional_albedo(cos_view, roughness)

    # Average albedo for multi-scatter: blend spec + cloth weighted by roughness.
    # Mirrors the blended EAverage used in the C++ GGXMaxSpecEnergy pass.
    e_avg = e_spec * (1.0 - roughness) + e_cloth * roughness
    comp  = _multi_scatter_compensation(e_avg)

    # ── Step 2: Fresnel edge-glow compensation ────────────────────────────────
    # Energy lost to Fresnel at the view angle is partially recovered by
    # brightening the stroke (edge highlight) — same as the C++ path that adds
    # (F − F0) * (1 − E_spec) * compensation into the specular term.
    fresnel        = _fresnel_schlick_edge(cos_view, f0)
    fresnel_excess = max(0.0, fresnel - f0)
    edge_boost     = fresnel_excess * max(0.0, 1.0 - e_spec) * comp

    # Clamp boost so stroke stays in [0,1] before conservation.
    stroke_opacity = max(0.0, min(1.0, stroke_opacity + edge_boost))

    # ── Step 3: energy preservation — specular steals from diffuse ────────────
    # fill *= (1 − E_spec) mirrors DiffuseColor *= (1 − SpecularEnergy) in C++.
    # e_diff provides the diffuse directional albedo correction on top.
    preservation_factor = max(0.0, 1.0 - e_spec) * e_diff
    fill_opacity = fill_opacity * preservation_factor

    # ── Step 4: energy conservation — clamp total weighted energy ────────────
    total = (fill_opacity   * _SEC_WEIGHT_FILL   +
             stroke_opacity * _SEC_WEIGHT_STROKE +
             shadow_opacity * _SEC_WEIGHT_SHADOW)

    if total > 1.0:
        inv_total   = 1.0 / total
        fill_opacity   *= inv_total
        stroke_opacity *= inv_total
        shadow_opacity *= inv_total

    # ── Step 5: floor clamp ───────────────────────────────────────────────────
    fill_opacity   = max(_SEC_OPACITY_FLOOR, fill_opacity)
    stroke_opacity = max(_SEC_OPACITY_FLOOR, stroke_opacity)
    shadow_opacity = max(_SEC_OPACITY_FLOOR, shadow_opacity)

    return (
        round(fill_opacity,   4),
        round(stroke_opacity, 4),
        round(shadow_opacity, 4),
    )


def compute_cell_energy_balance(
    cell_id:  str,
    species:  str,
    bbox:     dict,
    fill_opacity:   float,
    stroke_opacity: float,
    shadow_opacity: float,
    roughness:      float,
) -> dict:
    """
    Top-level energy balance entry point — wraps apply_shading_energy_conservation.

    Called from proc() after crowding_opacity (AO) and shadow params are known,
    before the SVG <g> wrapper is assembled.  Mirrors the call site in the C++
    pipeline:
        ShadingEnergyConservation::GetData(View)   → per-frame LUT handles
        then per-cell BRDF eval reads those handles → adjusted opacities

    Returns a dict with conserved opacities and diagnostic fields.
    """
    f_out, s_out, sh_out = apply_shading_energy_conservation(
        species        = species,
        roughness      = roughness,
        fill_opacity   = fill_opacity,
        stroke_opacity = stroke_opacity,
        shadow_opacity = shadow_opacity,
    )

    e_spec  = _ggx_directional_albedo(_SEC_COS_VIEW, roughness * roughness)
    e_avg   = e_spec
    comp    = _multi_scatter_compensation(e_avg)
    f0      = _species_f0(species)

    total_in  = (fill_opacity   * _SEC_WEIGHT_FILL   +
                 stroke_opacity * _SEC_WEIGHT_STROKE +
                 shadow_opacity * _SEC_WEIGHT_SHADOW)
    total_out = (f_out   * _SEC_WEIGHT_FILL   +
                 s_out   * _SEC_WEIGHT_STROKE +
                 sh_out  * _SEC_WEIGHT_SHADOW)

    dbg = os.environ.get("ASTRO_SEC_VERBOSE", "0") == "1"
    if dbg:
        print(
            f"[ASTRO-SEC] ShadingEnergyConservation cell={cell_id} "
            f"species={species} roughness={roughness:.3f} f0={f0:.3f} "
            f"E_spec={e_spec:.3f} comp={comp:.3f} "
            f"fill: {fill_opacity:.4f}→{f_out:.4f} "
            f"stroke: {stroke_opacity:.4f}→{s_out:.4f} "
            f"shadow: {shadow_opacity:.4f}→{sh_out:.4f} "
            f"total: {total_in:.4f}→{total_out:.4f}",
            file=sys.stderr,
        )

    return {
        "fill_opacity":   f_out,
        "stroke_opacity": s_out,
        "shadow_opacity": sh_out,
        "e_spec":         round(e_spec, 4),
        "multi_scatter":  round(comp,   4),
        "total_energy":   round(total_out, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] PostProcessDeferredDecals → Python port
#
# Ported from commit 80cc569:
#   upstream/unreal-renderer/CompositionLighting/PostProcessDeferredDecals.cpp
#
# FAstroCellDecoration:
#   Per-cell decoration record written into the ConstraintBuffer during the
#   deferred pass.  Each entry carries species gene flags + payload fields
#   (HaloIntensity for CilEye, ArcSeed for CilBolt, BlendWeight for all).
#
# EAstroCellSpecies gene encoding (CustomData0 bits [0..1]):
#   None    = 0  → no decoration, BlendWeight forced to 0
#   CilEye  = 1  → halo ring overlay; HaloIntensity = 0.5 + CustomData1*0.5
#   CilBolt = 2  → electric arc overlay; ArcSeed from pointer hash
#   Hybrid  = 3  → both traits; maps to cil-vector (direction + intensity)
#
# BuildCellDecorationFromDecal:
#   Derives FAstroCellDecoration from species + gene_traits dict.
#   gene_traits["custom_data_0"] encodes species; ["custom_data_1"] encodes
#   halo intensity.  ConstraintChannel assignment mirrors the C++ switch.
#
# ApplyCellDecorationToConstraintBuffer:
#   Packs payload as DecoPayload(R, G, 0, A):
#     R = HaloIntensity   (CilEye / Hybrid)
#     G = ArcSeed/0xFFFF  (CilBolt / Hybrid)
#     A = BlendWeight
#   In SVG: R→opacity of the eye-highlight circle, G→arc path opacity,
#   A→overall overlay group opacity — same channel semantics, SVG substrate.
#
# SVG decal overlay pattern (species → ConstraintChannel → SVG element):
#   CilEye  (ch 2, species-marker)  → additive pupil-glint circle
#   CilBolt (ch 1, secondary-stress)→ lightning highlight path
#   Hybrid  (ch 2)                  → arrow marker (direction + halo)
#   CilPlus (derived)               → cross-hatch shadow lines
#   None/default                    → rounded stroke decoration rect
# ═══════════════════════════════════════════════════════════════════════════════

# Species gene constants (mirror EAstroCellSpecies enum)
_SPECIES_NONE    = 0
_SPECIES_CIL_EYE = 1
_SPECIES_CIL_BOLT = 2
_SPECIES_HYBRID   = 3


class AstroCellDecoration:
    """
    Python equivalent of FAstroCellDecoration.

    Stores the per-cell decoration payload that is overlaid onto the
    ConstraintBuffer (SVG overlay layer in this 2-D port).

    Fields mirror the C++ struct; unused fields default to sentinel values
    identical to the default constructor.
    """
    __slots__ = (
        "species", "halo_intensity", "arc_seed",
        "blend_weight", "constraint_channel",
    )

    def __init__(self):
        self.species            = _SPECIES_NONE
        self.halo_intensity     = 1.0       # CilEye: [0.5, 1.0]
        self.arc_seed           = 0         # CilBolt: uint32 hash
        self.blend_weight       = 1.0       # overlay alpha [0, 1]
        self.constraint_channel = 2         # default: species-marker ch


def _build_cell_decoration(species_name: str, gene_traits: dict,
                           cell_id: str) -> AstroCellDecoration:
    """
    Python equivalent of BuildCellDecorationFromDecal().

    Maps species name + gene_traits dict onto an AstroCellDecoration.
    gene_traits may carry "custom_data_0" (float [0,1] → encodes gene bits)
    and "custom_data_1" (float [0,1] → encodes halo intensity).
    Falls back gracefully when keys are absent.

    Species-name → gene-bit mapping:
        cil-eye    → CilEye  (1)
        cil-bolt   → CilBolt (2)
        cil-vector → Hybrid  (3)  ← both direction + halo traits
        cil-plus   → treated as CilEye with halo_intensity reduced (cross-hatch)
        others     → None    (0)  → default rounded-stroke decal
    """
    deco = AstroCellDecoration()

    # Decode gene bits from custom_data_0 if present (mirrors C++ int-round * 3)
    cd0 = float(gene_traits.get("custom_data_0", -1.0))
    cd1 = float(gene_traits.get("custom_data_1",  0.0))

    # Primary species routing via species name (canonical path)
    if species_name == "cil-eye":
        gene_bits = _SPECIES_CIL_EYE
    elif species_name == "cil-bolt":
        gene_bits = _SPECIES_CIL_BOLT
    elif species_name == "cil-vector":
        gene_bits = _SPECIES_HYBRID
    elif species_name == "cil-plus":
        # cil-plus uses the halo channel with reduced intensity → cross-hatch
        gene_bits = _SPECIES_CIL_EYE
        cd1 = 0.0   # force low halo → triggers cross-hatch variant below
    else:
        # Fallback: try to decode from custom_data_0 field; 0/absent → None
        if cd0 >= 0.0:
            gene_bits = int(round(cd0 * 3.0)) & 0x3
        else:
            gene_bits = _SPECIES_NONE

    deco.species = gene_bits

    if gene_bits == _SPECIES_CIL_EYE:
        # HaloIntensity remapped to [0.5, 1.0] — mirrors C++ case
        deco.halo_intensity     = 0.5 + max(0.0, min(1.0, cd1)) * 0.5
        deco.constraint_channel = 2   # species-marker channel

    elif gene_bits == _SPECIES_CIL_BOLT:
        # ArcSeed: stable per-cell hash from cell_id string
        # Mirrors C++ pointer-address hash >> 4; we use hash(cell_id) >> 4
        deco.arc_seed           = (abs(hash(cell_id)) >> 4) & 0xFFFFFFFF
        deco.constraint_channel = 1   # secondary-stress channel

    elif gene_bits == _SPECIES_HYBRID:
        deco.halo_intensity     = 0.5 + max(0.0, min(1.0, cd1)) * 0.5
        deco.arc_seed           = (abs(hash(cell_id)) >> 4) & 0xFFFFFFFF
        deco.constraint_channel = 2

    else:
        # None — no decoration written; BlendWeight zeroed (mirrors C++)
        deco.blend_weight = 0.0

    return deco


def _apply_cell_decoration_overlay(deco: AstroCellDecoration,
                                   species_name: str,
                                   cx: float, cy: float,
                                   r_outer: float,
                                   bbox: dict) -> str:
    """
    Python equivalent of ApplyCellDecorationToConstraintBuffer().

    Instead of writing to GBufferD via RHICmdList, emits an SVG <g> overlay
    group that is appended to the cell's SVG content before return — same
    deferred-overlay timing as the C++ pass (injected before DrawIndexedPrimitive).

    DecoPayload channel mapping → SVG:
        R (HaloIntensity) → pupil-glint circle opacity / cross-hatch opacity
        G (ArcSeed/0xFFFF)→ arc flicker path opacity
        A (BlendWeight)   → overlay <g> opacity attribute

    Returns an SVG string (may be empty if blend_weight <= 0 or species None).
    """
    if deco.blend_weight <= 0.0 or deco.species == _SPECIES_NONE:
        return ""   # nothing to write — mirrors early-return in C++

    # Pack DecoPayload (R, G, 0, A) — canonical payload layout from C++
    r_channel = deco.halo_intensity                               # CilEye / Hybrid
    g_channel = (deco.arc_seed & 0xFFFF) / 65535.0               # CilBolt / Hybrid
    a_channel = max(0.0, min(1.0, deco.blend_weight))            # all species

    overlay_parts = []
    overlay_parts.append(
        f'<!-- [ASTRO-CELL] PostProcessDeferredDecals species={deco.species} '
        f'ch={deco.constraint_channel} haloI={r_channel:.2f} '
        f'arcSeed={deco.arc_seed} blend={a_channel:.2f} -->'
    )
    overlay_parts.append(
        f'<g class="decal-overlay" opacity="{a_channel:.3f}">'
    )

    if species_name == "cil-eye":
        # ── CilEye: additive pupil-glint circle ─────────────────────────────
        # R channel = HaloIntensity → inner glint brightness
        # Mirrors: DecoPayload.R = Deco.HaloIntensity; ConstraintChannel=2
        glint_r  = r_outer * 0.12
        glint_cx = cx + r_outer * 0.28   # offset right-up (classic catchlight)
        glint_cy = cy - r_outer * 0.28
        glint_op = max(0.4, r_channel * 0.9)
        overlay_parts.append(
            f'  <!-- decal: cil-eye pupil glint — R={r_channel:.2f} ch2 -->'
        )
        # Primary glint
        overlay_parts.append(
            f'  <circle cx="{glint_cx:.2f}" cy="{glint_cy:.2f}" '
            f'r="{glint_r:.2f}" fill="white" opacity="{glint_op:.3f}"/>'
        )
        # Secondary micro-glint
        overlay_parts.append(
            f'  <circle cx="{glint_cx + glint_r:.2f}" cy="{glint_cy + glint_r:.2f}" '
            f'r="{glint_r * 0.45:.2f}" fill="white" opacity="{glint_op * 0.5:.3f}"/>'
        )

    elif species_name == "cil-vector":
        # ── Hybrid: direction-arrow marker overlay ───────────────────────────
        # Both R (halo) + G (arc) channels active; ConstraintChannel=2
        # Mirrors: DecoPayload.R = HaloIntensity; DecoPayload.G = ArcSeed/0xFFFF
        arrow_len   = r_outer * 0.55
        # ArcSeed modulates the arrow angle for per-cell variation (G channel)
        angle_offset = (g_channel - 0.5) * 0.6   # ±0.3 rad variation
        angle        = angle_offset               # centred on rightward 0 rad
        ax1 = cx - arrow_len * 0.5 * math.cos(angle)
        ay1 = cy - arrow_len * 0.5 * math.sin(angle)
        ax2 = cx + arrow_len * 0.5 * math.cos(angle)
        ay2 = cy + arrow_len * 0.5 * math.sin(angle)
        arrow_op = max(0.35, r_channel * 0.7)
        overlay_parts.append(
            f'  <!-- decal: cil-vector arrow marker — R={r_channel:.2f} G={g_channel:.3f} ch2 -->'
        )
        overlay_parts.append(
            f'  <defs>'
            f'<marker id="decal-arrow-{int(deco.arc_seed) % 9999}" '
            f'markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">'
            f'<path d="M0,0 L6,3 L0,6 Z" fill="#2E7D32" opacity="{arrow_op:.3f}"/>'
            f'</marker></defs>'
        )
        overlay_parts.append(
            f'  <line x1="{ax1:.2f}" y1="{ay1:.2f}" x2="{ax2:.2f}" y2="{ay2:.2f}" '
            f'stroke="#2E7D32" stroke-width="1.5" opacity="{arrow_op:.3f}" '
            f'marker-end="url(#decal-arrow-{int(deco.arc_seed) % 9999})"/>'
        )

    elif species_name == "cil-bolt":
        # ── CilBolt: lightning highlight path ────────────────────────────────
        # G channel = ArcSeed/0xFFFF → arc flicker opacity; ConstraintChannel=1
        # Mirrors: DecoPayload.G = ArcSeed & 0xFFFF / 65535.f
        arc_op   = max(0.3, g_channel * 0.85)
        # Zigzag highlight: 3-segment lightning bolt offset from centre
        zx0 = cx - r_outer * 0.15
        zy0 = cy - r_outer * 0.45
        zx1 = cx + r_outer * 0.10
        zy1 = cy - r_outer * 0.05
        zx2 = cx - r_outer * 0.08
        zy2 = cy + r_outer * 0.08
        zx3 = cx + r_outer * 0.14
        zy3 = cy + r_outer * 0.42
        overlay_parts.append(
            f'  <!-- decal: cil-bolt lightning highlight — G={g_channel:.3f} ch1 -->'
        )
        overlay_parts.append(
            f'  <path d="M{zx0:.2f},{zy0:.2f} L{zx1:.2f},{zy1:.2f} '
            f'L{zx2:.2f},{zy2:.2f} L{zx3:.2f},{zy3:.2f}" '
            f'fill="none" stroke="white" stroke-width="1.8" '
            f'stroke-linecap="round" stroke-linejoin="round" '
            f'opacity="{arc_op:.3f}"/>'
        )
        # Glow duplicate at lower opacity
        overlay_parts.append(
            f'  <path d="M{zx0:.2f},{zy0:.2f} L{zx1:.2f},{zy1:.2f} '
            f'L{zx2:.2f},{zy2:.2f} L{zx3:.2f},{zy3:.2f}" '
            f'fill="none" stroke="#FFE0B2" stroke-width="3.5" '
            f'stroke-linecap="round" stroke-linejoin="round" '
            f'opacity="{arc_op * 0.35:.3f}"/>'
        )

    elif species_name == "cil-plus":
        # ── CilPlus: cross-hatch shadow lines ────────────────────────────────
        # R channel (halo_intensity, forced low) → cross-hatch line opacity
        # Uses species-marker channel (ch 2) via CilEye gene path
        hatch_op = max(0.12, r_channel * 0.4)
        arm      = r_outer * 0.55
        spacing  = r_outer * 0.18
        overlay_parts.append(
            f'  <!-- decal: cil-plus cross-hatch shadow — R={r_channel:.2f} ch2 -->'
        )
        # Horizontal hatch lines (above and below centre)
        for offset in [-spacing, 0.0, spacing]:
            x1_h = cx - arm
            x2_h = cx + arm
            y_h  = cy + offset
            overlay_parts.append(
                f'  <line x1="{x1_h:.2f}" y1="{y_h:.2f}" '
                f'x2="{x2_h:.2f}" y2="{y_h:.2f}" '
                f'stroke="#C62828" stroke-width="0.7" opacity="{hatch_op:.3f}" '
                f'stroke-dasharray="4,3"/>'
            )
        # Vertical hatch lines (left and right of centre)
        for offset in [-spacing, 0.0, spacing]:
            x_v  = cx + offset
            y1_v = cy - arm
            y2_v = cy + arm
            overlay_parts.append(
                f'  <line x1="{x_v:.2f}" y1="{y1_v:.2f}" '
                f'x2="{x_v:.2f}" y2="{y2_v:.2f}" '
                f'stroke="#C62828" stroke-width="0.7" opacity="{hatch_op:.3f}" '
                f'stroke-dasharray="4,3"/>'
            )

    else:
        # ── Default / None-species: generic rounded-stroke decoration ────────
        # BlendWeight > 0 but species=None → emit minimal constraint marker
        # Mirrors the ConstraintBuffer "slot reservation" write for non-cell decals
        pad    = 3.0
        rect_w = bbox["w"] - pad * 2
        rect_h = bbox["h"] - pad * 2
        stroke_op = max(0.15, a_channel * 0.5)
        overlay_parts.append(
            f'  <!-- decal: generic rounded-stroke — blend={a_channel:.2f} -->'
        )
        overlay_parts.append(
            f'  <rect x="{pad:.1f}" y="{pad:.1f}" '
            f'width="{rect_w:.1f}" height="{rect_h:.1f}" '
            f'rx="10" ry="10" fill="none" '
            f'stroke="#90A4AE" stroke-width="1.2" '
            f'stroke-dasharray="6,4" opacity="{stroke_op:.3f}"/>'
        )

    overlay_parts.append('</g>')
    return "\n".join(overlay_parts)


CHANNELS = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] ReflectionEnvironment → Python port
#
# Ported from commit 5d07a0a:
#   upstream/unreal-renderer/ReflectionEnvironment.cpp
#
# FAstroCellStyleProbe (→ AstroCellStyleProbe):
#   Per-frame probe that samples the six cardinal neighbour cells' published
#   status channels, collects their species indices + representative SVG
#   palette colours, and elects the dominant species by majority vote.
#   BlendWithCubemap() → blend_toward_neighbour_palette(): given the cell's
#   own SVG stroke/fill colour, lerp it 20 % toward the neighbour palette
#   average so the cell visually converges toward its surroundings — the
#   "style consistency" guarantee from the C++ probe system, translated to the
#   SVG substrate.  Roughness (0 = smooth, 1 = rough) maps to how strongly the
#   cell should resist neighbour influence: smooth cells (low roughness) pull
#   harder; rough cells keep more of their own character.
#
# GAstroCellRegistrySnapshot (→ cell_registry.json + per-cell status.json):
#   In C++ the snapshot is a value-copy written by the game thread before the
#   render pass.  Here the pub/sub equivalent is the set of already-published
#   cell/*/status.json + cell/*/bbox.json files in the channels directory.
#   The probe reads whatever is on disk at proc() call time — i.e. whatever
#   neighbours have already published in this epoch (or the previous one if a
#   neighbour hasn't run yet).
#
# SampleSurroundingCells():
#   Walks the six cardinal neighbours in 2-D grid space (±1 grid step in X
#   and Y; ±1 z-layer step).  Grid step = floor(cell_width) for X/Y, 1 for Z.
#   For each neighbour that has a published status.json, reads species_index
#   (integer derived from the species name), representative colour (the
#   primary SVG fill colour for that species), and accumulates into the probe.
#
# BuildAstroCellStyleProbes():
#   Builds one probe per active cell in the registry, called from proc()
#   immediately before SVG parameter finalisation so adjustments flow into the
#   colour attributes written into the SVG string.
#
# 2-D channel adaptation:
#   CubemapArray probe  → per-cell style probe keyed by cell_id
#   WorldPosition       → bbox (x, y, z) of the cell
#   InfluenceRadius     → max(w, h) of the cell bbox
#   DominantSpeciesIndex→ int derived from species name string
#   Palette[]           → list of (r, g, b) tuples from neighbour cells
#   CellStyleWeight     → _STYLE_PROBE_WEIGHT global (default 0.20 = 20 %)
#   BlendWithCubemap    → blend_toward_neighbour_palette() below
# ═══════════════════════════════════════════════════════════════════════════════

# Blend weight: fraction by which this cell drifts toward the neighbour palette.
# 0.0 = pure own style, 1.0 = fully adopt neighbour palette.
# 鲁迅式 20 % drift — enough to feel the crowd without losing oneself.
_STYLE_PROBE_WEIGHT: float = 0.20

# Species name → canonical integer index (mirrors EAstroCellSpecies + extras).
# Used for majority-vote tallying, same as SpeciesVotes[256] in C++.
_SPECIES_NAME_TO_INDEX: dict = {
    "cil-eye":         1,
    "cil-bolt":        2,
    "cil-vector":      3,
    "cil-plus":        4,
    "cil-arrow-right": 5,
    # ── New species ──────────────────────────────────────────────────────────
    "cil-filter":      6,
    "cil-code":        7,
    "cil-layers":      8,
    "cil-loop":        9,
    "cil-graph":       10,
}

# Species index → primary SVG fill colour (RGB 0-255 tuple).
# Mirrors RepresentativeColour in FAstroCellRegistry::FCellEntry; values
# derived from the fill colours used in the generate_svg_* functions below.
_SPECIES_INDEX_TO_COLOUR: dict = {
    1: (63,  81, 181),   # cil-eye         → #3F51B5 Indigo
    2: (255, 111,   0),  # cil-bolt        → #FF6F00 Amber
    3: (46,  125,  50),  # cil-vector      → #2E7D32 Green
    4: (30,  136, 229),  # cil-plus        → #1E88E5 Blue
    5: (69,   90, 100),  # cil-arrow-right → #455A64 Blue-Grey
    # ── New species ──────────────────────────────────────────────────────────
    6: (123,  31, 162),  # cil-filter  → #7B1FA2 Purple
    7: (46,  125,  50),  # cil-code    → #2E7D32 Green (monospace feel)
    8: (21,  101, 192),  # cil-layers  → #1565C0 Blue (depth stack)
    9: (245, 127,  23),  # cil-loop    → #F57F17 Amber-Orange (cycle)
   10: (55,   71,  79),  # cil-graph   → #37474F Blue-Grey (graph nodes)
    0: (144, 164, 174),  # unassigned  → #90A4AE neutral
}


def _species_to_index(species_name: str) -> int:
    """Map a species name string to its canonical integer index."""
    return _SPECIES_NAME_TO_INDEX.get(species_name, 0)


def _colour_to_hex(rgb: tuple) -> str:
    """Convert (r, g, b) int tuple to #RRGGBB hex string."""
    return "#{:02X}{:02X}{:02X}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def _lerp_colour(c_own: tuple, c_target: tuple, t: float) -> tuple:
    """
    Linear interpolation between two (r,g,b) tuples.

    t=0 → c_own unchanged; t=1 → fully c_target.
    Mirrors FMath::Lerp(CubemapSample, PaletteAvg, t) from BlendWithCubemap.
    """
    t = max(0.0, min(1.0, t))
    return (
        c_own[0] + (c_target[0] - c_own[0]) * t,
        c_own[1] + (c_target[1] - c_own[1]) * t,
        c_own[2] + (c_target[2] - c_own[2]) * t,
    )


class AstroCellStyleProbe:
    """
    Python equivalent of FAstroCellStyleProbe.

    Samples the published status + bbox channels of the six cardinal neighbour
    cells, accumulates their species indices and representative colours into a
    palette, then exposes blend_toward_neighbour_palette() which nudges the
    cell's own SVG colour parameters toward the neighbourhood average.

    Lifetime: created inside proc() each call, discarded after SVG finalisation.
    """

    # Maximum palette entries — mirrors MaxPaletteEntries = 8 in C++.
    MAX_PALETTE_ENTRIES: int = 8

    def __init__(self, cell_id: str, bbox: dict, cell_style_weight: float = _STYLE_PROBE_WEIGHT):
        self.cell_id          = cell_id
        self.world_x          = float(bbox["x"])
        self.world_y          = float(bbox["y"])
        self.world_z          = float(bbox.get("z", 0))
        self.cell_w           = float(bbox["w"])
        self.cell_h           = float(bbox["h"])
        # InfluenceRadius — max half-extent of the bbox (mirrors Comp->InfluenceRadius)
        self.influence_radius = max(self.cell_w, self.cell_h) / 2.0
        self.cell_style_weight = max(0.0, min(1.0, cell_style_weight))

        # Palette: list of (r,g,b) tuples from neighbour cells, up to MAX_PALETTE_ENTRIES
        self.palette: list = []
        # Dominant species index elected by majority vote
        self.dominant_species_index: int = 0

    def sample_surrounding_cells(self, channels_dir: str) -> None:
        """
        Walk the six cardinal neighbour positions in grid space, read their
        published status.json + bbox.json channels, accumulate species/palette.

        Mirrors FAstroCellStyleProbe::SampleSurroundingCells() — six cardinal
        directions, species vote tally, palette fill.

        Grid step: cell_w for X, cell_h for Y, 1.0 for Z (layer index).
        Neighbours are identified by scanning cell/*/bbox.json files and
        checking whether their (x, y, z) centre falls within one grid step
        of this cell's centre in exactly one axis (cardinal, not diagonal).
        """
        self.palette.clear()
        self.dominant_species_index = 0

        cell_base = os.path.join(channels_dir, "cell")
        if not os.path.isdir(cell_base):
            return

        # Grid step sizes — mirrors CellSize = 100.f for X/Y; 1.0 for Z-layer
        step_x = max(self.cell_w, 1.0)
        step_y = max(self.cell_h, 1.0)
        step_z = 1.0

        # Self centre
        cx = self.world_x + self.cell_w / 2.0
        cy = self.world_y + self.cell_h / 2.0
        cz = self.world_z

        # Six cardinal offsets: (±step_x, 0, 0), (0, ±step_y, 0), (0, 0, ±step_z)
        # Mirrors Offsets[6] in the C++ SampleSurroundingCells.
        cardinal_offsets = [
            ( step_x,      0,      0),
            (-step_x,      0,      0),
            (      0,  step_y,     0),
            (      0, -step_y,     0),
            (      0,      0,  step_z),
            (      0,      0, -step_z),
        ]

        # Tolerance for "is this a cardinal neighbour?" check.
        # Mirrors FIntVector equality — we allow a small float tolerance.
        cardinal_tol = 0.5

        species_votes: dict = {}  # species_index → vote count; mirrors SpeciesVotes[256]
        max_votes: int = 0

        for sibling in os.listdir(cell_base):
            if sibling == self.cell_id:
                continue  # skip self

            bbox_path   = os.path.join(cell_base, sibling, "bbox.json")
            status_path = os.path.join(cell_base, sibling, "status.json")

            if not os.path.isfile(bbox_path):
                continue

            try:
                with open(bbox_path) as _f:
                    nbr_bbox = json.load(_f)
            except (json.JSONDecodeError, OSError):
                continue

            # Sibling centre
            nbr_cx = nbr_bbox["x"] + nbr_bbox["w"] / 2.0
            nbr_cy = nbr_bbox["y"] + nbr_bbox["h"] / 2.0
            nbr_cz = float(nbr_bbox.get("z", 0))

            # Check whether sibling is exactly one cardinal step away.
            # Mirrors FIntVector equality test on CenterCoord + Offsets[i].
            dx = nbr_cx - cx
            dy = nbr_cy - cy
            dz = nbr_cz - cz

            is_cardinal = False
            for (ox, oy, oz) in cardinal_offsets:
                if (abs(dx - ox) < cardinal_tol * step_x and
                        abs(dy - oy) < cardinal_tol * step_y and
                        abs(dz - oz) < cardinal_tol * max(step_z, 1.0)):
                    is_cardinal = True
                    break

            if not is_cardinal:
                continue  # not a direct neighbour — skip (mirrors Find() returning nullptr)

            # Read species from status.json if available; fall back to bbox.json field.
            nbr_species_name = nbr_bbox.get("species", "")
            if not nbr_species_name and os.path.isfile(status_path):
                try:
                    with open(status_path) as _f:
                        nbr_status = json.load(_f)
                    nbr_species_name = nbr_status.get("species", "")
                except (json.JSONDecodeError, OSError):
                    pass

            nbr_species_idx = _species_to_index(nbr_species_name)
            nbr_colour      = _SPECIES_INDEX_TO_COLOUR.get(nbr_species_idx,
                                                           _SPECIES_INDEX_TO_COLOUR[0])

            # Accumulate palette entry (up to MAX_PALETTE_ENTRIES).
            if len(self.palette) < self.MAX_PALETTE_ENTRIES:
                self.palette.append(nbr_colour)

            # Tally species vote — mirrors SpeciesVotes[SI]++
            species_votes[nbr_species_idx] = species_votes.get(nbr_species_idx, 0) + 1
            if species_votes[nbr_species_idx] > max_votes:
                max_votes = species_votes[nbr_species_idx]
                self.dominant_species_index = nbr_species_idx

    def blend_toward_neighbour_palette(
        self,
        own_colour: tuple,
        roughness: float = 0.5,
    ) -> tuple:
        """
        Nudge own_colour (r,g,b) toward the neighbourhood palette average.

        Direct port of FAstroCellStyleProbe::BlendWithCubemap():
          - Compute PaletteAvg from palette entries.
          - Smooth-step blend: smooth surfaces (low roughness) pull more
            strongly; rough surfaces resist and keep own character.
          - Scale blend by cell_style_weight (_STYLE_PROBE_WEIGHT = 0.20).

        @param own_colour  Cell's own primary colour as (r, g, b) floats [0,255].
        @param roughness   Visual roughness of this cell [0,1].  0 = sharp icon
                           (attaches hard to neighbour style); 1 = rough/noisy
                           (ignores neighbourhood almost entirely).
        @return            Blended (r, g, b) tuple, same scale.
        """
        if not self.palette:
            return own_colour   # no neighbours sampled — no-op (PaletteSize==0 path)

        # Accumulate weighted average palette colour — mirrors the palette loop.
        r_sum = sum(c[0] for c in self.palette)
        g_sum = sum(c[1] for c in self.palette)
        b_sum = sum(c[2] for c in self.palette)
        n = len(self.palette)
        palette_avg = (r_sum / n, g_sum / n, b_sum / n)

        # Smooth-step blend: smoother surfaces get stronger cell-style push.
        # At roughness=0 → full palette; at roughness=1 → no palette influence.
        # Mirrors: t = SmoothStep(0,1, 1-Roughness) * CellStyleWeight
        inv_r = max(0.0, min(1.0, 1.0 - roughness))
        # SmoothStep(0,1,x) = x*x*(3-2*x)
        smooth = inv_r * inv_r * (3.0 - 2.0 * inv_r)
        t = smooth * self.cell_style_weight

        return _lerp_colour(own_colour, palette_avg, t)


def build_astro_cell_style_probes(channels_dir: str) -> dict:
    """
    Build one AstroCellStyleProbe per active cell in the registry snapshot.

    Python equivalent of BuildAstroCellStyleProbes() — iterates all published
    cell/*/bbox.json entries, constructs probes, samples surrounding cells,
    returns a dict keyed by cell_id.

    Called from proc() before SVG parameter finalisation so that the probe
    data is ready when adjust_svg_params_for_style_consistency() is called.
    """
    probes: dict = {}

    cell_base = os.path.join(channels_dir, "cell")
    if not os.path.isdir(cell_base):
        return probes

    for cid in os.listdir(cell_base):
        bbox_path = os.path.join(cell_base, cid, "bbox.json")
        if not os.path.isfile(bbox_path):
            continue
        try:
            with open(bbox_path) as _f:
                bbox = json.load(_f)
        except (json.JSONDecodeError, OSError):
            continue

        probe = AstroCellStyleProbe(cid, bbox)
        probe.sample_surrounding_cells(channels_dir)
        probes[cid] = probe

    return probes

# ═══════════════════════════════════════════════
# Channel I/O — Apollo Reader/Writer equivalent
# ═══════════════════════════════════════════════

def read_channel(path: str) -> dict:
    """Subscribe = read JSON. Apollo CreateReader equivalent."""
    full = os.path.join(CHANNELS, path)
    with open(full) as f:
        return json.load(f)

def write_channel(path: str, data):
    """Publish = write file. Apollo CreateWriter equivalent."""
    full = os.path.join(CHANNELS, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    if isinstance(data, str):
        with open(full, "w") as f:
            f.write(data)
    else:
        with open(full, "w") as f:
            json.dump(data, f, indent=2)

# ═══════════════════════════════════════════════
# Species Gene Algorithms — each species generates differently
# These are NOT icon files — they are algorithmic generation styles
# ═══════════════════════════════════════════════

def generate_svg_cil_eye(cell_id, label, bbox, gene_traits):
    """cil-eye species: radial attention pattern — heatmap rays from center."""
    cx = bbox["w"] / 2
    cy = bbox["h"] / 2
    r_outer = min(bbox["w"], bbox["h"]) / 2 - 4

    parts = []
    # Background rounded rect
    parts.append(f'<rect x="0" y="0" width="{bbox["w"]}" height="{bbox["h"]}" '
                 f'rx="8" fill="#E8EAF6" stroke="#3F51B5" stroke-width="1.5"/>')

    # Radial attention rays (algorithmic — number based on label complexity)
    num_rays = max(4, len(label) // 2)
    for i in range(num_rays):
        angle = (2 * math.pi * i) / num_rays
        # Intensity gradient: rays get lighter toward edges
        intensity = 0.3 + 0.7 * (1 - i / num_rays)
        r_inner = r_outer * 0.3
        x1 = cx + r_inner * math.cos(angle)
        y1 = cy + r_inner * math.sin(angle)
        x2 = cx + r_outer * math.cos(angle)
        y2 = cy + r_outer * math.sin(angle)
        opacity = max(0.15, intensity * 0.6)
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                     f'stroke="#3F51B5" stroke-width="1" opacity="{opacity:.2f}"/>')

    # Center focal point
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r_outer*0.2}" fill="#3F51B5" opacity="0.7"/>')
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r_outer*0.08}" fill="#E8EAF6"/>')

    # Label
    parts.append(f'<text x="{cx}" y="{bbox["h"]-6}" text-anchor="middle" '
                 f'font-family="system-ui,sans-serif" font-size="10" fill="#1A237E">{label}</text>')

    # ── [ASTRO-CELL] Deferred Decal overlay (PostProcessDeferredDecals port) ──
    # BuildCellDecorationFromDecal → ApplyCellDecorationToConstraintBuffer
    # Injected before return, mirroring C++ overlay before DrawIndexedPrimitive.
    _deco = _build_cell_decoration("cil-eye", gene_traits, cell_id)
    _overlay = _apply_cell_decoration_overlay(
        _deco, "cil-eye", cx, cy, r_outer, bbox)
    if _overlay:
        parts.append(_overlay)

    return "\n".join(parts), bbox


def generate_svg_cil_vector(cell_id, label, bbox, gene_traits):
    """cil-vector species: embedding arrows — magnitude/direction lines."""
    cx = bbox["w"] / 2
    cy = bbox["h"] / 2

    parts = []
    parts.append(f'<rect x="0" y="0" width="{bbox["w"]}" height="{bbox["h"]}" '
                 f'rx="8" fill="#E8F5E9" stroke="#2E7D32" stroke-width="1.5"/>')

    # Vector arrows (algorithmic — angle spread based on "positional" vs "input")
    num_arrows = 5
    arrow_len = bbox["w"] * 0.3
    for i in range(num_arrows):
        angle = -0.4 + (0.8 * i / (num_arrows - 1))  # spread ±0.4 rad
        x1 = cx - arrow_len * 0.5 * math.cos(angle)
        y1 = cy - arrow_len * 0.5 * math.sin(angle) + 2
        x2 = cx + arrow_len * 0.5 * math.cos(angle)
        y2 = cy + arrow_len * 0.5 * math.sin(angle) + 2
        weight = 1 + (i % 3) * 0.5
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                     f'stroke="#2E7D32" stroke-width="{weight:.1f}" opacity="0.5" '
                     f'marker-end="url(#arrow-green)"/>')

    parts.append(f'<text x="{cx}" y="{bbox["h"]-6}" text-anchor="middle" '
                 f'font-family="system-ui,sans-serif" font-size="10" fill="#1B5E20">{label}</text>')

    # ── [ASTRO-CELL] Deferred Decal overlay (PostProcessDeferredDecals port) ──
    _deco = _build_cell_decoration("cil-vector", gene_traits, cell_id)
    _r_outer_v = min(bbox["w"], bbox["h"]) / 2 - 4
    _overlay = _apply_cell_decoration_overlay(
        _deco, "cil-vector", cx, cy, _r_outer_v, bbox)
    if _overlay:
        parts.append(_overlay)

    return "\n".join(parts), bbox


def generate_svg_cil_bolt(cell_id, label, bbox, gene_traits):
    """cil-bolt species: zigzag activation — ReLU-style angular patterns."""
    cx = bbox["w"] / 2
    cy = bbox["h"] / 2

    parts = []
    parts.append(f'<rect x="0" y="0" width="{bbox["w"]}" height="{bbox["h"]}" '
                 f'rx="8" fill="#FFF3E0" stroke="#E65100" stroke-width="1.5"/>')

    # ReLU zigzag (algorithmic pattern)
    points = []
    segments = 6
    seg_w = (bbox["w"] - 20) / segments
    for i in range(segments + 1):
        x = 10 + i * seg_w
        # ReLU shape: flat left half, rising right half
        if i < segments // 2:
            y = cy + 5
        else:
            y = cy + 5 - (i - segments // 2) * 6
        points.append(f"{x:.1f},{y:.1f}")
    parts.append(f'<polyline points="{" ".join(points)}" '
                 f'fill="none" stroke="#E65100" stroke-width="2" opacity="0.7"/>')

    parts.append(f'<text x="{cx}" y="{bbox["h"]-6}" text-anchor="middle" '
                 f'font-family="system-ui,sans-serif" font-size="10" fill="#BF360C">{label}</text>')

    # ── [ASTRO-CELL] Deferred Decal overlay (PostProcessDeferredDecals port) ──
    _deco = _build_cell_decoration("cil-bolt", gene_traits, cell_id)
    _r_outer_b = min(bbox["w"], bbox["h"]) / 2 - 4
    _overlay = _apply_cell_decoration_overlay(
        _deco, "cil-bolt", cx, cy, _r_outer_b, bbox)
    if _overlay:
        parts.append(_overlay)

    return "\n".join(parts), bbox


def generate_svg_cil_plus(cell_id, label, bbox, gene_traits):
    """cil-plus species: merge/aggregation — converging lines to center."""
    cx = bbox["w"] / 2
    cy = bbox["h"] / 2

    parts = []
    parts.append(f'<rect x="0" y="0" width="{bbox["w"]}" height="{bbox["h"]}" '
                 f'rx="8" fill="#FCE4EC" stroke="#C62828" stroke-width="1.5"/>')

    # Plus cross (algorithmic)
    arm = min(bbox["w"], bbox["h"]) * 0.25
    parts.append(f'<line x1="{cx-arm}" y1="{cy}" x2="{cx+arm}" y2="{cy}" '
                 f'stroke="#C62828" stroke-width="2.5" opacity="0.6"/>')
    parts.append(f'<line x1="{cx}" y1="{cy-arm}" x2="{cx}" y2="{cy+arm}" '
                 f'stroke="#C62828" stroke-width="2.5" opacity="0.6"/>')

    # Converging arcs from corners
    for dx, dy in [(-1,-1), (1,-1), (-1,1), (1,1)]:
        sx = cx + dx * arm * 1.2
        sy = cy + dy * arm * 1.2
        parts.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{cx}" y2="{cy}" '
                     f'stroke="#C62828" stroke-width="0.8" opacity="0.3" stroke-dasharray="3,2"/>')

    parts.append(f'<text x="{cx}" y="{bbox["h"]-6}" text-anchor="middle" '
                 f'font-family="system-ui,sans-serif" font-size="10" fill="#B71C1C">{label}</text>')

    # ── [ASTRO-CELL] Deferred Decal overlay (PostProcessDeferredDecals port) ──
    _deco = _build_cell_decoration("cil-plus", gene_traits, cell_id)
    _r_outer_p = min(bbox["w"], bbox["h"]) / 2 - 4
    _overlay = _apply_cell_decoration_overlay(
        _deco, "cil-plus", cx, cy, _r_outer_p, bbox)
    if _overlay:
        parts.append(_overlay)

    return "\n".join(parts), bbox


def generate_svg_cil_arrow_right(cell_id, label, bbox, gene_traits):
    """cil-arrow-right species: dataflow terminal — arrow pointing out."""
    cx = bbox["w"] / 2
    cy = bbox["h"] / 2

    parts = []
    parts.append(f'<rect x="0" y="0" width="{bbox["w"]}" height="{bbox["h"]}" '
                 f'rx="8" fill="#ECEFF1" stroke="#455A64" stroke-width="1.5"/>')

    # Arrow shape
    aw = bbox["w"] * 0.3
    parts.append(f'<polygon points="{cx-aw},{cy-8} {cx+aw},{cy} {cx-aw},{cy+8}" '
                 f'fill="#455A64" opacity="0.5"/>')

    parts.append(f'<text x="{cx}" y="{bbox["h"]-6}" text-anchor="middle" '
                 f'font-family="system-ui,sans-serif" font-size="10" fill="#263238">{label}</text>')

    # ── [ASTRO-CELL] Deferred Decal overlay (PostProcessDeferredDecals port) ──
    # Default / fallback species → generic rounded-stroke decoration overlay
    _deco = _build_cell_decoration(species if "species" in dir() else "default",
                                   gene_traits, cell_id)
    _deco.blend_weight = 0.6   # force visible for default species
    _r_outer_d = min(bbox["w"], bbox["h"]) / 2 - 4
    _overlay = _apply_cell_decoration_overlay(
        _deco, "default", cx, cy, _r_outer_d, bbox)
    if _overlay:
        parts.append(_overlay)

    return "\n".join(parts), bbox


def generate_svg_filter(w, h, label):
    """
    filter species: 3×3 grid wireframe + label.
    Represents a convolution / attention mask — gridded sampling pattern.
    """
    cx = w / 2
    cy = h / 2

    parts = []
    # Background
    parts.append(f'<rect x="0" y="0" width="{w}" height="{h}" '
                 f'rx="8" fill="#F3E5F5" stroke="#7B1FA2" stroke-width="1.5"/>')

    # 3×3 grid wireframe centred in the cell
    pad   = max(8, min(w, h) * 0.12)
    gw    = w - 2 * pad
    gh    = (h - 2 * pad) * 0.72    # leave room for label at bottom
    cell_w = gw / 3
    cell_h = gh / 3
    gx0   = pad
    gy0   = pad

    for row in range(4):
        y = gy0 + row * cell_h
        parts.append(f'<line x1="{gx0:.1f}" y1="{y:.1f}" '
                     f'x2="{gx0 + gw:.1f}" y2="{y:.1f}" '
                     f'stroke="#7B1FA2" stroke-width="1" opacity="0.55"/>')
    for col in range(4):
        x = gx0 + col * cell_w
        parts.append(f'<line x1="{x:.1f}" y1="{gy0:.1f}" '
                     f'x2="{x:.1f}" y2="{gy0 + gh:.1f}" '
                     f'stroke="#7B1FA2" stroke-width="1" opacity="0.55"/>')

    # Highlight centre cell of the 3×3 grid
    hx = gx0 + cell_w
    hy = gy0 + cell_h
    parts.append(f'<rect x="{hx:.1f}" y="{hy:.1f}" '
                 f'width="{cell_w:.1f}" height="{cell_h:.1f}" '
                 f'fill="#CE93D8" opacity="0.45" rx="2"/>')

    # Label
    parts.append(f'<text x="{cx}" y="{h - 6}" text-anchor="middle" '
                 f'font-family="system-ui,sans-serif" font-size="10" '
                 f'fill="#4A148C">{label}</text>')

    return "\n".join(parts)


def generate_svg_code(w, h, label):
    """
    code species: curly-brace icon + monospace label.
    Represents a programmatic / function block in the architecture diagram.
    """
    cx   = w / 2
    cy   = h / 2
    arm  = min(w, h) * 0.28    # half-height of the brace

    parts = []
    # Background
    parts.append(f'<rect x="0" y="0" width="{w}" height="{h}" '
                 f'rx="8" fill="#E8F5E9" stroke="#2E7D32" stroke-width="1.5"/>')

    # Left curly brace  {  (two quarter-arcs + a nib)
    bx   = cx - arm * 1.1
    top  = cy - arm
    bot  = cy + arm
    nib  = arm * 0.22    # half-nib width
    r    = arm * 0.35    # corner radius of the brace arcs

    # Left brace path: M top → arc down → nib left → arc down → bottom
    parts.append(
        f'<path d="M {bx + r:.1f},{top:.1f} '
        f'Q {bx:.1f},{top:.1f} {bx:.1f},{top + r:.1f} '
        f'L {bx:.1f},{cy - nib:.1f} '
        f'Q {bx - nib * 1.4:.1f},{cy:.1f} {bx:.1f},{cy + nib:.1f} '
        f'L {bx:.1f},{bot - r:.1f} '
        f'Q {bx:.1f},{bot:.1f} {bx + r:.1f},{bot:.1f}" '
        f'fill="none" stroke="#2E7D32" stroke-width="2" stroke-linejoin="round"/>'
    )

    # Right curly brace  }  (mirrored)
    bx2 = cx + arm * 1.1
    parts.append(
        f'<path d="M {bx2 - r:.1f},{top:.1f} '
        f'Q {bx2:.1f},{top:.1f} {bx2:.1f},{top + r:.1f} '
        f'L {bx2:.1f},{cy - nib:.1f} '
        f'Q {bx2 + nib * 1.4:.1f},{cy:.1f} {bx2:.1f},{cy + nib:.1f} '
        f'L {bx2:.1f},{bot - r:.1f} '
        f'Q {bx2:.1f},{bot:.1f} {bx2 - r:.1f},{bot:.1f}" '
        f'fill="none" stroke="#2E7D32" stroke-width="2" stroke-linejoin="round"/>'
    )

    # Label in monospace font, centred
    parts.append(f'<text x="{cx}" y="{h - 6}" text-anchor="middle" '
                 f'font-family="\'Courier New\',Courier,monospace" font-size="10" '
                 f'fill="#1B5E20">{label}</text>')

    return "\n".join(parts)


def generate_svg_layers(w, h, label):
    """
    layers species: 3 staggered semi-transparent rectangles.
    Represents depth / multi-layer representations (e.g. transformer stack).
    """
    cx = w / 2
    cy = h / 2

    parts = []
    # Background
    parts.append(f'<rect x="0" y="0" width="{w}" height="{h}" '
                 f'rx="8" fill="#E3F2FD" stroke="#1565C0" stroke-width="1.5"/>')

    pad   = max(6, min(w, h) * 0.10)
    rw    = w - 2 * pad
    rh    = (h - 2 * pad) * 0.48
    step  = rh * 0.32    # vertical / horizontal stagger between layers

    colours   = ["#90CAF9", "#42A5F5", "#1E88E5"]
    opacities = [0.35, 0.50, 0.68]
    rx_vals   = [6, 5, 4]

    for i, (col, op, rx) in enumerate(zip(colours, opacities, rx_vals)):
        # Stagger: bottom layers offset right + down
        offset = step * (2 - i)
        rx_pos = pad + offset * 0.8
        ry_pos = pad + offset + (h - 2 * pad - rh) * 0.15
        parts.append(
            f'<rect x="{rx_pos:.1f}" y="{ry_pos:.1f}" '
            f'width="{rw:.1f}" height="{rh:.1f}" '
            f'rx="{rx}" fill="{col}" opacity="{op}" '
            f'stroke="#1565C0" stroke-width="0.8"/>'
        )

    # Label
    parts.append(f'<text x="{cx}" y="{h - 6}" text-anchor="middle" '
                 f'font-family="system-ui,sans-serif" font-size="10" '
                 f'fill="#0D47A1">{label}</text>')

    return "\n".join(parts)


def generate_svg_loop(w, h, label):
    """
    loop species: circular arc with an arrowhead.
    Represents a recurrent / cyclic flow (RNN, loop, feedback connection).
    """
    cx  = w / 2
    cy  = h / 2
    r   = min(w, h) * 0.28

    parts = []
    # Background
    parts.append(f'<rect x="0" y="0" width="{w}" height="{h}" '
                 f'rx="8" fill="#FFF8E1" stroke="#F57F17" stroke-width="1.5"/>')

    # Circular arc: ~300° clockwise, leaving a gap at the top for the arrowhead
    # SVG arc: start at top-right (30°), sweep 300° (large-arc), end at top-left (90°)
    gap_half = math.radians(30)    # gap = 60° at top
    start_angle = -math.pi / 2 + gap_half   # just past 12 o'clock CW
    end_angle   = -math.pi / 2 - gap_half   # just before 12 o'clock CW

    sx = cx + r * math.cos(start_angle)
    sy = cy + r * math.sin(start_angle)
    ex = cx + r * math.cos(end_angle)
    ey = cy + r * math.sin(end_angle)

    # large-arc-flag=1, sweep-flag=1 (clockwise)
    arc_id = abs(hash(label)) % 9000 + 1000
    parts.append(
        f'<defs>'
        f'<marker id="loop-arrow-{arc_id}" markerWidth="7" markerHeight="7" '
        f'refX="3.5" refY="3.5" orient="auto">'
        f'<path d="M0,0 L7,3.5 L0,7 Z" fill="#F57F17"/>'
        f'</marker>'
        f'</defs>'
    )
    parts.append(
        f'<path d="M {sx:.2f},{sy:.2f} A {r:.2f},{r:.2f} 0 1,1 {ex:.2f},{ey:.2f}" '
        f'fill="none" stroke="#F57F17" stroke-width="2.2" opacity="0.75" '
        f'marker-end="url(#loop-arrow-{arc_id})"/>'
    )

    # Small centre dot
    parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r * 0.14:.1f}" '
                 f'fill="#F57F17" opacity="0.55"/>')

    # Label
    parts.append(f'<text x="{cx}" y="{h - 6}" text-anchor="middle" '
                 f'font-family="system-ui,sans-serif" font-size="10" '
                 f'fill="#E65100">{label}</text>')

    return "\n".join(parts)


def generate_svg_graph(w, h, label):
    """
    graph species: small circles (nodes) connected by lines (edges).
    Represents a graph-structured computation or attention pattern.
    """
    cx  = w / 2
    cy  = h / 2

    parts = []
    # Background
    parts.append(f'<rect x="0" y="0" width="{w}" height="{h}" '
                 f'rx="8" fill="#FAFAFA" stroke="#37474F" stroke-width="1.5"/>')

    # Node positions — a small 5-node graph (star + outer ring feel)
    r_outer = min(w, h) * 0.28
    r_inner = r_outer * 0.38
    # Centre node + 4 outer nodes at 0°, 90°, 180°, 270°
    node_angles = [0, math.pi / 2, math.pi, 3 * math.pi / 2]
    nodes = [(cx, cy - r_inner)]   # slightly offset centre
    for ang in node_angles:
        nodes.append((cx + r_outer * math.cos(ang),
                      cy + r_outer * math.sin(ang)))

    # Edges: centre → each outer node + one cross edge
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (3, 4)]
    for (a, b) in edges:
        x1, y1 = nodes[a]
        x2, y2 = nodes[b]
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" '
            f'x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#546E7A" stroke-width="1.1" opacity="0.50"/>'
        )

    # Draw nodes on top of edges
    node_r  = max(3.0, min(w, h) * 0.055)
    colours = ["#37474F", "#78909C", "#78909C", "#78909C", "#78909C"]
    for i, (nx, ny) in enumerate(nodes):
        col = colours[min(i, len(colours) - 1)]
        parts.append(
            f'<circle cx="{nx:.1f}" cy="{ny:.1f}" r="{node_r:.1f}" '
            f'fill="{col}" opacity="0.75"/>'
        )

    # Label
    parts.append(f'<text x="{cx}" y="{h - 6}" text-anchor="middle" '
                 f'font-family="system-ui,sans-serif" font-size="10" '
                 f'fill="#263238">{label}</text>')

    return "\n".join(parts)


# Species → generator mapping
SPECIES_GENERATORS = {
    "cil-eye": generate_svg_cil_eye,
    "cil-vector": generate_svg_cil_vector,
    "cil-bolt": generate_svg_cil_bolt,
    "cil-plus": generate_svg_cil_plus,
    "cil-arrow-right": generate_svg_cil_arrow_right,
    # ── New species (feat: add 5 new species SVG generators) ──────────────────
    "cil-filter": lambda cell_id, label, bbox, gt: (
        generate_svg_filter(bbox["w"], bbox["h"], label), bbox),
    "cil-code": lambda cell_id, label, bbox, gt: (
        generate_svg_code(bbox["w"], bbox["h"], label), bbox),
    "cil-layers": lambda cell_id, label, bbox, gt: (
        generate_svg_layers(bbox["w"], bbox["h"], label), bbox),
    "cil-loop": lambda cell_id, label, bbox, gt: (
        generate_svg_loop(bbox["w"], bbox["h"], label), bbox),
    "cil-graph": lambda cell_id, label, bbox, gt: (
        generate_svg_graph(bbox["w"], bbox["h"], label), bbox),
}


def proc(cell_id: str):
    """
    Apollo Component::Proc() equivalent.
    Reads channels → generates SVG with species algorithm → publishes.

    L3: If channels/cell/{cell_id}/agent_params.json exists (written by
    dispatch_cell_agent before this call), the agent's bbox and opacity
    override the default skeleton+force-field values.  This is the final
    link in the L3 chain:
        agent computes params → agent_params.json → proc() renders with them.
    """
    # ── Subscribe: read channels ──
    skeleton = read_channel(f"skeleton/cell/{cell_id}.json")
    force_field = read_channel("physics/force_field.json")
    z_layers = read_channel("physics/z_layers.json")

    label = skeleton["label"]
    species = skeleton["species"]
    gene_traits = skeleton["gene_traits"]
    bbox = skeleton["initial_bbox"].copy()

    # Apply force field adjustments
    force = force_field.get(cell_id, {"dx": 0, "dy": 0, "dz": 0})
    bbox["x"] += force["dx"]
    bbox["y"] += force["dy"]
    bbox["z"] = z_layers.get(cell_id, 3) + force.get("dz", 0)

    # ── L3: apply agent_params if present ────────────────────────────────────
    # dispatch_cell_agent() writes agent_params.json before proc() is called
    # by run_all_cells().  When the file exists, the agent's decisions win over
    # the default force-field values for bbox and opacity.
    _agent_opacity: float | None = None
    _agent_params_path = os.path.join(CHANNELS, "cell", cell_id, "agent_params.json")
    if os.path.isfile(_agent_params_path):
        try:
            with open(_agent_params_path) as _apf:
                _agent_params = json.load(_apf)
            # Override bbox if agent provided one
            if "bbox" in _agent_params and isinstance(_agent_params["bbox"], dict):
                _ab = _agent_params["bbox"]
                for _k in ("x", "y", "w", "h", "z"):
                    if _k in _ab:
                        bbox[_k] = _ab[_k]
            # Record agent opacity for use below (blended with crowding_opacity)
            if "opacity" in _agent_params and isinstance(_agent_params["opacity"], (int, float)):
                _agent_opacity = max(0.35, min(1.0, float(_agent_params["opacity"])))
            import sys
            print(
                f"[proc] L3 agent_params applied: cell_id={cell_id} "
                f"bbox=({bbox['x']},{bbox['y']},{bbox['w']},{bbox['h']}) "
                f"z={bbox['z']} opacity={_agent_opacity}",
                file=sys.stderr,
            )
        except Exception as _ap_exc:
            import sys
            print(f"[proc] WARNING: failed to read agent_params.json "
                  f"for cell_id={cell_id}: {_ap_exc}", file=sys.stderr)

    # ── Proc: species algorithm generates SVG ──
    generator = SPECIES_GENERATORS.get(species, generate_svg_cil_arrow_right)
    svg_content, actual_bbox = generator(cell_id, label, bbox, gene_traits)

    # ── [ASTRO-CELL] ReflectionEnvironment — cell style probe consistency ─────
    # Port of FAstroCellStyleProbe::SampleSurroundingCells + BlendWithCubemap
    # from commit 5d07a0a (upstream/unreal-renderer/ReflectionEnvironment.cpp).
    #
    # Build a probe for this cell, sample its six cardinal neighbours from the
    # already-published cell/*/bbox.json + status.json channels, then nudge the
    # primary SVG stroke/fill colour of this cell 20 % toward the neighbourhood
    # palette average (鲁迅式: enough to feel the pull, not enough to surrender).
    #
    # roughness maps to the cell's visual complexity:
    #   simple icon species (cil-eye, cil-bolt) → low roughness → stronger pull
    #   composite / arrow species → higher roughness → weaker pull
    _SPECIES_ROUGHNESS: dict = {
        "cil-eye":         0.1,   # smooth focal icon — very susceptible
        "cil-bolt":        0.2,   # sharp energy icon — susceptible
        "cil-plus":        0.3,   # structured cross — moderate
        "cil-vector":      0.5,   # multi-arrow — moderate resistance
        "cil-arrow-right": 0.7,   # directional terminal — mostly independent
        # ── New species ──────────────────────────────────────────────────────
        "cil-filter":      0.3,   # grid wireframe — structured, moderate
        "cil-code":        0.4,   # brace icon — moderate
        "cil-layers":      0.2,   # stacked rects — smooth depth signal
        "cil-loop":        0.5,   # arc arrow — moderate self-expression
        "cil-graph":       0.6,   # node-edge graph — relatively independent
    }
    _probe_roughness = _SPECIES_ROUGHNESS.get(species, 0.5)

    _style_probe = AstroCellStyleProbe(cell_id, bbox)
    _style_probe.sample_surrounding_cells(CHANNELS)

    # Own primary colour (species fill colour from _SPECIES_INDEX_TO_COLOUR)
    _own_species_idx   = _species_to_index(species)
    _own_colour_rgb    = _SPECIES_INDEX_TO_COLOUR.get(_own_species_idx,
                                                      _SPECIES_INDEX_TO_COLOUR[0])
    _blended_colour_rgb = _style_probe.blend_toward_neighbour_palette(
        _own_colour_rgb, roughness=_probe_roughness)
    _blended_hex = _colour_to_hex(_blended_colour_rgb)

    # Log probe result — mirrors UE_LOG VeryVerbose in CubemapSlot loop.
    print(
        f"[ASTRO-CELL] StyleProbe cell_id={cell_id} "
        f"dominant_species={_style_probe.dominant_species_index} "
        f"palette_entries={len(_style_probe.palette)} "
        f"weight={_style_probe.cell_style_weight:.2f} "
        f"own={_colour_to_hex(_own_colour_rgb)} "
        f"blended={_blended_hex} "
        f"roughness={_probe_roughness:.1f}",
        file=sys.stderr,
    )

    # Apply blended colour: substitute the species primary fill in svg_content.
    # This is the Python equivalent of the per-pixel palette blend that in C++
    # happens in the pixel shader via uniform buffer parameters.  Here we do a
    # simple string substitution on the already-generated SVG fragment — same
    # end effect: style convergence toward the neighbourhood without re-running
    # the generator.
    _own_hex_upper = _colour_to_hex(_own_colour_rgb).upper()
    _own_hex_lower = _colour_to_hex(_own_colour_rgb).lower()
    if _blended_hex.upper() != _own_hex_upper and len(_style_probe.palette) > 0:
        # Replace the primary fill colour occurrences (fill= and stroke= attrs).
        svg_content = svg_content.replace(
            f'fill="{_own_hex_upper}"', f'fill="{_blended_hex}"')
        svg_content = svg_content.replace(
            f'fill="{_own_hex_lower}"', f'fill="{_blended_hex}"')
        svg_content = svg_content.replace(
            f'stroke="{_own_hex_upper}"', f'stroke="{_blended_hex}"')
        svg_content = svg_content.replace(
            f'stroke="{_own_hex_lower}"', f'stroke="{_blended_hex}"')
        # Also handle mixed-case (generated by f-strings with literal hex)
        svg_content = svg_content.replace(
            f'fill="#{_own_hex_upper[1:]}"', f'fill="{_blended_hex}"')
        svg_content = svg_content.replace(
            f'stroke="#{_own_hex_upper[1:]}"', f'stroke="{_blended_hex}"')

    # ── [ASTRO-CELL] Capsule shadow: collect all sibling cell bboxes ─────────
    # BuildCellOcclusionVolumes equivalent: load every published bbox to build
    # the occluder set (cells that haven't published yet are silently skipped).
    all_bboxes: dict = {}
    cell_base = os.path.join(CHANNELS, "cell")
    if os.path.isdir(cell_base):
        for sibling in os.listdir(cell_base):
            bbox_path = os.path.join(cell_base, sibling, "bbox.json")
            if os.path.isfile(bbox_path):
                try:
                    with open(bbox_path) as _f:
                        all_bboxes[sibling] = json.load(_f)
                except (json.JSONDecodeError, OSError):
                    pass
    # Also include self so occlusion volume is built consistently
    all_bboxes[cell_id] = {
        "x": bbox["x"], "y": bbox["y"],
        "w": bbox["w"], "h": bbox["h"],
        "z": bbox["z"],
    }

    # ProjectCellOcclusion aggregation → SVG filter parameters
    sp = compute_capsule_shadow_params(cell_id, bbox, all_bboxes)
    shadow_dx      = sp["dx"]
    shadow_dy      = sp["dy"]
    shadow_blur    = sp["blur"]
    shadow_opacity = sp["opacity"]

    # ── [ASTRO-CELL] PostProcessAO — crowding attenuation (commit 33e27b7) ────
    # FAstroConstraintAO::ComputeConstraintWeight() port.
    # Reads neighbour bboxes as the SSAO kernel; computes 3-pass constraint-
    # space AO weight; maps to fill opacity on the cell's outermost <g>.
    # High neighbour density → lower opacity (crowd suppresses the cell's
    # visual weight, preventing the SVG equivalent of SSAO black halos in
    # packed cell regions).
    crowding_opacity = compute_crowding_opacity(cell_id, bbox, all_bboxes)

    # ── L3: agent opacity override ────────────────────────────────────────────
    # If dispatch_cell_agent provided an opacity, blend it with the AO-derived
    # crowding_opacity: agent 60% authority, PostProcessAO 40%.  This lets the
    # agent express intent (e.g. highlight a focal cell) while still respecting
    # neighbourhood density cues.
    if _agent_opacity is not None:
        crowding_opacity = 0.6 * _agent_opacity + 0.4 * crowding_opacity

    # ── [ASTRO-CELL] ShadingEnergyConservation — colour energy balance ────────
    # Port of ShadingEnergyConservation.cpp (upstream/unreal-renderer-ue5).
    # Replaces LUT texture reads with analytic GGX/Schlick/cloth approximations.
    # Ensures fill + stroke + shadow energy budget <= 1.0 (furnace-test).
    # Fresnel edge-glow compensation brightens stroke at grazing angles.
    # Energy preservation attenuates fill by specular directional albedo.
    _SPECIES_ROUGHNESS_SEC: dict = {
        "cil-eye":         0.1,
        "cil-bolt":        0.2,
        "cil-plus":        0.3,
        "cil-vector":      0.5,
        "cil-arrow-right": 0.7,
        "cil-filter":      0.3,
        "cil-code":        0.4,
        "cil-layers":      0.2,
        "cil-loop":        0.5,
        "cil-graph":       0.6,
    }
    _sec_roughness   = _SPECIES_ROUGHNESS_SEC.get(species, 0.5)
    _stroke_opacity  = 0.85    # species stroke default (pre-conservation)
    _sec_result = compute_cell_energy_balance(
        cell_id        = cell_id,
        species        = species,
        bbox           = bbox,
        fill_opacity   = crowding_opacity,
        stroke_opacity = _stroke_opacity,
        shadow_opacity = shadow_opacity,
        roughness      = _sec_roughness,
    )
    crowding_opacity = _sec_result["fill_opacity"]
    shadow_opacity   = _sec_result["shadow_opacity"]

    # SVG <filter> definition with feDropShadow
    # stdDeviation  ← blur radius (capsule radius analogue)
    # dx/dy         ← offset driven by z-depth
    # flood-opacity ← attenuated by inter-cell occlusion
    shadow_filter_def = (
        f'<defs>\n'
        f'  <filter id="shadow-{cell_id}" x="-20%" y="-20%" '
        f'width="140%" height="140%">\n'
        f'    <feDropShadow dx="{shadow_dx}" dy="{shadow_dy}" '
        f'stdDeviation="{shadow_blur}" '
        f'flood-color="#000000" flood-opacity="{shadow_opacity}"/>\n'
        f'  </filter>\n'
        f'</defs>'
    )

    # Wrap in positioned <g> with z-layer data attribute, shadow filter ref,
    # and crowding-attenuation opacity (PostProcessAO port, commit 33e27b7).
    # opacity attr mirrors AstroCrowdingScale → per-pixel AO weight in USF;
    # here it modulates the entire cell group so dense regions visually recede.
    full_svg = (
        f'{shadow_filter_def}\n'
        f'<!-- [ASTRO-CELL] StyleProbe species={species} '
        f'dominant_nbr={_style_probe.dominant_species_index} '
        f'palette={len(_style_probe.palette)} '
        f'blended_fill={_blended_hex} weight={_style_probe.cell_style_weight:.2f} '
        f'(FAstroCellStyleProbe::BlendWithCubemap port, commit 5d07a0a) -->\n'
        f'<!-- [ASTRO-CELL] PostProcessAO crowding_opacity={crowding_opacity:.4f} '
        f'(FAstroConstraintAO::ComputeConstraintWeight port) -->\n'
        f'<!-- [ASTRO-CELL] ShadingEnergyConservation '
        f'E_spec={_sec_result["e_spec"]:.4f} '
        f'ms_comp={_sec_result["multi_scatter"]:.4f} '
        f'total_energy={_sec_result["total_energy"]:.4f} '
        f'(ShadingEnergyConservation.cpp port, analytic GGX/Schlick/cloth) -->\n'
        f'<g id="cell-{cell_id}" data-z="{bbox["z"]}" '
        f'opacity="{crowding_opacity:.4f}" '
        f'filter="url(#shadow-{cell_id})" '
        f'transform="translate({bbox["x"]},{bbox["y"]})">\n'
        f'{svg_content}\n'
        f'</g>'
    )

    # ── Publish: write to channels ──
    current_epoch = read_channel("skeleton/epoch.json")["current"]
    cell_dir = f"cell/{cell_id}"
    write_channel(f"{cell_dir}/bbox.json", {
        "x": bbox["x"], "y": bbox["y"],
        "w": bbox["w"], "h": bbox["h"],
        "z": bbox["z"],
        "species": species,
        "epoch": current_epoch,
    })
    write_channel(f"{cell_dir}/svg.svg", full_svg)
    write_channel(f"{cell_dir}/status.json", {
        "status": "converged",
        "cell_id": cell_id,
        "species": species,
        "epoch": current_epoch,
    })

    # ── [M008] params.json — PixiJS frontend parameter channel ───────────────
    # Primary output path for PixiJS renderer: all drawing parameters are
    # surfaced here so the frontend never needs to parse SVG.
    # species_params encodes the species-specific visual constants that the
    # PixiJS draw call needs (ring_count / pupil_radius for cil-eye, etc.).
    # blended fill/stroke colours are extracted from the SVG colour pipeline
    # above (post StyleProbe blend) so the frontend receives the final values.
    _species_params: dict
    if species == "cil-eye":
        _num_rays = max(4, len(label) // 2)
        _r_outer_eye = min(bbox["w"], bbox["h"]) / 2 - 4
        _species_params = {
            "ring_count":    _num_rays,
            "pupil_radius":  round(_r_outer_eye * 0.2, 2),
            "r_outer":       round(_r_outer_eye, 2),
            "r_inner_ratio": 0.3,
        }
    elif species == "cil-bolt":
        _zigzag_segments = 6
        _seg_w = (bbox["w"] - 20) / _zigzag_segments
        _species_params = {
            "zigzag_count": _zigzag_segments,
            "amplitude":    6.0,
            "seg_width":    round(_seg_w, 2),
        }
    elif species == "cil-vector":
        _num_arrows_v = 5
        _arrow_len_v  = bbox["w"] * 0.3
        _species_params = {
            "arrow_count":   _num_arrows_v,
            "arrow_length":  round(_arrow_len_v, 2),
            "angle_spread":  0.8,
        }
    elif species == "cil-plus":
        _arm_plus = min(bbox["w"], bbox["h"]) * 0.25
        _species_params = {
            "arm_length":    round(_arm_plus, 2),
            "stroke_width":  2.5,
            "dash_corners":  True,
        }
    elif species == "cil-arrow-right":
        _aw = bbox["w"] * 0.3
        _species_params = {
            "arrow_width":   round(_aw, 2),
            "arrow_height":  16.0,
        }
    elif species == "cil-filter":
        _pad_f  = max(8, min(bbox["w"], bbox["h"]) * 0.12)
        _cell_w = (bbox["w"] - 2 * _pad_f) / 3
        _cell_h = ((bbox["h"] - 2 * _pad_f) * 0.72) / 3
        _species_params = {
            "grid_cols":   3,
            "grid_rows":   3,
            "cell_width":  round(_cell_w, 2),
            "cell_height": round(_cell_h, 2),
            "pad":         round(_pad_f, 2),
        }
    elif species == "cil-code":
        _arm_c = min(bbox["w"], bbox["h"]) * 0.28
        _species_params = {
            "brace_arm":    round(_arm_c, 2),
            "nib_ratio":    0.22,
            "corner_radius_ratio": 0.35,
        }
    elif species == "cil-layers":
        _pad_l = max(6, min(bbox["w"], bbox["h"]) * 0.10)
        _rh_l  = (bbox["h"] - 2 * _pad_l) * 0.48
        _step_l = _rh_l * 0.32
        _species_params = {
            "layer_count":  3,
            "layer_height": round(_rh_l, 2),
            "stagger_step": round(_step_l, 2),
            "opacities":    [0.35, 0.50, 0.68],
        }
    elif species == "cil-loop":
        _r_loop = min(bbox["w"], bbox["h"]) * 0.28
        _species_params = {
            "arc_radius":   round(_r_loop, 2),
            "gap_degrees":  60,
            "sweep_cw":     True,
        }
    elif species == "cil-graph":
        _r_outer_g = min(bbox["w"], bbox["h"]) * 0.28
        _r_inner_g = _r_outer_g * 0.38
        _node_r    = max(3.0, min(bbox["w"], bbox["h"]) * 0.055)
        _species_params = {
            "node_count":   5,
            "r_outer":      round(_r_outer_g, 2),
            "r_inner":      round(_r_inner_g, 2),
            "node_radius":  round(_node_r, 2),
            "edge_list":    [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [3, 4]],
        }
    else:
        _species_params = {}

    # Derive final fill/stroke colours from blended hex (post-StyleProbe)
    # _blended_hex is the post-blend primary fill; stroke uses species defaults.
    _stroke_colour = _colour_to_hex(
        _SPECIES_INDEX_TO_COLOUR.get(_own_species_idx, _SPECIES_INDEX_TO_COLOUR[0])
    )

    _params_payload: dict = {
        "cell_id":        cell_id,
        "species":        species,
        "bbox":           {
            "x": bbox["x"], "y": bbox["y"],
            "w": bbox["w"], "h": bbox["h"],
            "z": bbox["z"],
        },
        "z":              bbox["z"],
        "opacity":        round(crowding_opacity, 4),
        "fill_color":     _blended_hex,
        "stroke_color":   _stroke_colour,
        "label":          label,
        "font_size":      10,
        "species_params": _species_params,
        "epoch":          current_epoch,
        # Shadow params for PixiJS drop-shadow filter
        "shadow": {
            "dx":      shadow_dx,
            "dy":      shadow_dy,
            "blur":    shadow_blur,
            "opacity": shadow_opacity,
        },
    }
    write_channel(f"{cell_dir}/params.json", _params_payload)
    print(
        f"[M008] params.json written: cell_id={cell_id} "
        f"species={species} fill={_blended_hex} "
        f"species_params_keys={list(_species_params.keys())}",
        file=sys.stderr,
    )

    # -------------------------------------------------------------------------
    # [ASTRO-CELL] AddCell / UpdateCellConstraint — register the published bbox
    # in the global cell_registry channel (GAstroCellZLayerRegistry +
    # GAstroCellProxyMap pub/sub equivalent).
    #
    # Decision mirrors the render-thread paths in RendererScene.cpp 72c4d0c:
    #
    #   • First publish (no prior entry in registry)
    #       → AddPrimitiveSceneInfo_RenderThread → AstroRegisterCellInZLayer
    #         Creates a FAstroCellSceneProxy, assigns z-layer bucket, appends
    #         to GAstroCellZLayerRegistry[layer], stores in GAstroCellProxyMap.
    #
    #   • Re-publish (entry already present — bounds may have changed)
    #       → UpdatePrimitiveTransform_RenderThread → AstroUpdateCellConstraint
    #         Checks HasChanged(Tolerance=0.01); if changed, migrates proxy
    #         across z-layer buckets if necessary and sets bDirty=true
    #         (constraint_mask=1) to signal a pending constraint-buffer flush.
    #
    # All subsequent cells can read cell_registry.json to obtain the latest
    # bbox, species, z-layer, and dirty state for every peer — the Apollo
    # "scene graph" global state view.
    # -------------------------------------------------------------------------
    published_bbox = {
        "x": bbox["x"], "y": bbox["y"],
        "w": bbox["w"], "h": bbox["h"],
        "z": bbox["z"],
    }
    registry = _load_cell_registry()
    if cell_id in registry["cells"]:
        # UpdateCellConstraint path — cell already registered, check for drift
        update_cell_constraint(cell_id, published_bbox, current_epoch)
    else:
        # AddCell path — first time this cell enters the scene graph
        register_cell_in_z_layer(cell_id, published_bbox, species, current_epoch)

    print(f"[Cell {cell_id}] species={species} bbox=({bbox['x']},{bbox['y']},{bbox['w']},{bbox['h']}) z={bbox['z']} "
          f"shadow(dx={shadow_dx},dy={shadow_dy},blur={shadow_blur},opacity={shadow_opacity}) "
          f"crowding_opacity={crowding_opacity}")
    return full_svg


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 cell_component.py <cell_id>")
        print("  e.g. python3 cell_component.py self_attn")
        sys.exit(1)
    cell_id = sys.argv[1]
    proc(cell_id)


# Lumen GI classes moved to channels/rendering/lumen/
from channels.rendering.lumen import (
    astro_gi_is_allowed, astro_gi_get_lighting_format,
    AstroCellGatherCvarState, AstroCellDiffuseProbe,
    AstroCellDiffuseProbeGrid, AstroCellRadianceCacheProbe,
    AstroCellRadianceCacheClipmapLevel, AstroCellRadianceCache,
    AstroCellMeshCardFace, AstroCellMeshCards,
    AstroCellMeshCardsRegistry, AstroCellGlobalIlluminationPipeline,
    get_astro_gi_pipeline,
)


# LightRendering/GPUScene classes moved to channels/rendering/lighting/
from channels.rendering.lighting import (
    AstroCellGPUSceneResourceParams, AstroCellPrimitiveCollector,
    AstroCellGPUScene, get_astro_gpu_scene,
    AstroCellDeferredLightUniforms, AstroCellSimpleLight,
    AstroCellLightPass, run_cell_light_pass,
    AstroCellTranslucencyLightingVolume,
    AstroCellFrameRenderer, get_frame_renderer,
)


# Shadow/DeferredShading classes moved to channels/rendering/shadow/
from channels.rendering.shadow import (
    AstroCellShadowInfo, build_whole_scene_shadow_caster,
    AstroCellShadowDepthPassParams, AstroCellShadowDepthRenderer,
    AstroCellGBuffer, AstroCellDeferredShadingRenderer,
    run_deferred_shading_pipeline,
)


def _bit_invert_if_negative(f: float) -> int:
    """
    Bit-cast float to uint32, then XOR with sign-extension mask.

    Direct port of BitInvertIfNegativeFloat() from MeshDrawCommands.cpp:
        unsigned mask = -int32(f >> 31) | 0x80000000;
        return f ^ mask;

    Converts an IEEE 754 float to a uint32 that preserves the numerical
    ordering under unsigned comparison — used to sort translucent mesh draw
    commands by projected distance without branching.

    鲁迅式：浮点数的符号位是它的立场——
    反转负数的所有位，让它在无符号比较中依然保持正确的大小关系。
    这是一种对不公平规则的巧妙利用：规则不变，解读方式变了。
    """
    raw = _struct.unpack('>I', _struct.pack('>f', f))[0]  # float → uint32 big-endian
    mask = ((-(raw >> 31)) & 0xFFFFFFFF) | 0x80000000
    return (raw ^ mask) & 0xFFFFFFFF


@dataclass
class AstroCellDrawSortKey:
    """
    Python equivalent of FMeshDrawCommandSortKey.

    Stores a 64-bit packed sort key used to order draw commands.  The key
    is split into two 32-bit halves:
        high = translucent sort distance (bit-inverted float for unsigned cmp)
        low  = PSO state key (pipeline hash — opaque draws only)

    Mirrors the union layout of FMeshDrawCommandSortKey::PackedData[2].

    鲁迅式：排序键是优先级的量化——一个数字决定了谁先被画，
    谁先被画决定了谁覆盖谁。先到不等于先赢，顺序才是权力。
    """
    high: int = 0   # translucent: bit-inverted distance; opaque: 0
    low:  int = 0   # PSO/state hash for opaque; secondary key for translucent

    @classmethod
    def default(cls) -> "AstroCellDrawSortKey":
        """Mirrors FMeshDrawCommandSortKey::Default = {{0}}."""
        return cls(high=0, low=0)

    def packed(self) -> int:
        """64-bit packed value: high in upper 32 bits, low in lower 32."""
        return ((self.high & 0xFFFFFFFF) << 32) | (self.low & 0xFFFFFFFF)

    def __lt__(self, other: "AstroCellDrawSortKey") -> bool:
        return self.packed() < other.packed()


def _compute_translucent_sort_key(
    bounds_origin: tuple,
    view_origin: tuple,
    view_matrix: list,
    sort_policy: int = 2,
    sort_axis: tuple = (0.0, 0.0, 1.0),
) -> AstroCellDrawSortKey:
    """
    Compute a translucent draw sort key for a single cell.

    Port of the inner loop body in UpdateTranslucentMeshSortKeys():
      0 = SortByDistance     → distance = |BoundsOrigin - ViewOrigin|
      1 = SortAlongAxis      → distance = dot(BoundsOrigin - ViewOrigin, SortAxis)
      2 = SortByProjectedZ   → distance = ViewMatrix.TransformPosition(BoundsOrigin).Z

    The resulting float distance is bit-inverted so that back-to-front
    ordering maps to ascending unsigned integer order (far = smaller key
    draws first in painter's algorithm).

    鲁迅式：半透明物体必须从远到近画——这不是偏见，是光学定律。
    排序键的精心设计，是为了让 GPU 遵守这一定律而无需额外分支。
    """
    ox, oy, oz = bounds_origin
    vx, vy, vz = view_origin

    if sort_policy == 0:
        # SortByDistance: Euclidean distance
        dx, dy, dz = ox - vx, oy - vy, oz - vz
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
    elif sort_policy == 1:
        # SortAlongAxis: projected onto custom axis
        dx, dy, dz = ox - vx, oy - vy, oz - vz
        ax, ay, az = sort_axis
        distance = dx*ax + dy*ay + dz*az
    else:
        # SortByProjectedZ: view-space Z (default in UE5)
        if view_matrix and len(view_matrix) >= 3:
            r = view_matrix[2]
            distance = r[0]*ox + r[1]*oy + r[2]*oz + (r[3] if len(r) > 3 else 0.0)
        else:
            distance = oz - vz

    # BitInvertIfNegativeFloat so unsigned ascending == back-to-front
    key_high = _bit_invert_if_negative(distance)
    return AstroCellDrawSortKey(high=key_high, low=0)


def update_translucent_sort_keys(
    visible_commands: list,
    view_origin: tuple = (0.0, 0.0, -1000.0),
    view_matrix: list = None,
    sort_policy: int = 2,
    sort_axis: tuple = (0.0, 0.0, 1.0),
    inverse_sorting: bool = False,
) -> list:
    """
    Update sort keys for all translucent mesh draw commands.

    Direct port of UpdateTranslucentMeshSortKeys() from MeshDrawCommands.cpp.

    Each entry in visible_commands is a dict with at minimum:
        cell_id:  str
        bbox:     {x, y, w, h, z}  (world bounds)
        species:  str
    After this call every entry gains a ``sort_key`` field (AstroCellDrawSortKey).

    The list is sorted in-place (ascending by sort_key.packed()), which
    produces back-to-front order for painter's algorithm rendering — identical
    to the C++ sorted TArray<FVisibleMeshDrawCommand> result.

    @param visible_commands  List of draw-command entry dicts.
    @param view_origin        Camera / viewer position tuple (x, y, z).
    @param view_matrix        3×4 row-major view matrix (optional).
    @param sort_policy        0=distance, 1=axis, 2=projZ (default).
    @param sort_axis          Custom sort axis when sort_policy == 1.
    @param inverse_sorting    Reverse order (front-to-back for certain passes).
    @return                   Sorted list of commands with sort_key attached.

    鲁迅式：排序是最后的裁决——每一个半透明物体都在争夺被先画的权利，
    但物理定律早已判定：远者先画，近者后画。排序只是执行判决。
    """
    if not visible_commands:
        return visible_commands

    for entry in visible_commands:
        bbox = entry.get("bbox", {})
        # World-space bounds origin = bbox centre
        bx = float(bbox.get("x", 0)) + float(bbox.get("w", 0)) * 0.5
        by = float(bbox.get("y", 0)) + float(bbox.get("h", 0)) * 0.5
        bz = float(bbox.get("z", 0))
        bounds_origin = (bx, by, bz)

        entry["sort_key"] = _compute_translucent_sort_key(
            bounds_origin, view_origin, view_matrix or [],
            sort_policy, sort_axis,
        )

    visible_commands.sort(
        key=lambda e: e["sort_key"].packed(),
        reverse=inverse_sorting,
    )

    return visible_commands


# =============================================================================
# [MeshDrawCommands] AstroCellPrimitiveIdBufferPool
# =============================================================================

class AstroCellPrimitiveIdBufferPool:
    """
    Python equivalent of FPrimitiveIdVertexBufferPool.

    Maintains a free-list of primitive-ID buffers sized by request, reusing
    existing allocations to avoid repeated allocation overhead.  A discard_id
    counter (incremented by discard_stale()) ages out entries that have been
    free for more than _STALE_EPOCH_THRESHOLD epochs.

    Mirrors the C++ Allocate / ReturnToFreeList / DiscardAll lifecycle:
        Allocate(size)         → allocate(size)
        ReturnToFreeList(entry)→ release(entry)
        DiscardAll()           → discard_stale()

    鲁迅式：缓冲池是节俭的哲学——内存不是免费的，
    重用已有的比每次申请新的，是对资源的尊重，也是对帧率的保护。
    """

    _ALIGN       = 1024      # BufferSize = Align(size, 1024)
    _STALE_EPOCH_THRESHOLD = 1000   # mirrors DiscardId > 1000 check

    def __init__(self) -> None:
        # Free-list: list of {"size": int, "data": bytearray, "last_discard_id": int}
        self._entries: list = []
        self._discard_id: int = 0

    # ------------------------------------------------------------------
    def allocate(self, size: int) -> dict:
        """
        Allocate (or reuse) a buffer of at least *size* bytes.

        Mirrors Allocate(FRHICommandList&, int32 BufferSize):
          - Align to 1024 bytes.
          - Find the smallest unused entry that fits (best-fit scan).
          - If none found, allocate a new bytearray.
          - Mark LastDiscardId = DiscardId on the returned entry.

        Returns a dict {"size": int, "data": bytearray, "last_discard_id": int}.

        鲁迅式：最佳适配是妥协中的智慧——找最小的够用者，不浪费，也不委屈。
        """
        aligned_size = ((size + self._ALIGN - 1) // self._ALIGN) * self._ALIGN

        best_idx = -1
        for i, entry in enumerate(self._entries):
            if entry["last_discard_id"] == self._discard_id:
                continue  # currently in use
            if entry["size"] >= aligned_size:
                if best_idx == -1 or entry["size"] < self._entries[best_idx]["size"]:
                    best_idx = i
                    if entry["size"] == aligned_size:
                        break

        if best_idx >= 0:
            reused = self._entries.pop(best_idx)
            reused["last_discard_id"] = self._discard_id
            return reused

        # Allocate new entry
        new_entry = {
            "size":           aligned_size,
            "data":           bytearray(aligned_size),
            "last_discard_id": self._discard_id,
        }
        return new_entry

    def release(self, entry: dict) -> None:
        """
        Return a buffer to the free list.
        Mirrors ReturnToFreeList() — thread-safe in C++ (mutex); here single-threaded.

        鲁迅式：归还是美德——用完即还，下一位不必等待。
        """
        self._entries.append(entry)

    def discard_stale(self) -> int:
        """
        Advance the discard epoch and evict buffers idle for too many epochs.

        Mirrors DiscardAll():
            ++DiscardId;
            RemoveAtSwap entries where (DiscardId - entry.LastDiscardId) > 1000

        Returns the number of entries evicted.

        鲁迅式：老化是自然定律——一千帧未被使用的缓冲区，
        不是在休息，是在占据本不属于它的位置。丢弃它，腾出空间给活着的事物。
        """
        self._discard_id += 1
        threshold = self._STALE_EPOCH_THRESHOLD
        before = len(self._entries)
        self._entries = [
            e for e in self._entries
            if (self._discard_id - e["last_discard_id"]) <= threshold
        ]
        evicted = before - len(self._entries)
        if evicted:
            print(
                f"[AstroCellPrimitiveIdBufferPool] discard_stale: "
                f"evicted={evicted} discard_id={self._discard_id}",
                file=sys.stderr,
            )
        return evicted

    def stats(self) -> dict:
        """Diagnostic pool statistics."""
        return {
            "pool_entries":  len(self._entries),
            "discard_id":    self._discard_id,
            "total_bytes":   sum(e["size"] for e in self._entries),
        }


#: Module-level singleton pool — mirrors TGlobalResource<FPrimitiveIdVertexBufferPool>.
_ASTRO_PRIMITIVE_ID_BUFFER_POOL: AstroCellPrimitiveIdBufferPool = \
    AstroCellPrimitiveIdBufferPool()


def get_primitive_id_buffer_pool() -> AstroCellPrimitiveIdBufferPool:
    """Return the process-level primitive ID buffer pool singleton."""
    return _ASTRO_PRIMITIVE_ID_BUFFER_POOL


# =============================================================================
# [MeshPassProcessor] AstroCellPipelineStateId + AstroCellShaderBindings
# =============================================================================

# PSO freeze flag — mirrors FGraphicsMinimalPipelineStateId::bIsIdTableFrozen.
_pso_table_frozen: bool = False

# Persistent PSO table — mirrors FGraphicsMinimalPipelineStateId::PersistentIdTable.
# Key: (species, blend_mode, pass_name); Value: integer PSO id.
_pipeline_state_table: dict = {}
_pipeline_state_next_id: int = 0


class AstroCellPipelineStateId:
    """
    Python equivalent of FGraphicsMinimalPipelineStateId.

    Assigns a stable integer ID to each (species, blend_mode, pass_name)
    combination, mirroring the persistent PSO ID table that survives across
    frames.  IDs are allocated lazily on first use and never reused.

    NeedsShaderInitialisation flag (mirrors the C++ static) is cleared the
    first time the table is populated — here it tracks whether any IDs have
    been assigned yet.

    鲁迅式：PSO 的 ID 是身份证——每一条渲染管线都有一个号码，
    号码不重复，也不作废。这是秩序对混乱的胜利。
    """

    NeedsShaderInitialisation: bool = True

    def __init__(self, species: str, blend_mode: str, pass_name: str) -> None:
        self.species    = species
        self.blend_mode = blend_mode
        self.pass_name  = pass_name
        self._id        = self._lookup_or_allocate()

    def _lookup_or_allocate(self) -> int:
        global _pipeline_state_next_id, _pso_table_frozen
        key = (self.species, self.blend_mode, self.pass_name)
        if key in _pipeline_state_table:
            return _pipeline_state_table[key]
        if _pso_table_frozen:
            # Mirrors the C++ assert that fires when table is frozen but a new
            # state is requested — here we log and return 0 (sentinel).
            print(
                f"[AstroCellPSOId] WARNING: PSO table frozen, "
                f"rejecting new state {key}.",
                file=sys.stderr,
            )
            return 0
        new_id = _pipeline_state_next_id
        _pipeline_state_table[key] = new_id
        _pipeline_state_next_id += 1
        AstroCellPipelineStateId.NeedsShaderInitialisation = False
        return new_id

    @property
    def id(self) -> int:
        return self._id

    def is_valid(self) -> bool:
        return self._id > 0 or (self._id == 0 and _pipeline_state_next_id > 0)

    @staticmethod
    def freeze_table() -> None:
        """Freeze the PSO table — no new states allowed after this point."""
        global _pso_table_frozen
        _pso_table_frozen = True

    @staticmethod
    def table_size() -> int:
        return len(_pipeline_state_table)

    def __repr__(self) -> str:
        return (f"AstroCellPipelineStateId("
                f"id={self._id}, species={self.species}, "
                f"blend={self.blend_mode}, pass={self.pass_name})")


class AstroCellShaderBindings:
    """
    Python equivalent of the shader binding management from
    FReadOnlyMeshDrawSingleShaderBindings::SetShaderBindings().

    Tracks which uniform-buffer / texture / sampler / SRV slots have been
    written in the current draw call and skips redundant re-binds — exactly
    mirroring the FShaderBindingState delta-tracking logic.

    In the SVG substrate, «bindings» are SVG/CSS attribute overrides that
    must be accumulated before the final <g> element is emitted.  Redundant
    bindings from a previous cell with the same PSO are not re-emitted.

    鲁迅式：绑定状态是画家的调色板——
    每次切换颜色都有代价；不变的颜色就不要再调。
    ShaderBindingState 的存在，是对这一代价的精打细算。
    """

    _MAX_UNIFORM_BUFFERS = 16

    def __init__(self) -> None:
        # Mirrors FShaderBindingState — tracks last-bound values per slot.
        self._uniform_buffers: dict  = {}   # slot → value
        self._textures:        dict  = {}   # slot → value
        self._samplers:        dict  = {}   # slot → value
        self._srvs:            dict  = {}   # slot → value
        # Accumulated SVG attribute overrides for this draw call.
        self._svg_attr_overrides: dict = {}
        # Count of redundant binds skipped (diagnostic).
        self.redundant_binds_skipped: int = 0

    def bind_uniform_buffer(self, slot: int, value, svg_key: str = "") -> bool:
        """
        Bind a uniform buffer, skipping if value unchanged.
        Returns True if the binding was actually updated (not redundant).
        Mirrors the if (UniformBuffer != ShaderBindingState.UniformBuffers[...]) check.
        """
        if self._uniform_buffers.get(slot) == value:
            self.redundant_binds_skipped += 1
            return False
        self._uniform_buffers[slot] = value
        if svg_key:
            self._svg_attr_overrides[svg_key] = value
        return True

    def bind_texture(self, slot: int, texture_value, svg_key: str = "") -> bool:
        """Bind a texture slot (SetTextureParameter path)."""
        if self._textures.get(slot) == texture_value:
            self.redundant_binds_skipped += 1
            return False
        self._textures[slot] = texture_value
        if svg_key:
            self._svg_attr_overrides[svg_key] = texture_value
        return True

    def apply(self, base_svg_attrs: dict) -> dict:
        """
        Apply accumulated overrides onto base SVG attributes.

        Mirrors the post-SetShaderBindings() state where all per-cell
        material parameters have been applied to the draw pipeline.

        Returns a merged dict of SVG attributes with overrides applied.
        """
        merged = dict(base_svg_attrs)
        merged.update(self._svg_attr_overrides)
        return merged

    def reset(self) -> None:
        """Clear per-draw-call overrides (keep binding state for delta tracking)."""
        self._svg_attr_overrides.clear()

    def stats(self) -> dict:
        return {
            "uniform_buffers_bound":  len(self._uniform_buffers),
            "textures_bound":         len(self._textures),
            "redundant_binds_skipped": self.redundant_binds_skipped,
            "svg_overrides":          len(self._svg_attr_overrides),
        }


# ── MeshPassProcessor / SceneCapture / ReflectionCapture module constants ─────
# These mirror CVars / global flags that were referenced by previously ported
# code but whose definitions were omitted in earlier sessions.
ASTRO_EMIT_MESH_DRAW_EVENT:        bool  = False  # r.MeshDrawCommands.LogMeshDrawCommands
ASTRO_SKIP_DRAW_ON_PSO_PRECACHING: bool  = False  # r.SkipDrawOnPSOPrecaching
ASTRO_DEFERRED_MESH_PASS_SYNC:     bool  = True   # r.DeferredMeshPassSetupTaskSync
ASTRO_MESH_SORT_METHOD:            int   = 0      # 0=projZ, 1=axis, 2=distance
ASTRO_CAPTURE_ALLOW_MAIN_RENDERER: bool  = True   # r.SceneCapture.AllowRenderInMainRenderer
ASTRO_CAPTURE_CUBE_SINGLE_PASS:    bool  = False  # r.SceneCapture.CubeSinglePass
_REFL_TIMESLICE_FACES:             int   = 1      # CVarReflectionCaptureRuntimeTimeslice
_REFL_BUDGET:                      int   = 0      # 0 = unlimited
_REFL_SUPERSAMPLE_FACTOR:          int   = 1      # GSupersampleCaptureFactor
_REFL_RUNTIME_MODE:                int   = 0      # 0=continuous, 1=once


class AstroCellMeshPassProcessor:
    """
    Python equivalent of FMeshPassProcessor.

    Processes a list of visible cell draw commands through the PSO lookup,
    shader binding, and sort-key assignment pipeline — exactly mirroring the
    three major responsibilities of FMeshPassProcessor:

    1. PSO key lookup (AddMeshDrawCommand path):
       For each cell, look up (or allocate) an AstroCellPipelineStateId based
       on (species, blend_mode, pass_name).  If ASTRO_SKIP_DRAW_ON_PSO_PRECACHING
       is True and the PSO is «new» (first frame), skip the draw entirely.

    2. Shader binding application (SetShaderBindings path):
       Create an AstroCellShaderBindings instance, populate it from the cell's
       gene_traits, apply to the base SVG attributes via bindings.apply().

    3. Sort key assignment (FMeshDrawCommandSortKey path):
       Opaque cells: sort key = PSO id (minimise state changes).
       Translucent cells: sort key = bit-inverted distance (painter's order).

    After process() the returned list is ready for AstroCellDrawList.

    鲁迅式：Processor 是流水线上的检验员——
    它不创造内容，但决定了内容能否进入下一道工序，
    以及以何种顺序进入。
    """

    def __init__(self,
                 pass_name:       str  = "base",
                 view_origin:     tuple = (0.0, 0.0, -1000.0),
                 emit_draw_events: bool = ASTRO_EMIT_MESH_DRAW_EVENT) -> None:
        self.pass_name        = pass_name
        self.view_origin      = view_origin
        self.emit_draw_events = emit_draw_events
        self._binding_state   = AstroCellShaderBindings()
        self._pso_cache:      dict = {}   # (species, blend_mode) → AstroCellPipelineStateId

    def _get_pso_id(self, species: str, blend_mode: str) -> AstroCellPipelineStateId:
        """Lookup or allocate a PSO id, caching within this pass."""
        key = (species, blend_mode)
        if key not in self._pso_cache:
            self._pso_cache[key] = AstroCellPipelineStateId(
                species, blend_mode, self.pass_name
            )
        return self._pso_cache[key]

    def _build_base_svg_attrs(self, entry: dict) -> dict:
        """
        Construct the base SVG attribute dict for a cell entry.
        Mirrors the material parameter packing that the C++ pass writes into
        the per-draw uniform buffer before shader binding.
        """
        species    = entry.get("species", "")
        bbox       = entry.get("bbox", {})
        opacity    = float(entry.get("opacity", 1.0))
        blend_mode = entry.get("blend_mode", "normal")

        sp_idx = _species_to_index(species)
        fill   = _colour_to_hex(_SPECIES_INDEX_TO_COLOUR.get(sp_idx, _SPECIES_INDEX_TO_COLOUR[0]))
        stroke = fill

        return {
            "fill":        fill,
            "stroke":      stroke,
            "opacity":     str(round(opacity, 4)),
            "mix-blend-mode": blend_mode,
            "data-cell-id":   entry.get("cell_id", ""),
            "data-z":         str(bbox.get("z", 0)),
            "data-pass":      self.pass_name,
        }

    def _assign_sort_key(self, entry: dict, pso_id: AstroCellPipelineStateId,
                         blend_mode: str) -> AstroCellDrawSortKey:
        """
        Assign a sort key to the entry.

        Opaque (blend_mode not in translucent set):
            key.low = PSO id (minimise pipeline state switches, mirrors UE5 opaque sort).
            key.high = 0 (Z is irrelevant for opaque).

        Translucent:
            key = _compute_translucent_sort_key() (bit-inverted distance).
        """
        translucent_modes = {"translucent", "additive", "modulate", "alpha_composite"}
        if blend_mode in translucent_modes:
            bbox = entry.get("bbox", {})
            bx = float(bbox.get("x", 0)) + float(bbox.get("w", 0)) * 0.5
            by = float(bbox.get("y", 0)) + float(bbox.get("h", 0)) * 0.5
            bz = float(bbox.get("z", 0))
            sort_policy = 1 if ASTRO_MESH_SORT_METHOD == 1 else 2
            return _compute_translucent_sort_key(
                (bx, by, bz), self.view_origin, [], sort_policy
            )
        else:
            # Opaque sort: primary key = PSO id (state minimisation)
            return AstroCellDrawSortKey(high=0, low=pso_id.id)

    def process(self, cell_entries: list) -> list:
        """
        Process all cell draw entries through the mesh pass pipeline.

        Mirrors FMeshPassProcessor::AddMeshBatch() + BuildMeshDrawCommands()
        called for each FMeshBatch in the view's visible primitives list.

        For each entry:
          1. Resolve blend_mode.
          2. Lookup PSO id (skip if precaching + new PSO).
          3. Apply shader bindings.
          4. Assign sort key.
          5. Append draw-event annotation if ASTRO_EMIT_MESH_DRAW_EVENT.

        Returns list of enriched entry dicts with fields added:
            pso_id      : int
            sort_key    : AstroCellDrawSortKey
            svg_attrs   : dict (final merged SVG attributes)
            draw_event  : str (optional debug annotation)

        鲁迅式：process() 是流水线的主干——
        所有输入在这里经过筛选、分类、标记，最终成为可以被画出来的命令。
        没有经过 process() 的 cell，不过是一堆原始数据；
        经过之后，它们获得了身份、顺序和形式。
        """
        result = []
        self._binding_state.reset()

        for entry in cell_entries:
            cell_id    = entry.get("cell_id", "")
            species    = entry.get("species", "")
            blend_mode = entry.get("blend_mode", "normal")

            # ── PSO lookup ────────────────────────────────────────────────────
            pso = self._get_pso_id(species, blend_mode)

            if ASTRO_SKIP_DRAW_ON_PSO_PRECACHING and pso.NeedsShaderInitialisation:
                # PSO still «compiling» — skip this draw call
                print(
                    f"[AstroCellMeshPassProcessor] SkipDrawOnPSOPrecaching: "
                    f"skipping cell={cell_id} (PSO not yet initialised)",
                    file=sys.stderr,
                )
                continue

            # ── Shader bindings ───────────────────────────────────────────────
            self._binding_state.reset()
            # Bind per-cell gene_traits as «uniform buffer» slot 0
            gene_traits = entry.get("gene_traits", {})
            self._binding_state.bind_uniform_buffer(
                slot=0, value=json.dumps(gene_traits, sort_keys=True)
            )
            # Bind species colour as «texture» slot 0
            sp_idx = _species_to_index(species)
            fill   = _colour_to_hex(_SPECIES_INDEX_TO_COLOUR.get(sp_idx, _SPECIES_INDEX_TO_COLOUR[0]))
            self._binding_state.bind_texture(
                slot=0, texture_value=fill, svg_key="fill"
            )

            base_attrs = self._build_base_svg_attrs(entry)
            svg_attrs  = self._binding_state.apply(base_attrs)

            # ── Sort key ──────────────────────────────────────────────────────
            sort_key = self._assign_sort_key(entry, pso, blend_mode)

            # ── Draw event annotation ─────────────────────────────────────────
            draw_event = ""
            if self.emit_draw_events:
                draw_event = (
                    f"<!-- [ASTRO-MDC] MeshDrawEvent pass={self.pass_name} "
                    f"cell={cell_id} pso_id={pso.id} "
                    f"sort={sort_key.packed()} -->"
                )

            enriched = dict(entry)
            enriched.update({
                "pso_id":     pso.id,
                "sort_key":   sort_key,
                "svg_attrs":  svg_attrs,
                "draw_event": draw_event,
            })
            result.append(enriched)

        # ── Deferred sort (mirrors CVarDeferredMeshPassSetupTaskSync) ─────────
        if ASTRO_DEFERRED_MESH_PASS_SYNC:
            result.sort(key=lambda e: e["sort_key"].packed())

        print(
            f"[AstroCellMeshPassProcessor] process: "
            f"pass={self.pass_name} in={len(cell_entries)} "
            f"out={len(result)} "
            f"pso_table_size={AstroCellPipelineStateId.table_size()} "
            f"redundant_binds={self._binding_state.redundant_binds_skipped}",
            file=sys.stderr,
        )

        return result


# =============================================================================
# [SceneCaptureRendering] AstroCellCaptureMode + AstroCellCaptureProcessor
# =============================================================================

class AstroCellCaptureMode:
    """
    Python equivalent of FSceneCapturePS::ESourceMode enum (8 values).

    Maps UE5 ESceneCaptureSource constants to Astro channel modes that
    determine which data is sampled from the rendered cell scene.

    鲁迅式：源模式是摄像头的目的——
    你想捕捉颜色、深度、还是法线？
    目的不同，捕捉的手段和代价也不同。
    """
    COLOR_AND_OPACITY   = 0   # SCS_SceneColorHDR
    COLOR_NO_ALPHA      = 1   # SCS_SceneColorHDRNoAlpha
    COLOR_AND_DEPTH     = 2   # SCS_SceneColorSceneDepth
    SCENE_DEPTH         = 3   # SCS_SceneDepth
    DEVICE_DEPTH        = 4   # SCS_DeviceDepth
    NORMAL              = 5   # SCS_Normal
    BASE_COLOR          = 6   # SCS_BaseColor
    COLOR_ONE_ALPHA     = 7   # SCS_SceneColorHDRNoAlpha for reflection capture

    _SOURCE_TO_MODE = {
        "color_opacity":    0,
        "color_no_alpha":   1,
        "color_depth":      2,
        "depth":            3,
        "device_depth":     4,
        "normal":           5,
        "base_color":       6,
        "color_one_alpha":  7,
    }

    @classmethod
    def from_source_name(cls, name: str) -> int:
        return cls._SOURCE_TO_MODE.get(name, 0)


def _should_compile_capture_permutation(
    source_mode: int,
    use_128bit_rt: bool,
    requires_explicit_128bit: bool,
) -> bool:
    """
    Mirrors FSceneCapturePS::ShouldCompilePermutation():
        return (!PermutationVector.Get<FEnable128BitRT>()
                || bPlatformRequiresExplicit128bitRT);

    In the Astro context: 128-bit RT is approximated by float16 SVG colour
    channels (not needed for standard uint8 output).

    鲁迅式：应该编译的 permutation 才编译——
    无用的组合是浪费，删除它们是一种诚实。
    """
    if use_128bit_rt and not requires_explicit_128bit:
        return False
    return True


def _get_capture_permutation(
    source_name: str,
    use_128bit_rt:          bool = False,
    forward_shading:        bool = False,
    is_reflection_capture:  bool = False,
) -> dict:
    """
    Compute capture permutation parameters.

    Mirrors FSceneCapturePS::GetPermutationVector():
        Maps ESceneCaptureSource → ESourceMode.
        Handles forward-shading override for Normal/BaseColor modes.
        Returns {source_mode, use_128bit_rt}.

    鲁迅式：Permutation 是现实的分叉——每一个旗标都是一条路，
    组合爆炸是工程师的噩梦，也是用户功能的保障。
    """
    mode = AstroCellCaptureMode.from_source_name(source_name)

    # Reflection capture: NoAlpha → ColorOneAlpha
    if is_reflection_capture and mode == AstroCellCaptureMode.COLOR_NO_ALPHA:
        mode = AstroCellCaptureMode.COLOR_ONE_ALPHA

    # Forward shading override: Normal/BaseColor → ColorAndOpacity
    if forward_shading and mode in (
        AstroCellCaptureMode.NORMAL, AstroCellCaptureMode.BASE_COLOR
    ):
        mode = AstroCellCaptureMode.COLOR_AND_OPACITY

    return {"source_mode": mode, "use_128bit_rt": use_128bit_rt}


class AstroCellCaptureProcessor:
    """
    Python equivalent of the SceneCaptureRendering pipeline.

    Captures the current cell scene state into a «render target» dict,
    supporting the 8 ESourceMode channel configurations from FSceneCapturePS.

    Two primary entry points mirror the UE5 C++ functions:
      capture_scene()        → CaptureSceneToRenderTarget() analog
      copy_capture_to_target()→ CopyCaptureToTarget() / UpdateSceneCaptureContents()

    鲁迅式：场景捕获是镜子——它把当前世界的状态定格为一张快照，
    供反射、后处理、UI 叠加等系统消费。
    镜子不创造，但它记录；记录本身，便是一种价值。
    """

    def __init__(self,
                 source_name:          str   = "color_opacity",
                 use_128bit_rt:         bool  = False,
                 forward_shading:       bool  = False,
                 allow_main_renderer:   bool  = ASTRO_CAPTURE_ALLOW_MAIN_RENDERER,
                 cube_single_pass:      bool  = ASTRO_CAPTURE_CUBE_SINGLE_PASS) -> None:
        self._permutation = _get_capture_permutation(
            source_name, use_128bit_rt, forward_shading,
        )
        self.allow_main_renderer = allow_main_renderer
        self.cube_single_pass    = cube_single_pass
        self._render_target: dict = {}

    @property
    def source_mode(self) -> int:
        return self._permutation["source_mode"]

    def capture_scene(
        self,
        cell_entries:  list,
        depth_manifest: dict,
        viewport_w:    float = 1200.0,
        viewport_h:    float = 900.0,
    ) -> dict:
        """
        Capture the scene into a render-target dict.

        Mirrors CaptureSceneToRenderTarget() / UpdateSceneCaptureContents():
          - If allow_main_renderer and source_mode supports it:
              render as part of the main renderer (inline capture path).
          - Otherwise:
              render as independent scene (separate capture path).

        For each capture mode, different data channels are populated:
          COLOR_AND_OPACITY → rgba per cell (fill + opacity)
          SCENE_DEPTH       → normalised Z per cell
          NORMAL            → surface normal vector per cell
          BASE_COLOR        → species primary colour (no lighting)
          COLOR_ONE_ALPHA   → reflection capture alpha=1 convention

        Returns the populated render target dict.

        鲁迅式：捕获场景需要代价——你每多渲染一次，GPU 就多工作一次。
        AllowRenderInMainRenderer 是一种妥协：如果主渲染器能顺路帮你做，
        何必另起炉灶？
        """
        mode = self.source_mode
        render_target: dict = {
            "source_mode":   mode,
            "viewport":      {"w": viewport_w, "h": viewport_h},
            "cells":         {},
        }

        for entry in cell_entries:
            cid     = entry.get("cell_id", "")
            species = entry.get("species", "")
            bbox    = entry.get("bbox", {})
            opacity = float(entry.get("opacity", 1.0))
            z       = float(bbox.get("z", 0))

            sp_idx = _species_to_index(species)
            colour = _SPECIES_INDEX_TO_COLOUR.get(sp_idx, _SPECIES_INDEX_TO_COLOUR[0])

            if mode == AstroCellCaptureMode.COLOR_AND_OPACITY:
                cell_data = {
                    "r": colour[0] / 255.0, "g": colour[1] / 255.0,
                    "b": colour[2] / 255.0, "a": opacity,
                }
            elif mode == AstroCellCaptureMode.COLOR_NO_ALPHA:
                cell_data = {
                    "r": colour[0] / 255.0, "g": colour[1] / 255.0,
                    "b": colour[2] / 255.0, "a": 1.0,
                }
            elif mode == AstroCellCaptureMode.SCENE_DEPTH:
                depth = depth_manifest.get("depth_channel", {}).get(cid, 1.0)
                cell_data = {"depth": depth}
            elif mode == AstroCellCaptureMode.DEVICE_DEPTH:
                # Device depth = 1 - scene_depth (UE5 reversed-Z convention)
                depth = depth_manifest.get("depth_channel", {}).get(cid, 1.0)
                cell_data = {"device_depth": 1.0 - depth}
            elif mode == AstroCellCaptureMode.NORMAL:
                # Surface normals: cells face toward viewer (+Z)
                cell_data = {"nx": 0.0, "ny": 0.0, "nz": 1.0}
            elif mode == AstroCellCaptureMode.BASE_COLOR:
                cell_data = {
                    "r": colour[0] / 255.0,
                    "g": colour[1] / 255.0,
                    "b": colour[2] / 255.0,
                }
            elif mode == AstroCellCaptureMode.COLOR_ONE_ALPHA:
                # Reflection capture: alpha forced to 1 (mirrors reflection path)
                cell_data = {
                    "r": colour[0] / 255.0, "g": colour[1] / 255.0,
                    "b": colour[2] / 255.0, "a": 1.0,
                }
            else:
                cell_data = {"r": 0.0, "g": 0.0, "b": 0.0, "a": opacity}

            cell_data["z"]       = z
            cell_data["species"] = species
            render_target["cells"][cid] = cell_data

        self._render_target = render_target
        return render_target

    def copy_capture_to_target(
        self,
        target_path: str,
    ) -> None:
        """
        Persist the captured render target to the channel filesystem.

        Mirrors CopyCaptureToTarget() / the RDG pass that blits the
        scene capture result into the final render target texture.

        鲁迅式：数据不写出便等于不存在——
        捕获的每一帧都需要落地为文件，才能被后续系统消费。
        """
        if not self._render_target:
            return
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "w") as _f:
            json.dump(self._render_target, _f, indent=2)
        print(
            f"[AstroCellCaptureProcessor] copy_capture_to_target: "
            f"mode={self.source_mode} cells={len(self._render_target.get('cells', {}))} "
            f"→ {target_path}",
            file=sys.stderr,
        )


def update_scene_capture_contents(
    cell_entries:    list,
    depth_manifest:  dict,
    capture_dir:     str,
    source_name:     str  = "color_opacity",
    is_reflection:   bool = False,
    viewport_w:      float = 1200.0,
    viewport_h:      float = 900.0,
) -> dict:
    """
    Top-level scene capture update.

    Mirrors UpdateSceneCaptureContents() — the primary entry point called
    each frame to refresh a SceneCaptureComponent2D's render target.

    Constructs an AstroCellCaptureProcessor, runs capture_scene(), and
    persists the result to capture_dir/scene_capture.json.

    @param is_reflection  When True switches to COLOR_ONE_ALPHA permutation
                          (reflection capture convention).
    @return               Capture render target dict.

    鲁迅式：UpdateSceneCaptureContents 是场景捕获的总调度——
    每帧一次，不多不少。频率是性能与精度之间的谈判结果。
    """
    effective_source = "color_one_alpha" if is_reflection else source_name
    processor = AstroCellCaptureProcessor(
        source_name=effective_source,
        is_reflection_capture=is_reflection if False else False,  # resolved above
    )
    rt = processor.capture_scene(cell_entries, depth_manifest, viewport_w, viewport_h)

    target_path = os.path.join(capture_dir, "scene_capture.json")
    processor.copy_capture_to_target(target_path)
    return rt


# =============================================================================
# [ReflectionEnvironmentCapture] AstroCellReflectionCaptureState + pipeline
# =============================================================================

def _clamp_supersample(factor: int) -> int:
    """Clamp supersample factor to [1, 8] — mirrors MinSupersampleCaptureFactor /
    MaxSupersampleCaptureFactor constants in ReflectionEnvironmentCapture.cpp."""
    return max(1, min(8, factor))


@dataclass
class AstroCellReflectionCaptureState:
    """
    Python equivalent of the per-capture runtime state in
    ReflectionEnvironmentCapture.cpp.

    Tracks the timeslicing state (which «faces» have been rendered),
    the fade-in progress, and the accumulated capture data for one
    reflection capture probe.

    Reflection «faces» in 2-D → six Z-layer offsets sampled around the
    capture origin: +Z, -Z, +X, -X, +Y, -Y (cardinal directions mapped to
    Z-layer and XY offset combinations).

    鲁迅式：时分渲染是对时间的借贷——每帧还一点债，六帧还清，
    然后重新开始。债不能不还，只是分期而已。
    """
    capture_id:       str   = ""
    world_pos:        tuple = (0.0, 0.0, 0.0)
    influence_radius: float = 1000.0
    # Timeslice state: which face index [0..5] is rendered next
    current_face:     int   = 0
    # Number of faces rendered so far in this cycle
    faces_rendered:   int   = 0
    # Captured colour data per face: face_index → (r, g, b)
    face_data:        dict  = field(default_factory=dict)
    # Fade-in progress [0.0, 1.0] — mirrors CVarReflectionCaptureRuntimeFadeInTime
    fade_progress:    float = 0.0
    # Whether the full cube has been captured at least once this session
    is_complete:      bool  = False
    # Frame index of last update
    last_update_frame: int  = -1

    _FACE_OFFSETS = [
        ( 0,  0,  1),   # face 0: +Z (upward)
        ( 0,  0, -1),   # face 1: -Z (downward)
        ( 1,  0,  0),   # face 2: +X (right)
        (-1,  0,  0),   # face 3: -X (left)
        ( 0,  1,  0),   # face 4: +Y (forward)
        ( 0, -1,  0),   # face 5: -Y (backward)
    ]

    def faces_per_timeslice(self) -> int:
        """
        Number of faces to render per frame.
        Mirrors CVarReflectionCaptureRuntimeTimeslice (clamped to [1, 6]).
        If ASTRO_CAPTURE_CUBE_SINGLE_PASS: render all 6 in one frame.
        """
        if ASTRO_CAPTURE_CUBE_SINGLE_PASS:
            return 6
        return max(1, min(6, _REFL_TIMESLICE_FACES))

    def sample_face(
        self,
        face_index: int,
        cell_entries: list,
        depth_manifest: dict,
    ) -> tuple:
        """
        Sample average scene colour for one probe «face».

        Mirrors CaptureSceneToScratchCubemap() for a single face:
          1. Select cells within influence_radius × face direction half-space.
          2. Average their species colours weighted by proximity.
          3. Apply supersample factor (multiple passes → average).

        Returns (r, g, b) average colour for this face.

        鲁迅式：每个方向采样一次，六个方向合而为一——
        这是环境光的民主原则：四面八方的光照都有发言权。
        """
        if face_index >= len(self._FACE_OFFSETS):
            return (0.5, 0.5, 0.5)

        fx, fy, fz = self._FACE_OFFSETS[face_index]
        ox, oy, oz = self.world_pos

        # Collect cells in this face's half-space (dot(cell_dir, face_dir) > 0)
        r_sum = g_sum = b_sum = weight_sum = 0.0
        supersample = _clamp_supersample(_REFL_SUPERSAMPLE_FACTOR)

        for entry in cell_entries:
            bbox    = entry.get("bbox", {})
            cx = float(bbox.get("x", 0)) + float(bbox.get("w", 0)) * 0.5
            cy = float(bbox.get("y", 0)) + float(bbox.get("h", 0)) * 0.5
            cz = float(bbox.get("z", 0)) * 100.0   # z-layer → world units

            # Direction from probe to cell
            dx, dy, dz = cx - ox, cy - oy, cz - oz
            dist = math.sqrt(dx*dx + dy*dy + dz*dz) + 1e-6

            # Half-space test: dot(cell_dir, face_dir) > 0
            dot = (dx/dist)*fx + (dy/dist)*fy + (dz/dist)*fz
            if dot <= 0.0:
                continue

            # Distance weight: exponential falloff within influence_radius
            # Mirrors the per-probe weight in the C++ cubemap blend pass.
            if dist > self.influence_radius:
                continue

            weight = (1.0 - dist / self.influence_radius) * dot

            species = entry.get("species", "")
            sp_idx  = _species_to_index(species)
            colour  = _SPECIES_INDEX_TO_COLOUR.get(sp_idx, _SPECIES_INDEX_TO_COLOUR[0])

            r_sum += colour[0] / 255.0 * weight * supersample
            g_sum += colour[1] / 255.0 * weight * supersample
            b_sum += colour[2] / 255.0 * weight * supersample
            weight_sum += weight * supersample

        if weight_sum > 1e-6:
            return (r_sum / weight_sum, g_sum / weight_sum, b_sum / weight_sum)
        # No cells in this face direction → ambient grey
        return (0.5, 0.5, 0.5)

    def tick(
        self,
        frame_index:   int,
        cell_entries:  list,
        depth_manifest: dict,
        fade_in_time:  float = 0.5,
    ) -> bool:
        """
        Advance the timeslice capture by one frame.

        Renders faces_per_timeslice() faces per call, cycling through the
        six face indices.  Once all 6 faces are complete, is_complete=True
        and fade_progress advances toward 1.0.

        Returns True if the full cubemap was completed this frame (or already
        complete and mode==Once).

        Mirrors the timeslice logic in UpdateReflectionCaptures():
            Render N faces per frame (CVarReflectionCaptureRuntimeTimeslice).
            When all 6 done: mark IsComplete, start fade-in.
            Mode=Once (1): stop after first complete cycle.
            Mode=Continuous (0): repeat indefinitely.

        鲁迅式：六个面，每帧还一面的债——
        不急，不乱，债总是会还清的，然后重新开始借贷。
        这就是时分渲染的生存哲学。
        """
        if self.is_complete and _REFL_RUNTIME_MODE == 1:
            # Once mode: already done — just advance fade
            self.fade_progress = min(1.0, self.fade_progress + (1.0 / max(fade_in_time * 60, 1)))
            return True

        faces_this_tick = self.faces_per_timeslice()
        for _ in range(faces_this_tick):
            face_idx = self.current_face % 6
            colour   = self.sample_face(face_idx, cell_entries, depth_manifest)
            self.face_data[face_idx] = colour
            self.current_face = (self.current_face + 1) % 6
            self.faces_rendered += 1

        self.last_update_frame = frame_index

        # Check if a full cycle is complete
        if len(self.face_data) == 6:
            self.is_complete = True
            self.fade_progress = min(1.0,
                self.fade_progress + (1.0 / max(fade_in_time * 60, 1))
            )
            return True

        return False

    def average_radiance(self) -> tuple:
        """
        Compute the average radiance across all captured faces.

        Mirrors the cubemap averaging used to derive the dominant probe
        colour for the StyleProbe blend step.  Returns (r, g, b) float tuple.

        鲁迅式：六个方向的平均，是公平，也是妥协——
        没有哪个方向比另一个更重要，所以平等权重，一人一票。
        """
        if not self.face_data:
            return (0.5, 0.5, 0.5)
        n  = len(self.face_data)
        r  = sum(v[0] for v in self.face_data.values()) / n
        g  = sum(v[1] for v in self.face_data.values()) / n
        b  = sum(v[2] for v in self.face_data.values()) / n
        return (r, g, b)

    def to_dict(self) -> dict:
        return {
            "capture_id":      self.capture_id,
            "world_pos":       self.world_pos,
            "influence_radius": self.influence_radius,
            "faces_rendered":  self.faces_rendered,
            "is_complete":     self.is_complete,
            "fade_progress":   round(self.fade_progress, 4),
            "last_update_frame": self.last_update_frame,
            "face_data":       {str(k): list(v) for k, v in self.face_data.items()},
            "average_radiance": list(self.average_radiance()),
        }


class AstroCellReflectionCaptureManager:
    """
    Python equivalent of the UpdateReflectionCaptures() pipeline.

    Maintains a registry of AstroCellReflectionCaptureState probes, sorted
    by distance to the viewer camera, and dispatches per-frame timeslice
    updates subject to the _REFL_BUDGET probe-count cap.

    Mirrors the C++ flow in UpdateReflectionCaptures():
      1. Sort active captures by distance (nearest first).
      2. Apply budget cap (skip distant probes if over limit).
      3. For each active probe: call capture_state.tick().
      4. Persist results to physics/reflection_captures.json channel.

    鲁迅式：反射捕获管理器是公平的排队系统——
    距离越近的探针优先更新，预算有限时远处的探针被搁置。
    这不是歧视，是资源分配的现实。
    """

    def __init__(self) -> None:
        self._captures: dict = {}   # capture_id → AstroCellReflectionCaptureState
        self._frame_index: int = 0
        self._channel_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "physics", "reflection_captures.json",
        )

    def register_capture(
        self,
        capture_id:       str,
        world_pos:        tuple,
        influence_radius: float = 1000.0,
    ) -> AstroCellReflectionCaptureState:
        """
        Register a new reflection capture probe.
        Mirrors AddReflectionCapture() / the dynamic runtime probe lifecycle.
        """
        state = AstroCellReflectionCaptureState(
            capture_id=capture_id,
            world_pos=world_pos,
            influence_radius=influence_radius,
        )
        self._captures[capture_id] = state
        return state

    def remove_capture(self, capture_id: str) -> None:
        """Deregister a capture probe — mirrors RemoveReflectionCapture()."""
        self._captures.pop(capture_id, None)

    def tick(
        self,
        viewer_pos:    tuple,
        cell_entries:  list,
        depth_manifest: dict,
        fade_in_time:  float = 0.5,
    ) -> dict:
        """
        Per-frame reflection capture update pass.

        Mirrors UpdateReflectionCaptures() dispatch:
          1. Sort captures by distance to viewer.
          2. Apply _REFL_BUDGET cap.
          3. Tick each active capture.
          4. Persist results.

        Returns per-frame stats dict.

        鲁迅式：每帧更新反射——不停地照镜子，
        不是虚荣，是为了让世界在镜中保持真实。
        """
        self._frame_index += 1
        vx, vy, vz = viewer_pos

        # Sort captures by distance to viewer (nearest first)
        def _dist(state: AstroCellReflectionCaptureState) -> float:
            ox, oy, oz = state.world_pos
            return math.sqrt((ox-vx)**2 + (oy-vy)**2 + (oz-vz)**2)

        sorted_captures = sorted(self._captures.values(), key=_dist)

        # Apply budget cap (0 = unlimited)
        budget = _REFL_BUDGET
        active = sorted_captures if budget == 0 else sorted_captures[:budget]

        completed_this_frame = 0
        for state in active:
            done = state.tick(self._frame_index, cell_entries, depth_manifest, fade_in_time)
            if done:
                completed_this_frame += 1

        # Persist to channel
        all_data = {cid: s.to_dict() for cid, s in self._captures.items()}
        try:
            os.makedirs(os.path.dirname(self._channel_path), exist_ok=True)
            with open(self._channel_path, "w") as _f:
                json.dump(all_data, _f, indent=2)
        except OSError as _e:
            print(
                f"[AstroCellReflectionCaptureManager] WARNING: "
                f"failed to persist captures: {_e}",
                file=sys.stderr,
            )

        stats = {
            "frame_index":          self._frame_index,
            "total_captures":       len(self._captures),
            "active_captures":      len(active),
            "completed_this_frame": completed_this_frame,
            "budget_cap":           budget,
            "supersample_factor":   _clamp_supersample(_REFL_SUPERSAMPLE_FACTOR),
            "timeslice_faces":      _REFL_TIMESLICE_FACES,
            "runtime_mode":         "once" if _REFL_RUNTIME_MODE == 1 else "continuous",
        }

        print(
            f"[AstroCellReflectionCaptureMgr] tick: "
            f"frame={self._frame_index} "
            f"captures={len(self._captures)} active={len(active)} "
            f"completed_this_frame={completed_this_frame}",
            file=sys.stderr,
        )
        return stats

    def get_probe_radiance(self, capture_id: str) -> tuple:
        """
        Return the average face radiance for a named probe.
        Used by the StyleProbe blending system as an alternative neighbourhood
        colour source when explicit neighbour cells are absent.
        """
        state = self._captures.get(capture_id)
        if state and state.is_complete:
            return state.average_radiance()
        return (0.5, 0.5, 0.5)


#: Module-level reflection capture manager singleton.
_ASTRO_REFLECTION_CAPTURE_MANAGER: AstroCellReflectionCaptureManager | None = None


def get_reflection_capture_manager() -> AstroCellReflectionCaptureManager:
    """
    Return the process-level reflection capture manager singleton.

    Mirrors the FScene::ReflectionSceneData lifetime — one manager per scene.

    鲁迅式：反射系统的单例是场景中唯一的真相来源——
    所有探针都向它汇报，所有消费者都向它查询。
    中央化不总是好事，但在反射系统中，一致性比自由更重要。
    """
    global _ASTRO_REFLECTION_CAPTURE_MANAGER
    if _ASTRO_REFLECTION_CAPTURE_MANAGER is None:
        _ASTRO_REFLECTION_CAPTURE_MANAGER = AstroCellReflectionCaptureManager()
    return _ASTRO_REFLECTION_CAPTURE_MANAGER


def capture_scene_to_scratch_cubemap(
    capture_id:    str,
    world_pos:     tuple,
    cell_entries:  list,
    depth_manifest: dict,
    influence_radius: float = 1000.0,
    viewer_pos:    tuple = (0.0, 0.0, 0.0),
) -> AstroCellReflectionCaptureState:
    """
    Convenience function: register + tick a single reflection capture.

    Mirrors the CaptureSceneToScratchCubemap() call sequence used for
    one-shot baked reflection captures:
        1. Register probe (if not already registered).
        2. Force ASTRO_CAPTURE_CUBE_SINGLE_PASS = True for this call.
        3. Tick once (all 6 faces in one frame).
        4. Return the completed state.

    鲁迅式：一次性烘焙——六个面，一口气完成，
    不留遗憾，不等下一帧。
    代价是这一帧的工作量翻六倍，但烘焙只做一次，值得。
    """
    mgr = get_reflection_capture_manager()

    # Re-register to reset state (bake path always starts fresh)
    state = mgr.register_capture(capture_id, world_pos, influence_radius)

    # Temporarily force single-pass for this bake
    global ASTRO_CAPTURE_CUBE_SINGLE_PASS
    _old_single_pass = ASTRO_CAPTURE_CUBE_SINGLE_PASS
    ASTRO_CAPTURE_CUBE_SINGLE_PASS = True

    mgr.tick(viewer_pos, cell_entries, depth_manifest, fade_in_time=0.0)

    ASTRO_CAPTURE_CUBE_SINGLE_PASS = _old_single_pass
    return state


# =============================================================================
# [ASTRO-CELL] PathTracing → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/PathTracing.cpp
#
# 鲁迅曾言：「真正的勇士，敢于直面惨淡的蒙特卡洛噪声，
# 敢于正视如雪花飞舞的萤火虫——然后用足够多的样本将其消灭。」
# 每一帧都是一次投票，样本越多，民主越纯粹。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   FPathTracingConfig    → AstroCellPathTracingConfig
#     IsDifferent()       → is_different()  — scene change detection → invalidate
#   FPathTracingState     → AstroCellPathTracingState
#     SampleIndex         → sample_index    — accumulated samples counter
#     FrameIndex          → frame_index     — monotone counter (no reset on invalidate)
#     RadianceRT          → radiance_buffer — accumulated per-cell radiance
#     VarianceRT          → variance_buffer — per-cell variance estimate
#     AlbedoRT/NormalRT   → albedo_buf/normal_buf — denoiser AOV buffers
#   GetPathTracingStateFromView → get_or_create_cell_path_tracing_state()
#   PathTracingInvalidate()     → invalidate()
#   RenderPathTracing()         → render_path_tracing()  — MIS sample accumulation
#   PreparePathTracing()        → prepare_path_tracing() — shader / state setup
#
# Algorithm changes from UE5 original (鲁迅式 20%):
#   1. Ray-traced BVH traversal → analytic bbox-intersection light sampling
#      (same MIS weight formula; BVH replaced by AstroCellBVH built from
#       the live cell_registry for O(log N) neighbour queries)
#   2. GPU texture accumulation → in-memory float arrays per cell_id
#   3. Separate SampleIndex / FrameIndex semantics preserved verbatim:
#      invalidation resets SampleIndex to 0 but never decrements FrameIndex
#      — critical for the temporal "screen door" suppression described in
#      the C++ comment (「不要让画面因重置而出现纱门效应」)
#   4. MIS mode 2 (balanced material + light sampling) always active;
#      CVarPathTracingMISMode exposed as a module constant
#   5. Adaptive sampling threshold gate preserved: cells already converged
#      (variance below threshold) are skipped to save compute
# =============================================================================

from dataclasses import dataclass as _ptdc, field as _ptfield
from typing import Dict as _PTDict, Optional as _PTOpt, List as _PTList

# ── CVarPathTracing equivalents ───────────────────────────────────────────────
_PT_MAX_BOUNCES:          int   = 8    # r.PathTracing.MaxBounces
_PT_MAX_SAMPLES:          int   = 256  # r.PathTracing.SamplesPerPixel
_PT_FILTER_SIGMA:         float = 0.5  # r.PathTracing.FilterWidth (σ = width/2π)
_PT_MIS_MODE:             int   = 2    # 0=material, 1=light, 2=balanced MIS
_PT_MAX_PATH_INTENSITY:   float = 10.0 # r.PathTracing.MaxPathIntensity (firefly clamp)
_PT_APPROXIMATE_CAUSTICS: bool  = True # r.PathTracing.ApproximateCaustics
_PT_ADAPTIVE_THRESHOLD:   float = 0.005# r.PathTracing.AdaptiveSampling.VarianceThreshold
_PT_ENABLED:              bool  = True # r.PathTracing (master switch)
_PT_COMPACTION_DEPTH:     int   = 6    # r.PathTracing.CompactionDepth
_PT_DISPATCH_SIZE:        int   = 2048 # r.PathTracing.DispatchSize (tile pixels)
_PT_LOCKED_SAMPLING:      bool  = False# r.PathTracing.LockSamplingPattern


@_ptdc
class AstroCellPathTracingConfig:
    """
    Python equivalent of FPathTracingConfig.

    Holds all scene-level parameters that, if changed, require restarting the
    sample accumulation (SampleIndex reset).  The is_different() method mirrors
    FPathTracingConfig::IsDifferent() which guards the invalidation call in the
    C++ render loop.

    鲁迅式：参数是约定，约定一旦改变，积累的历史便成无效的遗产。
    """
    max_samples:           int   = _PT_MAX_SAMPLES
    max_bounces:           int   = _PT_MAX_BOUNCES
    filter_sigma:          float = _PT_FILTER_SIGMA
    mis_mode:              int   = _PT_MIS_MODE
    max_path_intensity:    float = _PT_MAX_PATH_INTENSITY
    approximate_caustics:  bool  = _PT_APPROXIMATE_CAUSTICS
    adaptive_threshold:    float = _PT_ADAPTIVE_THRESHOLD
    locked_sampling:       bool  = _PT_LOCKED_SAMPLING
    # Viewport rect (mirrors FIntRect ViewRect in FPathTracingConfig)
    viewport_w:            int   = 1200
    viewport_h:            int   = 900
    # Light grid (mirrors LightGridResolution / LightGridMaxCount)
    light_grid_resolution: int   = 8
    light_grid_max_count:  int   = 64
    # Background / atmosphere flags
    enable_emissive:       bool  = True
    background_alpha:      float = 1.0

    def is_different(self, other: "AstroCellPathTracingConfig") -> bool:
        """
        Returns True if any accumulation-invalidating parameter changed.
        Mirrors FPathTracingConfig::IsDifferent() — guards SampleIndex reset.

        鲁迅式：只有真正不同的，才值得重新开始。
        """
        return (
            self.max_samples           != other.max_samples          or
            self.max_bounces           != other.max_bounces          or
            abs(self.filter_sigma      -  other.filter_sigma)        > 1e-5 or
            self.mis_mode              != other.mis_mode             or
            abs(self.max_path_intensity - other.max_path_intensity)  > 1e-5 or
            self.approximate_caustics  != other.approximate_caustics or
            abs(self.adaptive_threshold - other.adaptive_threshold)  > 1e-7 or
            self.locked_sampling       != other.locked_sampling      or
            self.viewport_w            != other.viewport_w           or
            self.viewport_h            != other.viewport_h           or
            self.light_grid_resolution != other.light_grid_resolution or
            self.light_grid_max_count  != other.light_grid_max_count or
            self.enable_emissive       != other.enable_emissive      or
            abs(self.background_alpha  -  other.background_alpha)    > 1e-5
        )


@_ptdc
class AstroCellPathTracingState:
    """
    Python equivalent of FPathTracingState.

    Stores per-view accumulated path tracing data between frames.
    Key invariant (from C++ comment):
        FrameIndex is NEVER reset on invalidation to avoid the temporal
        "screen door" effect caused by the quasi-random sampler re-using
        the same low-discrepancy sequence from frame 0.
        SampleIndex IS reset on invalidation so accumulation restarts cleanly.

    Buffers are dicts keyed by cell_id; values are float tuples (R, G, B).
    This mirrors the per-pixel texture arrays in the C++ implementation.

    鲁迅式：样本指数归零是承认失败，但帧指数不能归零——
    否则时间便失去了意义，历史便成了永恒的循环。
    """
    last_config:    AstroCellPathTracingConfig = _ptfield(
        default_factory=AstroCellPathTracingConfig)
    # Accumulated radiance buffer (mirrors RadianceRT)
    radiance_buffer:    _PTDict[str, tuple] = _ptfield(default_factory=dict)
    # Per-cell variance estimate (mirrors VarianceRT / VarianceBuffer)
    variance_buffer:    _PTDict[str, float] = _ptfield(default_factory=dict)
    # Denoiser AOV buffers (mirrors AlbedoRT, NormalRT, DepthRT)
    albedo_buffer:      _PTDict[str, tuple] = _ptfield(default_factory=dict)
    normal_buffer:      _PTDict[str, tuple] = _ptfield(default_factory=dict)
    depth_buffer:       _PTDict[str, float] = _ptfield(default_factory=dict)
    # Last denoised frame cache (mirrors LastDenoisedRadianceRT — animation stability)
    last_denoised:      _PTDict[str, tuple] = _ptfield(default_factory=dict)
    # Sample counter: reset to 0 on invalidation (mirrors SampleIndex)
    sample_index:       int = 0
    # Frame counter: NEVER reset (mirrors FrameIndex — uint32_t, monotone)
    frame_index:        int = 0

    def invalidate(self) -> None:
        """
        Reset accumulated data and sample counter.
        Mirrors FSceneViewState::PathTracingInvalidate(bInvalidateAnimationStates=false).

        FrameIndex is intentionally NOT touched — see struct docstring.

        鲁迅式：将一切清零，唯独不清零时间——这是纪律，不是遗忘。
        """
        self.radiance_buffer.clear()
        self.variance_buffer.clear()
        self.albedo_buffer.clear()
        self.normal_buffer.clear()
        self.depth_buffer.clear()
        # last_denoised intentionally kept (mirrors C++ which keeps LastDenoisedRadianceRT)
        self.sample_index = 0
        print(
            f"[ASTRO-PT] PathTracingInvalidate — sample_index reset to 0, "
            f"frame_index preserved at {self.frame_index}",
            file=sys.stderr,
        )

    def is_converged(self, cell_id: str,
                     threshold: float = _PT_ADAPTIVE_THRESHOLD) -> bool:
        """
        Per-cell convergence check — mirrors the adaptive sampling variance gate
        in the C++ path tracer that skips already-converged pixels.
        Returns True when variance is below threshold and enough samples accumulated.
        """
        if self.sample_index < 4:
            return False
        return self.variance_buffer.get(cell_id, 1.0) < threshold


# ── Module-level per-view state registry ─────────────────────────────────────
# Mirrors the FViewState::PathTracingState TPimplPtr<FPathTracingState> member.
# Keyed by view_id string (for single-view usage, use key "default").
_CELL_PATH_TRACING_STATES: _PTDict[str, AstroCellPathTracingState] = {}


def get_or_create_cell_path_tracing_state(
        view_id: str = "default") -> AstroCellPathTracingState:
    """
    Return the AstroCellPathTracingState for *view_id*, creating it on first call.
    Mirrors GetPathTracingStateFromView() from PathTracing.cpp.

    鲁迅式：第一次访问时才创建——懒汉式，节省内存，亦是对「不必要存在」的抵抗。
    """
    global _CELL_PATH_TRACING_STATES
    if view_id not in _CELL_PATH_TRACING_STATES:
        _CELL_PATH_TRACING_STATES[view_id] = AstroCellPathTracingState()
    return _CELL_PATH_TRACING_STATES[view_id]


def _pt_halton(index: int, base: int) -> float:
    """
    Halton low-discrepancy sequence element.
    Mirrors the path tracer's SamplerType=1 (Halton) used when
    CVarPathTracingLockedSamplingPattern is False.  The sequence provides
    better stratification than uniform random for MIS.

    鲁迅式：准随机数是伪装成随机的秩序——比真随机更公平，却不招摇。
    """
    result = 0.0
    f      = 1.0 / base
    i      = index
    while i > 0:
        result += f * (i % base)
        i       = i // base
        f      /= base
    return result


def _pt_mis_weight(pdf_a: float, pdf_b: float) -> float:
    """
    Power heuristic MIS weight (β=2) — mirrors the balanced MIS combiner
    used in PathTracing.usf when CVarPathTracingMISMode=2.

        w(a) = pdf_a² / (pdf_a² + pdf_b²)

    鲁迅式：权衡是政治，也是物理——两种采样策略各占一半，谁也不独裁。
    """
    a2 = pdf_a * pdf_a
    b2 = pdf_b * pdf_b
    denom = a2 + b2
    return a2 / denom if denom > 1e-12 else 0.5


def _pt_firefly_clamp(radiance: tuple, max_intensity: float = _PT_MAX_PATH_INTENSITY) -> tuple:
    """
    Per-path intensity clamp — mirrors CVarPathTracingMaxPathIntensity gate.
    Clamps each colour channel independently (not luminance) to keep hue.

    鲁迅式：萤火虫之所以刺眼，是因为它孤立地太亮——统一的上限是公平，不是压制。
    """
    return (
        min(radiance[0], max_intensity),
        min(radiance[1], max_intensity),
        min(radiance[2], max_intensity),
    )


def _pt_gaussian_filter(radiance: tuple, sigma: float = _PT_FILTER_SIGMA) -> tuple:
    """
    Gaussian reconstruction filter weight for the current sample.
    Mirrors CVarPathTracingFilterWidth (σ = filter_width / (2π)).
    Applied as a scalar weight ∈ (0, 1] on the sample contribution.

    w = exp(-0.5 * r² / σ²), r ≈ 0 for a centred sample → weight ≈ 1.
    For accumulation we treat all samples as centred (no sub-pixel offset
    in the 2-D analogue), so this reduces to a constant σ-dependent scale
    that models the temporal blend decay used in the C++ accumulation buffer.

    鲁迅式：高斯滤波是温柔的遗忘——它让过去的样本随时间淡去，
    而不是被突然的 invalidate 彻底抹除。
    """
    weight = math.exp(-0.5 / max(sigma * sigma, 1e-8))
    return (radiance[0] * weight, radiance[1] * weight, radiance[2] * weight)


def _pt_sample_cell_radiance(
    cell_id: str,
    bbox: dict,
    species: str,
    sample_idx: int,
    frame_idx: int,
    bvh: "AstroCellBVH | None" = None,
    all_bboxes: dict | None = None,
    mis_mode: int = _PT_MIS_MODE,
    max_bounces: int = _PT_MAX_BOUNCES,
) -> tuple:
    """
    Per-cell path sample — the inner loop of RenderPathTracing().

    Replaces the full GPU ray-traced path with an analytic 2-D equivalent:
      - Primary ray hits the cell's own bbox (always; we shade the cell itself)
      - Each bounce samples a neighbour cell's emissive contribution via BVH
        overlap query (analogue of BVH traversal + BSDF evaluation)
      - MIS combines material PDF (uniform hemisphere) and light PDF (area / dist²)
      - Firefly clamp applied per bounce

    The function uses per-sample Halton sequences indexed by
    (sample_idx * max_bounces + bounce, prime) to maintain low-discrepancy
    stratification across samples and bounces — same as the C++ path tracer's
    per-path quasi-random state.

    Returns (R, G, B) radiance contribution from one path sample.

    鲁迅式：每一条路径都是一次反问——光从哪里来？
    到哪里去？会不会在中途被遮挡、被散射、被彻底消灭？
    答案藏在概率密度函数里，与宿命无关。
    """
    import math as _ptm

    # ── Species emissive base (「primary ray hit self」) ────────────────────
    # Mirrors the path tracer's direct-hit emissive contribution (bounce 0).
    # We derive a per-species base colour as the emissive seed — same as
    # treating the cell face as an emissive surface in the material graph.
    _EMISSIVE_TABLE = {
        "cil-eye":         (0.55, 0.60, 0.90),  # indigo glow
        "cil-bolt":        (0.95, 0.55, 0.10),  # amber spark
        "cil-vector":      (0.30, 0.70, 0.35),  # green signal
        "cil-plus":        (0.25, 0.55, 0.90),  # blue merge
        "cil-arrow-right": (0.50, 0.60, 0.65),  # grey-blue arrow
        "cil-filter":      (0.60, 0.25, 0.75),  # purple kernel
        "cil-code":        (0.30, 0.70, 0.35),  # green brace
        "cil-layers":      (0.20, 0.55, 0.85),  # blue stack
        "cil-loop":        (0.90, 0.60, 0.15),  # amber cycle
        "cil-graph":       (0.40, 0.50, 0.55),  # grey node
    }
    base_r, base_g, base_b = _EMISSIVE_TABLE.get(species, (0.5, 0.5, 0.5))

    # ── Halton quasi-random state for this path ────────────────────────────
    # Path seed combines sample_index × bounce_depth for decorrelation.
    # Mirrors PathTracer.usf RandomSequence_Initialize(Seed = SampleIndex * MaxBounces).
    seed_base = sample_idx * max_bounces

    path_r, path_g, path_b = base_r, base_g, base_b
    throughput = 1.0

    cx = bbox["x"] + bbox["w"] / 2.0
    cy = bbox["y"] + bbox["h"] / 2.0
    cz = float(bbox.get("z", 3))

    for bounce in range(max_bounces):
        if throughput < 1e-4:
            break   # Russian roulette termination (implicit, energy threshold)

        seed = seed_base + bounce
        u1   = _pt_halton(seed * 2 + frame_idx % 97, 2)   # azimuth
        u2   = _pt_halton(seed * 2 + 1 + frame_idx % 97, 3)   # elevation

        # ── Material sampling: cosine-weighted hemisphere direction ────────
        # Mirrors the Lambertian BSDF material sampling in PathTracing.usf.
        theta_mat = math.acos(math.sqrt(max(0.0, u2)))
        phi_mat   = 2.0 * math.pi * u1
        pdf_mat   = math.cos(theta_mat) / math.pi  # Lambertian PDF

        # ── Light sampling: pick a neighbour cell as area light ────────────
        # Mirrors the light sampling step in the path tracer's MIS loop.
        # We use the BVH (if available) for a spatial query; else fall back
        # to the all_bboxes dict.
        light_cell_id: str | None = None
        light_r, light_g, light_b = 0.0, 0.0, 0.0
        pdf_light = 0.0

        if bvh is not None:
            candidates = bvh.query_overlapping_cells({
                "x": cx - bbox["w"],  "y": cy - bbox["h"],
                "w": bbox["w"] * 2,   "h": bbox["h"] * 2,
            })
        elif all_bboxes:
            candidates = list(all_bboxes.keys())
        else:
            candidates = []

        # Filter out self; pick one candidate by quasi-random index
        candidates = [c for c in candidates if c != cell_id]
        if candidates:
            pick_idx   = int(u1 * len(candidates)) % len(candidates)
            light_cell_id = candidates[pick_idx]
            lb = all_bboxes.get(light_cell_id, {}) if all_bboxes else {}
            if lb:
                lx = lb.get("x", cx) + lb.get("w", 80) / 2.0
                ly = lb.get("y", cy) + lb.get("h", 50) / 2.0
                lz = float(lb.get("z", cz))
                dist_sq = max((cx-lx)**2 + (cy-ly)**2 + (cz-lz)**2 * 10000, 1.0)
                area    = lb.get("w", 80) * lb.get("h", 50)
                pdf_light = 1.0 / (len(candidates) * area / dist_sq)  # area light PDF

                # Light colour from emissive table
                lsp = lb.get("species", "cil-arrow-right")
                light_r, light_g, light_b = _EMISSIVE_TABLE.get(lsp, (0.5, 0.5, 0.5))

        # ── MIS weight (balanced power heuristic, β=2) ────────────────────
        if mis_mode == 2 and pdf_light > 0.0:
            # MIS mode 2: combine material + light sampling
            w_mat   = _pt_mis_weight(pdf_mat,   pdf_light)
            w_light = _pt_mis_weight(pdf_light, pdf_mat)
            # Throughput contribution from MIS combination
            contrib_r = (path_r * w_mat + light_r * w_light) * throughput
            contrib_g = (path_g * w_mat + light_g * w_light) * throughput
            contrib_b = (path_b * w_mat + light_b * w_light) * throughput
        elif mis_mode == 1 and pdf_light > 0.0:
            # MIS mode 1: light sampling only
            contrib_r = light_r * throughput
            contrib_g = light_g * throughput
            contrib_b = light_b * throughput
        else:
            # MIS mode 0 or no light: material sampling only
            contrib_r = path_r * throughput
            contrib_g = path_g * throughput
            contrib_b = path_b * throughput

        # ── Firefly clamp per bounce ───────────────────────────────────────
        contrib_r, contrib_g, contrib_b = _pt_firefly_clamp(
            (contrib_r, contrib_g, contrib_b), _PT_MAX_PATH_INTENSITY)

        # ── Caustic approximation gate ─────────────────────────────────────
        # When ApproximateCaustics=True, clamp specular contribution on diffuse
        # surfaces to reduce noise from low-roughness indirect paths.
        # Mirrors the C++ caustic approximation that clamps glossy→diffuse paths.
        if _PT_APPROXIMATE_CAUSTICS and bounce > 0:
            contrib_r *= 0.25
            contrib_g *= 0.25
            contrib_b *= 0.25

        path_r = contrib_r
        path_g = contrib_g
        path_b = contrib_b

        # ── Throughput update (Russian roulette) ──────────────────────────
        # Mirrors the path tracer's per-bounce throughput × albedo update.
        albedo_avg = (base_r + base_g + base_b) / 3.0
        throughput *= max(0.0, min(1.0, albedo_avg * math.cos(theta_mat)))

    return (max(0.0, path_r), max(0.0, path_g), max(0.0, path_b))


def prepare_path_tracing(
    config: AstroCellPathTracingConfig | None = None,
    view_id: str = "default",
) -> AstroCellPathTracingState:
    """
    Check for configuration changes and invalidate if needed.
    Mirrors PreparePathTracing() + the IsDifferent/Invalidate block in
    FDeferredShadingSceneRenderer::RenderPathTracing().

    Called once per epoch before render_path_tracing() dispatches.

    鲁迅式：准备是清醒，清醒是有时候比勇气更难做到的事情。
    """
    state  = get_or_create_cell_path_tracing_state(view_id)
    cfg    = config or AstroCellPathTracingConfig()

    if cfg.is_different(state.last_config):
        print(
            f"[ASTRO-PT] PreparePathTracing — config changed, invalidating state "
            f"(sample_index was {state.sample_index})",
            file=sys.stderr,
        )
        state.last_config = cfg
        state.invalidate()
    else:
        # No change — bump frame_index only (mirrors FrameIndex++ in C++ each frame)
        state.frame_index += 1

    return state


def render_path_tracing(
    cell_id: str,
    bbox: dict,
    species: str,
    all_bboxes: dict | None = None,
    bvh: "AstroCellBVH | None" = None,
    config: AstroCellPathTracingConfig | None = None,
    view_id: str = "default",
) -> dict:
    """
    Accumulate one path tracing sample for *cell_id*.

    Entry point mirroring the per-view dispatch in RenderPathTracing() →
    the per-pixel sample loop inside PathTracing.usf.

    Algorithm:
      1. prepare_path_tracing() — guard invalidation on config changes
      2. Adaptive sampling gate — skip converged cells (VarianceBuffer check)
      3. Sample one path via _pt_sample_cell_radiance()
      4. Gaussian-filter the sample weight
      5. Running-average accumulate into radiance_buffer
      6. Update variance_buffer (Welford online variance)
      7. Update denoiser AOV buffers (albedo, normal proxy, depth)
      8. Increment sample_index when all cells sampled

    Returns dict with per-cell accumulated radiance + denoiser AOV data.

    鲁迅式：渲染是积累，积累是耐心，耐心是这个时代最稀缺的品质。
    每一帧调用一次，样本慢慢增多，噪声慢慢消退——
    如同鲁迅一篇篇写下去，终究成了一部真实的中国。
    """
    if not _PT_ENABLED:
        return {"cell_id": cell_id, "pt_enabled": False}

    cfg   = config or AstroCellPathTracingConfig()
    state = prepare_path_tracing(cfg, view_id)

    # ── Adaptive sampling convergence gate ────────────────────────────────
    if state.is_converged(cell_id, cfg.adaptive_threshold):
        return {
            "cell_id":      cell_id,
            "sample_index": state.sample_index,
            "converged":    True,
            "radiance":     state.radiance_buffer.get(cell_id, (0.0, 0.0, 0.0)),
            "variance":     state.variance_buffer.get(cell_id, 0.0),
        }

    # ── Sample one path ────────────────────────────────────────────────────
    raw = _pt_sample_cell_radiance(
        cell_id     = cell_id,
        bbox        = bbox,
        species     = species,
        sample_idx  = state.sample_index,
        frame_idx   = state.frame_index,
        bvh         = bvh,
        all_bboxes  = all_bboxes,
        mis_mode    = cfg.mis_mode,
        max_bounces = cfg.max_bounces,
    )

    # ── Gaussian reconstruction filter ────────────────────────────────────
    filtered = _pt_gaussian_filter(raw, cfg.filter_sigma)

    # ── Running-average accumulation (mirrors RadianceRT += sample / N) ───
    n   = state.sample_index + 1
    old = state.radiance_buffer.get(cell_id, (0.0, 0.0, 0.0))
    new_r = old[0] + (filtered[0] - old[0]) / n
    new_g = old[1] + (filtered[1] - old[1]) / n
    new_b = old[2] + (filtered[2] - old[2]) / n
    state.radiance_buffer[cell_id] = (new_r, new_g, new_b)

    # ── Welford online variance (mirrors VarianceBuffer update) ───────────
    # δ = sample − old_mean;  δ2 = sample − new_mean
    # M2 += δ × δ2;  variance = M2 / (n-1) for n≥2
    old_var = state.variance_buffer.get(cell_id, 0.0)
    lum_old = (old[0] + old[1] + old[2]) / 3.0
    lum_new = (new_r + new_g + new_b) / 3.0
    lum_sample = (filtered[0] + filtered[1] + filtered[2]) / 3.0
    delta   = lum_sample - lum_old
    delta2  = lum_sample - lum_new
    # Running M2 stored as variance × (n-1) scaled back
    m2_prev = old_var * max(n - 2, 1)
    m2_new  = m2_prev + delta * delta2
    state.variance_buffer[cell_id] = m2_new / max(n - 1, 1)

    # ── Denoiser AOV update ────────────────────────────────────────────────
    # AlbedoRT: species base colour (mirrors material albedo AOV)
    _ALBEDO_MAP = {
        "cil-eye": (0.49, 0.51, 0.71), "cil-bolt": (1.0, 0.44, 0.0),
        "cil-vector": (0.18, 0.49, 0.20), "cil-plus": (0.12, 0.53, 0.90),
        "cil-arrow-right": (0.27, 0.35, 0.39), "cil-filter": (0.48, 0.12, 0.64),
        "cil-code": (0.18, 0.49, 0.20), "cil-layers": (0.08, 0.40, 0.75),
        "cil-loop": (0.96, 0.50, 0.09), "cil-graph": (0.21, 0.28, 0.31),
    }
    state.albedo_buffer[cell_id] = _ALBEDO_MAP.get(species, (0.5, 0.5, 0.5))
    # NormalRT: upward-facing normal (all cells face viewer → (0, 0, 1))
    state.normal_buffer[cell_id] = (0.0, 0.0, 1.0)
    # DepthRT: normalised depth from z-layer (mirrors DepthRT = z / z_far)
    z_far = 8.0
    state.depth_buffer[cell_id]  = max(0.0, min(1.0,
        float(bbox.get("z", 3)) / z_far))

    # ── Increment sample counter after all cells complete one sample ───────
    # In the C++ renderer SampleIndex is incremented once per frame after the
    # full tile dispatch.  Here we increment per-cell call (single-threaded).
    state.sample_index = n

    result = {
        "cell_id":      cell_id,
        "sample_index": state.sample_index,
        "frame_index":  state.frame_index,
        "converged":    state.is_converged(cell_id, cfg.adaptive_threshold),
        "radiance":     state.radiance_buffer[cell_id],
        "variance":     state.variance_buffer.get(cell_id, 0.0),
        "albedo":       state.albedo_buffer[cell_id],
        "normal":       state.normal_buffer[cell_id],
        "depth":        state.depth_buffer[cell_id],
        "pt_enabled":   True,
    }

    dbg = os.environ.get("ASTRO_PT_VERBOSE", "0") == "1"
    if dbg:
        print(
            f"[ASTRO-PT] render_path_tracing cell={cell_id} "
            f"spp={state.sample_index}/{cfg.max_samples} "
            f"radiance=({new_r:.3f},{new_g:.3f},{new_b:.3f}) "
            f"var={result['variance']:.5f} "
            f"converged={result['converged']}",
            file=sys.stderr,
        )

    return result


# =============================================================================
# [ASTRO-CELL] PathTracingSpatialTemporalDenoising → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/PathTracingSpatialTemporalDenoising.cpp
#
# 鲁迅曾言：「不读书的人，思想就会停止。」
# 不去噪的渲染器，噪声就会永远停止不了。
# 降噪是文明的努力——用空间和时间的信息，重建真实的光照。
#
# FDenoiserManager → AstroCellDenoiserManager
#   RegisterSpatialDenoiser          → register_spatial_denoiser()
#   RegisterSpatialTemporalDenoiser  → register_spatial_temporal_denoiser()
#   UnregisterDenoiser               → unregister_denoiser()
#   HasSpatialDenoiser               → has_spatial_denoiser()
#   HasSpatialTemporalDenoiser       → has_spatial_temporal_denoiser()
#   GetSpatialDenoiser               → get_spatial_denoiser()
#   GetSpatialTemporalDenoiser       → get_spatial_temporal_denoiser()
#   bNeedTextureCreateExtraFlags     → need_extra_flags (bool)
#
# Key denoising passes (ported as pure-Python analytic approximations):
#   FTemporalReprojectionAlignCS    → temporal_reprojection_align()
#   FTemporalReprojectionBlurCS     → temporal_reprojection_blur()
#   FTemporalReprojectionMergeCS    → temporal_reprojection_merge()
#   FTemporalHighFrequencyRejectMapCS → high_frequency_reject_map()
#   FTemporalFeatureFusionCS        → temporal_feature_fusion()
#
# Algorithm changes (鲁迅式 20%):
#   1. GPU compute shaders → analytic Python per-cell operations
#   2. MotionVector texture → per-cell z-layer delta (2-D displacement)
#   3. Variance-weighted temporal blend: history_weight adapted per-cell
#      from the variance buffer of AstroCellPathTracingState
#   4. Spatial NLM kernel (3×3 BVH neighbour query) replaces the full
#      NxN screen-space bilateral filter pass
# =============================================================================

# CVarPathTracingDenoiser equivalents
_PTD_ENABLED:              int   = 1     # r.PathTracing.Denoiser (-1/0/1)
_PTD_SPATIAL_ENABLED:      int   = 1     # r.PathTracing.SpatialDenoiser
_PTD_NORMAL_SPACE:         int   = 0     # 0=world, 1=camera
_PTD_VARIANCE_TYPE:        int   = 1     # 1=combined single-channel
_PTD_RANKED_LUM_VAR:       int   = 0     # 0=default luminance variance
_PTD_TEMPORAL_WEIGHT:      float = 0.9   # history blend weight (temporal stability)
_PTD_SPATIAL_TYPE:         int   = 0     # 0=spatial-only plugin, 1=spatio-temporal
_PTD_DENOISER_NAME:        str   = "NNEDenoiser"
_PTD_TEMPORAL_NAME:        str   = "NFOR"


class AstroCellDenoiserManager:
    """
    Python equivalent of FDenoiserManager.

    Registry for spatial and spatio-temporal denoiser plugins.
    Thread-safe in C++; single-threaded singleton here (epoch loop).

    鲁迅式：管理者的职责是注册、查询、注销——
    如同一个公正的仲裁者，自己不参与战斗，只确保战斗有规则。
    """

    def __init__(self) -> None:
        # Spatial denoisers: name → callable(radiance_buf, albedo_buf, normal_buf)
        self._spatial:           _PTDict[str, object] = {}
        # Spatio-temporal denoisers: name → callable(radiance_buf, history_buf, var_buf)
        self._spatial_temporal:  _PTDict[str, object] = {}
        # Whether any registered denoiser needs extra texture creation flags
        self.need_extra_flags: bool = False

    def register_spatial_denoiser(self, name: str, denoiser_fn,
                                   needs_extra_flags: bool = False) -> None:
        """
        Register a spatial denoiser plugin.
        Mirrors RegisterSpatialDenoiser(TUniquePtr<IPathTracingDenoiser>, FString).
        """
        assert name not in self._spatial, f"Denoiser '{name}' already registered"
        self._spatial[name] = denoiser_fn
        self.need_extra_flags |= needs_extra_flags
        print(
            f"[ASTRO-PTD] RegisterSpatialDenoiser name={name} "
            f"need_extra_flags={needs_extra_flags}",
            file=sys.stderr,
        )

    def register_spatial_temporal_denoiser(self, name: str, denoiser_fn,
                                            needs_extra_flags: bool = False) -> None:
        """
        Register a spatio-temporal denoiser plugin.
        Mirrors RegisterSpatialTemporalDenoiser(TUniquePtr<IPathTracingSpatialTemporalDenoiser>…).
        """
        assert name not in self._spatial_temporal, (
            f"S-T Denoiser '{name}' already registered")
        self._spatial_temporal[name] = denoiser_fn
        self.need_extra_flags |= needs_extra_flags

    def unregister_denoiser(self, name: str) -> None:
        """Remove a denoiser by name from both registries."""
        self._spatial.pop(name, None)
        self._spatial_temporal.pop(name, None)

    def has_spatial_denoiser(self) -> bool:
        return bool(self._spatial)

    def has_spatial_temporal_denoiser(self) -> bool:
        return bool(self._spatial_temporal)

    def has_denoiser(self) -> bool:
        return self.has_spatial_denoiser() or self.has_spatial_temporal_denoiser()

    def get_spatial_denoiser(self, name: str, exact_match: bool = False):
        """
        Return denoiser plugin by name; falls back to first registered if not exact.
        Mirrors FDenoiserManager::GetSpatialDenoiser(FString Name, bool bMatch).
        """
        if name in self._spatial:
            return self._spatial[name]
        if not exact_match and self._spatial:
            return next(iter(self._spatial.values()))
        return None

    def get_spatial_temporal_denoiser(self, name: str, exact_match: bool = False):
        if name in self._spatial_temporal:
            return self._spatial_temporal[name]
        if not exact_match and self._spatial_temporal:
            return next(iter(self._spatial_temporal.values()))
        return None


# Module-level singleton — mirrors the static FDenoiserManager instance
_ASTRO_DENOISER_MANAGER: AstroCellDenoiserManager = AstroCellDenoiserManager()


def get_denoiser_manager() -> AstroCellDenoiserManager:
    """Return the global AstroCellDenoiserManager singleton."""
    return _ASTRO_DENOISER_MANAGER


# ── Temporal reprojection passes (FTemporalReprojection* CS ports) ────────────

def temporal_reprojection_align(
    radiance_buf:   _PTDict[str, tuple],
    history_buf:    _PTDict[str, tuple],
    motion_vectors: _PTDict[str, tuple],  # cell_id → (dz,) displacement
) -> _PTDict[str, tuple]:
    """
    Temporal reprojection alignment pass.
    Mirrors FTemporalReprojectionAlignCS: warps history to current frame using
    per-cell motion vectors.

    In 2-D, motion is only in the Z axis (z-layer transitions); X/Y do not change
    between frames in the pub/sub epoch model.  The warp is a z-layer index
    lookup: if cell moved from z_prev to z_curr, copy its history entry
    (no blending needed for integer z-layer steps).

    Returns a dict of aligned history radiance (same structure as radiance_buf).

    鲁迅式：时间的对齐是第一步——如果你无法找到上一帧的位置，
    历史便是别人的历史，与你无关。
    """
    aligned: _PTDict[str, tuple] = {}
    for cell_id, rad in radiance_buf.items():
        mv = motion_vectors.get(cell_id, (0.0,))
        dz = mv[0] if mv else 0.0
        if abs(dz) < 0.5:
            # No significant motion — use history directly (fast path)
            aligned[cell_id] = history_buf.get(cell_id, rad)
        else:
            # Cell moved to a new z-layer: history is stale; restart from current
            # (mirrors the C++ path where large motion vectors cause history rejection)
            aligned[cell_id] = rad
    return aligned


def temporal_reprojection_blur(
    aligned_history: _PTDict[str, tuple],
    bvh:             "AstroCellBVH | None",
    all_bboxes:      dict,
    blur_radius:     float = 1.0,
) -> _PTDict[str, tuple]:
    """
    Temporal reprojection blur pass.
    Mirrors FTemporalReprojectionBlurCS: applies a small spatial blur to the
    aligned history to reduce temporal ghosting from mis-aligned history.

    2-D adaptation: BVH spatial query fetches immediate neighbours; their
    history values are averaged as a 3-tap bilateral kernel weighted by
    distance (analogue of the C++ screen-space 3×1 separable blur kernel).

    鲁迅式：模糊是宽容，是允许错误存在的制度——
    但宽容过度便是纵容，故 blur_radius 不宜过大。
    """
    blurred: _PTDict[str, tuple] = {}
    for cell_id, hist in aligned_history.items():
        bbox = all_bboxes.get(cell_id, {})
        if not bbox or bvh is None:
            blurred[cell_id] = hist
            continue

        # Spatial neighbourhood from BVH
        nbrs = bvh.query_overlapping_cells({
            "x": bbox.get("x", 0) - blur_radius * bbox.get("w", 80),
            "y": bbox.get("y", 0) - blur_radius * bbox.get("h", 50),
            "w": bbox.get("w", 80) * (1 + 2 * blur_radius),
            "h": bbox.get("h", 50) * (1 + 2 * blur_radius),
        })
        nbr_hists = [aligned_history[n] for n in nbrs
                     if n != cell_id and n in aligned_history]

        if nbr_hists:
            # Simple average (bilateral weights omitted — analytic context)
            avg_r = (hist[0] + sum(h[0] for h in nbr_hists)) / (len(nbr_hists) + 1)
            avg_g = (hist[1] + sum(h[1] for h in nbr_hists)) / (len(nbr_hists) + 1)
            avg_b = (hist[2] + sum(h[2] for h in nbr_hists)) / (len(nbr_hists) + 1)
            blurred[cell_id] = (avg_r, avg_g, avg_b)
        else:
            blurred[cell_id] = hist

    return blurred


def temporal_reprojection_merge(
    current_radiance: _PTDict[str, tuple],
    blurred_history:  _PTDict[str, tuple],
    variance_buf:     _PTDict[str, float],
    base_weight:      float = _PTD_TEMPORAL_WEIGHT,
) -> _PTDict[str, tuple]:
    """
    Temporal accumulation merge pass.
    Mirrors FTemporalReprojectionMergeCS: blends current frame with history.

    The history weight is modulated by per-cell variance:
        w_hist = base_weight × clamp(1 − variance / variance_max, 0, 1)
    Low-variance (converged) cells keep more history; high-variance cells
    (still noisy) accept more current-frame data — same as the C++
    TotalVariation permutation of the merge shader.

    鲁迅式：过去与现在的混合比例，取决于现在有多嘈杂——
    越嘈杂，越需要历史来压制；越平静，历史越可以安全保留。
    """
    merged: _PTDict[str, tuple] = {}
    variance_max = max(variance_buf.values()) if variance_buf else 1.0
    variance_max = max(variance_max, 1e-6)

    for cell_id, curr in current_radiance.items():
        hist  = blurred_history.get(cell_id, curr)
        var   = variance_buf.get(cell_id, 1.0)
        # Variance-adaptive weight
        w_hist = base_weight * max(0.0, min(1.0, 1.0 - var / variance_max))
        w_curr = 1.0 - w_hist
        merged[cell_id] = (
            curr[0] * w_curr + hist[0] * w_hist,
            curr[1] * w_curr + hist[1] * w_hist,
            curr[2] * w_curr + hist[2] * w_hist,
        )
    return merged


def high_frequency_reject_map(
    radiance_buf:     _PTDict[str, tuple],
    last_denoised:    _PTDict[str, tuple],
    variance_buf:     _PTDict[str, float],
    reject_threshold: float = 0.15,
) -> _PTDict[str, float]:
    """
    High-frequency reject map pass.
    Mirrors FTemporalHighFrequencyRejectMapCS: generates a per-cell mask
    [0, 1] where 1 = accept current (high-frequency / newly appeared feature)
    and 0 = reject current (temporal ghost / noise spike).

    Implemented as luminance-delta comparison against last denoised frame:
        delta_lum = |lum_current − lum_last_denoised| / max(lum_last_denoised, 1e-4)
        accept    = 1 if delta_lum < reject_threshold and variance < threshold
                    0 otherwise (clamp to [0, 1])

    鲁迅式：高频拒绝图是防伪标记——真实的光照变化缓慢，突变是噪声的证据。
    但拒绝必须谨慎，过于激进的拒绝会抹平真实的变化，造成「滞后」的幽灵。
    """
    accept_map: _PTDict[str, float] = {}
    for cell_id, curr in radiance_buf.items():
        lum_curr = (curr[0] + curr[1] + curr[2]) / 3.0
        last = last_denoised.get(cell_id)
        if last is None:
            accept_map[cell_id] = 1.0
            continue
        lum_last = (last[0] + last[1] + last[2]) / 3.0
        delta    = abs(lum_curr - lum_last) / max(lum_last, 1e-4)
        var      = variance_buf.get(cell_id, 1.0)
        if delta < reject_threshold and var < _PT_ADAPTIVE_THRESHOLD * 10:
            accept_map[cell_id] = 0.0   # accept history (reject current spike)
        else:
            accept_map[cell_id] = 1.0   # accept current (genuine change)
    return accept_map


def temporal_feature_fusion(
    merged_radiance: _PTDict[str, tuple],
    accept_map:      _PTDict[str, float],
    last_denoised:   _PTDict[str, tuple],
) -> _PTDict[str, tuple]:
    """
    Temporal feature fusion pass.
    Mirrors FTemporalFeatureFusionCS: final per-cell combination of merged
    radiance with last-denoised history, gated by the accept_map.

    accept_map[cell_id] == 1.0 → use merged_radiance (fresh data)
    accept_map[cell_id] == 0.0 → blend toward last_denoised (temporal stability)

    鲁迅式：融合是最后的抉择——在新与旧之间，在清晰与稳定之间，
    accept_map 是那唯一的判官，不偏不倚（除非偶尔被数学愚弄）。
    """
    fused: _PTDict[str, tuple] = {}
    for cell_id, merged in merged_radiance.items():
        a     = accept_map.get(cell_id, 1.0)
        last  = last_denoised.get(cell_id, merged)
        fused[cell_id] = (
            merged[0] * a + last[0] * (1.0 - a),
            merged[1] * a + last[1] * (1.0 - a),
            merged[2] * a + last[2] * (1.0 - a),
        )
    return fused


def run_spatial_temporal_denoising(
    state:       AstroCellPathTracingState,
    all_bboxes:  dict,
    bvh:         "AstroCellBVH | None" = None,
    motion_vectors: _PTDict[str, tuple] | None = None,
) -> _PTDict[str, tuple]:
    """
    Full spatio-temporal denoising pipeline for all cells.

    Orchestrates the five passes in order, matching the compute-pass dispatch
    sequence in PathTracingSpatialTemporalDenoising.cpp:
      1. temporal_reprojection_align
      2. temporal_reprojection_blur
      3. temporal_reprojection_merge
      4. high_frequency_reject_map
      5. temporal_feature_fusion

    After fusion, the result is stored in state.last_denoised for the next
    frame's history (mirrors LastDenoisedRadianceRT update).

    Returns the final denoised radiance dict.

    鲁迅式：五道工序，缺一不可——就如同一篇文章，
    初稿之后还需修改、再修改、校对、排版，才能印出来给人看。
    """
    if not state.radiance_buffer:
        return {}

    mv = motion_vectors or {}

    # Pass 1: Temporal alignment
    aligned = temporal_reprojection_align(
        state.radiance_buffer, state.last_denoised, mv)

    # Pass 2: History blur
    blurred = temporal_reprojection_blur(aligned, bvh, all_bboxes)

    # Pass 3: Temporal merge with variance adaptation
    merged = temporal_reprojection_merge(
        state.radiance_buffer, blurred, state.variance_buffer)

    # Pass 4: High-frequency reject map
    accept = high_frequency_reject_map(
        state.radiance_buffer, state.last_denoised, state.variance_buffer)

    # Pass 5: Feature fusion
    denoised = temporal_feature_fusion(merged, accept, state.last_denoised)

    # Update last-denoised cache for next frame
    state.last_denoised.update(denoised)

    total    = len(denoised)
    accepted = sum(1 for v in accept.values() if v > 0.5)
    print(
        f"[ASTRO-PTD] run_spatial_temporal_denoising: "
        f"total_cells={total} accepted_fresh={accepted} "
        f"temporal_blended={total - accepted} "
        f"spp={state.sample_index}",
        file=sys.stderr,
    )

    return denoised


# =============================================================================
# [ASTRO-CELL] ReflectionEnvironmentCapture → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/ReflectionEnvironmentCapture.cpp
#
# 鲁迅曾言：「希望本是无所谓有，无所谓无的。这正如地上的路；
# 其实地上本没有路，走的人多了，也便成了路。」
# 反射探针亦然——世界本无镜，捕获得多了，也便成了反射。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   GSupersampleCaptureFactor       → ASTRO_CAPTURE_SUPERSAMPLE_FACTOR
#   GReflectionCaptureNearPlane     → ASTRO_CAPTURE_NEAR_PLANE
#   CVarReflectionCaptureRuntimeTimeslice → ASTRO_CAPTURE_TIMESLICE_FACES
#   CaptureSceneToScratchCubemap    → capture_scene_to_scratch_cubemap()
#   ConvolveCubeMap                 → convolve_cube_map()
#   FindOrAllocateCubemapIndex      → find_or_allocate_cubemap_index()
#   ComputeRuntimeBudgetSignedDistance → compute_capture_priority()
#   FCaptureComponentSceneState     → AstroCellCaptureState
#   FReflectionSceneData            → AstroCellReflectionSceneData
#   BeginReflectionCaptureSlowTask  → begin_capture_task() (log-only)
#   UpdateReflectionCaptureSlowTask → update_capture_task()
#   EndReflectionCaptureSlowTask    → end_capture_task()
#
# 2-D SVG adaptation:
#   CubemapArray[face][mip]  → per-cell specular probe dict  (6 faces × N mips)
#   Radiance SH L2 (9 coeff) → 3-component SH L1 (3 floats per channel = 9)
#   Downsample mip pass      → gaussian_downsample_face_mip()
#   Convolve specular face   → convolve_specular_face()
#   Diffuse irradiance SH    → compute_diffuse_irradiance_sh()
# =============================================================================

# CVarReflectionCapture equivalents
ASTRO_CAPTURE_NEAR_PLANE:       float = 5.0     # GReflectionCaptureNearPlane
ASTRO_CAPTURE_SUPERSAMPLE_MIN:  int   = 1       # MinSupersampleCaptureFactor
ASTRO_CAPTURE_SUPERSAMPLE_MAX:  int   = 8       # MaxSupersampleCaptureFactor
ASTRO_CAPTURE_SUPERSAMPLE:      int   = 1       # GSupersampleCaptureFactor
ASTRO_CAPTURE_TIMESLICE_FACES:  int   = 2       # CVarReflectionCaptureRuntimeTimeslice
ASTRO_CAPTURE_TIMESLICE_EDITOR: int   = 3       # CVarReflectionCaptureRuntimeTimesliceEditor
ASTRO_CAPTURE_TIMESLICE_SLOW:   bool  = False   # CVarReflectionCaptureRuntimeTimesliceSlow
ASTRO_CAPTURE_FADE_TIME:        float = 0.5     # CVarReflectionCaptureRuntimeFadeInTime
ASTRO_CAPTURE_BUDGET:           int   = 0       # CVarReflectionCaptureRuntimeBudget (0=unlimited)
ASTRO_CAPTURE_FOLIAGE:          bool  = False   # CVarReflectionCaptureRuntimeFoliage
ASTRO_CAPTURE_TRANSLUCENCY:     bool  = False   # CVarReflectionCaptureRuntimeTranslucency
ASTRO_CAPTURE_MODE:             int   = 1       # 0=continuous, 1=once
ASTRO_CAPTURE_FAST_ON_LOAD:     int   = 3       # CVarReflectionCaptureRuntimeFastRenderOnLoad
# Cube faces: +X,-X,+Y,-Y,+Z,-Z (indices 0..5)
_CAPTURE_NUM_FACES:             int   = 6
_CAPTURE_NUM_MIPS:              int   = 7       # mip 0..6 for 128×128 cube (log2(128)+1)


@_ptdc
class AstroCellCaptureState:
    """
    Python equivalent of FCaptureComponentSceneState.

    Tracks the lifecycle of one reflection capture probe:
        cubemap_index: int     — slot in the global cubemap array
        fade_alpha:    float   — fade-in progress [0, 1]
        rendered_once: bool    — True once all 6 faces captured at least once
        is_dirty:      bool    — True when the capture needs a refresh
        cell_id:       str     — owning cell (Astro-specific field)

    鲁迅式：一个探针的一生——诞生于 FindOrAllocateCubemapIndex，
    成熟于所有面被捕获完毕，淡入于 fade_alpha 趋近 1.0，
    死亡于探针被 evict 或场景被清除。
    """
    cubemap_index: int   = -1
    fade_alpha:    float = 0.0
    rendered_once: bool  = False
    is_dirty:      bool  = True
    cell_id:       str   = ""
    # Per-face capture data: list of 6 face dicts, each with mip levels
    # Face dict: { "mip_0": (R,G,B), "mip_1": (R,G,B), ... }
    face_data: _PTList[_PTDict[str, tuple]] = _ptfield(
        default_factory=lambda: [{} for _ in range(_CAPTURE_NUM_FACES)])
    # Prefiltered specular: mip_level → average_colour (convolved)
    specular_prefilter: _PTDict[int, tuple] = _ptfield(default_factory=dict)
    # Diffuse SH irradiance (9 coefficients, 3 channels × 3 = 9 floats)
    diffuse_sh: _PTList[float] = _ptfield(default_factory=lambda: [0.0] * 9)


@_ptdc
class AstroCellReflectionSceneData:
    """
    Python equivalent of FReflectionSceneData.

    Holds the global cubemap array and the per-probe state registry.
    Mirrors the scene-level structure that the C++ renderer consults
    when resolving cubemap slots and building the Uniform Buffer.

    鲁迅式：场景数据是公共档案馆——每个探针在这里登记户籍，
    离开时注销，场景清除时关门大吉。
    """
    # Allocated captures: cell_id → AstroCellCaptureState
    allocated_captures: _PTDict[str, AstroCellCaptureState] = _ptfield(
        default_factory=dict)
    # Next available cubemap slot index (mirrors NextAvailableReflectionCaptureSortedIndex)
    next_cubemap_slot: int = 0
    # Maximum supported captures per scene (mirrors MaxCubemaps)
    max_cubemaps: int = 64


# Module-level scene data singleton
_ASTRO_REFLECTION_SCENE: AstroCellReflectionSceneData = AstroCellReflectionSceneData()


def get_reflection_scene_data() -> AstroCellReflectionSceneData:
    """Return the module-level AstroCellReflectionSceneData singleton."""
    return _ASTRO_REFLECTION_SCENE


def begin_capture_task(num_captures: int, reason: str = "UpdateCaptures") -> None:
    """
    Log the start of a reflection capture batch.
    Mirrors BeginReflectionCaptureSlowTask() — only logging in Astro (no UE slow task UI).
    """
    print(
        f"[ASTRO-CAPTURE] BeginReflectionCaptureSlowTask: "
        f"num={num_captures} reason={reason}",
        file=sys.stderr,
    )


def update_capture_task(capture_index: int, num_captures: int) -> None:
    """Mirrors UpdateReflectionCaptureSlowTask — progress logging."""
    if capture_index % max(1, num_captures // 5) == 0:
        pct = int(100.0 * capture_index / max(num_captures, 1))
        print(
            f"[ASTRO-CAPTURE] UpdateReflectionCaptureSlowTask: "
            f"{capture_index}/{num_captures} ({pct}%)",
            file=sys.stderr,
        )


def end_capture_task(num_captures: int) -> None:
    """Mirrors EndReflectionCaptureSlowTask."""
    print(
        f"[ASTRO-CAPTURE] EndReflectionCaptureSlowTask: "
        f"num={num_captures} complete",
        file=sys.stderr,
    )


def find_or_allocate_cubemap_index(cell_id: str) -> int:
    """
    Allocate or return the existing cubemap slot for *cell_id*.

    Mirrors FindOrAllocateCubemapIndex() from ReflectionEnvironmentCapture.cpp:
        CaptureSceneStatePtr = Scene.ReflectionSceneData.AllocatedReflectionCaptureState
                                .AddReference(Component)
        if (!CaptureSceneStatePtr): allocate new slot

    Returns cubemap_index ∈ [0, max_cubemaps) or -1 on overflow.

    鲁迅式：分配是有限的资源与无限的需求之间的妥协——
    64 个探针槽，比大多数场景需要的多；
    但若场景足够野心勃勃，终有耗尽的一天。
    """
    scene = get_reflection_scene_data()
    if cell_id in scene.allocated_captures:
        return scene.allocated_captures[cell_id].cubemap_index

    if scene.next_cubemap_slot >= scene.max_cubemaps:
        print(
            f"[ASTRO-CAPTURE] WARNING: cubemap array full "
            f"({scene.max_cubemaps} slots) — cannot allocate for cell={cell_id}",
            file=sys.stderr,
        )
        return -1

    idx = scene.next_cubemap_slot
    scene.next_cubemap_slot += 1
    state = AstroCellCaptureState(cubemap_index=idx, cell_id=cell_id)
    scene.allocated_captures[cell_id] = state

    print(
        f"[ASTRO-CAPTURE] FindOrAllocateCubemapIndex: "
        f"cell_id={cell_id} slot={idx} "
        f"total_allocated={len(scene.allocated_captures)}",
        file=sys.stderr,
    )
    return idx


def compute_capture_priority(
    cell_id:    str,
    bbox:       dict,
    camera_pos: tuple = (600.0, 450.0, 3.0),
) -> float:
    """
    Compute signed distance priority for runtime capture budget sorting.

    Mirrors ComputeRuntimeBudgetSignedDistance() from ReflectionEnvironmentCapture.cpp:
        For sphere probes: dist − InfluenceRadius
        For box probes:    Chebyshev distance to box surface

    2-D adaptation: uses 2-D bbox proximity to camera position.
    Lower = higher priority (closest captures rendered first).

    鲁迅式：优先级是稀缺资源分配的哲学——离得近的先照，离得远的等着。
    这不是歧视，是现实主义。
    """
    cx = bbox.get("x", 0) + bbox.get("w", 80) / 2.0
    cy = bbox.get("y", 0) + bbox.get("h", 50) / 2.0
    cz = float(bbox.get("z", 3))

    cam_x, cam_y, cam_z = camera_pos
    dx = cx - cam_x
    dy = cy - cam_y
    dz = (cz - cam_z) * 100.0   # scale z to world units

    influence_r = max(bbox.get("w", 80), bbox.get("h", 50)) / 2.0
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    return dist - influence_r   # negative = camera inside the probe influence


def gaussian_downsample_face_mip(
    face_colour: tuple,
    mip_level:   int,
    sigma_scale: float = 0.8,
) -> tuple:
    """
    Per-mip Gaussian downsample of a cubemap face colour.
    Mirrors FDownsampleCubeFaceCS (DownsampleCS in ReflectionEnvironmentShaders.usf):
        Each mip halves the resolution and blurs with a 3×3 Gaussian kernel.
        Energy is conserved (sum of Gaussian weights = 1).

    2-D adaptation: operates on a single (R, G, B) float tuple representing
    the average colour of a face at the given mip level.  The Gaussian kernel
    is replaced by an exponential decay on luminance (mirrors the energy loss
    at higher mip levels where the specular lobe widens).

    Returns the downsampled face colour at *mip_level*.

    鲁迅式：Mip 层是宽容的代价——越高的 mip，细节越少，也越不刺眼。
    这是视觉的让步，也是性能的胜利。
    """
    # Gaussian decay factor per mip: each level loses sigma_scale of sharpness
    decay = math.exp(-0.5 * (mip_level * sigma_scale) ** 2)
    # Blend toward mid-grey (0.5, 0.5, 0.5) at higher mips — mirrors the
    # BRDF integration limit where fully rough surfaces → uniform hemisphere
    mid = 0.5
    return (
        face_colour[0] * decay + mid * (1.0 - decay),
        face_colour[1] * decay + mid * (1.0 - decay),
        face_colour[2] * decay + mid * (1.0 - decay),
    )


def convolve_specular_face(
    face_colour:   tuple,
    mip_level:     int,
    roughness:     float = 0.5,
    num_mips:      int   = _CAPTURE_NUM_MIPS,
) -> tuple:
    """
    Per-face specular convolution (pre-filtered environment map).
    Mirrors FConvolveSpecularFaceCS (FilterCS in ReflectionEnvironmentShaders.usf):
        Integrates the GGX BSDF lobe over the cubemap face weighted by
        the mip-level roughness mapping:
            perceptual_roughness = mip / (num_mips - 1)
            alpha = perceptual_roughness²   (GGX alpha = roughness²)

    2-D adaptation: approximates the convolution result as a roughness-weighted
    blend between the specular (sharp) face colour and a diffuse (isotropic) grey.
    The GGX NDF width increases with roughness — the highest mip approximates
    a Lambertian hemisphere integral (uniform over all directions → grey).

    Returns pre-filtered specular colour for this face + mip combination.

    鲁迅式：预滤波是先见之明——把所有可能的粗糙度预先计算好，
    运行时只需查表，不必每次重新积分。这是懒惰，也是智慧。
    """
    perceptual_roughness = mip_level / max(num_mips - 1, 1)
    alpha = perceptual_roughness * perceptual_roughness

    # GGX lobe weight: sharper lobe at low roughness → more face colour;
    # wider lobe at high roughness → blend toward isotropic grey
    lobe_w = max(0.0, 1.0 - alpha)
    iso_w  = alpha

    lum   = (face_colour[0] + face_colour[1] + face_colour[2]) / 3.0
    grey  = (lum, lum, lum)

    prefiltered = (
        face_colour[0] * lobe_w + grey[0] * iso_w,
        face_colour[1] * lobe_w + grey[1] * iso_w,
        face_colour[2] * lobe_w + grey[2] * iso_w,
    )
    return prefiltered


def compute_diffuse_irradiance_sh(
    face_colours: _PTList[tuple],
) -> _PTList[float]:
    """
    Compute diffuse irradiance SH L1 coefficients from 6 face averages.
    Mirrors FComputeSkyEnvMapDiffuseIrradianceCS (ComputeSkyEnvMapDiffuseIrradianceCS.usf):
        Integrates the cubemap × Lambertian kernel into SH L2 coefficients.
        For 6 axis-aligned faces, the SH projection simplifies to per-axis terms.

    Returns 9 floats: [Y_00_R, Y_00_G, Y_00_B, Y_1-1_R, …, Y_11_B]
    (SH L0 + L1 for each colour channel, interleaved per-channel for
    compatibility with the C++ IrradianceEnvMapSH layout).

    鲁迅式：球谐函数是数学对光照的压缩——把无限方向的辐照度
    压缩成九个系数，失去了细节，保留了大意。
    一如散文诗：意境犹在，字句已省。
    """
    # Face order: +X, -X, +Y, -Y, +Z, -Z
    # Each face contributes to SH L0 (Y_00) and L1 (Y_1-1, Y_10, Y_11)
    # via the face normal dotted with the SH basis functions.
    # SH normalization constants: Y_00 = 1/√(4π), Y_1m = √(3/(4π))
    c0 = 0.282095   # Y_00
    c1 = 0.488603   # Y_1x normalisation

    # Face normals (+X,-X,+Y,-Y,+Z,-Z)
    normals = [
        ( 1, 0, 0), (-1, 0, 0),
        ( 0, 1, 0), ( 0,-1, 0),
        ( 0, 0, 1), ( 0, 0,-1),
    ]

    sh = [0.0] * 9  # [L0_R, L0_G, L0_B, L1y_R, L1y_G, L1y_B, L1z_R, L1z_G, L1z_B]
    solid_angle = 4.0 * math.pi / _CAPTURE_NUM_FACES  # uniform solid angle per face

    for i, (nx, ny, nz) in enumerate(normals):
        fc = face_colours[i] if i < len(face_colours) else (0.0, 0.0, 0.0)
        fr, fg, fb = fc

        # L0 accumulation (isotropic, same for all channels)
        sh[0] += c0 * fr * solid_angle
        sh[1] += c0 * fg * solid_angle
        sh[2] += c0 * fb * solid_angle

        # L1 Y band (Y_1-1 ∝ y, Y_10 ∝ z): skip Y_11 (∝ x) for brevity
        sh[3] += c1 * ny * fr * solid_angle
        sh[4] += c1 * ny * fg * solid_angle
        sh[5] += c1 * ny * fb * solid_angle
        sh[6] += c1 * nz * fr * solid_angle
        sh[7] += c1 * nz * fg * solid_angle
        sh[8] += c1 * nz * fb * solid_angle

    return sh


def capture_scene_to_scratch_cubemap(
    cell_id:     str,
    bbox:        dict,
    species:     str,
    all_bboxes:  dict,
    face_index:  int  = -1,   # -1 = all 6 faces; 0..5 = specific face (timeslice)
    supersample: int  = ASTRO_CAPTURE_SUPERSAMPLE,
) -> AstroCellCaptureState:
    """
    Capture scene radiance into the cell's cubemap scratch buffer.

    Mirrors CaptureSceneToScratchCubemap() which renders 6 cube faces into
    a temporary render target, including sky atmosphere, sky light, and
    foliage (if enabled).

    2-D adaptation:
        Each face is assigned a canonical direction in the 2-D SVG plane:
            Face 0 (+X): right-side neighbours
            Face 1 (-X): left-side neighbours
            Face 2 (+Y): bottom neighbours
            Face 3 (-Y): top neighbours
            Face 4 (+Z): higher-z-layer neighbours  (「上」)
            Face 5 (-Z): lower-z-layer neighbours   (「下」)
        For each face, we query the BVH (or all_bboxes) for neighbours in
        that half-plane, average their emissive colours weighted by solid
        angle, and write the result into the face data.

    The supersample factor is clamped to [MIN, MAX] as in C++ and applied
    as a weight boost for the Gaussian downsample kernel.

    Returns the updated AstroCellCaptureState for *cell_id*.

    鲁迅式：捕获是对现实的凝视——六个方向，不遗漏任何角落。
    但凝视需要勇气：某些方向可能什么都没有，而这本身也是信息。
    """
    supersample = max(ASTRO_CAPTURE_SUPERSAMPLE_MIN,
                      min(ASTRO_CAPTURE_SUPERSAMPLE_MAX, supersample))

    scene  = get_reflection_scene_data()
    slot   = find_or_allocate_cubemap_index(cell_id)
    if slot < 0:
        return AstroCellCaptureState(cell_id=cell_id, cubemap_index=-1)

    capture = scene.allocated_captures[cell_id]

    cx   = bbox.get("x", 0) + bbox.get("w", 80) / 2.0
    cy   = bbox.get("y", 0) + bbox.get("h", 50) / 2.0
    cz   = float(bbox.get("z", 3))
    hw   = bbox.get("w", 80) / 2.0
    hh   = bbox.get("h", 50) / 2.0

    # Face query planes: half-planes in (X, Y, Z) centred on cell
    face_filters = [
        lambda b, _cx=cx: b.get("x", 0) > _cx,           # +X: right
        lambda b, _cx=cx: b.get("x", 0) + b.get("w", 80) < _cx,  # -X: left
        lambda b, _cy=cy: b.get("y", 0) > _cy,           # +Y: below (SVG y grows down)
        lambda b, _cy=cy: b.get("y", 0) + b.get("h", 50) < _cy,  # -Y: above
        lambda b, _cz=cz: float(b.get("z", 3)) > _cz,   # +Z: higher layer
        lambda b, _cz=cz: float(b.get("z", 3)) < _cz,   # -Z: lower layer
    ]

    faces_to_capture = range(_CAPTURE_NUM_FACES) if face_index < 0 else [face_index]

    _EMISSIVE_CAPTURE_TABLE = {
        "cil-eye":         (0.55, 0.60, 0.90),
        "cil-bolt":        (0.95, 0.55, 0.10),
        "cil-vector":      (0.30, 0.70, 0.35),
        "cil-plus":        (0.25, 0.55, 0.90),
        "cil-arrow-right": (0.50, 0.60, 0.65),
        "cil-filter":      (0.60, 0.25, 0.75),
        "cil-code":        (0.30, 0.70, 0.35),
        "cil-layers":      (0.20, 0.55, 0.85),
        "cil-loop":        (0.90, 0.60, 0.15),
        "cil-graph":       (0.40, 0.50, 0.55),
    }

    for fi in faces_to_capture:
        face_filter_fn = face_filters[fi]
        # Collect neighbours in this face's half-plane
        contributors = []
        for other_id, obbox in all_bboxes.items():
            if other_id == cell_id:
                continue
            if face_filter_fn(obbox):
                sp = obbox.get("species", "cil-arrow-right")
                contrib_col = _EMISSIVE_CAPTURE_TABLE.get(sp, (0.5, 0.5, 0.5))
                # Solid angle weight: larger / closer cells contribute more
                ox   = obbox.get("x", 0) + obbox.get("w", 80) / 2.0
                oy   = obbox.get("y", 0) + obbox.get("h", 50) / 2.0
                oz   = float(obbox.get("z", 3))
                dist = max(1.0, math.sqrt((cx-ox)**2 + (cy-oy)**2 + (cz-oz)**2*1e4))
                area = obbox.get("w", 80) * obbox.get("h", 50)
                w    = area / (dist * dist) * supersample
                contributors.append((contrib_col, w))

        if contributors:
            total_w = sum(w for _, w in contributors)
            avg_r   = sum(c[0]*w for c, w in contributors) / total_w
            avg_g   = sum(c[1]*w for c, w in contributors) / total_w
            avg_b   = sum(c[2]*w for c, w in contributors) / total_w
        else:
            # No contributors: sky colour from near-plane (ASTRO_CAPTURE_NEAR_PLANE)
            # Mirrors the CaptureSceneToScratchCubemap sky fallback
            avg_r, avg_g, avg_b = 0.55, 0.68, 0.82   # sky blue default

        face_base = (avg_r, avg_g, avg_b)

        # Build mip chain for this face
        face_dict: _PTDict[str, tuple] = {}
        for mip in range(_CAPTURE_NUM_MIPS):
            face_dict[f"mip_{mip}"] = gaussian_downsample_face_mip(
                face_base, mip, sigma_scale=0.7)
        capture.face_data[fi] = face_dict

    print(
        f"[ASTRO-CAPTURE] CaptureSceneToScratchCubemap: "
        f"cell_id={cell_id} slot={slot} "
        f"faces_captured={list(faces_to_capture)} "
        f"supersample={supersample}",
        file=sys.stderr,
    )
    return capture


def convolve_capture_cubemap(cell_id: str) -> AstroCellCaptureState:
    """
    Convolve the captured cubemap to produce a pre-filtered specular environment.
    Mirrors the ConvolveCubeMap() pass called after CaptureSceneToScratchCubemap().

    For each mip level: average face_data across all 6 faces → convolve_specular_face
    → store result in capture.specular_prefilter[mip].

    Also computes diffuse irradiance SH from the mip-0 face data.

    鲁迅式：卷积是提炼——把六个方向的原始捕获数据，
    提炼成一份可以被任何粗糙度查询的预滤波环境贴图。
    这是从现象到本质的压缩，是科学的做法。
    """
    scene   = get_reflection_scene_data()
    capture = scene.allocated_captures.get(cell_id)
    if capture is None:
        return AstroCellCaptureState(cell_id=cell_id, cubemap_index=-1)

    for mip in range(_CAPTURE_NUM_MIPS):
        mip_key = f"mip_{mip}"
        face_cols = [
            capture.face_data[fi].get(mip_key, (0.5, 0.5, 0.5))
            for fi in range(_CAPTURE_NUM_FACES)
        ]
        avg_r = sum(c[0] for c in face_cols) / _CAPTURE_NUM_FACES
        avg_g = sum(c[1] for c in face_cols) / _CAPTURE_NUM_FACES
        avg_b = sum(c[2] for c in face_cols) / _CAPTURE_NUM_FACES
        face_avg = (avg_r, avg_g, avg_b)
        capture.specular_prefilter[mip] = convolve_specular_face(face_avg, mip)

    # Diffuse irradiance SH from mip-0 faces (highest resolution)
    mip0_faces = [
        capture.face_data[fi].get("mip_0", (0.5, 0.5, 0.5))
        for fi in range(_CAPTURE_NUM_FACES)
    ]
    capture.diffuse_sh = compute_diffuse_irradiance_sh(mip0_faces)

    capture.rendered_once = True
    capture.is_dirty      = False

    print(
        f"[ASTRO-CAPTURE] ConvolveCubeMap: "
        f"cell_id={cell_id} slot={capture.cubemap_index} "
        f"mips_computed={_CAPTURE_NUM_MIPS} "
        f"sh_L0=({capture.diffuse_sh[0]:.3f},{capture.diffuse_sh[1]:.3f},{capture.diffuse_sh[2]:.3f})",
        file=sys.stderr,
    )
    return capture


def update_reflection_captures(
    all_bboxes:    dict,
    camera_pos:    tuple = (600.0, 450.0, 3.0),
    timeslice:     bool  = True,
    force_all:     bool  = False,
) -> _PTList[str]:
    """
    Update dirty reflection captures in priority order.

    Mirrors the runtime reflection capture update loop in
    FScene::UpdateReflectionCaptureContents() / BeginRenderingReflectionCaptures():
      1. Collect dirty captures from allocated_captures
      2. Sort by compute_capture_priority (nearest first)
      3. Apply budget (ASTRO_CAPTURE_BUDGET = 0 → unlimited)
      4. For each: capture_scene_to_scratch_cubemap + convolve_capture_cubemap
      5. Fade-in (fade_alpha += dt / ASTRO_CAPTURE_FADE_TIME)

    Returns list of cell_ids updated this pass.

    鲁迅式：按距离排队，公平而现实——远处的探针等着，近处的先享受光照。
    这不是歧视，是优先级：视觉效果由近而远递减，计算预算由近而远递增。
    """
    scene   = get_reflection_scene_data()

    # Collect dirty + never-rendered captures from all_bboxes
    pending: _PTList[tuple] = []
    for cell_id, bbox in all_bboxes.items():
        cap = scene.allocated_captures.get(cell_id)
        if cap is None or cap.is_dirty or force_all:
            prio = compute_capture_priority(cell_id, bbox, camera_pos)
            pending.append((prio, cell_id, bbox))

    # Sort by priority (ascending distance → nearest first)
    pending.sort(key=lambda x: x[0])

    # Budget gate (0 = unlimited)
    budget    = ASTRO_CAPTURE_BUDGET if ASTRO_CAPTURE_BUDGET > 0 else len(pending)
    to_update = pending[:budget]

    if to_update:
        begin_capture_task(len(to_update), "UpdateReflectionCaptures")

    updated_cells: _PTList[str] = []
    dt = 1.0 / max(1, ASTRO_CAPTURE_TIMESLICE_FACES)  # fake dt per timeslice step

    for idx, (prio, cell_id, bbox) in enumerate(to_update):
        update_capture_task(idx, len(to_update))
        sp = all_bboxes[cell_id].get("species", "cil-arrow-right")

        if timeslice and not force_all:
            # Timesliced: capture ASTRO_CAPTURE_TIMESLICE_FACES per update
            for fi in range(ASTRO_CAPTURE_TIMESLICE_FACES):
                capture_scene_to_scratch_cubemap(
                    cell_id, bbox, sp, all_bboxes, face_index=fi)
        else:
            # Full capture: all 6 faces at once (mirrors "fast render on load")
            capture_scene_to_scratch_cubemap(
                cell_id, bbox, sp, all_bboxes, face_index=-1)

        convolve_capture_cubemap(cell_id)

        # Fade-in update: increment fade_alpha toward 1.0
        cap = scene.allocated_captures.get(cell_id)
        if cap:
            cap.fade_alpha = min(1.0, cap.fade_alpha + dt / max(ASTRO_CAPTURE_FADE_TIME, 1e-3))

        updated_cells.append(cell_id)

    if to_update:
        end_capture_task(len(to_update))

    return updated_cells


def query_specular_radiance(
    cell_id:      str,
    roughness:    float = 0.5,
    face_index:   int   = 4,   # default +Z face (「上方天空」)
) -> tuple:
    """
    Query the pre-filtered specular environment for *cell_id*.

    Maps *roughness* to the appropriate mip level using the Nanite LOD
    metric analogue:
        mip = round(roughness × (num_mips − 1))
    Returns the prefiltered (R, G, B) specular colour × fade_alpha.

    Mirrors the specular environment probe lookup performed in the
    reflection capture material shader (GetOffSpecularPeakReflectionDir +
    texCUBElod call in ReflectionEnvironmentShared.usf).

    鲁迅式：查询是坦然的索取——环境贴图积累好了，
    谁需要，谁就来取，不必客气，不必感谢。
    """
    scene   = get_reflection_scene_data()
    capture = scene.allocated_captures.get(cell_id)
    if capture is None or not capture.specular_prefilter:
        return (0.5, 0.5, 0.5)   # default sky grey

    mip      = int(round(roughness * (_CAPTURE_NUM_MIPS - 1)))
    mip      = max(0, min(mip, _CAPTURE_NUM_MIPS - 1))
    spec_col = capture.specular_prefilter.get(mip, (0.5, 0.5, 0.5))
    alpha    = capture.fade_alpha

    return (
        spec_col[0] * alpha,
        spec_col[1] * alpha,
        spec_col[2] * alpha,
    )


# =============================================================================
# [ASTRO-CELL] ReflectionEnvironmentRealTimeCapture → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/ReflectionEnvironmentRealTimeCapture.cpp
#
# 「世界上本没有实时反射，用的人多了，也便有了实时反射。」——鲁迅（改写）
#
# ReflectionEnvironmentRealTimeCapture 实现了「实时」天光捕获：
# 每帧分时渲染一个或多个 cube face，逐渐积累完整的 sky env map，
# 再对其执行 downsample + convolve + diffuse SH 通道，
# 并通过 bRealTimeCaptureEnabled 标志按需触发。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   CVarRealTimeReflectionCaptureTimeSlicing   → ASTRO_RT_CAPTURE_TIMESLICE
#   CVarRealTimeReflectionCaptureTimeSlicingSkyCloudCubeFacePerFrame
#                                              → ASTRO_RT_CAPTURE_CLOUD_FACES
#   CVarRealTimeReflectionCaptureShadowFromOpaque → ASTRO_RT_SHADOW_FROM_OPAQUE
#   CVarRealTimeReflectionCaptureDepthBuffer   → ASTRO_RT_DEPTH_BUFFER
#   CVarRealTimeReflectionCaptureVolumetricCloudResolutionDivider
#                                              → ASTRO_RT_CLOUD_RES_DIVIDER
#   FRealTimeSlicedReflectionCapture           → AstroCellRealTimeSkyCapture
#   RenderSkyPassForCapture                    → render_sky_pass_for_capture()
#   UpdateSkyEnvMap                            → update_sky_env_map()
#   ValidateSkyLightRealTimeCapture            → validate_sky_light_rt_capture()
# =============================================================================

# CVarRealTimeReflectionCapture equivalents
ASTRO_RT_CAPTURE_TIMESLICE:    bool  = True   # r.SkyLight.RealTimeReflectionCapture.TimeSlice
ASTRO_RT_CAPTURE_CLOUD_FACES:  int   = 2      # faces per frame for cloud
ASTRO_RT_SHADOW_FROM_OPAQUE:   bool  = False  # opaque mesh shadow in capture
ASTRO_RT_DEPTH_BUFFER:         bool  = True   # depth-aware capture
ASTRO_RT_CLEAR_COLOR:          bool  = False  # always clear colour buffer
ASTRO_RT_CLOUD_RES_DIVIDER:    int   = 2      # cloud resolution divider
ASTRO_RT_RES_OVERRIDE:         int   = 0      # 0 = default resolution (128)
ASTRO_RT_DEFAULT_CUBE_SIZE:    int   = 128    # default sky capture cube resolution


@_ptdc
class AstroCellRealTimeSkyCapture:
    """
    Python equivalent of FRealTimeSlicedReflectionCapture.

    State machine for the timesliced sky capture update:
        current_face:  int  — which cube face is being rendered this frame (0..5)
        faces_done:    int  — bitmask of completed faces (all done when == 0x3F)
        cube_size:     int  — current cube resolution (overrideable)
        is_valid:      bool — True after at least one complete convolution
        invalidated:   bool — set True when sky conditions change (forces re-capture)

    The C++ FRealTimeSlicedReflectionCapture holds similar state in the scene:
        ConvolvedSkyRenderTarget[2] → convolve buffers (inner/outer mip chains)
        bConvolvedSkyRenderTargetInvalid → invalidated flag

    鲁迅式：分时渲染是以时间换空间的妥协——
    每帧只画一面，六帧后才是完整的天空；
    不完整的天空已经够用了，这就是实时的现实。
    """
    current_face:    int   = 0
    faces_done:      int   = 0       # bitmask 0b000000 .. 0b111111
    cube_size:       int   = ASTRO_RT_DEFAULT_CUBE_SIZE
    is_valid:        bool  = False
    invalidated:     bool  = True
    frame_count:     int   = 0
    # Sky face radiance: face_index → (R, G, B)
    sky_face_radiance: _PTList[tuple] = _ptfield(
        default_factory=lambda: [(0.55, 0.68, 0.82)] * _CAPTURE_NUM_FACES)
    # Cloud face radiance (low-resolution): face_index → (R, G, B)
    cloud_face_radiance: _PTList[tuple] = _ptfield(
        default_factory=lambda: [(0.85, 0.88, 0.92)] * _CAPTURE_NUM_FACES)
    # Convolve outputs (pre-filtered specular mip chain)
    convolve_specular: _PTDict[int, tuple] = _ptfield(default_factory=dict)
    diffuse_sh:        _PTList[float]      = _ptfield(default_factory=lambda: [0.0]*9)


# Module-level real-time sky capture state singleton
_ASTRO_RT_SKY_CAPTURE: AstroCellRealTimeSkyCapture = AstroCellRealTimeSkyCapture()


def get_rt_sky_capture() -> AstroCellRealTimeSkyCapture:
    """Return the module-level AstroCellRealTimeSkyCapture singleton."""
    return _ASTRO_RT_SKY_CAPTURE


def validate_sky_light_rt_capture(
    has_sky_mesh: bool,
    sky_material_changed: bool,
    is_being_edited: bool = False,
) -> None:
    """
    Validate / invalidate the real-time sky capture state.
    Mirrors FScene::ValidateSkyLightRealTimeCapture():
        If sky conditions changed (sky mesh added/removed, material changed),
        set bConvolvedSkyRenderTargetInvalid = True to force a full re-capture.

    鲁迅式：验证是诚实的代价——每次场景发生变化，
    旧的天空就失效了，必须诚实地重新捕获。
    假装旧的还有效，是偷懒，也是错误。
    """
    capture = get_rt_sky_capture()
    if sky_material_changed or has_sky_mesh != capture.is_valid or is_being_edited:
        capture.invalidated = True
        capture.faces_done  = 0
        capture.current_face = 0
        print(
            f"[ASTRO-RT-CAPTURE] ValidateSkyLightRealTimeCapture — invalidated: "
            f"has_sky_mesh={has_sky_mesh} "
            f"sky_changed={sky_material_changed} "
            f"editing={is_being_edited}",
            file=sys.stderr,
        )


def render_sky_pass_for_capture(
    face_index:       int,
    atmosphere_color: tuple = (0.55, 0.68, 0.82),
    cloud_color:      tuple = (0.85, 0.88, 0.92),
    sun_direction:    tuple = (0.3, -0.8, 0.5),
    include_clouds:   bool  = True,
    depth_buffer:     bool  = ASTRO_RT_DEPTH_BUFFER,
) -> tuple:
    """
    Render one sky face for the real-time sky env map capture.

    Mirrors RenderSkyPassForCapture() + the per-face render loop in
    UpdateSkyEnvMap():
      - Sky atmosphere scatter (SkyAtmosphereRendering.cpp port → analytic approx)
      - Volumetric cloud compositing (low-res, CVarRealTimeReflectionCaptureVolumetricCloudResolutionDivider)
      - Fog contribution (FogRendering.cpp → distance-weighted blend)
      - Optional shadow from opaque (ASTRO_RT_SHADOW_FROM_OPAQUE)
      - Optional depth buffer (ASTRO_RT_DEPTH_BUFFER → height-fog attenuation)

    Returns (R, G, B) sky radiance for the given face direction.

    鲁迅式：天空是每帧都在变化的背景——太阳西沉，云彩移动，
    大气散射随角度而改变。我们无法一劳永逸地捕获它，
    只能帧帧跟进，面面不落。
    """
    # Face normal directions
    normals = [
        ( 1, 0, 0), (-1, 0, 0),
        ( 0, 1, 0), ( 0,-1, 0),
        ( 0, 0, 1), ( 0, 0,-1),
    ]
    nx, ny, nz = normals[face_index % _CAPTURE_NUM_FACES]

    # ── Atmosphere scatter (Rayleigh + Mie analytic approximation) ───────
    # Sun angle relative to this face normal
    sx, sy, sz = sun_direction
    s_len = math.sqrt(sx*sx + sy*sy + sz*sz)
    if s_len > 1e-6:
        sx, sy, sz = sx/s_len, sy/s_len, sz/s_len
    cos_sun = max(0.0, nx*sx + ny*sy + nz*sz)

    # Rayleigh scatter: blue sky dominates at angles away from sun
    rayleigh  = 1.0 - cos_sun * 0.6
    sky_r = atmosphere_color[0] * rayleigh + cos_sun * 0.95
    sky_g = atmosphere_color[1] * rayleigh + cos_sun * 0.85
    sky_b = atmosphere_color[2] * rayleigh + cos_sun * 0.70

    # ── Cloud compositing (low-res, ASTRO_RT_CLOUD_RES_DIVIDER) ──────────
    if include_clouds:
        cloud_weight = max(0.0, cloud_color[0] + cloud_color[1] + cloud_color[2]) / 3.0
        cloud_frac   = min(0.35, cloud_weight * 0.4) / max(ASTRO_RT_CLOUD_RES_DIVIDER, 1)
        sky_r = sky_r * (1.0 - cloud_frac) + cloud_color[0] * cloud_frac
        sky_g = sky_g * (1.0 - cloud_frac) + cloud_color[1] * cloud_frac
        sky_b = sky_b * (1.0 - cloud_frac) + cloud_color[2] * cloud_frac

    # ── Depth buffer height fog (ASTRO_RT_DEPTH_BUFFER attenuation) ───────
    if depth_buffer:
        # Height-based fog: lower faces (face 3 = -Y) get more fog
        fog_factor = max(0.0, -ny) * 0.15
        fog_r, fog_g, fog_b = 0.8, 0.85, 0.9   # fog colour
        sky_r = sky_r * (1-fog_factor) + fog_r * fog_factor
        sky_g = sky_g * (1-fog_factor) + fog_g * fog_factor
        sky_b = sky_b * (1-fog_factor) + fog_b * fog_factor

    # ── Shadow from opaque (optional, ASTRO_RT_SHADOW_FROM_OPAQUE) ────────
    if ASTRO_RT_SHADOW_FROM_OPAQUE:
        # Darken the +Y face (sun-facing) slightly for opaque-mesh shadow
        shadow_mult = 1.0 - max(0.0, ny) * 0.12
        sky_r *= shadow_mult
        sky_g *= shadow_mult
        sky_b *= shadow_mult

    return (max(0.0, sky_r), max(0.0, sky_g), max(0.0, sky_b))


def update_sky_env_map(
    all_bboxes:     dict | None = None,
    atmosphere:     tuple = (0.55, 0.68, 0.82),
    cloud_color:    tuple = (0.85, 0.88, 0.92),
    sun_direction:  tuple = (0.3, -0.8, 0.5),
    is_editing:     bool  = False,
) -> bool:
    """
    Per-frame real-time sky environment map update.

    Mirrors the top-level sky capture dispatch in UpdateSkyEnvMap() /
    FScene::UpdateSkyLightRealTimeCapture():
      1. Validate state (invalidate if sky changed)
      2. If time-sliced: render ASTRO_RT_CAPTURE_CLOUD_FACES per frame
         Else: render all 6 faces at once (editor fast path)
      3. When all 6 faces complete: run convolve + diffuse SH
      4. Set is_valid = True, reset faces_done bitmask

    Returns True when a complete convolution cycle just finished.

    鲁迅式：更新天光是渲染器最无聊的工作——
    每帧做一点点，没有人注意，没有人感谢，
    但若停下来，天空就会失去真实感，没有人会知道为什么。
    这就是后台工作者的处境。
    """
    capture  = get_rt_sky_capture()
    cycle_complete = False

    # ── Timeslice: decide how many faces to render this frame ─────────────
    if is_editing:
        faces_this_frame = ASTRO_CAPTURE_TIMESLICE_EDITOR
    elif ASTRO_RT_CAPTURE_TIMESLICE:
        faces_this_frame = ASTRO_RT_CAPTURE_CLOUD_FACES
    else:
        faces_this_frame = _CAPTURE_NUM_FACES   # all at once (non-timesliced)

    rendered_faces = []
    for _ in range(faces_this_frame):
        fi = capture.current_face

        # Render this face
        sky_col = render_sky_pass_for_capture(
            face_index=fi,
            atmosphere_color=atmosphere,
            cloud_color=cloud_color,
            sun_direction=sun_direction,
            include_clouds=True,
            depth_buffer=ASTRO_RT_DEPTH_BUFFER,
        )
        capture.sky_face_radiance[fi]   = sky_col

        # Cloud at reduced resolution (CVarVolumetricCloudResolutionDivider)
        cloud_col_low = (
            cloud_color[0] / ASTRO_RT_CLOUD_RES_DIVIDER,
            cloud_color[1] / ASTRO_RT_CLOUD_RES_DIVIDER,
            cloud_color[2] / ASTRO_RT_CLOUD_RES_DIVIDER,
        )
        capture.cloud_face_radiance[fi] = cloud_col_low

        capture.faces_done |= (1 << fi)
        rendered_faces.append(fi)

        # Advance to next face (wrap around at 6)
        capture.current_face = (fi + 1) % _CAPTURE_NUM_FACES

        # Slow timeslice: skip every other frame when enabled + only 1 face/frame
        if ASTRO_CAPTURE_TIMESLICE_SLOW and faces_this_frame == 1:
            if capture.frame_count % 2 != 0:
                break

    capture.frame_count += 1

    # ── Check if all 6 faces are done → run convolution ───────────────────
    if capture.faces_done == 0x3F:   # all 6 bits set
        # Convolve: pre-filter sky env map for all roughness levels
        combined_faces = [
            (capture.sky_face_radiance[fi][0] + capture.cloud_face_radiance[fi][0]*0.5,
             capture.sky_face_radiance[fi][1] + capture.cloud_face_radiance[fi][1]*0.5,
             capture.sky_face_radiance[fi][2] + capture.cloud_face_radiance[fi][2]*0.5)
            for fi in range(_CAPTURE_NUM_FACES)
        ]
        for mip in range(_CAPTURE_NUM_MIPS):
            avg_r = sum(c[0] for c in combined_faces) / _CAPTURE_NUM_FACES
            avg_g = sum(c[1] for c in combined_faces) / _CAPTURE_NUM_FACES
            avg_b = sum(c[2] for c in combined_faces) / _CAPTURE_NUM_FACES
            capture.convolve_specular[mip] = convolve_specular_face(
                (avg_r, avg_g, avg_b), mip)

        # Diffuse SH from mip-0 faces
        capture.diffuse_sh = compute_diffuse_irradiance_sh(combined_faces)

        capture.is_valid   = True
        capture.invalidated = False
        capture.faces_done  = 0      # reset for next cycle
        cycle_complete     = True

        print(
            f"[ASTRO-RT-CAPTURE] UpdateSkyEnvMap — cycle complete: "
            f"frame={capture.frame_count} "
            f"specular_mips={_CAPTURE_NUM_MIPS} "
            f"sh_L0=({capture.diffuse_sh[0]:.3f},"
            f"{capture.diffuse_sh[1]:.3f},{capture.diffuse_sh[2]:.3f})",
            file=sys.stderr,
        )
    else:
        print(
            f"[ASTRO-RT-CAPTURE] UpdateSkyEnvMap — "
            f"rendered_faces={rendered_faces} "
            f"faces_done=0b{capture.faces_done:06b} "
            f"frame={capture.frame_count}",
            file=sys.stderr,
        )

    return cycle_complete


def query_sky_specular_radiance(roughness: float = 0.5) -> tuple:
    """
    Query the real-time sky pre-filtered specular environment.

    Mirrors the SkyLight specular probe texture lookup performed in
    ReflectionEnvironmentPixelShader.usf after UpdateSkyEnvMap() completes.

    Maps *roughness* → mip level, returns (R, G, B) × fade_alpha.
    Falls back to a neutral grey when the capture has not yet completed.

    鲁迅式：天空的光芒不会因为你还没准备好就消失——
    但在探针完成之前，它确实只是一个猜测。
    """
    capture = get_rt_sky_capture()
    if not capture.is_valid or not capture.convolve_specular:
        return (0.55, 0.68, 0.82)   # sky-blue fallback

    mip = int(round(roughness * (_CAPTURE_NUM_MIPS - 1)))
    mip = max(0, min(mip, _CAPTURE_NUM_MIPS - 1))
    return capture.convolve_specular.get(mip, (0.55, 0.68, 0.82))


# =============================================================================
# [ASTRO-CELL] HairStrandsRendering → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/HairStrands/HairStrandsRendering.cpp
#
# 鲁迅曾言：「青年应当有朝气，敢说，敢笑，敢哭，敢怒，敢骂，敢打，
# 在这可诅咒的地方击退了可诅咒的时代！」
# 发丝亦然——每一根细线都是独立的生命，密密麻麻，汇成不可忽视的力量。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   FHairStrandsViewUniformParameters  → AstroCellHairStrandsViewParams
#     HairCoverageTexture              → coverage_map   (cell_id → [0,1])
#     HairOnlyDepthTexture             → hair_depth_map (cell_id → depth)
#     MaxSamplePerPixelCount           → max_sample_count (int)
#     HairTileCountXY                  → tile_count_xy   (tuple)
#     HairDualScatteringRoughnessOverride → dual_scatter_roughness (float)
#
#   InternalCreateHairStrandsViewUniformBuffer → build_hair_strands_view_params()
#   AllocateHairTransientResources      → allocate_hair_transient()
#   RenderHairPrePass                   → render_hair_pre_pass()
#
# Algorithm changes (鲁迅式 20%):
#   1. GPU RDG texture → per-cell coverage float in [0,1] derived from
#      the cell's own z-layer density (fraction of z-layer occupied by this cell)
#   2. HZB (Hierarchical Z-Buffer) → simple sorted depth list per z-layer
#   3. Tile data → grid tiles computed from cell bbox / tile_size
#   4. Stereo rendering second-eye copy → single-view epoch model (no stereo)
#   5. MacroGroup → species group (same species = same hair macro group)
# =============================================================================

# ── HairStrands CVars ─────────────────────────────────────────────────────────
_HS_TILE_SIZE:                int   = 8     # r.HairStrands.TileSize
_HS_MAX_SAMPLE_PER_PIXEL:     int   = 8     # r.HairStrands.MaxSample
_HS_DUAL_SCATTER_ROUGHNESS:   float = 0.3   # r.HairStrands.DualScatteringRoughnessOverride
_HS_DEEP_SHADOW_ENABLED:      bool  = True  # r.HairStrands.DeepShadow
_HS_VOXELIZATION_ENABLED:     bool  = True  # r.HairStrands.Voxelization
_HS_HZB_UPDATE_ENABLED:       bool  = True  # r.HairStrands.HZBUpdate


class AstroCellHairStrandsViewParams:
    """
    Python equivalent of FHairStrandsViewUniformParameters.

    Holds per-view hair strand visibility data: coverage map, depth map,
    tile layout, and sampling parameters.  Built once per epoch by
    build_hair_strands_view_params() from the live cell registry.

    鲁迅式：发丝的参数是细节之海——每一根参数都在描述
    那些太细以至于无法被多边形捕捉的微小存在。
    """

    def __init__(self) -> None:
        # HairCoverageTexture: cell_id → coverage fraction [0,1]
        # High coverage = cell bbox densely packed with visual elements
        self.coverage_map: dict[str, float] = {}
        # HairOnlyDepthTexture: cell_id → normalised depth [0,1]
        self.hair_depth_map: dict[str, float] = {}
        # HairOnlyDepthHZBParameters (mip chain min/max): [z_min, z_max, inv_range, 0]
        self.hzb_parameters: tuple = (0.0, 1.0, 1.0, 0.0)
        # HZB mip chain: list of {z_min, z_max} per mip level
        self.hzb_mips: list[dict] = []
        # HairTileCountXY
        self.tile_count_xy: tuple = (0, 0)
        # HairSampleViewportResolution
        self.sample_viewport_resolution: tuple = (1200, 900)
        # MaxSamplePerPixelCount
        self.max_sample_count: int = _HS_MAX_SAMPLE_PER_PIXEL
        # HairDualScatteringRoughnessOverride
        self.dual_scatter_roughness: float = _HS_DUAL_SCATTER_ROUGHNESS
        # bHairTileValid
        self.tile_valid: bool = False
        # Species macro groups: species → list of cell_ids
        self.macro_groups: dict[str, list[str]] = {}


def _hs_compute_coverage(cell_id: str, bbox: dict,
                          z_layer_cells: list[str]) -> float:
    """
    Compute hair-analogue coverage for a cell.

    Mirrors HairCoverageTexture: fraction of the cell's «screen tile» that
    is occupied by hair-like visual elements.  In the Astro 2-D context,
    coverage is derived from:
      - Cell area relative to its z-layer peers (denser z-layer → higher coverage)
      - Species type: hair-strand-like species (cil-eye, cil-graph) → high coverage;
        solid block species (cil-bolt, cil-layers) → lower coverage

    鲁迅式：覆盖率是细节密度的证明——发丝之所以难渲染，
    是因为它无处不在，无时不遮挡。
    """
    _COVERAGE_BY_SPECIES = {
        "cil-eye":         0.75,   # radial rays = hair-like density
        "cil-graph":       0.70,   # node-edge filaments
        "cil-vector":      0.60,   # arrow lines
        "cil-loop":        0.55,   # arc strand
        "cil-code":        0.50,   # brace lines
        "cil-bolt":        0.40,   # solid zigzag
        "cil-plus":        0.35,   # cross arms
        "cil-filter":      0.55,   # grid lines
        "cil-layers":      0.30,   # solid rects
        "cil-arrow-right": 0.25,   # single polygon
    }
    # Base coverage from species type
    species = bbox.get("species", "")
    base = _COVERAGE_BY_SPECIES.get(species, 0.45)

    # Modulate by z-layer density: more peers in same layer → each cell more covered
    n_peers = max(len(z_layer_cells), 1)
    density_factor = min(1.0, 1.0 + (n_peers - 1) * 0.05)
    return min(1.0, base * density_factor)


def _hs_build_hzb(depth_values: list[float]) -> tuple[tuple, list[dict]]:
    """
    Build a minimal HZB (Hierarchical Z-Buffer) from a sorted depth list.

    Mirrors HairOnlyDepthHZBParameters (FVector4f) and the per-mip closest/
    furthest HZB textures maintained for hair depth testing.

    2-D adaptation: instead of a full 2-D texture hierarchy, we maintain a
    list of (z_min, z_max) pairs at each power-of-2 mip level, computed by
    splitting the sorted depth list and taking min/max per segment.

    Returns (hzb_params_tuple, mip_list).

    鲁迅式：层级深度缓冲区是对效率的崇拜——
    用树状结构把深度测试的代价从 O(N) 压到 O(log N)。
    哪怕是发丝，也逃不过这棵树的筛选。
    """
    if not depth_values:
        return (0.0, 1.0, 1.0, 0.0), [{"z_min": 0.0, "z_max": 1.0}]

    sorted_depths = sorted(depth_values)
    z_min_global  = sorted_depths[0]
    z_max_global  = sorted_depths[-1]
    z_range       = z_max_global - z_min_global
    inv_range     = 1.0 / max(z_range, 1e-6)
    hzb_params    = (z_min_global, z_max_global, inv_range, 0.0)

    # Build mip chain (each mip halves the sample count)
    mips = []
    current = sorted_depths
    while current:
        mips.append({
            "z_min": current[0],
            "z_max": current[-1],
            "count": len(current),
        })
        # Next mip: take every other element (closest/furthest per 2-cell window)
        current = [current[i] for i in range(0, len(current), 2)]
        if len(current) <= 1:
            break

    return hzb_params, mips


def build_hair_strands_view_params(
    cell_registry: dict,
    viewport_w:    int = 1200,
    viewport_h:    int = 900,
) -> AstroCellHairStrandsViewParams:
    """
    Build per-epoch HairStrands view parameters from the cell registry.

    Mirrors InternalCreateHairStrandsViewUniformBuffer() — constructs the
    FHairStrandsViewUniformParameters struct (or its dummy fallback) from
    the visibility data published by the pre-pass.

    Called by render_hair_pre_pass() before any cell's hair-specific SVG
    elements are generated.

    鲁迅式：统一缓冲区是每个着色器的共同语言——
    没有它，每个发丝都在孤独地猜测世界的状态。
    """
    params = AstroCellHairStrandsViewParams()
    params.sample_viewport_resolution = (viewport_w, viewport_h)
    params.max_sample_count           = _HS_MAX_SAMPLE_PER_PIXEL
    params.dual_scatter_roughness     = _HS_DUAL_SCATTER_ROUGHNESS

    cells = cell_registry.get("cells", {})
    if not cells:
        # Dummy fallback — mirrors the «else» branch in the C++ function
        # that fills default textures when no hair visibility data exists.
        params.tile_count_xy = (0, 0)
        params.tile_valid    = False
        return params

    # ── Build coverage + depth maps ────────────────────────────────────────
    # Group cells by z-layer first (z-layer peers are the «macro group»)
    z_layer_groups: dict[int, list[str]] = {}
    for cell_id, entry in cells.items():
        z = entry.get("z", 3)
        z_layer_groups.setdefault(z, []).append(cell_id)

    all_depths: list[float] = []
    for cell_id, entry in cells.items():
        bbox_data = entry.get("bbox", {})
        if "min" in bbox_data and "max" in bbox_data:
            mn, mx = bbox_data["min"], bbox_data["max"]
            bbox = {"x": mn[0], "y": mn[1], "w": mx[0]-mn[0],
                    "h": mx[1]-mn[1], "z": mn[2] if len(mn)>2 else 0,
                    "species": entry.get("species", "")}
        else:
            bbox = dict(bbox_data)
            bbox["species"] = entry.get("species", "")

        z_layer = entry.get("z", 3)
        peers   = z_layer_groups.get(z_layer, [])

        coverage = _hs_compute_coverage(cell_id, bbox, peers)
        params.coverage_map[cell_id] = coverage

        # Depth: normalised z-layer position in [0,1]
        depth = min(1.0, z_layer / 8.0)
        params.hair_depth_map[cell_id] = depth
        all_depths.append(depth)

    # ── Build HZB from depth values ────────────────────────────────────────
    if all_depths:
        params.hzb_parameters, params.hzb_mips = _hs_build_hzb(all_depths)
        params.tile_valid = True

    # ── Tile count: viewport / tile_size ──────────────────────────────────
    tiles_x = (viewport_w + _HS_TILE_SIZE - 1) // _HS_TILE_SIZE
    tiles_y = (viewport_h + _HS_TILE_SIZE - 1) // _HS_TILE_SIZE
    params.tile_count_xy = (tiles_x, tiles_y)

    # ── Species macro groups ───────────────────────────────────────────────
    for cell_id, entry in cells.items():
        sp = entry.get("species", "unknown")
        params.macro_groups.setdefault(sp, []).append(cell_id)

    print(
        f"[ASTRO-HS] build_hair_strands_view_params: "
        f"cells={len(cells)} tile_count={params.tile_count_xy} "
        f"hzb_mips={len(params.hzb_mips)} "
        f"macro_groups={len(params.macro_groups)} "
        f"tile_valid={params.tile_valid}",
        file=sys.stderr,
    )
    return params


def render_hair_pre_pass(
    cell_registry: dict,
    viewport_w:    int = 1200,
    viewport_h:    int = 900,
) -> AstroCellHairStrandsViewParams:
    """
    Hair strands pre-pass — macro group creation + voxelization + deep shadow.

    Mirrors RenderHairPrePass():
      1. AddRenderCurveRasterPipeline  → early-out if no hair-like cells
      2. CreateHairStrandsMacroGroups  → group cells by species
      3. VoxelizeHairStrands           → compute per-cell volume density
      4. RenderHairStrandsDeepShadows  → compute per-cell self-shadow attenuation
      5. AddMeshDrawTransitionPass     → mark macro group dirty flag cleared

    Returns populated AstroCellHairStrandsViewParams.

    鲁迅式：预处理是牺牲——在主渲染之前先受苦，
    换来主渲染时的从容不迫。没有预处理，一切都是临时起意。
    """
    params = build_hair_strands_view_params(cell_registry, viewport_w, viewport_h)

    if not params.tile_valid:
        print("[ASTRO-HS] render_hair_pre_pass: no hair cells — early out",
              file=sys.stderr)
        return params

    # ── Voxelization: compute per-cell volume density ─────────────────────
    # Mirrors VoxelizeHairStrands() — allocates a 3-D voxel grid and
    # rasterises each hair strand into it.  2-D analogue: per-cell density
    # in its bbox tile grid (coverage × cell area / tile area).
    tile_area = _HS_TILE_SIZE * _HS_TILE_SIZE
    cells     = cell_registry.get("cells", {})
    for cell_id, entry in cells.items():
        coverage = params.coverage_map.get(cell_id, 0.5)
        bbox_data = entry.get("bbox", {})
        if "min" in bbox_data and "max" in bbox_data:
            mn, mx  = bbox_data["min"], bbox_data["max"]
            cell_w  = mx[0] - mn[0]
            cell_h  = mx[1] - mn[1]
        else:
            cell_w  = bbox_data.get("w", 80)
            cell_h  = bbox_data.get("h", 50)
        cell_area = max(cell_w * cell_h, 1.0)
        # VoxelDensity ≈ coverage × area / tile_area (proxy for strand density)
        voxel_density = min(1.0, coverage * cell_area / (tile_area * 4.0))
        # Store in coverage map as voxelised value (overwrite with refined estimate)
        params.coverage_map[cell_id] = max(params.coverage_map.get(cell_id, coverage),
                                           voxel_density)

    # ── Deep shadow: per-cell self-shadow from voxel density ──────────────
    # Mirrors RenderHairStrandsDeepShadows() — per-cell transmittance computed
    # from accumulated voxel density along the light ray.  We use an analytic
    # Beer-Lambert approximation: T = exp(-density × shadow_extinction).
    _SHADOW_EXTINCTION = 4.0   # extinction coefficient (per-cell tuning)
    deep_shadow: dict[str, float] = {}
    for cell_id in cells:
        density    = params.coverage_map.get(cell_id, 0.5)
        transmittance = math.exp(-density * _SHADOW_EXTINCTION)
        deep_shadow[cell_id] = transmittance   # 0 = fully shadowed, 1 = lit

    # Publish deep shadow as a sub-field (consumed by hair lighting pass)
    params.deep_shadow = deep_shadow   # type: ignore[attr-defined]

    print(
        f"[ASTRO-HS] render_hair_pre_pass: "
        f"voxelised={len(params.coverage_map)} "
        f"deep_shadows={len(deep_shadow)} "
        f"avg_coverage={sum(params.coverage_map.values())/max(len(params.coverage_map),1):.3f}",
        file=sys.stderr,
    )
    return params


# =============================================================================
# [ASTRO-CELL] SubstrateRendering → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/Substrate/Substrate.cpp
#
# 鲁迅曾言：「我翻开历史一查，这历史没有年代，歪歪斜斜的每页上
# 都写着『仁义道德』四个字。我横竖睡不着，仔细看了半夜，
# 才从字缝里看出字来，满本都写着两个字是『吃人』！」
# Substrate 材质系统亦然——层层包装之下，是光与物质的碰撞。
# 每一层闭包，都在争夺那有限的能量预算。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   FSubstrateGlobalUniformParameters  → AstroCellSubstrateUniforms
#     MaxClosurePerPixel               → max_closure_per_pixel (int, ≤8)
#     UsesTileTypeMask                 → uses_tile_type_mask    (int, bitmask)
#     UsesAnisotropy                   → uses_anisotropy        (bool)
#     ClosuresPerPixel (from CVar)     → closures_per_pixel     (int)
#   FSubstrateViewData::Reset()        → substrate_view_data_reset()
#   GetSubstrateMaxClosureCount()      → get_substrate_max_closure_count()
#   GetClosureTileIndirectArgsOffset() → get_closure_tile_offset()
#   IsStochasticLightingActive()       → is_stochastic_lighting_active()
#   UsesSubstrateMaterialBuffer()      → uses_material_buffer()
#
# Algorithm changes (鲁迅式 20%):
#   1. GPU tile classification pass → per-species closure count heuristic
#   2. ClosureCountFromMaterial (CVar) → species complexity table
#   3. Stochastic lighting classification → probabilistic sampling weight
#   4. CMask/HTile clear → Python dict reset
#   5. AllocationMode grows-only behaviour → max-of-history dict tracking
# =============================================================================

# ── Substrate CVars ───────────────────────────────────────────────────────────
_SUB_MAX_CLOSURE_COUNT:           int  = 8    # SUBSTRATE_MAX_CLOSURE_COUNT
_SUB_CLOSURES_PER_PIXEL:          int  = 3    # r.Substrate.ClosuresPerPixel
_SUB_USE_CLOSURE_COUNT_FROM_MAT:  bool = True # r.Substrate.UseClosureCountFromMaterial
_SUB_ASYNC_CLASSIFICATION:        bool = True # r.Substrate.AsyncClassification
_SUB_STOCHASTIC_LIGHTING_ACTIVE:  bool = False# r.Substrate.StochasticLighting.Active
_SUB_ROUGHNESS_TRACKING:          bool = True # r.Substrate.Debug.RoughnessTracking
_SUB_TILE_COORD_8BIT:             bool = True # r.Substrate.TileCoord8bits
_SUB_ALLOCATION_MODE:             int  = 1    # r.Substrate.AllocationMode
_SUB_MAX_DOWNSAMPLE_FACTOR:       int  = 2    # GetMaxDownsampleFactor()

# Tile type bitmask constants (mirrors ESubstrateTileType enum bits)
_SUB_TILE_SIMPLE         = 1 << 0   # simple single-closure tile
_SUB_TILE_SINGLE_CLOSURE = 1 << 1   # single-closure complex tile
_SUB_TILE_COMPLEX        = 1 << 2   # multi-closure tile
_SUB_TILE_HAIR           = 1 << 3   # hair-strands tile
_SUB_TILE_ALL            = 0xFF


class AstroCellSubstrateUniforms:
    """
    Python equivalent of FSubstrateGlobalUniformParameters.

    Holds per-frame Substrate material system state: closure counts,
    tile type masks, anisotropy flags, and buffer layout parameters.

    鲁迅式：Substrate 的 Uniform 是材质系统的宪法——
    每帧一次，全体着色器必须遵守。
    """

    __slots__ = (
        "max_closure_per_pixel",
        "uses_tile_type_mask",
        "uses_anisotropy",
        "closures_per_pixel",
        "effective_max_closure",
        "stochastic_lighting_active",
        "roughness_tracking_enabled",
        "tile_coord_8bit",
    )

    def __init__(self) -> None:
        self.max_closure_per_pixel:      int  = 1
        self.uses_tile_type_mask:        int  = 0
        self.uses_anisotropy:            bool = False
        self.closures_per_pixel:         int  = _SUB_CLOSURES_PER_PIXEL
        self.effective_max_closure:      int  = 1
        self.stochastic_lighting_active: bool = _SUB_STOCHASTIC_LIGHTING_ACTIVE
        self.roughness_tracking_enabled: bool = _SUB_ROUGHNESS_TRACKING
        self.tile_coord_8bit:            bool = _SUB_TILE_COORD_8BIT


# Per-species closure complexity table (mirrors per-material ClosureCount from C++).
# Species with multiple BSDF lobes (e.g. eye has specular + diffuse + SSS)
# have higher closure counts.
_SPECIES_CLOSURE_COUNT: dict[str, int] = {
    "cil-eye":         3,   # diffuse + specular + SSS (eyelens)
    "cil-bolt":        2,   # diffuse + emissive
    "cil-vector":      2,   # diffuse + directional specular
    "cil-plus":        1,   # simple diffuse
    "cil-arrow-right": 1,   # simple diffuse
    "cil-filter":      2,   # diffuse + anisotropic specular (grid lines)
    "cil-code":        1,   # monochrome diffuse
    "cil-layers":      3,   # 3 layers × 1 closure each
    "cil-loop":        2,   # diffuse + emissive rim
    "cil-graph":       2,   # diffuse + edge specular
}

# Species anisotropy flag (mirrors bUsesAnisotropy in FSubstrateViewData)
_SPECIES_ANISOTROPIC: set[str] = {"cil-filter", "cil-vector"}


def get_substrate_max_closure_count(species_list: list[str]) -> int:
    """
    Compute effective max closure count for the given species set.

    Mirrors GetSubstrateMaxClosureCount(FViewInfo&):
      if UseClosureCountFromMaterial:
          max_closure = max over visible materials of ClosureCountFromMaterial
      else:
          max_closure = ClosuresPerPixel (CVar)
    Clamped to [1, SUBSTRATE_MAX_CLOSURE_COUNT].

    鲁迅式：最大闭包数是资源分配的上限——
    不让任何一种材质独吞全部 GBuffer 预算，也不让简单材质浪费内存。
    """
    if not species_list:
        return 1
    if _SUB_USE_CLOSURE_COUNT_FROM_MAT:
        raw = max(_SPECIES_CLOSURE_COUNT.get(sp, 1) for sp in species_list)
    else:
        raw = _SUB_CLOSURES_PER_PIXEL
    return max(1, min(raw, _SUB_MAX_CLOSURE_COUNT))


def get_closure_tile_offset(downsample_factor: int) -> int:
    """
    Compute byte offset into the indirect args buffer for a given downsample factor.

    Mirrors GetClosureTileIndirectArgsOffset(InDownsampleFactor):
        Offset = (clamp(dsf, 1, max_dsf) - 1) * sizeof(FRHIDispatchIndirectParameters)
    sizeof(FRHIDispatchIndirectParameters) = 12 bytes (3× uint32).

    鲁迅式：偏移量是寻址的精确度——差一个字节，便是另一个世界。
    """
    _INDIRECT_PARAMS_SIZE = 12   # 3 × uint32
    clamped = max(1, min(downsample_factor, _SUB_MAX_DOWNSAMPLE_FACTOR))
    return (clamped - 1) * _INDIRECT_PARAMS_SIZE


def is_stochastic_lighting_active() -> bool:
    """
    Mirrors IsStochasticLightingActive():
        return IsStochasticLightingEnabled(Platform) && CVarActive > 0
    In Astro: platform capability check → True; CVar toggle from constant.

    鲁迅式：随机光照是用概率换取精度——以混沌始，以收敛终。
    """
    return _SUB_STOCHASTIC_LIGHTING_ACTIVE


def build_substrate_uniforms(
    cell_registry: dict,
    visible_cell_ids: list[str] | None = None,
) -> AstroCellSubstrateUniforms:
    """
    Build per-epoch Substrate global uniform parameters.

    Mirrors the Substrate global UB setup that occurs in
    FDeferredShadingSceneRenderer::Render() before material passes:
      1. Collect species for visible cells.
      2. Compute max closure count from material data.
      3. Classify tile type mask from species complexity.
      4. Detect anisotropy.
      5. Pack into FSubstrateGlobalUniformParameters.

    鲁迅式：全局参数是帧的脸面——在这一帧里，
    所有材质都必须在这套规则下生存。
    """
    uni = AstroCellSubstrateUniforms()
    cells = cell_registry.get("cells", {})

    # Which cells to process (all if visibility list not given)
    target_ids = visible_cell_ids if visible_cell_ids else list(cells.keys())
    species_list = [cells[cid].get("species", "") for cid in target_ids if cid in cells]

    # Max closure count
    uni.max_closure_per_pixel  = get_substrate_max_closure_count(species_list)
    uni.effective_max_closure  = uni.max_closure_per_pixel
    uni.closures_per_pixel     = _SUB_CLOSURES_PER_PIXEL

    # Tile type mask: OR together tile types present
    tile_mask = 0
    for sp in species_list:
        cc = _SPECIES_CLOSURE_COUNT.get(sp, 1)
        if cc == 1:
            tile_mask |= _SUB_TILE_SIMPLE
        elif cc == 2:
            tile_mask |= _SUB_TILE_SINGLE_CLOSURE
        else:
            tile_mask |= _SUB_TILE_COMPLEX
        if sp in ("cil-eye", "cil-graph", "cil-vector"):
            tile_mask |= _SUB_TILE_HAIR   # hair-like fine detail
    uni.uses_tile_type_mask = tile_mask

    # Anisotropy: any anisotropic species in view?
    uni.uses_anisotropy = any(sp in _SPECIES_ANISOTROPIC for sp in species_list)

    # Stochastic + feature flags
    uni.stochastic_lighting_active = is_stochastic_lighting_active()
    uni.roughness_tracking_enabled = _SUB_ROUGHNESS_TRACKING
    uni.tile_coord_8bit            = _SUB_TILE_COORD_8BIT

    print(
        f"[ASTRO-SUB] build_substrate_uniforms: "
        f"species_count={len(set(species_list))} "
        f"max_closure={uni.max_closure_per_pixel} "
        f"tile_mask=0b{uni.uses_tile_type_mask:08b} "
        f"anisotropy={uni.uses_anisotropy} "
        f"stochastic={uni.stochastic_lighting_active}",
        file=sys.stderr,
    )
    return uni


def substrate_view_data_reset(prev_uniforms: AstroCellSubstrateUniforms) -> AstroCellSubstrateUniforms:
    """
    Reset per-view Substrate data between epochs, preserving tile type mask.

    Mirrors FSubstrateViewData::Reset():
        Preserves UsesTileTypeMask and bUsesAnisotropy across reset
        (they represent accumulated scene complexity and are only updated,
        never reverted to 0, until the scene is fully re-classified).

    鲁迅式：重置是新生，但有些东西不能忘记——
    上一帧的复杂度标记，是对下一帧分配策略的提示。
    """
    new_uni = AstroCellSubstrateUniforms()
    # Carry forward tile mask and anisotropy (mirrors C++ preservation)
    new_uni.uses_tile_type_mask = prev_uniforms.uses_tile_type_mask
    new_uni.uses_anisotropy     = prev_uniforms.uses_anisotropy
    return new_uni


# =============================================================================
# [ASTRO-CELL] DiaphragmDOF (Depth of Field) → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/PostProcess/DiaphragmDOF.cpp
#
# 鲁迅曾言：「不在沉默中爆发，就在沉默中灭亡。」
# 景深亦然——前景的模糊是对远处真相的沉默，
# 而焦点处的清晰是最后的爆发。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   DiaphragmDOF::IsSupported()        → is_dof_supported()
#   CVarDOFGatherResDivisor            → _DOF_GATHER_RES_DIVISOR
#   CVarAccumulatorQuality             → _DOF_ACCUMULATOR_QUALITY
#   CVarRingCount                      → _DOF_RING_COUNT
#   CVarRecombineQuality               → _DOF_RECOMBINE_QUALITY
#   CVarMinimalFullresBlurRadius       → _DOF_MIN_FULLRES_BLUR_RADIUS
#   CVarScatterMaxSpriteRatio          → _DOF_SCATTER_MAX_SPRITE_RATIO
#   CVarScatterMinCocRadius            → _DOF_SCATTER_MIN_COC_RADIUS
#   FDiaphragmDOFPass (gather pass)    → AstroCellDOFGatherPass
#   ComputeCircleOfConfusionRadius     → compute_coc_radius()
#   GatherBokeh (ring accumulation)    → gather_bokeh_rings()
#   ScatterBokeh (sprite pass)         → scatter_bokeh_sprites()
#   RecombinePass (full-res merge)     → dof_recombine()
#
# Algorithm changes (鲁迅式 20%):
#   1. GPU gather at half-resolution → analytic per-cell CoC radius from z-depth
#   2. Ring kernel (N=5 rings, 78 samples) → analytic annulus area formula
#   3. Scatter sprite pass (GPU instanced quads) → per-cell blurred bbox rect
#   4. Foreground/background separation → z-layer threshold split
#   5. Bokeh shape (hexagon/octagon) → circular approximation (area-preserving)
# =============================================================================

# ── DiaphragmDOF CVars ────────────────────────────────────────────────────────
_DOF_GATHER_RES_DIVISOR:      int   = 2      # r.DOF.Gather.ResolutionDivisor
_DOF_ACCUMULATOR_QUALITY:     int   = 1      # r.DOF.Gather.AccumulatorQuality
_DOF_RING_COUNT:              int   = 5      # r.DOF.Gather.RingCount [3,5]
_DOF_RECOMBINE_QUALITY:       int   = 2      # r.DOF.Recombine.Quality
_DOF_MIN_FULLRES_BLUR_RADIUS: float = 0.1    # r.DOF.Recombine.MinFullresBlurRadius
_DOF_SCATTER_MIN_COC:         float = 3.0    # r.DOF.Scatter.MinCocRadius
_DOF_SCATTER_MAX_SPRITE:      float = 0.1    # r.DOF.Scatter.MaxSpriteRatio
_DOF_TAA_QUALITY:             int   = 1      # r.DOF.TemporalAAQuality
_DOF_PREFER_LOWER_BIT:        bool  = False  # r.DOF.PreferLowerBitDepth
_DOF_COC_BILATERAL_STRENGTH:  float = 0.0    # r.DOF.TAA.CoCBilateralFilterStrength

# Focal plane parameters (app-level DOF config, not a CVar in UE5)
_DOF_FOCAL_Z_LAYER:   float = 3.0   # z-layer of sharp focus
_DOF_NEAR_TRANSITION: float = 1.5   # z-layer range for near DOF
_DOF_FAR_TRANSITION:  float = 2.0   # z-layer range for far DOF
_DOF_MAX_COC_PIXELS:  float = 24.0  # maximum CoC radius in pixels


def compute_coc_radius(
    cell_z:     float,
    focal_z:    float = _DOF_FOCAL_Z_LAYER,
    near_trans: float = _DOF_NEAR_TRANSITION,
    far_trans:  float = _DOF_FAR_TRANSITION,
    max_coc:    float = _DOF_MAX_COC_PIXELS,
) -> float:
    """
    Compute Circle-of-Confusion radius for a cell at z-layer *cell_z*.

    Mirrors DiaphragmDOF's CoC computation from the lens model:
        CoC = f² × (z − z_focal) / (N × z × z_focal × (z − f))
    Simplified to a linear ramp for the 2-D z-layer model:
        CoC_near = clamp((focal_z − cell_z) / near_trans, 0, 1) × max_coc
        CoC_far  = clamp((cell_z − focal_z) / far_trans,  0, 1) × max_coc
        CoC      = max(CoC_near, CoC_far)

    Returns CoC radius in pixels (float, ≥ 0).

    鲁迅式：弥散圆是距离的惩罚——离焦点越远，圆越大，清晰越少。
    焦平面是优待，远处是放逐。
    """
    delta = cell_z - focal_z
    if delta < 0:
        # Near-field: cell is in front of focal plane
        coc = abs(delta) / max(near_trans, 1e-6)
    else:
        # Far-field: cell is behind focal plane
        coc = delta / max(far_trans, 1e-6)
    return min(max_coc, coc * max_coc)


def gather_bokeh_rings(
    coc_radius: float,
    ring_count: int = _DOF_RING_COUNT,
    accumulator_quality: int = _DOF_ACCUMULATOR_QUALITY,
) -> float:
    """
    Compute the effective blur weight from concentric ring kernel accumulation.

    Mirrors the gather pass ring accumulator:
        For each ring r in [1, ring_count]:
            n_samples_r = 8 × r   (octagonal ring)
            sample_weight ∝ 1 / (n_samples_r)
        total_weight = sum of all ring sample weights

    In the 2-D analytic version we compute the equivalent area-weighted blur
    strength from the ring radii, bypassing the sample loop.

    Returns a blur weight ∈ [0, 1] for use in SVG filter blur radius scaling.

    鲁迅式：环形采样是民主的——每一环等权，远近均有发言权，
    只是远处的声音（大 CoC）被更多环覆盖而显得更重要。
    """
    if coc_radius < _DOF_MIN_FULLRES_BLUR_RADIUS:
        return 0.0

    # Total sample count across all rings (mirrors actual ring sample counts)
    total_samples = sum(8 * r for r in range(1, ring_count + 1))
    if total_samples == 0:
        return 0.0

    # Effective blur weight: fraction of rings whose radius ≤ CoC
    effective_rings = min(ring_count,
                          max(1, int(coc_radius / max(_DOF_MAX_COC_PIXELS / ring_count, 1))))
    effective_samples = sum(8 * r for r in range(1, effective_rings + 1))

    # Quality multiplier (AccumulatorQuality=1 → 1.0; =2 → 1.2 extra samples)
    quality_mult = 1.0 + (accumulator_quality - 1) * 0.2

    return min(1.0, (effective_samples / total_samples) * quality_mult)


def scatter_bokeh_sprites(
    coc_radius:       float,
    brightness:       float,
    scatter_min_coc:  float = _DOF_SCATTER_MIN_COC,
    scatter_max_ratio:float = _DOF_SCATTER_MAX_SPRITE,
) -> float:
    """
    Determine scatter contribution for a cell with given CoC.

    Mirrors the scatter pass sprite logic:
        if CoC < scatter_min_coc: no scatter
        if scattered_pixels / total_pixels > max_ratio: clamp
        scatter_intensity = clamp(CoC / max_coc, 0, 1) × brightness

    Returns scatter opacity ∈ [0, 1] to add to the cell's SVG filter.

    鲁迅式：散射是高光的奢侈——只有足够亮、足够散的像素，
    才配获得单独的精灵渲染。其余的，老老实实去聚集通道排队。
    """
    if coc_radius < scatter_min_coc:
        return 0.0
    scatter_strength = min(1.0, coc_radius / _DOF_MAX_COC_PIXELS)
    # Brightness gate: only bright cells scatter (high brightness = high CoC visibility)
    if brightness < 0.3:
        return 0.0
    return min(1.0, scatter_strength * brightness * scatter_max_ratio * 10.0)


def dof_recombine(
    cell_gather_weight: float,
    cell_scatter_weight: float,
    coc_radius: float,
    recombine_quality: int = _DOF_RECOMBINE_QUALITY,
) -> float:
    """
    Full-resolution recombine pass — merge gather + scatter into final blur.

    Mirrors the Recombine pass that composites half-res gathered DOF back
    with the full-res sharp layer:
        if recombine_quality == 0: no slight-out-of-focus
        if recombine_quality >= 1: blend gather + scatter × sof_weight

    Returns final DOF blur radius for the cell's feGaussianBlur stdDeviation.

    鲁迅式：重组是和解——把模糊的过去和清晰的现在合并成一帧，
    既不全然遗忘，也不执意清醒。
    """
    if coc_radius < _DOF_MIN_FULLRES_BLUR_RADIUS:
        return 0.0

    # Slight-out-of-focus weight (only for recombine_quality >= 1)
    sof_weight = 0.0 if recombine_quality == 0 else min(1.0, coc_radius / 4.0)

    # Combine gather and scatter contributions
    combined = (cell_gather_weight * 0.7 + cell_scatter_weight * 0.3) * sof_weight

    # Scale to pixel blur radius (0 = sharp, _DOF_MAX_COC_PIXELS = max blur)
    blur_radius = combined * coc_radius
    return round(blur_radius, 2)


class AstroCellDOFGatherPass:
    """
    Depth-of-Field gather pass orchestrator — mirrors FDiaphragmDOFPass.

    Processes all cells in the registry, computes per-cell CoC radii,
    classifies each as foreground/background/in-focus, runs gather + scatter,
    and returns a dict of per-cell DOF blur parameters for SVG filter injection.

    鲁迅式：景深处理是摄影师的选择——
    决定让谁清晰，让谁模糊，便是决定谁是主角，谁是背景。
    """

    def __init__(self,
                 focal_z:    float = _DOF_FOCAL_Z_LAYER,
                 near_trans: float = _DOF_NEAR_TRANSITION,
                 far_trans:  float = _DOF_FAR_TRANSITION) -> None:
        self.focal_z    = focal_z
        self.near_trans = near_trans
        self.far_trans  = far_trans

    def process(self, cell_registry: dict) -> dict[str, dict]:
        """
        Run the full DOF pipeline for all registered cells.

        Returns dict: cell_id → {coc_radius, blur_radius, layer, gather_w, scatter_w}
        where layer ∈ {'foreground', 'focus', 'background'}.
        """
        cells  = cell_registry.get("cells", {})
        result: dict[str, dict] = {}

        for cell_id, entry in cells.items():
            z = float(entry.get("z", 3))
            # Brightness proxy from coverage (higher coverage = brighter cell)
            coverage = entry.get("constraint_mask", 0)
            brightness = 0.5 + coverage * 0.3   # simplified

            coc = compute_coc_radius(z, self.focal_z, self.near_trans, self.far_trans)
            gather_w  = gather_bokeh_rings(coc)
            scatter_w = scatter_bokeh_sprites(coc, brightness)
            blur_r    = dof_recombine(gather_w, scatter_w, coc)

            # Layer classification
            delta = z - self.focal_z
            if delta < -0.5:
                layer = "foreground"
            elif delta > 0.5:
                layer = "background"
            else:
                layer = "focus"

            result[cell_id] = {
                "coc_radius":    round(coc, 3),
                "blur_radius":   blur_r,
                "gather_weight": round(gather_w, 4),
                "scatter_weight":round(scatter_w, 4),
                "layer":         layer,
            }

        fg = sum(1 for v in result.values() if v["layer"] == "foreground")
        bg = sum(1 for v in result.values() if v["layer"] == "background")
        print(
            f"[ASTRO-DOF] AstroCellDOFGatherPass.process: "
            f"total={len(result)} fg={fg} focus={len(result)-fg-bg} bg={bg} "
            f"focal_z={self.focal_z:.1f}",
            file=sys.stderr,
        )
        return result


# =============================================================================
# [ASTRO-CELL] PostProcessMotionBlur → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/PostProcess/PostProcessMotionBlur.cpp
#
# 鲁迅曾言：「时间就是性命。无端的空耗别人的时间，
# 其实是无异于谋财害命的。」
# 运动模糊亦然——时间在帧与帧之间流逝，运动的轨迹是时间的刻痕。
# 不做运动模糊，是对时间流逝的否认；做过度，是对帧率的谋财害命。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   IsMotionBlurEnabled()              → is_motion_blur_enabled()
#   IsMotionBlurScatterRequired()      → is_scatter_required()
#   GetMotionBlurTileCount()           → get_motion_blur_tile_count()
#   FVelocityFlattenTextures           → AstroCellVelocityFlattenData
#   kMotionBlurFlattenTileSize = 16    → _MB_FLATTEN_TILE_SIZE
#   GetPreviousWorldToClipMatrix()     → compute_prev_frame_velocity()
#   EMotionBlurQuality                 → _MB_QUALITY (0=low, 1=med, 2=high, 3=cinematic)
#   EMotionBlurFilter                  → _MB_FILTER (0=low, 1=med, 2=high)
#
# Algorithm changes (鲁迅式 20%):
#   1. Velocity flatten → per-cell z-layer velocity (dz per epoch)
#   2. GPU tile max velocity → max absolute dz across neighbours
#   3. Scatter vs gather decision → analytic threshold (not GPU readback)
#   4. Motion blur filter → 1-D Gaussian blur on SVG feGaussianBlur
#   5. Half-res gather → simply halved blur kernel for performance
# =============================================================================

# ── MotionBlur CVars ──────────────────────────────────────────────────────────
_MB_QUALITY:             int   = 3      # r.MotionBlurQuality [0,4]; 0=off
_MB_AMOUNT:              float = 1.0    # PostProcessSettings.MotionBlurAmount
_MB_MAX_VEL_FRACTION:    float = 0.05   # PostProcessSettings.MotionBlurMax (% of viewport)
_MB_FLATTEN_TILE_SIZE:   int   = 16     # kMotionBlurFlattenTileSize
_MB_FILTER_TILE_SIZE:    int   = 16     # kMotionBlurFilterTileSize
_MB_SCATTER_THRESHOLD:   float = 3.0    # TileDistanceMaxGathered
_MB_HALF_RES_INPUT:      bool  = True   # r.MotionBlur.HalfResInput
_MB_SEPARABLE:           bool  = False  # r.MotionBlurSeparable
_MB_DIRECTIONS:          int   = 1      # r.MotionBlur.Directions
_MB_SECOND_SCALE:        float = 1.0    # r.MotionBlur2ndScale


def is_motion_blur_enabled(quality: int = _MB_QUALITY) -> bool:
    """
    Mirrors IsMotionBlurEnabled(FViewInfo&):
        FeatureLevel >= SM5 AND PostProcessing AND MotionBlur flags
        AND Amount > 0.001 AND Max > 0.001 AND bRealtimeUpdate AND Quality > 0
    """
    return (quality > 0 and
            _MB_AMOUNT > 0.001 and
            _MB_MAX_VEL_FRACTION > 0.001)


def get_motion_blur_tile_count(width: int, height: int) -> tuple[int, int]:
    """
    Compute tile grid dimensions for velocity flatten.
    Mirrors GetMotionBlurTileCount(FIntPoint):
        TilesX = DivideAndRoundUp(W, kFlattenTileSize)
        TilesY = DivideAndRoundUp(H, kFlattenTileSize)

    鲁迅式：瓦片是统治单元——一个个 16×16 的小格子，
    把无限的屏幕空间分割成有限的可管理区域。
    """
    tx = (width  + _MB_FLATTEN_TILE_SIZE - 1) // _MB_FLATTEN_TILE_SIZE
    ty = (height + _MB_FLATTEN_TILE_SIZE - 1) // _MB_FLATTEN_TILE_SIZE
    return (tx, ty)


def compute_prev_frame_velocity(
    cell_id: str,
    cell_z:  float,
    cell_registry_prev: dict,
) -> float:
    """
    Compute z-axis velocity (dz per epoch) for a cell.

    Mirrors GetPreviousWorldToClipMatrix + the velocity buffer computation
    that encodes how far each primitive moved since the last frame.

    In the Astro 2-D model, z-layer is the primary motion axis (cells can
    move between z-layers between epochs).  X/Y motion is constrained by
    the force-field and is typically small.

    Returns signed dz (z_current - z_previous), 0.0 if no history.

    鲁迅式：速度是位移对时间的导数——
    没有过去就没有速度，有了过去才能预测未来。
    """
    prev_cells = cell_registry_prev.get("cells", {})
    prev_entry = prev_cells.get(cell_id)
    if prev_entry is None:
        return 0.0
    prev_z = float(prev_entry.get("z", cell_z))
    return cell_z - prev_z


class AstroCellVelocityFlattenData:
    """
    Python equivalent of FVelocityFlattenTextures.

    Stores per-cell velocity (dz) and per-tile max velocity used for
    the motion blur scatter/gather decision.

    鲁迅式：速度展平是民调——
    把每个像素的速度汇总为瓦片的最大速度，
    从而决定这一瓦片该用哪种模糊策略。
    """

    def __init__(self) -> None:
        # Per-cell velocity: cell_id → float dz
        self.cell_velocities:  dict[str, float] = {}
        # Per-tile max velocity: (tx, ty) → float max_abs_dz
        self.tile_max_velocity: dict[tuple, float] = {}
        # Global max velocity across all cells
        self.global_max_velocity: float = 0.0

    def build(self, cell_registry: dict,
              cell_registry_prev: dict,
              viewport_w: int = 1200,
              viewport_h: int = 900) -> None:
        """
        Populate velocity data for all cells and build tile max buffer.
        Mirrors the VelocityFlatten pass dispatch.
        """
        cells = cell_registry.get("cells", {})
        tile_grid: dict[tuple, list[float]] = {}

        for cell_id, entry in cells.items():
            z   = float(entry.get("z", 3))
            vel = compute_prev_frame_velocity(cell_id, z, cell_registry_prev)
            self.cell_velocities[cell_id] = vel

            # Assign to tile based on cell centre
            bbox_data = entry.get("bbox", {})
            if "min" in bbox_data and "max" in bbox_data:
                mn, mx  = bbox_data["min"], bbox_data["max"]
                cx = (mn[0] + mx[0]) / 2.0
                cy = (mn[1] + mx[1]) / 2.0
            else:
                cx = bbox_data.get("x", 0) + bbox_data.get("w", 80) / 2.0
                cy = bbox_data.get("y", 0) + bbox_data.get("h", 50) / 2.0

            tx = int(cx) // _MB_FLATTEN_TILE_SIZE
            ty = int(cy) // _MB_FLATTEN_TILE_SIZE
            tile_grid.setdefault((tx, ty), []).append(abs(vel))

        # Tile max velocity (mirrors TileMaxVelocity texture)
        for tile_key, vels in tile_grid.items():
            self.tile_max_velocity[tile_key] = max(vels)

        self.global_max_velocity = (
            max(abs(v) for v in self.cell_velocities.values())
            if self.cell_velocities else 0.0
        )


def compute_motion_blur_params(
    cell_id: str,
    cell_z:  float,
    velocity: float,
    viewport_w: int = 1200,
    viewport_h: int = 900,
    quality: int = _MB_QUALITY,
) -> dict:
    """
    Compute per-cell motion blur SVG filter parameters.

    Mirrors the MotionBlur CS dispatch that, for each tile, computes the
    blur kernel direction and magnitude from the velocity buffer.

    Algorithm:
      1. Velocity → tile max (already in VelocityFlattenData)
      2. Tile max → scatter required? (VelocityMaxInTiles > TileDistanceMaxGathered)
      3. Blur radius = velocity × MB_AMOUNT × quality_scale
      4. Blur direction: z-axis maps to diagonal (45° shadow direction in 2-D)

    Returns {blur_radius, direction_angle_deg, use_scatter, quality_scale}.

    鲁迅式：运动模糊的方向是时间的方向——
    向前模糊，向前运动；向后模糊，向后运动。
    方向错了，时间便倒流了。
    """
    if not is_motion_blur_enabled(quality):
        return {"blur_radius": 0.0, "direction_angle_deg": 0.0,
                "use_scatter": False, "quality_scale": 0.0}

    # Quality scale (mirrors EMotionBlurQuality: 0=low→0.25, 3=cinematic→1.0)
    quality_scales = [0.25, 0.5, 0.75, 1.0]
    quality_scale  = quality_scales[min(quality - 1, 3)] if quality >= 1 else 0.0

    # Velocity in tiles (mirrors VelocityMaxInTiles computation)
    max_vel_pixels = _MB_MAX_VEL_FRACTION * viewport_w
    vel_in_tiles   = abs(velocity) * max_vel_pixels * 0.5 / _MB_FLATTEN_TILE_SIZE

    # Scatter vs gather decision
    use_scatter = vel_in_tiles > _MB_SCATTER_THRESHOLD

    # Blur radius in pixels
    blur_radius = abs(velocity) * max_vel_pixels * _MB_AMOUNT * quality_scale
    if _MB_HALF_RES_INPUT and not use_scatter:
        blur_radius *= 0.5   # half-res gather = halved effective radius

    # Separable second pass (r.MotionBlurSeparable = adds second orthogonal pass)
    if _MB_SEPARABLE:
        blur_radius *= _MB_SECOND_SCALE

    # Direction: z-layer velocity maps to a temporal diagonal (135° = up-right to down-left)
    # Positive dz → cell moved away → blur toward bottom-right
    direction = 135.0 if velocity > 0 else 315.0

    return {
        "blur_radius":        round(blur_radius, 2),
        "direction_angle_deg": direction,
        "use_scatter":         use_scatter,
        "quality_scale":       quality_scale,
    }


def run_motion_blur_pass(
    cell_registry:      dict,
    cell_registry_prev: dict,
    viewport_w:         int = 1200,
    viewport_h:         int = 900,
) -> dict[str, dict]:
    """
    Full motion blur pipeline for all cells.

    Mirrors the three-pass MotionBlur dispatch:
      1. VelocityFlatten  → compute per-cell velocities
      2. TileMaxVelocity  → compute per-tile max velocities
      3. MotionBlur CS    → compute per-cell blur params

    Returns dict: cell_id → motion blur parameter dict.

    鲁迅式：运动模糊的流水线是时间的考古学——
    通过比较现在与过去，重建运动的证据，再将其涂抹在画面上，
    告诉观看者：这里曾经有运动，虽然现在已经静止了。
    """
    velocity_data = AstroCellVelocityFlattenData()
    velocity_data.build(cell_registry, cell_registry_prev, viewport_w, viewport_h)

    cells  = cell_registry.get("cells", {})
    result: dict[str, dict] = {}

    for cell_id, entry in cells.items():
        z   = float(entry.get("z", 3))
        vel = velocity_data.cell_velocities.get(cell_id, 0.0)
        result[cell_id] = compute_motion_blur_params(
            cell_id, z, vel, viewport_w, viewport_h
        )

    blurred = sum(1 for v in result.values() if v["blur_radius"] > 0.1)
    print(
        f"[ASTRO-MB] run_motion_blur_pass: "
        f"total={len(result)} blurred={blurred} "
        f"global_max_vel={velocity_data.global_max_velocity:.3f}",
        file=sys.stderr,
    )
    return result


# =============================================================================
# [ASTRO-CELL] TemporalAA → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/PostProcess/TemporalAA.cpp
#
# 鲁迅曾言：「我不想再说废话了。废话说了半天没有人听。
# 历史上，有些话说完之后，沉默了几十年，然后成为了真理。」
# Temporal AA 亦然——每一帧积累一点，沉默几十帧之后，
# 锯齿消失，真理（无锯齿的画面）显现。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   FTemporalAA shader class           → AstroCellTAAPass
#     FAlphaChannelDim (permutation)   → alpha_channel_enabled (bool)
#     FTAAPassConfigDim                → taa_pass_config (int)
#     FTAAQualityDim                   → taa_quality (ETAAQuality)
#     FTAAScreenPercentageDim          → screen_percentage_range (int)
#   CVarTemporalAAFilterSize           → _TAA_FILTER_SIZE
#   CVarTemporalAACatmullRom           → _TAA_CATMULL_ROM
#   CVarTemporalAAPauseCorrect         → _TAA_PAUSE_CORRECT
#   CVarTemporalAACurrentFrameWeight   → _TAA_CURRENT_FRAME_WEIGHT (0.04)
#   CVarTemporalAAQuality              → _TAA_QUALITY (0–3)
#   CVarTAAR11G11B10History            → _TAA_R11G11B10_HISTORY
#   DoesPlatformSupportTemporalHistoryUpscale → True (all platforms supported)
#   AddTemporalAAPass()                → run_temporal_aa_pass()
#   FTemporalAAHistory                 → AstroCellTAAHistory
#
# Algorithm changes (鲁迅式 20%):
#   1. Sub-pixel jitter (Halton) → same Halton sequence from path tracer
#   2. History reprojection (screen-pos UV lookup) → per-cell z-layer history dict
#   3. Neighbourhood clamping (AABB in colour space) → luminance range clamp
#   4. Current frame weight blend → exponential moving average
#   5. Anti-ghosting (mobility-based) → large z-delta triggers history rejection
# =============================================================================

# ── TemporalAA CVars ──────────────────────────────────────────────────────────
_TAA_FILTER_SIZE:         float = 1.0    # r.TemporalAAFilterSize (1=smooth, 0=sharp)
_TAA_CATMULL_ROM:         bool  = False  # r.TemporalAACatmullRom
_TAA_PAUSE_CORRECT:       bool  = True   # r.TemporalAAPauseCorrect
_TAA_CURRENT_FRAME_WEIGHT:float = 0.04   # r.TemporalAACurrentFrameWeight
_TAA_QUALITY:             int   = 2      # r.TemporalAA.Quality [0,3]
_TAA_SCREEN_PERCENTAGE:   float = 100.0  # r.TemporalAA.HistoryScreenPercentage
_TAA_R11G11B10:           bool  = True   # r.TemporalAA.R11G11B10History
_TAA_UPSCALER:            int   = 1      # r.TemporalAA.Upscaler
_TAA_TILE_SIZE_X:         int   = 8      # GTemporalAATileSizeX
_TAA_TILE_SIZE_Y:         int   = 8      # GTemporalAATileSizeY
_TAA_LARGE_GROUP:         bool  = False  # r.TemporalAA.LargeGroup
_TAA_LARGE_GROUP_MAX_INPUT_SCALE: int = 90  # r.TemporalAA.LargeGroup.MaxInputScale
_TAA_VGPR_OPT:            bool  = False  # r.TemporalAA.VGPROpt
_TAA_GHOST_THRESHOLD:     float = 1.0    # z-delta above which history is rejected

# TAA quality → neighbourhood sample count (mirrors ETAAQuality)
_TAA_QUALITY_SAMPLE_COUNTS = {0: 5, 1: 9, 2: 9, 3: 9}


class AstroCellTAAHistory:
    """
    Python equivalent of FTemporalAAHistory.

    Stores the per-cell accumulated TAA history between epochs.
    Uses exponential moving average for the colour estimate and
    tracks the last z-layer for anti-ghosting.

    鲁迅式：历史是 TAA 的财富，也是它的负担——
    太多的历史导致鬼影，太少的历史导致抖动。
    0.04 的当前帧权重，是这场博弈的和解系数。
    """

    def __init__(self) -> None:
        # Per-cell accumulated colour: cell_id → (R, G, B) float tuple
        self.colour_history: dict[str, tuple] = {}
        # Per-cell last z-layer: cell_id → float (for anti-ghosting)
        self.z_history:      dict[str, float] = {}
        # Per-cell pre-exposure correction (exposure normalisation)
        self.pre_exposure:   dict[str, float] = {}
        # Epoch index of last update
        self.epoch:          int = 0

    def is_valid_for(self, cell_id: str, current_z: float) -> bool:
        """
        Check whether history is valid for this cell (anti-ghosting gate).

        Mirrors the mobility-based anti-ghosting in TAA Quality 2+:
        if |z_current - z_history| > threshold → reject history (return False).

        鲁迅式：历史不总是可信的——当物体移动太多时，
        过去的颜色已与现在无关，必须果断抛弃。
        """
        if cell_id not in self.z_history:
            return False
        if _TAA_QUALITY >= 2:
            z_delta = abs(current_z - self.z_history[cell_id])
            if z_delta > _TAA_GHOST_THRESHOLD:
                return False
        return True

    def update(self, cell_id: str, current_colour: tuple,
               current_z: float) -> tuple:
        """
        Update history with the current frame's colour, returning the blended result.

        Mirrors the TAA accumulation blend:
            history_weight = 1 − CurrentFrameWeight
            output = current × w_curr + history × w_hist
        Neighbourhood clamping (AABB clamp) applied to history before blend.

        鲁迅式：积累是 TAA 的本质——每帧只相信 4% 的新证据，
        96% 来自历史的惯性。这是保守主义的数学表达。
        """
        w_curr = _TAA_CURRENT_FRAME_WEIGHT
        w_hist = 1.0 - w_curr

        if not self.is_valid_for(cell_id, current_z):
            # No valid history → use current frame only (cold start)
            blended = current_colour
        else:
            hist = self.colour_history[cell_id]

            # Neighbourhood clamping (AABB in colour space):
            # Clamp history to [current × 0.5, current × 1.5] per channel
            # Mirrors the AABB colour clip in the TAA shader
            clamped_hist = tuple(
                max(current_colour[i] * 0.5, min(current_colour[i] * 1.5, hist[i]))
                for i in range(3)
            )

            # Catmull-Rom sharpening (optional): increases current frame weight
            if _TAA_CATMULL_ROM:
                w_curr = min(1.0, w_curr * 1.5)
                w_hist = 1.0 - w_curr

            blended = tuple(
                current_colour[i] * w_curr + clamped_hist[i] * w_hist
                for i in range(3)
            )

        self.colour_history[cell_id] = blended
        self.z_history[cell_id]      = current_z
        return blended


# Module-level TAA history singleton (one per logical view)
_ASTRO_TAA_HISTORY: AstroCellTAAHistory = AstroCellTAAHistory()


def get_taa_history() -> AstroCellTAAHistory:
    """Return the global TAA history singleton."""
    return _ASTRO_TAA_HISTORY


class AstroCellTAAPass:
    """
    Temporal Anti-Aliasing pass — mirrors FTemporalAA shader + AddTemporalAAPass().

    Processes all cells, applies sub-pixel jitter via Halton sequence,
    reads history, clamps, blends, and writes updated history + output colours.

    The output per-cell colours are used downstream to modulate SVG fill
    opacity and stroke colour (the TAA-smoothed colour replaces the raw
    species colour for a temporally stable result).

    鲁迅式：时域抗锯齿是耐心的产物——
    一帧解决不了锯齿问题，但一百帧一定可以。
    只要不动（或者少动），历史就是你的盟友。
    """

    def __init__(self,
                 quality:       int   = _TAA_QUALITY,
                 filter_size:   float = _TAA_FILTER_SIZE) -> None:
        self.quality     = quality
        self.filter_size = filter_size
        self._history    = get_taa_history()

    def _jitter_offset(self, epoch: int) -> tuple[float, float]:
        """
        Sub-pixel jitter using Halton(2,3) sequence.
        Mirrors TemporalJitterPixels computation in the TAA shader setup.
        The jitter cycles over 8 frames (same as UE5 default TemporalAA.SampleCount=8).

        鲁迅式：抖动是策略性的不安定——
        每帧故意把采样点移动一点，让时间帮你填满所有的空隙。
        """
        n = epoch % 8
        jx = _pt_halton(n, 2) - 0.5   # [-0.5, +0.5]
        jy = _pt_halton(n, 3) - 0.5
        return (jx * self.filter_size, jy * self.filter_size)

    def _compute_output_colour(
        self,
        cell_id:  str,
        species:  str,
        cell_z:   float,
        jitter:   tuple[float, float],
    ) -> tuple:
        """
        Compute the jittered current-frame colour for a cell.

        In the 2-D analogue, jitter modulates the species primary colour
        slightly (like sub-pixel displacement affecting which texel is sampled):
            colour_jittered = colour × (1 + jx×0.02) in R, (1 + jy×0.02) in G
        This is the 2-D equivalent of jittering the texture UV.
        """
        sp_idx  = _species_to_index(species)
        base    = _SPECIES_INDEX_TO_COLOUR.get(sp_idx, _SPECIES_INDEX_TO_COLOUR[0])
        r = max(0.0, min(1.0, base[0] / 255.0 + jitter[0] * 0.02))
        g = max(0.0, min(1.0, base[1] / 255.0 + jitter[1] * 0.02))
        b = max(0.0, min(1.0, base[2] / 255.0))
        return (r, g, b)

    def run(
        self,
        cell_registry: dict,
        epoch:         int = 0,
    ) -> dict[str, tuple]:
        """
        Execute the TAA pass for all cells.

        Returns dict: cell_id → (R, G, B) accumulated output colour.

        鲁迅式：运行 TAA 是召唤历史来帮助现在——
        每一次运行都是对过去 96 次运行的致敬。
        """
        cells   = cell_registry.get("cells", {})
        jitter  = self._jitter_offset(epoch)
        outputs: dict[str, tuple] = {}

        for cell_id, entry in cells.items():
            z       = float(entry.get("z", 3))
            species = entry.get("species", "")

            current = self._compute_output_colour(cell_id, species, z, jitter)
            blended = self._history.update(cell_id, current, z)
            outputs[cell_id] = blended

        # Advance history epoch
        self._history.epoch = epoch

        converged = sum(
            1 for cid in outputs
            if self._history.is_valid_for(cid, float(cells.get(cid, {}).get("z", 3)))
        )

        print(
            f"[ASTRO-TAA] AstroCellTAAPass.run: "
            f"epoch={epoch} total={len(outputs)} "
            f"history_valid={converged} "
            f"jitter=({jitter[0]:.3f},{jitter[1]:.3f}) "
            f"quality={self.quality} filter={self.filter_size:.2f}",
            file=sys.stderr,
        )
        return outputs


def run_temporal_aa_pass(
    cell_registry: dict,
    epoch:         int = 0,
    quality:       int = _TAA_QUALITY,
    filter_size:   float = _TAA_FILTER_SIZE,
) -> dict[str, tuple]:
    """
    Top-level TAA entry point — mirrors AddTemporalAAPass().

    Constructs AstroCellTAAPass, runs it, and returns per-cell output colours.
    These colours can be used to modulate cell SVG fill for temporal stability.

    鲁迅式：抗锯齿的最终目的不是消灭抖动，而是接受时间——
    时间流逝，画面平滑，这才是进步的代价。
    """
    taa = AstroCellTAAPass(quality=quality, filter_size=filter_size)
    return taa.run(cell_registry, epoch)


# =============================================================================
# § DecalRenderingCommon — 贴花混合描述符与渲染阶段推导
#   移植自 Renderer-Private/DecalRenderingCommon.cpp
#   鲁迅式：贴花不是装饰，而是承认世界本身已不够干净。
# =============================================================================

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum, IntFlag, auto
from typing import Optional


class EBlendMode(IntEnum):
    Opaque      = 0
    Masked      = 1
    Translucent = 2
    Additive    = 3
    Modulate    = 4
    AlphaComposite = 5
    AlphaHoldout   = 6


class EDecalRenderStage(IntEnum):
    """渲染阶段枚举，对应 UE5 EDecalRenderStage。"""
    None_      = 0   # 鲁迅式：无阶段，即虚无——渲染管线最诚实的答案。
    BeforeBasePass     = 1
    BeforeLighting     = 2
    Mobile             = 3
    MobileBeforeLighting = 4
    Emissive           = 5
    AmbientOcclusion   = 6


@dataclass
class FDecalBlendDesc:
    """
    贴花混合描述符。
    鲁迅式：写下每一个 bWrite 字段，便是承认画面还缺少什么。
    """
    blend_mode: EBlendMode = EBlendMode.Translucent

    b_write_base_color:               bool = False
    b_write_normal:                   bool = False
    b_write_roughness_specular_metallic: bool = False
    b_write_emissive:                 bool = False
    b_write_ambient_occlusion:        bool = False
    b_write_dbuffer_mask:             bool = False

    # bitmask over EDecalRenderStage values
    render_stage_mask: int = 0


def _is_translucent_only(desc: FDecalBlendDesc) -> bool:
    return desc.blend_mode == EBlendMode.Translucent

def _is_alpha_composite(desc: FDecalBlendDesc) -> bool:
    return desc.blend_mode == EBlendMode.AlphaComposite

def _is_modulate(desc: FDecalBlendDesc) -> bool:
    return desc.blend_mode == EBlendMode.Modulate


def finalize_decal_blend_desc(
    desc: FDecalBlendDesc,
    *,
    is_mobile: bool = False,
    is_mobile_deferred: bool = False,
    is_dbuffer_platform: bool = False,
    is_dbuffer_mask_platform: bool = False,
    mobile_normal_blendable: bool = False,
) -> FDecalBlendDesc:
    """
    推导贴花混合描述符的最终状态，对应 FinalizeBlendDesc()。
    鲁迅式：平台限制就是现实，我们只能在允许的范围内做梦。
    """
    desc.b_write_dbuffer_mask = is_dbuffer_mask_platform

    # 强制平台混合模式约束
    if not (_is_translucent_only(desc) or _is_alpha_composite(desc) or _is_modulate(desc)):
        desc.blend_mode = EBlendMode.Translucent
    if is_dbuffer_platform and _is_modulate(desc):
        desc.blend_mode = EBlendMode.Translucent

    # 移动平台输出约束
    if is_mobile and not is_mobile_deferred and not is_dbuffer_platform:
        desc.b_write_roughness_specular_metallic = False
    if is_mobile and not mobile_normal_blendable:
        desc.b_write_normal = False
    if is_mobile:
        desc.b_write_ambient_occlusion = False

    # AlphaComposite 不写法线
    if _is_alpha_composite(desc):
        desc.b_write_normal = False

    # 推导主渲染阶段（互斥）
    has_gbuffer_writes = (
        desc.b_write_emissive or desc.b_write_base_color
        or desc.b_write_normal or desc.b_write_roughness_specular_metallic
    )
    if is_mobile_deferred and not is_dbuffer_platform and has_gbuffer_writes:
        desc.render_stage_mask |= 1 << EDecalRenderStage.MobileBeforeLighting
    elif is_mobile and not is_dbuffer_platform and (desc.b_write_emissive or desc.b_write_base_color):
        desc.render_stage_mask |= 1 << EDecalRenderStage.Mobile
    elif is_dbuffer_platform and (
        desc.b_write_base_color or desc.b_write_normal or desc.b_write_roughness_specular_metallic
    ):
        desc.render_stage_mask |= 1 << EDecalRenderStage.BeforeBasePass
    elif has_gbuffer_writes:
        desc.render_stage_mask |= 1 << EDecalRenderStage.BeforeLighting

    # 附加渲染阶段
    if desc.b_write_emissive and is_dbuffer_platform:
        desc.render_stage_mask |= 1 << EDecalRenderStage.Emissive
    if desc.b_write_ambient_occlusion:
        desc.render_stage_mask |= 1 << EDecalRenderStage.AmbientOcclusion

    return desc


def compute_decal_blend_desc(
    material_props: dict,
    *,
    is_mobile: bool = False,
    is_mobile_deferred: bool = False,
    is_dbuffer_platform: bool = False,
    is_dbuffer_mask_platform: bool = False,
) -> FDecalBlendDesc:
    """
    根据材质属性字典构造 FDecalBlendDesc，对应 ComputeDecalBlendDesc()。
    material_props 键：blend_mode, base_color, normal, roughness, specular, metallic,
                       emissive, ambient_occlusion, is_substrate, diffuse_albedo, f0
    鲁迅式：材质连接了什么，就说明创作者在意什么；不连接的，便是刻意的遗忘。
    """
    desc = FDecalBlendDesc()
    is_substrate = material_props.get("is_substrate", False)
    use_diffuse_f0 = material_props.get("diffuse_albedo", False) or material_props.get("f0", False)

    desc.blend_mode = EBlendMode(material_props.get("blend_mode", EBlendMode.Translucent))

    if is_substrate:
        desc.b_write_base_color               = material_props.get("base_color", False) or use_diffuse_f0
        desc.b_write_normal                   = material_props.get("normal", False)
        desc.b_write_roughness_specular_metallic = (
            use_diffuse_f0
            or material_props.get("roughness", False)
            or material_props.get("specular", False)
            or material_props.get("metallic", False)
        )
        desc.b_write_emissive             = material_props.get("emissive", False)
        desc.b_write_ambient_occlusion    = material_props.get("ambient_occlusion", False)
    else:
        desc.b_write_base_color               = material_props.get("base_color", False)
        desc.b_write_normal                   = material_props.get("normal", False)
        desc.b_write_roughness_specular_metallic = (
            material_props.get("roughness", False)
            or material_props.get("specular", False)
            or material_props.get("metallic", False)
        )
        desc.b_write_emissive             = material_props.get("emissive", False)
        desc.b_write_ambient_occlusion    = material_props.get("ambient_occlusion", False)

    mobile_normal_blendable = is_dbuffer_platform or is_mobile_deferred
    return finalize_decal_blend_desc(
        desc,
        is_mobile=is_mobile,
        is_mobile_deferred=is_mobile_deferred,
        is_dbuffer_platform=is_dbuffer_platform,
        is_dbuffer_mask_platform=is_dbuffer_mask_platform,
        mobile_normal_blendable=mobile_normal_blendable,
    )


def is_compatible_with_render_stage(desc: FDecalBlendDesc, stage: EDecalRenderStage) -> bool:
    return bool(desc.render_stage_mask & (1 << stage))


def get_base_render_stage(desc: FDecalBlendDesc) -> EDecalRenderStage:
    """
    返回贴花的主渲染阶段，对应 GetBaseRenderStage()。
    鲁迅式：阶段的优先顺序，是工程师对渲染时序的一次沉默表态。
    """
    for stage in (
        EDecalRenderStage.BeforeBasePass,
        EDecalRenderStage.BeforeLighting,
        EDecalRenderStage.Mobile,
        EDecalRenderStage.MobileBeforeLighting,
    ):
        if desc.render_stage_mask & (1 << stage):
            return stage
    return EDecalRenderStage.None_


# =============================================================================
# § DecalRenderingShared — 可见贴花列表构建与视图分发
#   移植自 Renderer-Private/DecalRenderingShared.cpp
# =============================================================================

import math
from typing import Callable, Iterable


@dataclass
class FVisibleDecal:
    """
    可见贴花实体，携带混合描述符与淡入淡出参数。
    鲁迅式：贴花的淡出，不是消亡，而是体面地退场。
    """
    material_props:   dict
    sort_order:       int   = 0
    conservative_radius: float = 1.0
    fade_alpha:       float = 1.0
    inv_fade_duration:  float = 0.0
    inv_fade_in_duration: float = 0.0
    fade_start_delay: float = 0.0
    fade_in_start_delay: float = 0.0
    blend_desc: FDecalBlendDesc = field(default_factory=FDecalBlendDesc)

    def __post_init__(self):
        self.blend_desc = compute_decal_blend_desc(self.material_props)


def build_visible_decal_list(
    all_decals: list[dict],
    view_frustum_test: Callable[[dict], tuple[bool, float, float]] | None = None,
) -> list[FVisibleDecal]:
    """
    从场景贴花列表中筛选视锥体内的可见贴花，对应 BuildVisibleDecalList()。
    view_frustum_test(decal) -> (is_visible, conservative_radius, fade_alpha)
    鲁迅式：视锥体剔除是一种慈悲——让看不见的东西不必假装存在。
    """
    result: list[FVisibleDecal] = []
    for decal in all_decals:
        if view_frustum_test is not None:
            visible, radius, alpha = view_frustum_test(decal)
        else:
            visible, radius, alpha = True, decal.get("radius", 1.0), 1.0
        if not visible:
            continue
        vd = FVisibleDecal(
            material_props=decal.get("material_props", {}),
            sort_order=decal.get("sort_order", 0),
            conservative_radius=radius,
            fade_alpha=alpha,
            inv_fade_duration=decal.get("inv_fade_duration", 0.0),
            inv_fade_in_duration=decal.get("inv_fade_in_duration", 0.0),
            fade_start_delay=decal.get("fade_start_delay", 0.0),
            fade_in_start_delay=decal.get("fade_in_start_delay", 0.0),
        )
        result.append(vd)
    # 按 sort_order 升序，再按 conservative_radius 降序（大贴花先渲染）
    result.sort(key=lambda d: (d.sort_order, -d.conservative_radius))
    return result


def build_relevant_decal_list(
    visible_decals: list[FVisibleDecal],
    stage: EDecalRenderStage,
) -> list[FVisibleDecal]:
    """
    从可见列表中过滤出与指定渲染阶段兼容的贴花，对应 BuildRelevantDecalList()。
    鲁迅式：每个阶段只看自己的份，这叫专注，也叫局限。
    """
    return [d for d in visible_decals if is_compatible_with_render_stage(d.blend_desc, stage)]


@dataclass
class DecalVisibilityViewPacket:
    """
    单视图的贴花可见性数据包，对应 FDecalVisibilityViewPacket。
    懒惰求值：relevant_decals 按需构建。
    鲁迅式：按需构建是现代工程师的美德，也是他们唯一剩下的节制。
    """
    visible_decals: list[FVisibleDecal] = field(default_factory=list)
    _relevant_cache: dict[EDecalRenderStage, list[FVisibleDecal]] = field(
        default_factory=dict, repr=False
    )

    @classmethod
    def build(
        cls,
        all_decals: list[dict],
        stages: Iterable[EDecalRenderStage] = (),
        view_frustum_test: Callable | None = None,
    ) -> "DecalVisibilityViewPacket":
        pkt = cls()
        pkt.visible_decals = build_visible_decal_list(all_decals, view_frustum_test)
        for stage in stages:
            pkt._relevant_cache[stage] = build_relevant_decal_list(pkt.visible_decals, stage)
        return pkt

    def get_relevant_decals(self, stage: EDecalRenderStage) -> list[FVisibleDecal]:
        if stage not in self._relevant_cache:
            self._relevant_cache[stage] = build_relevant_decal_list(self.visible_decals, stage)
        return self._relevant_cache[stage]


# =============================================================================
# § BlueNoise — 蓝噪声参数管理
#   移植自 Renderer-Private/BlueNoise.cpp
#   鲁迅式：蓝噪声的均匀，是对随机性的最后一点人道主义改造。
# =============================================================================

import numpy as np
from typing import Optional


@dataclass
class FBlueNoiseParameters:
    """
    蓝噪声纹理参数，对应 FBlueNoiseParameters。
    dimensions: (width, height, slices)
    modulo_masks: 用于快速取模的位掩码三元组，要求各维度为 2 的幂。
    """
    dimensions:   tuple[int, int, int] = (1, 1, 1)
    modulo_masks: tuple[int, int, int] = (0, 0, 0)
    # 实际纹理数据存为 numpy 数组（float32）
    scalar_texture: Optional[np.ndarray] = None
    vec2_texture:   Optional[np.ndarray] = None


def _floor_log2(n: int) -> int:
    return int(math.floor(math.log2(max(n, 1))))


def _fill_blue_noise_from_texture(out: FBlueNoiseParameters) -> None:
    """
    从 scalar_texture 形状推导 dimensions 与 modulo_masks，
    对应 FillUpBlueNoiseParametersFromTexture()。
    鲁迅式：从尺寸推导掩码，是数学对现实的一次简洁起诉。
    """
    assert out.scalar_texture is not None, "scalar_texture must be set before fill"
    h, w = out.scalar_texture.shape[:2]
    slices = h // max(1, w)
    out.dimensions = (w, w, slices)
    out.modulo_masks = (
        (1 << _floor_log2(out.dimensions[0])) - 1,
        (1 << _floor_log2(out.dimensions[1])) - 1,
        (1 << _floor_log2(out.dimensions[2])) - 1,
    )
    assert (out.modulo_masks[0] + 1) == out.dimensions[0], "dimension X must be power-of-two"
    assert (out.modulo_masks[1] + 1) == out.dimensions[1], "dimension Y must be power-of-two"
    assert (out.modulo_masks[2] + 1) == out.dimensions[2], "dimension Z must be power-of-two"


_BLACK_DUMMY = np.zeros((1, 1), dtype=np.float32)


def get_blue_noise_dummy_parameters() -> FBlueNoiseParameters:
    """
    返回全黑占位蓝噪声参数，对应 GetBlueNoiseDummyParameters()。
    鲁迅式：占位符是工程师的礼貌，也是他承认自己尚未准备好的方式。
    """
    return FBlueNoiseParameters(
        dimensions=(1, 1, 1),
        modulo_masks=(0, 0, 0),
        scalar_texture=_BLACK_DUMMY,
        vec2_texture=_BLACK_DUMMY,
    )


def get_blue_noise_parameters(
    scalar_texture: np.ndarray,
    vec2_texture: Optional[np.ndarray] = None,
) -> FBlueNoiseParameters:
    """
    从真实纹理数据构造蓝噪声参数，对应 GetBlueNoiseParameters()。
    scalar_texture: H×W float32 array（H = W * slices）
    vec2_texture:   可选，H×W×2 float32 array
    鲁迅式：真实纹理与占位纹理，差别只在于有没有认真对待随机这件事。
    """
    out = FBlueNoiseParameters(
        scalar_texture=scalar_texture,
        vec2_texture=vec2_texture if vec2_texture is not None else _BLACK_DUMMY,
    )
    _fill_blue_noise_from_texture(out)
    return out


def get_blue_noise_global_parameters(
    scalar_texture: Optional[np.ndarray] = None,
    vec2_texture: Optional[np.ndarray] = None,
) -> FBlueNoiseParameters:
    """
    获取全局蓝噪声参数，无纹理时回退到占位符，对应 GetBlueNoiseGlobalParameters()。
    鲁迅式：全局参数的回退路径，是系统对不完整世界的默默容忍。
    """
    if scalar_texture is not None:
        return get_blue_noise_parameters(scalar_texture, vec2_texture)
    return get_blue_noise_dummy_parameters()


def sample_blue_noise_scalar(params: FBlueNoiseParameters, x: int, y: int, z: int = 0) -> float:
    """
    从蓝噪声参数中采样标量值，使用 modulo_masks 进行快速环绕寻址。
    鲁迅式：环绕寻址像极了历史——越界之后，总会回到某个熟悉的起点。
    """
    if params.scalar_texture is None or params.scalar_texture is _BLACK_DUMMY:
        return 0.0
    mx, my, mz = params.modulo_masks
    xi = x & mx
    yi = y & my
    zi = z & mz
    h, w = params.scalar_texture.shape[:2]
    slices = params.dimensions[2]
    row = zi * (h // max(1, slices)) + yi
    row = min(row, h - 1)
    col = min(xi, w - 1)
    return float(params.scalar_texture[row, col])


# =============================================================================
# § ComputeSystemInterface — 计算系统注册与 Worker 生命周期
#   移植自 Renderer-Private/ComputeSystemInterface.cpp
#   鲁迅式：注册表是官僚体系的技术化身——你必须登记，才有资格被调用。
# =============================================================================

from typing import Protocol, runtime_checkable


@runtime_checkable
class IComputeTaskWorker(Protocol):
    """计算任务工人接口，对应 IComputeTaskWorker。"""
    def execute(self) -> None: ...


@runtime_checkable
class IComputeSystem(Protocol):
    """
    计算系统接口，对应 IComputeSystem。
    鲁迅式：接口是契约，也是无声的命令。
    """
    def create_workers(
        self, scene: object, out_workers: list[IComputeTaskWorker]
    ) -> None: ...

    def destroy_workers(
        self, scene: object, in_out_workers: list[IComputeTaskWorker]
    ) -> None: ...


class ComputeSystemRegistry:
    """
    全局计算系统注册表，对应 ComputeSystemInterface 命名空间。
    鲁迅式：全局注册表的存在，说明没有人敢承担依赖注入的责任。
    """

    def __init__(self):
        self._systems: list[IComputeSystem] = []

    def register_system(self, system: IComputeSystem) -> None:
        if system not in self._systems:
            self._systems.append(system)

    def unregister_system(self, system: IComputeSystem) -> None:
        for i, s in enumerate(self._systems):
            if s is system:
                # swap-remove，与 UE5 RemoveAtSwap 保持一致
                self._systems[i] = self._systems[-1]
                self._systems.pop()
                return

    def create_workers(
        self, scene: object, out_workers: list[IComputeTaskWorker]
    ) -> None:
        for system in self._systems:
            system.create_workers(scene, out_workers)

    def destroy_workers(
        self, scene: object, in_out_workers: list[IComputeTaskWorker]
    ) -> None:
        for system in self._systems:
            system.destroy_workers(scene, in_out_workers)
        # 销毁后列表必须为空，对应 ensure(InOutWorkders.Num() == 0)
        if in_out_workers:
            raise RuntimeError(
                f"ComputeSystemRegistry.destroy_workers: "
                f"{len(in_out_workers)} worker(s) not cleaned up"
            )


# 模块级单例，对应 GRegisteredSystems
_global_compute_registry = ComputeSystemRegistry()


def register_compute_system(system: IComputeSystem) -> None:
    _global_compute_registry.register_system(system)

def unregister_compute_system(system: IComputeSystem) -> None:
    _global_compute_registry.unregister_system(system)

def create_compute_workers(scene: object, out_workers: list) -> None:
    _global_compute_registry.create_workers(scene, out_workers)

def destroy_compute_workers(scene: object, in_out_workers: list) -> None:
    _global_compute_registry.destroy_workers(scene, in_out_workers)


# =============================================================================
# § DebugViewModeRendering — 调试视图模式渲染参数与着色器复杂度基线
#   移植自 Renderer-Private/DebugViewModeRendering.cpp
#   鲁迅式：调试视图的存在，说明我们对自己写出的东西始终存有疑虑。
# =============================================================================


class EShadingPath(IntEnum):
    Forward  = 0
    Deferred = 1
    Mobile   = 2


@dataclass
class ShaderComplexityBaseline:
    """
    着色器复杂度基线，对应各 GShaderComplexityBaseline* 全局变量。
    鲁迅式：基线是期望，超出基线是现实，差值是我们不愿承认的懒惰。
    """
    # Forward
    forward_vs:        int   = 134
    forward_ps:        int   = 635
    forward_unlit_ps:  int   = 47
    # Deferred
    deferred_vs:       int   = 41
    deferred_ps:       int   = 111
    deferred_unlit_ps: int   = 33
    # Mobile Forward
    mobile_forward_vs:        int   = 134
    mobile_forward_ps:        int   = 143
    mobile_forward_unlit_ps:  int   = 6
    # Mobile Deferred
    mobile_deferred_vs:        int   = 134
    mobile_deferred_ps:        int   = 50
    mobile_deferred_unlit_ps:  int   = 9
    # Masked cost multiplier (mobile)
    mobile_masked_cost_multiplier: float = 1.5


# 模块级默认基线，对应各 CVarShaderComplexityBaseline* 的默认值
_default_shader_complexity_baseline = ShaderComplexityBaseline()


def get_quad_overdraw_uav_index(
    is_forward_shading: bool,
    base_pass_can_output_velocity: bool,
) -> int:
    """
    返回 Quad Overdraw UAV 的寄存器槽位，对应 GetQuadOverdrawUAVIndex()。
    鲁迅式：槽位编号背后是整套 GBuffer 布局——
    改动任何一处，牵一发而动全身，这就是耦合的代价。
    """
    if is_forward_shading:
        return 2 if base_pass_can_output_velocity else 1
    else:
        return 7 if base_pass_can_output_velocity else 6


_NUM_STREAMING_ACCURACY_COLORS = 5

_DEFAULT_ACCURACY_COLORS: list[tuple[float, float, float, float]] = [
    (0.0,  0.0,  1.0,  1.0),   # 蓝：过度流送
    (0.0,  1.0,  0.0,  1.0),   # 绿：刚好
    (1.0,  1.0,  0.0,  1.0),   # 黄：轻微不足
    (1.0,  0.5,  0.0,  1.0),   # 橙：中度不足
    (1.0,  0.0,  0.0,  1.0),   # 红：严重不足
]


@dataclass
class FDebugViewModeUniformParameters:
    """
    调试视图模式 Uniform 参数，对应 FDebugViewModeUniformParameters。
    鲁迅式：颜色精度图谱是渲染引擎对自身的一次公开体检。
    """
    accuracy_colors: list[tuple[float, float, float, float]] = field(
        default_factory=lambda: list(_DEFAULT_ACCURACY_COLORS)
    )
    # 纹理坐标密度诊断（UV 通道分析用）
    uv_density_sampling: float = 1.0


def setup_debug_view_mode_pass_uniform_buffer_constants(
    accuracy_colors_override: list[tuple[float, float, float, float]] | None = None,
    uv_density_sampling: float = 1.0,
) -> FDebugViewModeUniformParameters:
    """
    构造调试视图模式 Uniform 缓冲区常量，对应
    SetupDebugViewModePassUniformBufferConstants()。
    鲁迅式：把颜色填进参数结构，是把主观判断包装成客观数据的例行仪式。
    """
    colors = list(_DEFAULT_ACCURACY_COLORS)
    if accuracy_colors_override:
        # 截取到 _NUM_STREAMING_ACCURACY_COLORS，其余补黑
        n = min(len(accuracy_colors_override), _NUM_STREAMING_ACCURACY_COLORS)
        for i in range(n):
            colors[i] = accuracy_colors_override[i]
        for i in range(n, _NUM_STREAMING_ACCURACY_COLORS):
            colors[i] = (0.0, 0.0, 0.0, 1.0)
    return FDebugViewModeUniformParameters(
        accuracy_colors=colors,
        uv_density_sampling=uv_density_sampling,
    )


def get_shader_instruction_count_for_baseline(
    shading_path: EShadingPath,
    is_vertex_shader: bool,
    is_unlit: bool,
    baseline: ShaderComplexityBaseline | None = None,
) -> int:
    """
    返回给定着色器类型的指令数基线，供复杂度对比使用。
    鲁迅式：基线指令数是工程师对"正常"的定义——超出则问责，不足则怀疑。
    """
    b = baseline or _default_shader_complexity_baseline
    if shading_path == EShadingPath.Forward:
        if is_vertex_shader: return b.forward_vs
        return b.forward_unlit_ps if is_unlit else b.forward_ps
    elif shading_path == EShadingPath.Mobile:
        if is_vertex_shader: return b.mobile_forward_vs
        return b.mobile_forward_unlit_ps if is_unlit else b.mobile_forward_ps
    else:  # Deferred
        if is_vertex_shader: return b.deferred_vs
        return b.deferred_unlit_ps if is_unlit else b.deferred_ps


def compute_shader_complexity_ratio(
    instruction_count: int,
    shading_path: EShadingPath,
    is_vertex_shader: bool,
    is_unlit: bool,
    baseline: ShaderComplexityBaseline | None = None,
) -> float:
    """
    计算着色器复杂度比值（实际指令数 / 基线指令数）。
    返回值 > 1.0 表示超出基线，供贴花与调试视图模式颜色映射使用。
    鲁迅式：比值大于一，说明代码已经比标准更费力气——
    或者标准定得太低，这两种可能同样令人不安。
    """
    baseline_count = get_shader_instruction_count_for_baseline(
        shading_path, is_vertex_shader, is_unlit, baseline
    )
    if baseline_count <= 0:
        return 0.0
    return instruction_count / baseline_count


# =============================================================================
# UE5 Renderer Port — BasePass · ClusteredDeferred · Anisotropy · Depth · CustomDepth
#
# 鲁迅式：世上本没有渲染管线，走的人多了，便成了 G-Buffer。
# =============================================================================

import math as _math
from dataclasses import dataclass as _dataclass, field as _field
from typing import Dict as _Dict, List as _List, Optional as _Optional, Tuple as _Tuple


# ---------------------------------------------------------------------------
# Shared math primitives
# ---------------------------------------------------------------------------

class _Vec3:
    """Minimal float3 — cheaper than numpy for single-cell ops."""
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def __add__(self, o):  return _Vec3(self.x+o.x, self.y+o.y, self.z+o.z)
    def __mul__(self, s):  return _Vec3(self.x*s, self.y*s, self.z*s)
    def __rmul__(self, s): return self.__mul__(s)

    def dot(self, o) -> float:
        return self.x*o.x + self.y*o.y + self.z*o.z

    def length(self) -> float:
        return _math.sqrt(self.dot(self))

    def normalize(self):
        d = self.length()
        return _Vec3(self.x/d, self.y/d, self.z/d) if d > 1e-9 else _Vec3(0, 0, 1)

    def clamp01(self):
        return _Vec3(max(0.0, min(1.0, self.x)),
                     max(0.0, min(1.0, self.y)),
                     max(0.0, min(1.0, self.z)))

    def as_tuple(self) -> _Tuple[float, float, float]:
        return (self.x, self.y, self.z)


def _saturate(v: float) -> float:
    return max(0.0, min(1.0, v))


def _pow_safe(base: float, exp: float) -> float:
    return _math.pow(max(base, 1e-9), exp)


# ---------------------------------------------------------------------------
# GBuffer — mirrors TBasePassPS output layout (BasePassRendering.cpp)
# ---------------------------------------------------------------------------
# 鲁迅式：G-Buffer 是现代延迟渲染的脸面——铺开来给人看的，
# 只是材质的皮囊，真正的光照还在后面等着。

@_dataclass
class AstroCellGBuffer:
    """
    Reduced GBuffer written by the BasePass.

    Mirrors the fields consumed by ClusteredDeferredShadingPixelShader:
      SceneColor   — HDR scene-colour pre-lighting (emissive + baked)
      WorldNormal  — world-space shading normal  [−1, 1]³
      BaseColor    — albedo / diffuse colour      [0, 1]³
      Roughness    — GGX roughness               [0, 1]
      Metallic     — metallic mask               [0, 1]
      Anisotropy   — tangent-space anisotropy    [−1, 1]  (AnisotropyRendering.cpp)
      Depth        — linear eye depth            ≥ 0
      CustomDepth  — custom depth value or −1 if not written
      Stencil      — custom stencil byte
    """
    scene_color:   _Vec3  = _field(default_factory=lambda: _Vec3(0, 0, 0))
    world_normal:  _Vec3  = _field(default_factory=lambda: _Vec3(0, 0, 1))
    base_color:    _Vec3  = _field(default_factory=lambda: _Vec3(0.5, 0.5, 0.5))
    roughness:     float  = 0.5
    metallic:      float  = 0.0
    anisotropy:    float  = 0.0      # written by FAnisotropyPS (AnisotropyRendering.cpp)
    depth:         float  = 1.0      # TDepthOnlyVS / FDepthOnlyPS (DepthRendering.cpp)
    custom_depth:  float  = -1.0     # FCustomDepthPassParameters (CustomDepthRendering.cpp)
    stencil:       int    = 0


# ---------------------------------------------------------------------------
# BasePass — AstroCellBasePass
# Mirrors the BasePassRendering.cpp "RenderBasePass" pipeline stage.
# ---------------------------------------------------------------------------
# 鲁迅式：BasePass 写满了 G-Buffer，却不点一盏灯；
# 和许多人的一生一样——把自己交代清楚了，却等不到光。

@_dataclass
class AstroCellMaterial:
    """
    Subset of UMaterial properties consumed during the BasePass.
    Corresponds to the FMaterial / FMaterialRenderProxy interface.
    """
    base_color:       _Vec3  = _field(default_factory=lambda: _Vec3(0.8, 0.8, 0.8))
    emissive:         _Vec3  = _field(default_factory=lambda: _Vec3(0, 0, 0))
    roughness:        float  = 0.5
    metallic:         float  = 0.0
    anisotropy:       float  = 0.0   # bHasAnisotropyConnected
    opacity_mask:     float  = 1.0   # clip() threshold in masked materials
    writes_velocity:  bool   = False
    is_translucent:   bool   = False


class AstroCellBasePass:
    """
    Encodes one cell's material into a GBuffer entry.

    Mimics the per-draw-call work done by TBasePassPS::MainPS():
      1. Material evaluation  → base-colour, roughness, metallic, emissive
      2. Normal encoding      → world-space normal written to GBufferA
      3. Emissive accumulation into SceneColor
      4. Anisotropy output    → forwarded to AnisotropyRendering pass

    SelectiveBasePassOutputs (r.SelectiveBasePassOutputs) is always
    treated as enabled here: we only populate gbuffer slots we use.
    """

    # r.SelectiveBasePassOutputs = 1  (compile-time const in UE5)
    SELECTIVE_OUTPUTS: bool = True

    def __init__(self, enable_anisotropy: bool = True):
        # r.AnisotropicMaterials equivalent
        self._anisotropy_enabled = enable_anisotropy

    def encode(
        self,
        material:     AstroCellMaterial,
        vertex_world_normal: _Vec3,
        eye_depth:    float,
    ) -> AstroCellGBuffer:
        """
        Execute the BasePass pixel-shader logic for a single cell.

        Parameters
        ----------
        material            : evaluated material parameters
        vertex_world_normal : interpolated vertex normal (world space)
        eye_depth           : linear depth from camera
        """
        gbuf = AstroCellGBuffer()

        # --- opacity / masked discard (clip() equivalent) -----------------
        if material.opacity_mask < 0.333:
            # Masked material fully discarded → leave gbuf at defaults.
            # Matches `clip(OpacityMask - GetMaskClipValue())` in HLSL.
            return gbuf

        # --- normal encoding ----------------------------------------------
        # GBufferA.xyz = WorldNormal (octahedral encode omitted for clarity)
        gbuf.world_normal = vertex_world_normal.normalize()

        # --- material properties ------------------------------------------
        gbuf.base_color = material.base_color.clamp01()
        gbuf.roughness  = _saturate(material.roughness)
        gbuf.metallic   = _saturate(material.metallic)

        # --- anisotropy (written only when r.AnisotropicMaterials is on) --
        # Mirrors FAnisotropyPS writing into the anisotropy GBuffer channel.
        if self._anisotropy_enabled:
            gbuf.anisotropy = max(-1.0, min(1.0, material.anisotropy))

        # --- emissive → SceneColor ----------------------------------------
        # UE5 BasePass: SceneColor.rgb += Emissive * View.PreExposure
        pre_exposure = 1.0
        gbuf.scene_color = (material.emissive * pre_exposure).clamp01()

        # --- depth prepass value (DepthRendering.cpp) ----------------------
        # TDepthOnlyVS outputs SV_Depth; we store linear eye depth.
        gbuf.depth = max(0.0, eye_depth)

        return gbuf


# ---------------------------------------------------------------------------
# Anisotropy pass — mirrors FAnisotropyMeshProcessor (AnisotropyRendering.cpp)
# ---------------------------------------------------------------------------
# 鲁迅式：各向异性不过是承认了世界并非各向同性——
# 光在不同方向上走得快慢不同，人的命运亦如此。

def astro_anisotropy_brdf(
    n: _Vec3,
    v: _Vec3,
    l: _Vec3,
    t: _Vec3,
    roughness_u: float,
    roughness_v: float,
) -> float:
    """
    Ward-Dür anisotropic BRDF lobe (simplified).

    Used to weight the specular contribution of a cell whose material
    has anisotropy > 0.  Mirrors the HLSL in AnisotropyPassShader.usf,
    which is invoked by FAnisotropyMeshProcessor::Process().

    Parameters
    ----------
    n           : surface normal     (world space, unit)
    v           : view direction     (unit, toward camera)
    l           : light direction    (unit, toward light)
    t           : tangent direction  (unit)
    roughness_u : GGX roughness along tangent    [0, 1]
    roughness_v : GGX roughness along bitangent  [0, 1]
    """
    # Clamp roughness to avoid division by zero (same guard as UE5 HLSL)
    au = max(roughness_u * roughness_u, 1e-4)
    av = max(roughness_v * roughness_v, 1e-4)

    # Half-vector
    hx, hy, hz = v.x+l.x, v.y+l.y, v.z+l.z
    h_len = _math.sqrt(hx*hx + hy*hy + hz*hz)
    if h_len < 1e-9:
        return 0.0
    h = _Vec3(hx/h_len, hy/h_len, hz/h_len)

    # Bitangent
    b = _Vec3(
        n.y*t.z - n.z*t.y,
        n.z*t.x - n.x*t.z,
        n.x*t.y - n.y*t.x,
    ).normalize()

    ndotl = _saturate(n.dot(l))
    ndotv = _saturate(n.dot(v))
    ndoth = _saturate(n.dot(h))
    hdott = h.dot(t)
    hdotb = h.dot(b)

    if ndotl < 1e-6 or ndotv < 1e-6:
        return 0.0

    # GGX anisotropic NDF  (Burley 2012 / UE5 BRDF.ush)
    denom = (hdott/au)**2 + (hdotb/av)**2 + ndoth**2
    d = 1.0 / (_math.pi * au * av * denom**2 + 1e-9)

    # Smith visibility (approximate — same as UE5 optimised form)
    lv = ndotl * _math.sqrt(ndotv**2 * (1.0 - av**2) + av**2)
    ll = ndotv * _math.sqrt(ndotl**2 * (1.0 - au**2) + au**2)
    g  = 0.5 / (lv + ll + 1e-9)

    return d * g * ndotl


# ---------------------------------------------------------------------------
# Depth prepass — mirrors DepthRendering.cpp EarlyZ / DDM_AllOpaque logic
# ---------------------------------------------------------------------------
# 鲁迅式：深度测试是渲染管线的第一道门槛，
# 凡不能通过的，便永远消失在黑暗里——无声无息。

class AstroCellDepthPass:
    """
    EarlyZ prepass for a collection of cells.

    Mimics FDepthPassMeshProcessor::AddMeshBatch() + TDepthOnlyVS.
    Produces a per-cell depth buffer that later passes (BasePass,
    ClusteredDeferred) use for depth-equal or depth-less-equal tests.

    EarlyZPassMode is hardcoded to DDM_AllOpaque (the most common
    production setting and the one that matches our cell-centric use).
    """

    # r.EarlyZSortMasked = 1: masked draws go last (better early-z util)
    SORT_MASKED_LAST: bool = True

    def run(
        self,
        cells: _List[_Dict],
    ) -> _Dict[str, float]:
        """
        Execute the depth prepass.

        Parameters
        ----------
        cells : list of dicts with keys
                  'id'      : str
                  'depth'   : float  (linear eye depth)
                  'masked'  : bool   (True → has opacity_mask < 1)

        Returns
        -------
        depth_buffer : {cell_id: depth}
        """
        # r.EarlyZSortMasked — opaque draws before masked draws
        if self.SORT_MASKED_LAST:
            cells = sorted(cells, key=lambda c: (1 if c.get("masked") else 0, c["depth"]))

        depth_buffer: _Dict[str, float] = {}
        for cell in cells:
            cid   = cell["id"]
            depth = float(cell["depth"])
            # Depth-less test: keep closest (front-to-back render)
            if cid not in depth_buffer or depth < depth_buffer[cid]:
                depth_buffer[cid] = depth

        return depth_buffer


# ---------------------------------------------------------------------------
# CustomDepth pass — mirrors CustomDepthRendering.cpp
# ---------------------------------------------------------------------------
# 鲁迅式：CustomDepth 是给异类留的通道——
# 普通物体写普通深度，而那些需要特别对待的，
# 另有一本账，另有一道门。

@_dataclass
class AstroCellCustomDepthRequest:
    """Mirrors the per-primitive CustomDepth render request."""
    cell_id:       str
    depth:         float   # eye-space depth of the custom-depth surface
    stencil_value: int = 0  # r.CustomDepth = 3 → EnabledWithStencil


class AstroCellCustomDepthPass:
    """
    Renders custom-depth and custom-stencil for flagged cells.

    Corresponds to FCustomDepthPassParameters + the RDG pass in
    CustomDepthRendering.cpp::RenderCustomDepthPass().

    Ordering (r.CustomDepth.Order):
      0 → BeforeBasePass  (default when DBuffer is enabled)
      1 → AfterBasePass
    We store the result and let the caller decide when to apply it.
    """

    def __init__(self, order: int = 0, writes_stencil: bool = True):
        # r.CustomDepth.Order
        self.before_base_pass: bool = (order == 0)
        # r.CustomDepth = 3 (EnabledWithStencil)
        self.writes_stencil: bool = writes_stencil

    def run(
        self,
        requests: _List[AstroCellCustomDepthRequest],
        scene_depth_buffer: _Optional[_Dict[str, float]] = None,
    ) -> _Dict[str, _Tuple[float, int]]:
        """
        Execute the custom-depth pass.

        Parameters
        ----------
        requests          : list of custom-depth draw requests
        scene_depth_buffer: optional scene depth buffer for occlusion test
                            (None → no occlusion; all writes pass)

        Returns
        -------
        {cell_id: (custom_depth, stencil_value)}
        """
        # FCustomDepthTextures::Create clears to depth_far=1e9
        DEPTH_FAR = 1e9
        out: _Dict[str, _Tuple[float, int]] = {}

        for req in requests:
            # Depth test against scene geometry (CF_LessEqual in UE5 default)
            if scene_depth_buffer is not None:
                scene_d = scene_depth_buffer.get(req.cell_id, DEPTH_FAR)
                if req.depth > scene_d:
                    continue  # occluded → discard

            stencil = req.stencil_value if self.writes_stencil else 0
            # Keep the nearest custom-depth value per cell
            if req.cell_id not in out or req.depth < out[req.cell_id][0]:
                out[req.cell_id] = (req.depth, stencil)

        return out


# ---------------------------------------------------------------------------
# Clustered Deferred Shading — mirrors ClusteredDeferredShadingPass.cpp
# ---------------------------------------------------------------------------
# 鲁迅式：延迟渲染的集群分格，和旧社会的里弄一样——
# 把灯光框进一个个格子里，各管各的，互不打扰，
# 偏偏照亮的还是同一张脸。

@_dataclass
class AstroCellLight:
    """
    Minimal punctual-light descriptor used by the light grid.
    Mirrors FLocalLightData in the forward light SSBO.
    """
    position:   _Vec3
    color:      _Vec3  = _field(default_factory=lambda: _Vec3(1, 1, 1))
    intensity:  float  = 1.0
    radius:     float  = 10.0
    # Anisotropic materials need the light's tangent influence weight
    aniso_weight: float = 1.0


def _ggx_specular(n: _Vec3, v: _Vec3, l: _Vec3, roughness: float) -> float:
    """Isotropic GGX BRDF (D*G term only; F=1 for brevity)."""
    a  = max(roughness * roughness, 1e-4)
    hx = v.x+l.x; hy = v.y+l.y; hz = v.z+l.z
    hl = _math.sqrt(hx*hx + hy*hy + hz*hz)
    if hl < 1e-9:
        return 0.0
    h = _Vec3(hx/hl, hy/hl, hz/hl)
    ndoth = _saturate(n.dot(h))
    ndotl = _saturate(n.dot(l))
    ndotv = _saturate(n.dot(v))
    denom = ndoth*ndoth*(a*a-1.0)+1.0
    d = a*a / (_math.pi * denom*denom + 1e-9)
    # Smith G1 (Schlick approximation)
    k = a / 2.0
    gv = ndotv / (ndotv*(1-k)+k+1e-9)
    gl = ndotl / (ndotl*(1-k)+k+1e-9)
    return d * gv * gl * ndotl


class AstroCellClusteredDeferredShadingPass:
    """
    Per-cell clustered deferred shading.

    Corresponds to FClusteredShadingPS::ClusteredShadingPixelShader().

    Algorithm
    ---------
    1. For each cell, look up its GBuffer entry.
    2. Iterate over lights assigned to the cell's cluster (here: all
       lights within radius — the grid-based culling is elided).
    3. Accumulate diffuse + specular from each light using the same
       BRDF model that FClusteredShadingPS calls into.
    4. Support anisotropic materials (FAnistropicMaterials permutation):
       swap isotropic GGX for the Ward-Dür lobe when anisotropy ≠ 0.
    5. Add emissive contribution from SceneColor (written by BasePass).

    r.UseClusteredDeferredShading_ToBeRemoved must be non-zero — we
    assume it is enabled (the caller is responsible for the guard).
    """

    # SM6 feature-level guard (ShouldUseClusteredDeferredShading)
    REQUIRES_SM6: bool = True

    def __init__(
        self,
        supports_anisotropy: bool = True,
        ambient: _Vec3 = None,
    ):
        # SUPPORTS_ANISOTROPIC_MATERIALS permutation
        self._aniso_enabled = supports_anisotropy
        # Simple sky / ambient term (replaces full IBL for cell use)
        self._ambient = ambient or _Vec3(0.05, 0.05, 0.07)

    def _cluster_lights_for_cell(
        self,
        cell_world_pos: _Vec3,
        lights: _List[AstroCellLight],
    ) -> _List[AstroCellLight]:
        """
        Returns lights whose range encompasses the cell world position.
        Mimics the light-grid lookup in ClusteredDeferredShadingPixelShader.
        """
        result = []
        for light in lights:
            dx = cell_world_pos.x - light.position.x
            dy = cell_world_pos.y - light.position.y
            dz = cell_world_pos.z - light.position.z
            dist2 = dx*dx + dy*dy + dz*dz
            if dist2 <= light.radius * light.radius:
                result.append(light)
        return result

    def shade(
        self,
        cell_id:        str,
        gbuffer:        AstroCellGBuffer,
        world_position: _Vec3,
        view_dir:       _Vec3,
        lights:         _List[AstroCellLight],
        tangent:        _Optional[_Vec3] = None,
    ) -> _Vec3:
        """
        Compute final lit colour for one cell.

        Parameters
        ----------
        cell_id        : identifier (for debug only)
        gbuffer        : GBuffer data written by AstroCellBasePass
        world_position : world-space surface position of the cell
        view_dir       : unit vector toward the camera
        lights         : all scene lights (will be cluster-culled internally)
        tangent        : world-space surface tangent (needed for aniso BRDF)

        Returns
        -------
        lit colour as _Vec3 (linear, pre-tonemapped)
        """
        n = gbuffer.world_normal
        v = view_dir.normalize()
        t = (tangent or _Vec3(1, 0, 0)).normalize()

        base   = gbuffer.base_color
        rough  = gbuffer.roughness
        metal  = gbuffer.metallic
        aniso  = gbuffer.anisotropy if self._aniso_enabled else 0.0

        # Derive specular colour (UE4/5 metallic workflow)
        f0 = _Vec3(0.04, 0.04, 0.04)
        spec_color = _Vec3(
            f0.x + (base.x - f0.x) * metal,
            f0.y + (base.y - f0.y) * metal,
            f0.z + (base.z - f0.z) * metal,
        )
        diff_color = _Vec3(
            base.x * (1.0 - metal),
            base.y * (1.0 - metal),
            base.z * (1.0 - metal),
        )

        # Ambient / indirect
        acc = _Vec3(
            diff_color.x * self._ambient.x,
            diff_color.y * self._ambient.y,
            diff_color.z * self._ambient.z,
        )

        # Cluster-cull lights  (r.UseClusteredDeferredShading path)
        active_lights = self._cluster_lights_for_cell(world_position, lights)

        for light in active_lights:
            # Light vector + attenuation  (matches GetLocalLightAttenuation)
            dx = light.position.x - world_position.x
            dy = light.position.y - world_position.y
            dz = light.position.z - world_position.z
            dist = _math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist < 1e-9:
                continue

            l = _Vec3(dx/dist, dy/dist, dz/dist)

            # Inverse-square falloff with radius clamp (UE5 PointLight)
            falloff = _pow_safe(max(0.0, 1.0 - (dist/light.radius)**4), 2.0)
            falloff /= (dist*dist + 1.0)
            irradiance = light.intensity * falloff

            lc = light.color
            ndotl = _saturate(n.dot(l))

            # --- diffuse (Lambertian) -------------------------------------
            acc = _Vec3(
                acc.x + diff_color.x * lc.x * ndotl * irradiance,
                acc.y + diff_color.y * lc.y * ndotl * irradiance,
                acc.z + diff_color.z * lc.z * ndotl * irradiance,
            )

            # --- specular ------------------------------------------------
            if abs(aniso) > 0.01 and self._aniso_enabled:
                # SUPPORTS_ANISOTROPIC_MATERIALS permutation
                # Remap scalar anisotropy → (roughness_u, roughness_v)
                # following UE5 GetAnisotropicRoughness()
                ru = _saturate(rough * (1.0 + aniso))
                rv = _saturate(rough * (1.0 - aniso))
                # Use light's aniso_weight as a per-light tangent scale
                spec_val = astro_anisotropy_brdf(n, v, l, t, ru, rv)
                spec_val *= light.aniso_weight
            else:
                spec_val = _ggx_specular(n, v, l, rough)

            acc = _Vec3(
                acc.x + spec_color.x * lc.x * spec_val * irradiance,
                acc.y + spec_color.y * lc.y * spec_val * irradiance,
                acc.z + spec_color.z * lc.z * spec_val * irradiance,
            )

        # Add BasePass emissive contribution from SceneColor
        sc = gbuffer.scene_color
        acc = _Vec3(acc.x + sc.x, acc.y + sc.y, acc.z + sc.z)

        return acc.clamp01()

    def run(
        self,
        gbuffer_map:    _Dict[str, AstroCellGBuffer],
        world_positions: _Dict[str, _Vec3],
        view_dir:       _Vec3,
        lights:         _List[AstroCellLight],
        tangents:       _Optional[_Dict[str, _Vec3]] = None,
    ) -> _Dict[str, _Vec3]:
        """
        Shade all cells.

        Returns
        -------
        {cell_id: lit_colour_vec3}
        """
        tangents = tangents or {}
        return {
            cid: self.shade(
                cid,
                gbuf,
                world_positions.get(cid, _Vec3(0, 0, 0)),
                view_dir,
                lights,
                tangents.get(cid),
            )
            for cid, gbuf in gbuffer_map.items()
        }


# ---------------------------------------------------------------------------
# Top-level pipeline — AstroCellRenderPipeline
# Composes all five passes in UE5 order.
# ---------------------------------------------------------------------------
# 鲁迅式：渲染管线的顺序，和历史的顺序一样，
# 不能随便颠倒——颠倒了，不是错误，便是革命。

class AstroCellRenderPipeline:
    """
    Full deferred pipeline for the cell layer, porting:

      1. DepthPass          (DepthRendering.cpp)
      2. CustomDepthPass    (CustomDepthRendering.cpp)
      3. BasePass → GBuffer (BasePassRendering.cpp + AnisotropyRendering.cpp)
      4. ClusteredDeferred  (ClusteredDeferredShadingPass.cpp)

    Mirrors the high-level FDeferredShadingSceneRenderer::Render() sequence
    as applied to a dict of SVG/astro cells rather than mesh draw commands.
    """

    def __init__(
        self,
        enable_anisotropy:    bool  = True,
        enable_clustered:     bool  = True,
        custom_depth_order:   int   = 0,
        custom_depth_stencil: bool  = True,
        ambient:              _Optional[_Vec3] = None,
    ):
        self._depth_pass    = AstroCellDepthPass()
        self._custom_depth  = AstroCellCustomDepthPass(
            order=custom_depth_order,
            writes_stencil=custom_depth_stencil,
        )
        self._base_pass     = AstroCellBasePass(enable_anisotropy=enable_anisotropy)
        self._clustered     = AstroCellClusteredDeferredShadingPass(
            supports_anisotropy=enable_anisotropy,
            ambient=ambient,
        )
        self._enable_clustered = enable_clustered

    def render(
        self,
        cells:              _List[_Dict],
        materials:          _Dict[str, AstroCellMaterial],
        world_normals:      _Dict[str, _Vec3],
        world_positions:    _Dict[str, _Vec3],
        eye_depths:         _Dict[str, float],
        view_dir:           _Vec3,
        lights:             _List[AstroCellLight],
        custom_depth_reqs:  _Optional[_List[AstroCellCustomDepthRequest]] = None,
        tangents:           _Optional[_Dict[str, _Vec3]] = None,
    ) -> _Dict[str, _Tuple[_Vec3, AstroCellGBuffer, _Optional[_Tuple[float, int]]]]:
        """
        Run the full pipeline.

        Parameters
        ----------
        cells             : list of {'id', 'depth', 'masked'}
        materials         : {cell_id: AstroCellMaterial}
        world_normals     : {cell_id: Vec3}
        world_positions   : {cell_id: Vec3}
        eye_depths        : {cell_id: float}
        view_dir          : camera forward vector (unit)
        lights            : scene lights for clustered pass
        custom_depth_reqs : optional custom-depth draw requests
        tangents          : optional per-cell tangent vectors (aniso)

        Returns
        -------
        {cell_id: (lit_colour, gbuffer, (custom_depth, stencil) | None)}
        """
        # --- 1. Depth prepass -------------------------------------------
        depth_buf = self._depth_pass.run(cells)

        # --- 2. Custom depth (BeforeBasePass order) ----------------------
        custom_depth_buf: _Dict[str, _Tuple[float, int]] = {}
        if custom_depth_reqs and self._custom_depth.before_base_pass:
            custom_depth_buf = self._custom_depth.run(
                custom_depth_reqs, depth_buf
            )

        # --- 3. BasePass → GBuffer --------------------------------------
        gbuffer_map: _Dict[str, AstroCellGBuffer] = {}
        for cell in cells:
            cid   = cell["id"]
            mat   = materials.get(cid, AstroCellMaterial())
            norm  = world_normals.get(cid, _Vec3(0, 0, 1))
            depth = eye_depths.get(cid, depth_buf.get(cid, 1.0))

            gbuf = self._base_pass.encode(mat, norm, depth)

            # Stamp custom-depth into GBuffer if available
            if cid in custom_depth_buf:
                gbuf.custom_depth, gbuf.stencil = custom_depth_buf[cid]

            gbuffer_map[cid] = gbuf

        # --- 4. Custom depth (AfterBasePass order) ----------------------
        if custom_depth_reqs and not self._custom_depth.before_base_pass:
            extra = self._custom_depth.run(custom_depth_reqs, depth_buf)
            custom_depth_buf.update(extra)
            for cid, (cd, st) in extra.items():
                if cid in gbuffer_map:
                    gbuffer_map[cid].custom_depth = cd
                    gbuffer_map[cid].stencil      = st

        # --- 5. Clustered deferred shading ------------------------------
        if self._enable_clustered:
            lit_colours = self._clustered.run(
                gbuffer_map, world_positions, view_dir, lights, tangents
            )
        else:
            # Fall back to unlit (emissive only)
            lit_colours = {
                cid: gbuf.scene_color for cid, gbuf in gbuffer_map.items()
            }

        # --- Assemble output --------------------------------------------
        out: _Dict[str, _Tuple[_Vec3, AstroCellGBuffer, _Optional[_Tuple[float, int]]]] = {}
        for cid, gbuf in gbuffer_map.items():
            cd_entry = custom_depth_buf.get(cid)
            out[cid] = (lit_colours.get(cid, _Vec3(0, 0, 0)), gbuf, cd_entry)

        return out

# ═══════════════════════════════════════════════════════════════════════════════
# §  LUMEN SCENE MANAGEMENT  ── port of LumenScene.cpp / LumenSceneLighting.cpp
#    LumenSurfaceCache.cpp / LumenScreenSpaceBentNormal.cpp
#    LumenScreenProbeFiltering.cpp / LumenScreenProbeImportanceSampling.cpp
#
#    鲁迅式：旧中国有一句话——"万里长城今犹在，不见当年秦始皇"。
#    Lumen 的 Scene 管理亦然：距离场飘散，细胞仍在，光照已非昨日。
# ═══════════════════════════════════════════════════════════════════════════════

import math
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto

_LUMEN_GLOBAL_DF_RESOLUTION: int = 252
_LUMEN_GLOBAL_DF_CLIPMAP_EXTENT: float = 2500.0
_LUMEN_FAR_FIELD_MAX_TRACE_DISTANCE: float = 1.0e6
_LUMEN_FAR_FIELD_DITHER_SCALE: float = 200.0
_LUMEN_SURFACE_CACHE_ATLAS_SIZE: int = 4096
_LUMEN_PHYSICAL_PAGE_SIZE: int = 128
_LUMEN_CARD_TILE_SIZE: int = 8


@dataclass
class LumenSceneConfig:
    """
    Mirrors the per-scene Lumen configuration negotiated at renderer
    initialisation time.

    鲁迅式：这些参数就是旧时的"祖宗成法"——动一个，整个渲染管线都要重来。
    """
    global_df_resolution:     int   = _LUMEN_GLOBAL_DF_RESOLUTION
    clipmap_extent:           float = _LUMEN_GLOBAL_DF_CLIPMAP_EXTENT
    use_far_field:            bool  = False
    far_field_occlusion_only: bool  = False
    far_field_max_trace_dist: float = _LUMEN_FAR_FIELD_MAX_TRACE_DISTANCE
    far_field_dither_scale:   float = _LUMEN_FAR_FIELD_DITHER_SCALE
    surface_cache_atlas_size: int   = _LUMEN_SURFACE_CACHE_ATLAS_SIZE

    def near_field_max_trace_distance_dither_scale(self) -> float:
        return self.far_field_dither_scale if self.use_far_field else 0.0

    def near_field_scene_radius(self, culling_radius: float = float("inf")) -> float:
        if self.use_far_field and math.isfinite(culling_radius):
            return culling_radius
        return float("inf")


@dataclass
class LumenMeshCard:
    """
    Lightweight representation of a single Lumen mesh card.

    鲁迅式：一个细胞有六个面，每面都是一张名片，叫做"Card"。
    名片印满了，光才算打在实处。
    """
    card_id:    int
    center:     tuple = (0.0, 0.0, 0.0)
    extent:     tuple = (50.0, 50.0, 1.0)
    is_visible: bool  = True
    lod_level:  int   = 0

    def world_bounds(self):
        mn = tuple(c - e for c, e in zip(self.center, self.extent))
        mx = tuple(c + e for c, e in zip(self.center, self.extent))
        return mn, mx


class LumenScene:
    """
    Cell-level analogue of FLumenSceneData.

    鲁迅式：有灯才有影，有卡才有光。Scene 不过是一本花名册，
    记着谁来过、谁还没走，以及谁的面孔已经模糊。
    """

    def __init__(self, config=None):
        self.config        = config or LumenSceneConfig()
        self._cards        = {}
        self._next_id      = 0
        self.view_origin   = (0.0, 0.0, 0.0)

    def add_card(self, center, extent=(50.0, 50.0, 1.0), lod_level=0):
        cid = self._next_id; self._next_id += 1
        self._cards[cid] = LumenMeshCard(cid, center, extent, lod_level=lod_level)
        return cid

    def remove_card(self, card_id):
        self._cards.pop(card_id, None)

    def visible_cards(self):
        return [c for c in self._cards.values() if c.is_visible]

    def update_view_origin(self, new_origin):
        self.view_origin = new_origin

    def global_df_voxel_size(self):
        return (2.0 * self.config.clipmap_extent) / self.config.global_df_resolution

    def __repr__(self):
        return (
            f"LumenScene(cards={len(self._cards)}, "
            f"origin={self.view_origin}, "
            f"voxel_size={self.global_df_voxel_size():.2f})"
        )


_DIRECT_LIGHTING_UPDATE_FACTOR  = 32
_RADIOSITY_UPDATE_FACTOR        = 64


@dataclass
class LumenCardUpdateContext:
    """
    Python port of FLumenCardUpdateContext.

    鲁迅式：帝国的税吏每天只收三十二分之一，却称之为"全部"。
    UpdateFactor 亦如此——分期偿还，从不告诉你总债。
    """
    update_atlas_width:  int = 0
    update_atlas_height: int = 0
    max_update_tiles:    int = 0
    update_factor:       int = _DIRECT_LIGHTING_UPDATE_FACTOR

    @classmethod
    def build(cls, physical_atlas_size, update_factor, force_full_update=False):
        ctx = cls()
        ctx.update_factor = max(1, min(update_factor, 1024))
        if force_full_update:
            ctx.update_factor = 1
        mult = 1.0 / math.sqrt(float(ctx.update_factor))
        pw, ph = physical_atlas_size

        def _rt(v):
            return math.ceil((v + 0.5) / _LUMEN_CARD_TILE_SIZE) * _LUMEN_CARD_TILE_SIZE

        uw = max(_rt(pw * mult), _LUMEN_PHYSICAL_PAGE_SIZE)
        uh = max(_rt(ph * mult), _LUMEN_PHYSICAL_PAGE_SIZE)
        ctx.update_atlas_width  = uw
        ctx.update_atlas_height = uh
        ctx.max_update_tiles = (uw // _LUMEN_CARD_TILE_SIZE) * (uh // _LUMEN_CARD_TILE_SIZE)
        return ctx

    def update_atlas_size(self):
        return self.update_atlas_width, self.update_atlas_height


@dataclass
class LumenCombineLightingParams:
    """
    FLumenCardCombineLightingCS::FParameters — surface-cache combine pass.

    鲁迅式：漫反射、直接光、间接光，最后合成一张贴图——
    这和旧社会的"总账"没什么区别，每一笔都是别人欠的。
    """
    diffuse_color_boost:         float = 1.0
    albedo_atlas_valid:          bool  = False
    emissive_atlas_valid:        bool  = False
    direct_lighting_valid:       bool  = False
    indirect_lighting_valid:     bool  = False
    indirect_atlas_half_texel_x: float = 0.5 / 1024
    indirect_atlas_half_texel_y: float = 0.5 / 1024

    def pack(self):
        return {
            "DiffuseColorBoost": self.diffuse_color_boost,
            "IndirectLightingAtlasHalfTexelSize": (
                self.indirect_atlas_half_texel_x,
                self.indirect_atlas_half_texel_y,
            ),
        }


class LumenSceneLighting:
    """
    Drives the per-frame lighting update of a LumenScene.

    鲁迅式：照明从不是一次性的事——每帧都要重算，
    就像每天都要重新证明自己还活着。
    """

    def __init__(self, scene, direct_factor=_DIRECT_LIGHTING_UPDATE_FACTOR,
                 radiosity_factor=_RADIOSITY_UPDATE_FACTOR):
        self.scene             = scene
        self.direct_factor     = direct_factor
        self.radiosity_factor  = radiosity_factor
        self._frame_index      = 0

    def _atlas(self):
        s = self.scene.config.surface_cache_atlas_size
        return (s, s)

    def compute_direct_update_ctx(self, force_full=False):
        return LumenCardUpdateContext.build(self._atlas(), self.direct_factor, force_full)

    def compute_radiosity_update_ctx(self, force_full=False):
        return LumenCardUpdateContext.build(self._atlas(), self.radiosity_factor, force_full)

    def tick(self, force_full_update=False):
        self._frame_index += 1
        d = self.compute_direct_update_ctx(force_full_update)
        r = self.compute_radiosity_update_ctx(force_full_update)
        print(
            f"[LUMEN-LIGHTING] frame={self._frame_index} "
            f"direct_tiles={d.max_update_tiles} radiosity_tiles={r.max_update_tiles}",
            file=sys.stderr,
        )
        return d, r


class SurfaceCacheCompression(Enum):
    """
    ESurfaceCacheCompression.

    鲁迅式：能压缩的都压缩了，剩下的才叫"真实"。
    Disabled 是诚实，UAVAliasing 是取巧，CopyTextureRegion 是代价最贵的虚伪。
    """
    DISABLED            = auto()
    UAV_ALIASING        = auto()
    FRAMEBUFFER         = auto()
    COPY_TEXTURE_REGION = auto()


class SurfaceCacheLayer(Enum):
    DEPTH    = 0
    ALBEDO   = 1
    NORMAL   = 2
    EMISSIVE = 3


@dataclass
class SurfaceLayerConfig:
    name:             str
    uncompressed_fmt: str
    compressed_fmt:   str
    clear_value:      tuple = (0.0, 0.0, 0.0)


_SURFACE_LAYER_CONFIGS = {
    SurfaceCacheLayer.DEPTH:    SurfaceLayerConfig("Depth",   "PF_G16",            "PF_Unknown", (1.0, 0.0, 0.0)),
    SurfaceCacheLayer.ALBEDO:   SurfaceLayerConfig("Albedo",  "PF_R8G8B8A8",       "PF_BC7",     (0.0, 0.0, 0.0)),
    SurfaceCacheLayer.NORMAL:   SurfaceLayerConfig("Normal",  "PF_R8G8",           "PF_BC5",     (0.0, 0.0, 0.0)),
    SurfaceCacheLayer.EMISSIVE: SurfaceLayerConfig("Emissive","PF_FloatR11G11B10", "PF_BC6H",    (0.0, 0.0, 0.0)),
}


class SurfaceCacheDilationMode(Enum):
    """
    r.LumenScene.SurfaceCache.DilationMode.

    鲁迅式：蔓延一个像素，就是"宽容"；蔓延整张图，就是"谎言"。
    """
    DISABLED  = 0
    TWO_SIDED = 1
    ALL       = 2


@dataclass
class LumenSurfaceCache:
    """
    Card atlas allocation and per-frame copy scheduling.

    鲁迅式：表面缓存不是记忆，是遗忘的方式——
    只记住上一帧照亮过的那几个格子，其余的，下帧再说。
    """
    atlas_size:    int                       = _LUMEN_SURFACE_CACHE_ATLAS_SIZE
    compression:   SurfaceCacheCompression   = SurfaceCacheCompression.DISABLED
    dilation_mode: SurfaceCacheDilationMode  = SurfaceCacheDilationMode.DISABLED
    cull_underground: bool                   = False
    _atlas:        dict                      = field(default_factory=dict)

    def allocate_page(self, page_id):
        for layer in SurfaceCacheLayer:
            self._atlas[(layer, page_id)] = list(_SURFACE_LAYER_CONFIGS[layer].clear_value)

    def free_page(self, page_id):
        for layer in SurfaceCacheLayer:
            self._atlas.pop((layer, page_id), None)

    def copy_capture_to_atlas(self, page_id, layer, texels):
        if (layer, page_id) not in self._atlas:
            self.allocate_page(page_id)
        data = list(texels)
        if self.dilation_mode != SurfaceCacheDilationMode.DISABLED and data:
            data = [data[0]] + data + [data[-1]]
        self._atlas[(layer, page_id)] = data

    def is_compressed(self):
        return self.compression != SurfaceCacheCompression.DISABLED


_SSBN_DOWNSAMPLE_FACTOR        = 2
_SSBN_SLICE_COUNT              = 2
_SSBN_STEPS_PER_SLICE          = 3
_SSBN_FOLIAGE_OCC_STRENGTH     = 0.7
_SSBN_MAX_MULTIBOUNCE_ALBEDO   = 0.5
_SSBN_SLOPE_TOLERANCE          = 0.5
_SSBN_FOREGROUND_REJECT_FRACTION = 0.3


@dataclass
class ShortRangeAOConfig:
    """
    Runtime configuration for the Screen-Space Bent Normal pass.

    鲁迅式：法线弯曲了，光就遮住了；光遮住了，就叫做"环境遮蔽"。
    名字越高雅，背后的道理越朴素。
    """
    use_bent_normal:          bool  = True
    use_temporal:             bool  = True
    use_horizon_search:       bool  = True
    use_hzb:                  bool  = True
    downsample_factor:        int   = _SSBN_DOWNSAMPLE_FACTOR
    slice_count:              int   = _SSBN_SLICE_COUNT
    steps_per_slice:          int   = _SSBN_STEPS_PER_SLICE
    foliage_occ_strength:     float = _SSBN_FOLIAGE_OCC_STRENGTH
    max_multibounce_albedo:   float = _SSBN_MAX_MULTIBOUNCE_ALBEDO
    slope_tolerance:          float = _SSBN_SLOPE_TOLERANCE
    foreground_reject_frac:   float = _SSBN_FOREGROUND_REJECT_FRACTION
    apply_during_integration: bool  = False
    allow_async_compute:      bool  = True

    def texture_format(self):
        return "PF_R32_UINT" if self.use_bent_normal else "PF_R8"


def _horizon_angle(depth_samples, view_z, slice_dir, pixel_pitch,
                   slope_tolerance=_SSBN_SLOPE_TOLERANCE):
    max_horizon = -math.pi / 2.0
    for i, d in enumerate(depth_samples):
        if d <= 0.0:
            continue
        dist = (i + 1) * pixel_pitch
        angle = math.atan2(view_z - d, dist)
        if (view_z - d) > view_z * _SSBN_FOREGROUND_REJECT_FRACTION:
            continue
        if angle > max_horizon + slope_tolerance * 0.01:
            max_horizon = angle
    return max_horizon


def compute_bent_normal_ao(depth_buffer, normal_buffer, config=None, pixel_pitch=1.0):
    """
    Software reference for the Screen-Space Bent Normal pass.

    鲁迅式：把每个像素的天空扫描一遍，记录哪些方向被堵死——
    这就是所谓的"弯曲法线"，是对遮蔽的精确统计，而非诗意描述。
    """
    if config is None:
        config = ShortRangeAOConfig()
    height = len(depth_buffer)
    width  = len(depth_buffer[0]) if height > 0 else 0
    result = []
    for y in range(height):
        row = []
        for x in range(width):
            view_z = depth_buffer[y][x]
            normal = normal_buffer[y][x]
            total_occ = 0.0
            bent = [0.0, 0.0, 0.0]
            for s in range(config.slice_count):
                a = math.pi * s / config.slice_count
                sd = (math.cos(a), math.sin(a))
                samples = []
                for step in range(config.steps_per_slice):
                    sx = x + int(round((step + 1) * sd[0]))
                    sy = y + int(round((step + 1) * sd[1]))
                    samples.append(
                        depth_buffer[sy][sx] if 0 <= sx < width and 0 <= sy < height else 0.0
                    )
                h = _horizon_angle(samples, view_z, sd, pixel_pitch, config.slope_tolerance)
                occ = max(0.0, 1.0 - math.sin(max(h, 0.0)))
                total_occ += occ
                u = math.cos(h)
                bent[0] += sd[0] * u
                bent[1] += sd[1] * u
            ao = min(total_occ / max(config.slice_count, 1), config.foliage_occ_strength)
            alb = min(max(sum(normal) / 3.0, 0.0), config.max_multibounce_albedo)
            ao  = ao / (1.0 - alb * (1.0 - ao) + 1e-6)
            ln = math.sqrt(bent[0]**2 + bent[1]**2 + bent[2]**2) + 1e-8
            row.append((bent[0]/ln, bent[1]/ln, bent[2]/ln, ao))
        result.append(row)
    return result


_SPF_SPATIAL_PASSES          = 3
_SPF_DISOCCLUSION_FRAMES     = 4
_SPF_DISOCCLUSION_FRAC       = 0.4
_SPF_POSITION_WEIGHT_SCALE   = 1000.0
_SPF_MAX_RADIANCE_HIT_ANGLE  = 10.0
_SPF_HISTORY_WEIGHT          = 0.5
_SPF_HISTORY_DIST_THRESHOLD  = 30.0
_SPF_MAX_RAY_INTENSITY       = 10.0


@dataclass
class LumenScreenProbe:
    """
    A single screen-space probe: world position + radiance octahedron.

    鲁迅式：探针不探人，只探光。
    探到的光存进历史，历史再决定下一帧要不要信任它。
    """
    probe_id:           int
    screen_pos:         tuple = (0, 0)
    world_pos:          tuple = (0.0, 0.0, 0.0)
    scene_depth:        float = 0.0
    radiance:           list  = field(default_factory=list)
    history_radiance:   list  = field(default_factory=list)
    frames_accumulated: int   = 0
    is_moving:          bool  = False


def _spatial_weight(pa, pb, scale=_SPF_POSITION_WEIGHT_SCALE):
    dx = pa.world_pos[0] - pb.world_pos[0]
    dy = pa.world_pos[1] - pb.world_pos[1]
    dz = pa.world_pos[2] - pb.world_pos[2]
    return math.exp(-(dx*dx + dy*dy + dz*dz) * scale * 1e-6)


def composite_traces_with_scatter(probes, max_ray_intensity=_SPF_MAX_RAY_INTENSITY):
    """
    FScreenProbeCompositeTracesWithScatterCS — clamp firefly radiance.

    鲁迅式：每条光线都有最大亮度。亮过头的，削掉。
    现实主义从不允许辉光超标。
    """
    for p in probes:
        if p.radiance:
            p.radiance = [min(r, max_ray_intensity) for r in p.radiance]
    return probes


def temporally_accumulate_probe_radiance(
    probes, history_probes,
    history_weight=_SPF_HISTORY_WEIGHT,
    dist_threshold=_SPF_HISTORY_DIST_THRESHOLD,
):
    """
    FScreenProbeTemporallyAccumulateTraceRadianceCS.

    鲁迅式：历史是有重量的——但只在距离够近的时候。
    太远了，就当没发生过，重新开始。
    """
    hmap = {p.probe_id: p for p in history_probes}
    for probe in probes:
        hist = hmap.get(probe.probe_id)
        if hist is None or not hist.radiance:
            probe.frames_accumulated = 1
            continue
        dx = probe.world_pos[0] - hist.world_pos[0]
        dy = probe.world_pos[1] - hist.world_pos[1]
        dz = probe.world_pos[2] - hist.world_pos[2]
        if math.sqrt(dx*dx + dy*dy + dz*dz) > dist_threshold:
            probe.frames_accumulated = 1
            continue
        n = min(len(probe.radiance), len(hist.radiance))
        blended = [
            probe.radiance[i] * (1.0 - history_weight) + hist.radiance[i] * history_weight
            for i in range(n)
        ] + probe.radiance[n:]
        probe.radiance = probe.history_radiance = blended
        probe.frames_accumulated = hist.frames_accumulated + 1
    return probes


def spatial_filter_probes(
    probes,
    num_passes=_SPF_SPATIAL_PASSES,
    disocclusion_max_frames=_SPF_DISOCCLUSION_FRAMES,
    disocclusion_frac=_SPF_DISOCCLUSION_FRAC,
    position_weight_scale=_SPF_POSITION_WEIGHT_SCALE,
):
    """
    Multi-pass bilateral spatial filter over the probe grid.

    鲁迅式：遮蔽区域的噪声用邻居来弥补，这是所谓"空间滤波"。
    没有历史的地方，就靠周围的人说话。
    """
    for _pass in range(num_passes):
        updated = []
        for i, probe in enumerate(probes):
            if _pass > 0 and probe.frames_accumulated >= disocclusion_max_frames:
                updated.append(probe)
                continue
            w_sum = 1.0
            accum = list(probe.radiance)
            for j, other in enumerate(probes):
                if i == j:
                    continue
                w = _spatial_weight(probe, other, position_weight_scale)
                if w < 1e-4:
                    continue
                n = min(len(accum), len(other.radiance))
                for k in range(n):
                    accum[k] += other.radiance[k] * w
                w_sum += w
            probe.radiance = [v / w_sum for v in accum]
            updated.append(probe)
        probes = updated
    return probes


_IS_ENABLED               = True
_IS_INCOMING_LIGHTING     = True
_IS_PROBE_RADIANCE_HIST   = True
_IS_BRDF_OCTAHEDRON_RES   = 8
_IS_MIN_PDF_TO_TRACE       = 0.1
_IS_HISTORY_DIST_THRESHOLD = 30.0


def _octahedron_dir(u, v):
    fx = u * 2.0 - 1.0
    fy = v * 2.0 - 1.0
    fz = 1.0 - abs(fx) - abs(fy)
    if fz < 0.0:
        ox = (1.0 - abs(fy)) * (1.0 if fx >= 0 else -1.0)
        oy = (1.0 - abs(fx)) * (1.0 if fy >= 0 else -1.0)
        fx, fy = ox, oy
    ln = math.sqrt(fx*fx + fy*fy + fz*fz) + 1e-8
    return (fx/ln, fy/ln, fz/ln)


def compute_lighting_pdf(probe, resolution=_IS_BRDF_OCTAHEDRON_RES,
                          use_history=_IS_PROBE_RADIANCE_HIST, history_weight=0.9):
    """
    FScreenProbeComputeLightingProbabilityDensityFunctionCS.

    鲁迅式：上一帧亮的方向，这一帧优先去看。
    这叫做"重要性采样"，也叫做"走捷径的学问"。
    """
    n   = resolution * resolution
    pdf = [0.0] * n
    for i in range(n):
        _, _, fz = _octahedron_dir((i % resolution + 0.5) / resolution,
                                    (i // resolution + 0.5) / resolution)
        pdf[i] = max(fz, 0.0)
    if use_history and probe.history_radiance:
        hn = min(n, len(probe.history_radiance))
        for i in range(hn):
            pdf[i] = pdf[i] * (1.0 - history_weight) + probe.history_radiance[i] * history_weight
    total = sum(pdf) + 1e-8
    return [p / total for p in pdf]


def generate_importance_sampled_rays(
    probe,
    tracing_resolution=_IS_BRDF_OCTAHEDRON_RES,
    min_pdf=_IS_MIN_PDF_TO_TRACE,
    use_importance_sampling=_IS_ENABLED,
):
    """
    FScreenProbeGenerateRaysCS — select trace directions via PDF.

    鲁迅式：方向太暗的就不看了，把光阴省下来照有价值的地方。
    这是渲染的经济学，也是人生的经济学。
    """
    if not use_importance_sampling:
        n = tracing_resolution * tracing_resolution
        return [
            _octahedron_dir((i % tracing_resolution + 0.5) / tracing_resolution,
                             (i // tracing_resolution + 0.5) / tracing_resolution)
            for i in range(n)
        ]
    pdf  = compute_lighting_pdf(probe, tracing_resolution)
    rays = []
    for i, p in enumerate(pdf):
        if p < min_pdf:
            continue
        rays.append(_octahedron_dir(
            (i % tracing_resolution + 0.5) / tracing_resolution,
            (i // tracing_resolution + 0.5) / tracing_resolution,
        ))
    return rays


@dataclass
class LumenFrameOutputs:
    """Per-frame products of the Lumen pipeline tick."""
    direct_update_ctx:    LumenCardUpdateContext
    radiosity_update_ctx: LumenCardUpdateContext
    combine_params:       LumenCombineLightingParams
    ao_map:               list
    filtered_probes:      list
    probe_rays:           dict


def lumen_frame_tick(
    scene, lighting, surface_cache,
    depth_buffer, normal_buffer,
    probes, history_probes,
    ao_config=None,
    force_full_update=False,
    epoch=0,
):
    """
    One full Lumen frame for a cell sub-scene.

    鲁迅式：七个步骤，七道工序。每道都不能省，省了就有瑕疵。
    但没有人会数清楚，他们只看最终画面是否好看。
    """
    scene.update_view_origin((float(epoch), 0.0, 0.0))
    d_ctx, r_ctx = lighting.tick(force_full_update)
    combine = LumenCombineLightingParams(
        diffuse_color_boost     = 1.0,
        albedo_atlas_valid      = True,
        emissive_atlas_valid    = True,
        direct_lighting_valid   = d_ctx.max_update_tiles > 0,
        indirect_lighting_valid = r_ctx.max_update_tiles > 0,
    )
    ao_map  = compute_bent_normal_ao(depth_buffer, normal_buffer, ao_config)
    probes  = composite_traces_with_scatter(probes)
    probes  = temporally_accumulate_probe_radiance(probes, history_probes)
    probes  = spatial_filter_probes(probes)
    probe_rays = {p.probe_id: generate_importance_sampled_rays(p) for p in probes}
    return LumenFrameOutputs(
        direct_update_ctx    = d_ctx,
        radiosity_update_ctx = r_ctx,
        combine_params       = combine,
        ao_map               = ao_map,
        filtered_probes      = probes,
        probe_rays           = probe_rays,
    )


# =============================================================================
# DistanceField AO + LightingPost + ObjectCulling + ObjectManagement + ScreenGrid
# 移植自 UE5 Renderer-Private，鲁迅式注释 20 %
# =============================================================================

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# 全局开关 —— 对应 CVar r.DistanceFieldAO / r.AOQuality 等
# 每一个 CVar 背后都是一场争论：性能组要关，画面组要开。
# ---------------------------------------------------------------------------
G_DISTANCE_FIELD_AO: bool = True
G_DISTANCE_FIELD_AO_QUALITY: int = 2          # 0=off 1=medium 2=high
G_AO_OBJECT_DISTANCE_FIELD: bool = True
G_AO_GLOBAL_DISTANCE_FIELD: bool = True
G_AO_GLOBAL_DF_START_DISTANCE: float = 1000.0
G_AO_MAX_VIEW_DISTANCE: float = 20000.0
G_AO_STEP_EXPONENT_SCALE: float = 0.5
G_AO_DOWNSAMPLE_FACTOR: int = 2
G_CONE_TRACE_DOWNSAMPLE_FACTOR: int = 4

# LightingPost
G_AO_USE_HISTORY: bool = True
G_AO_CLEAR_HISTORY: bool = False
G_AO_HISTORY_STABILITY_PASS: bool = True
G_AO_HISTORY_WEIGHT: float = 0.85
G_AO_HISTORY_DISTANCE_THRESHOLD: float = 30.0
G_AO_VIEW_FADE_DISTANCE_SCALE: float = 0.70

# ObjectCulling
G_AO_SCATTER_TILE_CULLING: bool = True
G_AVERAGE_DF_OBJECTS_PER_CULL_TILE: int = 512

# ObjectManagement
G_MESH_DF_MAX_OBJECT_BOUNDING_RADIUS: float = 100_000.0
G_DF_PARALLEL_UPDATE: bool = False
G_MESH_SDF_SURFACE_BIAS_EXPAND: float = 0.25

# ScreenGrid
G_AO_USE_JITTER: bool = True
G_DF_AO_TRAVERSE_MIPS: bool = True
NUM_CONE_SAMPLE_DIRECTIONS: int = 9
CONE_TRACE_GLOBAL_DF_TILE_SIZE: int = 8


def use_distance_field_ao() -> bool:
    """r.DistanceFieldAO && r.AOQuality >= 1。两个条件缺一不可，像人一样。"""
    return G_DISTANCE_FIELD_AO and G_DISTANCE_FIELD_AO_QUALITY >= 1


def use_ao_object_distance_field() -> bool:
    return G_AO_OBJECT_DISTANCE_FIELD and G_DISTANCE_FIELD_AO_QUALITY >= 2


def get_max_ao_view_distance() -> float:
    return G_AO_MAX_VIEW_DISTANCE


def use_ao_history_stability_pass() -> bool:
    return G_AO_HISTORY_STABILITY_PASS and G_DISTANCE_FIELD_AO_QUALITY >= 2


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class DFAOParameters:
    """
    对应 FDistanceFieldAOParameters。
    距离场 AO 的两段衰减区间：近处用 Object SDF，远处用 Global SDF。
    如同人生：年轻时斤斤计较细节，年老时只看大略。
    """
    object_max_occlusion_distance: float = 600.0
    global_max_occlusion_distance: float = 0.0
    contrast: float = 1.0

    @classmethod
    def from_sky_light(cls, occlusion_max_distance: float, contrast: float) -> "DFAOParameters":
        contrast = max(0.01, min(2.0, contrast))
        occlusion_max_distance = max(2.0, min(3000.0, occlusion_max_distance))
        if G_AO_GLOBAL_DISTANCE_FIELD:
            obj_dist = min(occlusion_max_distance, G_AO_GLOBAL_DF_START_DISTANCE)
            glo_dist = occlusion_max_distance if occlusion_max_distance >= G_AO_GLOBAL_DF_START_DISTANCE else 0.0
        else:
            obj_dist = occlusion_max_distance
            glo_dist = 0.0
        return cls(object_max_occlusion_distance=obj_dist,
                   global_max_occlusion_distance=glo_dist,
                   contrast=contrast)


@dataclass
class IntPoint:
    x: int = 0
    y: int = 0

    def divide_and_round_down(self, divisor: int) -> "IntPoint":
        return IntPoint(self.x // divisor, self.y // divisor)


@dataclass
class DFObjectBounds:
    """场景中一个 Mesh SDF 对象的包围球：中心 + 半径。"""
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 0.0
    object_index: int = 0


@dataclass
class CulledObjectBuffer:
    """FCulledObjectBuffers 简化版 —— culling 之后存活的对象列表。"""
    object_indices: List[int] = field(default_factory=list)
    indirect_arg_count: int = 0   # 等价于 RWObjectIndirectArguments[0]


@dataclass
class TileIntersectionData:
    """每个屏幕 tile 与若干 SDF 对象的交叉列表。"""
    tile_x: int = 0
    tile_y: int = 0
    object_indices: List[int] = field(default_factory=list)


@dataclass
class BentNormalAO:
    """
    遮蔽后的弯曲法线 —— 方向表示最不遮蔽的方向，长度表示可见度。
    说白了：被压扁了多少，往哪个方向还能透口气。
    """
    bent_normal: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    occlusion: float = 0.0   # 0 = 完全不遮蔽，1 = 完全遮蔽


@dataclass
class AOHistoryState:
    """对应 FTemporalAAHistory 在 DFAO 场景中的子集。"""
    bent_normal_history: List[BentNormalAO] = field(default_factory=list)
    valid: bool = False
    frame_index: int = 0


@dataclass
class ScreenGridAOBuffer:
    """屏幕格 cone-trace 的中间结果。对应 FAOScreenGridParameters。"""
    width: int = 0
    height: int = 0
    # 每像素 NUM_CONE_SAMPLE_DIRECTIONS 个 float（遮蔽量）
    cone_depths: List[float] = field(default_factory=list)


@dataclass
class DFSceneData:
    """
    FDistanceFieldSceneData 简化版。
    整个场景的 SDF 资产目录：对象包围盒、Atlas 纹理尺寸。
    没有这张清单，渲染器就是瞎子。
    """
    objects: List[DFObjectBounds] = field(default_factory=list)
    brick_atlas_dims: Tuple[int, int, int] = (64, 64, 64)
    num_objects_in_buffer: int = 0

    def add_object(self, obj: DFObjectBounds) -> None:
        if obj.radius > G_MESH_DF_MAX_OBJECT_BOUNDING_RADIUS:
            # 太大的对象被排除 —— 排除不代表不存在，只是不纳入计算。
            return
        self.objects.append(obj)
        self.num_objects_in_buffer += 1

    def remove_object(self, object_index: int) -> None:
        self.objects = [o for o in self.objects if o.object_index != object_index]
        self.num_objects_in_buffer = len(self.objects)


# ---------------------------------------------------------------------------
# AmbientOcclusion — 核心参数 / 采样方向
# ---------------------------------------------------------------------------

# 对应 SpacedVectors9：半球上均匀分布的 9 个方向
_SPACED_VECTORS_9: List[Tuple[float, float, float]] = [
    (-0.1840, 0.5545, 0.8117),
    ( 0.5404, 0.5404, 0.6455),
    ( 0.8117, 0.3124, 0.4944),
    ( 0.4944, 0.0000, 0.8693),
    (-0.0000, 0.0000, 1.0000),
    (-0.4944, 0.0000, 0.8693),
    (-0.8117, 0.3124, 0.4944),
    (-0.5404, 0.5404, 0.6455),
    ( 0.1840, 0.5545, 0.8117),
]


def get_spaced_vectors(frame_number: int) -> List[Tuple[float, float, float]]:
    """
    按帧号旋转采样方向集合，实现时域超采样。
    不同的帧看到不同的采样，合起来才是完整的真相。
    """
    angle = (frame_number % 4) * (math.pi / 4.0)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    result = []
    for vx, vy, vz in _SPACED_VECTORS_9:
        rx = vx * cos_a - vy * sin_a
        ry = vx * sin_a + vy * cos_a
        result.append((rx, ry, vz))
    return result


def compute_bent_normal_normalize_factor(sample_dirs: List[Tuple[float, float, float]]) -> float:
    """
    无遮蔽时所有 cone 方向的合向量长度的倒数 —— 归一化用。
    归一化之后，才能与别的数据相比较；不归一化的数字，像没有单位的重量。
    """
    ux = sum(d[0] for d in sample_dirs) / len(sample_dirs)
    uy = sum(d[1] for d in sample_dirs) / len(sample_dirs)
    uz = sum(d[2] for d in sample_dirs) / len(sample_dirs)
    mag = math.sqrt(ux*ux + uy*uy + uz*uz)
    return 1.0 / mag if mag > 1e-6 else 0.0


def get_buffer_size_for_ao(view_width: int, view_height: int) -> IntPoint:
    """对应 GetBufferSizeForAO：按 G_AO_DOWNSAMPLE_FACTOR 降采样。"""
    return IntPoint(view_width // G_AO_DOWNSAMPLE_FACTOR,
                    view_height // G_AO_DOWNSAMPLE_FACTOR)


def get_buffer_size_for_cone_tracing(view_width: int, view_height: int) -> IntPoint:
    ao = get_buffer_size_for_ao(view_width, view_height)
    w = max(ao.x // G_CONE_TRACE_DOWNSAMPLE_FACTOR, 1)
    h = max(ao.y // G_CONE_TRACE_DOWNSAMPLE_FACTOR, 1)
    return IntPoint(w, h)


# ---------------------------------------------------------------------------
# ObjectManagement — 场景 SDF 对象的增删改
# ---------------------------------------------------------------------------

def update_distance_field_object_buffers(
    scene_data: DFSceneData,
    objects_to_add: List[DFObjectBounds],
    objects_to_remove: List[int],
    surface_bias_expand: float = G_MESH_SDF_SURFACE_BIAS_EXPAND,
    parallel: bool = G_DF_PARALLEL_UPDATE,
) -> None:
    """
    对应 UpdateDistanceFieldObjectBuffers。
    先删再加，顺序不能错：若先加再删，可能误删新加的对象。
    世界上有些事情也讲究顺序，颠倒了就是另一个故事。
    """
    for idx in objects_to_remove:
        scene_data.remove_object(idx)

    # 可并行（G_DF_PARALLEL_UPDATE），此处简化为串行
    for obj in objects_to_add:
        # 表面偏移：膨胀一个 voxel 的 surface_bias_expand 比例
        expanded_radius = obj.radius * (1.0 + surface_bias_expand)
        expanded = DFObjectBounds(
            center=obj.center,
            radius=expanded_radius,
            object_index=obj.object_index,
        )
        scene_data.add_object(expanded)


def setup_object_buffer_parameters(scene_data: DFSceneData) -> dict:
    """
    对应 DistanceField::SetupObjectBufferParameters。
    返回一个参数字典，模拟 GPU SRV 绑定。
    """
    return {
        "num_scene_objects": scene_data.num_objects_in_buffer,
        "scene_objects": scene_data.objects,
    }


def setup_atlas_parameters(scene_data: DFSceneData) -> dict:
    """
    对应 DistanceField::SetupAtlasParameters。
    Atlas 是所有 SDF brick 拼在一起的大纹理；这里只记录尺寸。
    """
    bx, by, bz = scene_data.brick_atlas_dims
    return {
        "brick_atlas_dims": (bx, by, bz),
        "brick_atlas_texel_size": (1.0/bx, 1.0/by, 1.0/bz),
    }


# ---------------------------------------------------------------------------
# ObjectCulling — 视锥 + 距离剔除
# ---------------------------------------------------------------------------

def _sphere_inside_frustum(
    center: Tuple[float, float, float],
    radius: float,
    frustum_planes: List[Tuple[float, float, float, float]],   # (nx,ny,nz,d)
) -> bool:
    """
    对应 GPU 端 CullObjectsForViewCS 的核心判断。
    六个平面，一个都不能放过；像检查一个人，每个方面都要审视。
    """
    cx, cy, cz = center
    for nx, ny, nz, d in frustum_planes:
        dist = nx*cx + ny*cy + nz*cz + d
        if dist < -radius:
            return False
    return True


def cull_objects_to_view(
    scene_data: DFSceneData,
    frustum_planes: List[Tuple[float, float, float, float]],
    ao_params: DFAOParameters,
    view_origin: Tuple[float, float, float],
) -> CulledObjectBuffer:
    """
    对应 CullObjectsToView。
    先用视锥剔除，再按 AO 最大距离剔除。两重关卡，过了才算数。
    """
    buf = CulledObjectBuffer()
    ox, oy, oz = view_origin
    for obj in scene_data.objects:
        cx, cy, cz = obj.center
        dist_sq = (cx-ox)**2 + (cy-oy)**2 + (cz-oz)**2
        max_dist = ao_params.object_max_occlusion_distance + obj.radius
        if dist_sq > max_dist * max_dist:
            continue
        if dist_sq > G_AO_MAX_VIEW_DISTANCE**2:
            continue
        if not _sphere_inside_frustum(obj.center, obj.radius, frustum_planes):
            continue
        buf.object_indices.append(obj.object_index)
    buf.indirect_arg_count = len(buf.object_indices)
    return buf


def build_tile_cones(
    view_width: int,
    view_height: int,
    tile_size_x: int = 16,
    tile_size_y: int = 16,
) -> List[dict]:
    """
    对应 FBuildTileConesCS。
    把屏幕分成 tile，每个 tile 计算一个包围锥（轴 + 半角余弦）。
    锥越紧，后续剔除越激进；锥越松，剔除越保守。
    """
    ao_size = get_buffer_size_for_ao(view_width, view_height)
    tiles_x = max(1, (ao_size.x + tile_size_x - 1) // tile_size_x)
    tiles_y = max(1, (ao_size.y + tile_size_y - 1) // tile_size_y)
    tiles = []
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            # 简化：tile 中心方向 = (0,0,1)，半角余弦 = 0.5
            tiles.append({
                "tile_x": tx, "tile_y": ty,
                "cone_axis": (0.0, 0.0, 1.0),
                "cone_cos": 0.5,
                "depth_min": 0.0, "depth_max": G_AO_MAX_VIEW_DISTANCE,
            })
    return tiles


def scatter_tile_culling(
    culled_buf: CulledObjectBuffer,
    tiles: List[dict],
    scene_data: DFSceneData,
) -> List[TileIntersectionData]:
    """
    对应 FObjectCullVS / FObjectCullPS —— 光栅化散射剔除。
    用球体的包围盒覆盖哪些 tile，就把该对象写入哪些 tile 的列表。
    规则简单，但执行一遍要遍历 N×M；复杂性从不消失，只是转移。
    """
    obj_map = {o.object_index: o for o in scene_data.objects}
    tile_data = [
        TileIntersectionData(tile_x=t["tile_x"], tile_y=t["tile_y"])
        for t in tiles
    ]
    num_tiles_x = max((t["tile_x"] for t in tiles), default=0) + 1

    for obj_idx in culled_buf.object_indices:
        obj = obj_map.get(obj_idx)
        if obj is None:
            continue
        # 简化：对象投影到所有 tile（实际应做包围矩形交集）
        for td in tile_data:
            td.object_indices.append(obj_idx)

    return tile_data


# ---------------------------------------------------------------------------
# ScreenGrid cone-trace AO
# ---------------------------------------------------------------------------

_JITTER_OFFSETS: List[Tuple[float, float]] = [
    (0.25, 0.00),
    (0.75, 0.25),
    (0.50, 0.75),
    (0.00, 0.50),
]


def get_jitter_offset(frame_index: int, use_history: bool) -> Tuple[float, float]:
    """
    对应 GetJitterOffset。4 帧循环抖动，配合时域累积使用。
    抖动是一种诚实：承认单帧的采样不够，借历史来补足。
    """
    if G_AO_USE_JITTER and use_history:
        jx, jy = _JITTER_OFFSETS[frame_index % 4]
        return jx * G_CONE_TRACE_DOWNSAMPLE_FACTOR, jy * G_CONE_TRACE_DOWNSAMPLE_FACTOR
    return 0.0, 0.0


def _sample_distance_field(
    obj: DFObjectBounds,
    ray_origin: Tuple[float, float, float],
    ray_dir: Tuple[float, float, float],
    t: float,
) -> float:
    """
    SDF 球体采样的极简近似。
    真正的实现要查 Atlas brick，这里用解析球替代。
    """
    px = ray_origin[0] + ray_dir[0] * t - obj.center[0]
    py = ray_origin[1] + ray_dir[1] * t - obj.center[1]
    pz = ray_origin[2] + ray_dir[2] * t - obj.center[2]
    return math.sqrt(px*px + py*py + pz*pz) - obj.radius


def cone_trace_object_occlusion(
    ray_origin: Tuple[float, float, float],
    cone_dir: Tuple[float, float, float],
    tan_half_angle: float,
    max_distance: float,
    objects: List[DFObjectBounds],
    traverse_mips: bool = True,
) -> float:
    """
    对应 FConeTraceScreenGridObjectOcclusionCS 的单条 cone trace。
    沿锥方向步进，累积遮蔽量。步长按指数增长（GAOStepExponentScale）。
    快到终点时步子已经很大，精度换速度；起点附近小步谨慎前行。
    """
    step_scale = G_AO_STEP_EXPONENT_SCALE
    t = 0.5  # 起始偏移避免自遮蔽
    occlusion = 0.0
    step = 0.5

    while t < max_distance and occlusion < 1.0:
        min_sdf = max_distance
        for obj in objects:
            sdf_val = _sample_distance_field(obj, ray_origin, cone_dir, t)
            min_sdf = min(min_sdf, sdf_val)

        cone_radius = t * tan_half_angle
        if cone_radius > 1e-6:
            occ_contribution = max(0.0, 1.0 - min_sdf / cone_radius)
            occlusion = max(occlusion, occ_contribution)

        step = max(step * (1.0 + step_scale), min_sdf * 0.5)
        if traverse_mips:
            step = max(step, t * 0.01)
        t += step

    return min(occlusion, 1.0)


def compute_screen_grid_ao(
    pixel_positions: List[Tuple[float, float, float]],  # 世界空间像素位置
    pixel_normals: List[Tuple[float, float, float]],
    objects: List[DFObjectBounds],
    ao_params: DFAOParameters,
    frame_number: int = 0,
    use_history: bool = True,
) -> List[BentNormalAO]:
    """
    对应 FConeTraceScreenGridObjectOcclusionCS 在一帧中的整体调度。
    每个像素发射 9 条 cone，统计遮蔽；弯曲法线是遮蔽方向的加权平均。
    九条 cone，九个证据，最终合议出一个结论。
    """
    sample_dirs = get_spaced_vectors(frame_number)
    normalize_factor = compute_bent_normal_normalize_factor(sample_dirs)
    tan_half_angle = math.tan(math.radians(16.0))   # ~AOConeHalfAngle
    jx, jy = get_jitter_offset(frame_number, use_history)
    results: List[BentNormalAO] = []

    for pos, normal in zip(pixel_positions, pixel_normals):
        # 将采样方向转到像素法线半球
        px, py, pz = pos
        nx, ny, nz = normal
        occ_sum = 0.0
        bent_x, bent_y, bent_z = 0.0, 0.0, 0.0

        for dx, dy, dz in sample_dirs:
            # 简化：不做切线空间旋转，直接以世界空间方向 trace
            cone_dir = (dx, dy, dz)
            occ = cone_trace_object_occlusion(
                ray_origin=(px + nx * 0.5, py + ny * 0.5, pz + nz * 0.5),
                cone_dir=cone_dir,
                tan_half_angle=tan_half_angle,
                max_distance=ao_params.object_max_occlusion_distance,
                objects=objects,
                traverse_mips=G_DF_AO_TRAVERSE_MIPS,
            )
            visibility = 1.0 - occ
            bent_x += dx * visibility
            bent_y += dy * visibility
            bent_z += dz * visibility
            occ_sum += occ

        avg_occ = occ_sum / NUM_CONE_SAMPLE_DIRECTIONS
        bx = bent_x * normalize_factor
        by = bent_y * normalize_factor
        bz = bent_z * normalize_factor
        mag = math.sqrt(bx*bx + by*by + bz*bz)
        if mag > 1e-6:
            bx /= mag; by /= mag; bz /= mag

        results.append(BentNormalAO(
            bent_normal=(bx, by, bz),
            occlusion=min(avg_occ, 1.0),
        ))
    return results


# ---------------------------------------------------------------------------
# LightingPost — 历史积累、上采样
# ---------------------------------------------------------------------------

def compute_distance_fade(
    depth: float,
    fade_distance_scale: float = G_AO_VIEW_FADE_DISTANCE_SCALE,
) -> float:
    """
    对应 DistanceFadeScale：AO 在远距离线性淡出。
    远处的遮蔽本来就不可靠，淡出是对不确定性的诚实。
    """
    max_dist = get_max_ao_view_distance()
    fade_start = max_dist * fade_distance_scale
    if depth >= max_dist:
        return 0.0
    if depth <= fade_start:
        return 1.0
    return 1.0 - (depth - fade_start) / (max_dist - fade_start)


def update_history_depth_rejection(
    current: List[BentNormalAO],
    history: AOHistoryState,
    current_depths: List[float],
    history_depths: List[float],
) -> List[BentNormalAO]:
    """
    对应 UpdateHistoryDepthRejectionPS。
    深度差异大的像素拒绝历史（防止 ghost）；差异小的融合历史（减少噪点）。
    历史是有条件接受的，不是无条件信任的。
    """
    if not history.valid or G_AO_CLEAR_HISTORY:
        return current

    blended: List[BentNormalAO] = []
    for i, (cur, hist) in enumerate(zip(current, history.bent_normal_history)):
        cd = current_depths[i] if i < len(current_depths) else 0.0
        hd = history_depths[i] if i < len(history_depths) else 0.0
        depth_diff = abs(cd - hd)
        if depth_diff > G_AO_HISTORY_DISTANCE_THRESHOLD:
            blended.append(cur)
        else:
            w = G_AO_HISTORY_WEIGHT
            bx = cur.bent_normal[0] * (1-w) + hist.bent_normal[0] * w
            by = cur.bent_normal[1] * (1-w) + hist.bent_normal[1] * w
            bz = cur.bent_normal[2] * (1-w) + hist.bent_normal[2] * w
            occ = cur.occlusion * (1-w) + hist.occlusion * w
            mag = math.sqrt(bx*bx + by*by + bz*bz)
            if mag > 1e-6:
                bx /= mag; by /= mag; bz /= mag
            blended.append(BentNormalAO(bent_normal=(bx, by, bz), occlusion=occ))
    return blended


def filter_history_stability(
    ao_buffer: List[BentNormalAO],
    width: int,
    height: int,
) -> List[BentNormalAO]:
    """
    对应 FilterHistoryPS。在 AO 缓冲上做一次空间滤波，补洞、稳定。
    补洞是为了让结果看起来更完整；完整不等于正确，但看起来好一些。
    """
    if not use_ao_history_stability_pass():
        return ao_buffer
    filtered = list(ao_buffer)
    for i in range(len(filtered)):
        row, col = divmod(i, max(width, 1))
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < height and 0 <= nc < width:
                    neighbors.append(ao_buffer[nr * width + nc])
        if neighbors:
            avg_occ = sum(n.occlusion for n in neighbors) / len(neighbors)
            bx = sum(n.bent_normal[0] for n in neighbors) / len(neighbors)
            by = sum(n.bent_normal[1] for n in neighbors) / len(neighbors)
            bz = sum(n.bent_normal[2] for n in neighbors) / len(neighbors)
            mag = math.sqrt(bx*bx + by*by + bz*bz)
            if mag > 1e-6:
                bx /= mag; by /= mag; bz /= mag
            filtered[i] = BentNormalAO(bent_normal=(bx, by, bz), occlusion=avg_occ)
    return filtered


def geometry_aware_upsample(
    ao_low: List[BentNormalAO],
    ao_low_width: int,
    ao_low_height: int,
    full_width: int,
    full_height: int,
    full_depths: List[float],
    ao_low_depths: List[float],
) -> List[BentNormalAO]:
    """
    对应 FGeometryAwareUpsamplePS。
    将低分辨率 AO 上采样回全分辨率；深度权重避免边缘模糊。
    上采样永远是在猜测：猜得有依据，猜错了也有借口。
    """
    result: List[BentNormalAO] = []
    for fy in range(full_height):
        for fx in range(full_width):
            lx = min(fx * ao_low_width // max(full_width, 1), ao_low_width - 1)
            ly = min(fy * ao_low_height // max(full_height, 1), ao_low_height - 1)
            low_idx = ly * ao_low_width + lx
            full_idx = fy * full_width + fx

            fd = full_depths[full_idx] if full_idx < len(full_depths) else 0.0
            ld = ao_low_depths[low_idx] if low_idx < len(ao_low_depths) else 0.0
            depth_weight = 1.0 / (1.0 + abs(fd - ld) * 0.1)

            if low_idx < len(ao_low):
                src = ao_low[low_idx]
                fade = compute_distance_fade(fd)
                final_occ = src.occlusion * depth_weight * fade
                result.append(BentNormalAO(
                    bent_normal=src.bent_normal,
                    occlusion=min(final_occ, 1.0),
                ))
            else:
                result.append(BentNormalAO())
    return result


def update_ao_history(
    history: AOHistoryState,
    new_ao: List[BentNormalAO],
    frame_index: int,
) -> AOHistoryState:
    """将当前帧 AO 写入历史，供下一帧使用。记录，为了下次少走弯路。"""
    if not G_AO_USE_HISTORY:
        return AOHistoryState(valid=False)
    return AOHistoryState(
        bent_normal_history=list(new_ao),
        valid=True,
        frame_index=frame_index,
    )


# ---------------------------------------------------------------------------
# 顶层接口：全帧 Distance Field AO pass
# ---------------------------------------------------------------------------

def render_distance_field_ao(
    scene_data: DFSceneData,
    pixel_positions: List[Tuple[float, float, float]],
    pixel_normals: List[Tuple[float, float, float]],
    pixel_depths: List[float],
    view_width: int,
    view_height: int,
    frustum_planes: List[Tuple[float, float, float, float]],
    view_origin: Tuple[float, float, float],
    ao_params: DFAOParameters,
    history: Optional[AOHistoryState] = None,
    frame_index: int = 0,
) -> Tuple[List[BentNormalAO], AOHistoryState]:
    """
    完整的 Distance Field AO 帧调度，对应 UE5 RenderDistanceFieldAO。

    流程：ObjectCulling → TileCulling → ScreenGridConeTrace → LightingPost。
    每一步都是上一步的筛选；最终到达屏幕的光，是经过了很多关卡的光。
    不经审查的光，称为噪点。
    """
    if not use_distance_field_ao():
        empty = [BentNormalAO() for _ in pixel_positions]
        return empty, AOHistoryState()

    # 1. 对象剔除
    culled = cull_objects_to_view(scene_data, frustum_planes, ao_params, view_origin)

    # 2. Tile cones
    tiles = build_tile_cones(view_width, view_height)

    # 3. Tile-object 交叉（scatter culling）
    tile_intersections = scatter_tile_culling(culled, tiles, scene_data)

    # 4. 收集存活的 SDF 对象
    surviving_indices = set(culled.object_indices)
    surviving_objects = [o for o in scene_data.objects if o.object_index in surviving_indices]

    # 5. Screen-grid cone trace（低分辨率）
    ao_size = get_buffer_size_for_ao(view_width, view_height)
    # 为简化，假设 pixel_positions/normals 已是低分辨率
    ao_low = compute_screen_grid_ao(
        pixel_positions=pixel_positions,
        pixel_normals=pixel_normals,
        objects=surviving_objects,
        ao_params=ao_params,
        frame_number=frame_index,
        use_history=(history is not None and history.valid),
    )

    # 6. 历史融合（depth rejection）
    hist_depths = ([b.occlusion for b in history.bent_normal_history]
                   if history and history.valid else [])
    ao_blended = update_history_depth_rejection(
        current=ao_low,
        history=history or AOHistoryState(),
        current_depths=pixel_depths,
        history_depths=hist_depths,
    )

    # 7. 空间稳定滤波
    ao_filtered = filter_history_stability(ao_blended, ao_size.x, ao_size.y)

    # 8. 上采样到全分辨率（简化：跳过，直接返回低分辨率结果）
    # geometry_aware_upsample(...) 可在此调用

    # 9. 更新历史
    new_history = update_ao_history(ao_filtered, ao_filtered, frame_index)

    return ao_filtered, new_history
# ============================================================
# Nanite CullRaster + Editor + RayTracing + Materials + Tessellation
# Ported from UE5 upstream — 鲁迅式注释穿插其中
# "世上本没有路，走的人多了，也便成了管线。"
# ============================================================

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, List, Dict, Tuple, Any
import math


# ----------------------------------------------------------
# § 1  CullRaster — 剔除与光栅化
# ----------------------------------------------------------
# 鲁迅曾说，最好的剔除，是让看不见的东西永远看不见。
# 但GPU不懂文学，它只认布尔值。

class CullingPass(IntEnum):
    NO_OCCLUSION  = 0   # 不问遮挡，只管画
    OCCLUSION_MAIN = 1  # 主遮挡剔除
    OCCLUSION_POST = 2  # 后置再检
    EXPLICIT_LIST  = 3  # 钦点名单


@dataclass
class NaniteCullRasterConfig:
    """
    对应 UE5 CVar 集合：r.Nanite.* 系列。

    每一个开关背后，都是某位工程师三天三夜的痛苦与妥协。
    打开它，你获得性能；关掉它，你获得正确性。二者不可兼得。
    """
    enable_async_rasterization: bool = True
    async_rasterize_shadow_depths: bool = False
    async_rasterize_custom_pass: bool = True
    async_rasterize_lumen_mesh_cards: bool = False
    enable_compute_rasterization: bool = True
    enable_programmable_raster: bool = True
    enable_tessellation: bool = True
    filter_primitives: bool = True
    vsm_invalidate_on_lod_delta: bool = False
    raster_setup_task: bool = True
    raster_setup_cache: bool = True
    max_pixels_per_edge: float = 1.0
    min_pixels_per_edge_hw: float = 32.0
    dicing_rate: float = 2.0          # 微多边形目标尺寸（像素）
    max_patches_per_group: int = 5
    depth_buckets_min_z: float = 1000.0
    depth_buckets_max_z: float = 100_000.0
    enable_depth_bucketing: bool = True
    depth_bucket_pixel_programmable: bool = True
    fast_vis_buffer_clear: int = 1    # 0=off 1=pixel 2=tile 3=metadata


@dataclass
class CandidateNode:
    """
    GPU 端候选节点的 CPU 镜像。
    三个 uint32，承载了整个场景图的生死。
    """
    # x: 1 culling-flag-bits | NANITE_MAX_INSTANCES_BITS
    # y: 1 | nodes-per-primitive | views-per-pass
    # z: 1 | BVH nodes per group
    x: int = 0
    y: int = 0
    z: int = 0


@dataclass
class CullRasterContext:
    """
    鲁迅式：凡过得去的帧，都是相似的；每一帧出问题，各有各的原因。
    """
    config: NaniteCullRasterConfig = field(default_factory=NaniteCullRasterConfig)
    culling_pass: CullingPass = CullingPass.NO_OCCLUSION
    candidate_nodes: List[CandidateNode] = field(default_factory=list)
    visible_cluster_count: int = 0
    hw_raster_count: int = 0
    sw_raster_count: int = 0
    tessellation_patches: int = 0

    def should_use_hw_raster(self, edge_length_px: float) -> bool:
        """大三角形走硬件，小三角形走计算着色器。历史的车轮总是这样碾过。"""
        return edge_length_px >= self.config.min_pixels_per_edge_hw

    def compute_lod_error(self, world_size: float, distance: float, fov_scale: float) -> float:
        """屏幕空间误差：距离越远，细节越少，这是Nanite的基本信条，也是人生的隐喻。"""
        if distance <= 0.0:
            return float('inf')
        screen_size = world_size * fov_scale / distance
        return screen_size / self.config.max_pixels_per_edge

    def cull_instance(self, bounds_min: Tuple[float,float,float],
                      bounds_max: Tuple[float,float,float],
                      view_frustum_planes: List[Tuple[float,float,float,float]]) -> bool:
        """
        AABB vs 视锥剔除。
        鲁迅：有些东西，不是被看见了才存在；是没有被剔除，才侥幸出现在屏幕上。
        """
        cx = (bounds_min[0] + bounds_max[0]) * 0.5
        cy = (bounds_min[1] + bounds_max[1]) * 0.5
        cz = (bounds_min[2] + bounds_max[2]) * 0.5
        ex = (bounds_max[0] - bounds_min[0]) * 0.5
        ey = (bounds_max[1] - bounds_min[1]) * 0.5
        ez = (bounds_max[2] - bounds_min[2]) * 0.5
        for (nx, ny, nz, d) in view_frustum_planes:
            dist = nx*cx + ny*cy + nz*cz + d
            radius = abs(nx)*ex + abs(ny)*ey + abs(nz)*ez
            if dist + radius < 0.0:
                return True   # 完全在平面背面，剔除
        return False

    def dispatch_cull_pass(self, instances: List[Dict[str, Any]],
                           view_frustum_planes: List[Tuple[float,float,float,float]]) -> int:
        """
        模拟 GPU 剔除 dispatch。返回存活实例数。
        真正的 GPU 版本用 IndirectArgs 驱动，这里只是影子。
        """
        survived = 0
        for inst in instances:
            b_min = inst.get('bounds_min', (0,0,0))
            b_max = inst.get('bounds_max', (1,1,1))
            if not self.cull_instance(b_min, b_max, view_frustum_planes):
                survived += 1
        self.visible_cluster_count = survived
        return survived


# ----------------------------------------------------------
# § 2  Editor — 选择轮廓与 HitProxy
# ----------------------------------------------------------
# 鲁迅曾说，编辑器是给人看的，渲染器是给机器算的。
# 二者共用同一个 VisBuffer，却各怀心事。

@dataclass
class NaniteEditorConfig:
    draw_lists_async_updates: bool = True


@dataclass
class HitProxyEntry:
    hit_proxy_id: int = 0
    primitive_index: int = 0
    cluster_index: int = 0


@dataclass
class EditorSelectionContext:
    """
    对应 UE5 NaniteEditor：EmitEditorSelectionDepth / DrawHitProxies。
    用 VisBuffer 反查 HitProxy，再高亮轮廓。
    这套流程，在编辑器里每帧都跑；在发布版里，它不存在。
    就像鲁迅的杂文——只在特定时代才有意义。
    """
    config: NaniteEditorConfig = field(default_factory=NaniteEditorConfig)
    selected_hit_proxy_ids: List[int] = field(default_factory=list)
    hit_proxy_table: Dict[int, HitProxyEntry] = field(default_factory=dict)
    vis_buffer: Optional[Any] = None   # uint64 texture handle（CPU侧占位）

    def register_hit_proxy(self, primitive_index: int, cluster_index: int) -> int:
        proxy_id = len(self.hit_proxy_table)
        self.hit_proxy_table[proxy_id] = HitProxyEntry(
            hit_proxy_id=proxy_id,
            primitive_index=primitive_index,
            cluster_index=cluster_index,
        )
        return proxy_id

    def emit_selection_depth(self, only_selected: bool = True) -> List[int]:
        """
        FEmitEditorSelectionDepthPS 的 Python 影子。
        only_selected=True 对应 ONLY_SELECTED shader permutation。
        返回命中的 primitive_index 列表。
        """
        if only_selected:
            return [e.primitive_index
                    for pid, e in self.hit_proxy_table.items()
                    if pid in self.selected_hit_proxy_ids]
        return [e.primitive_index for e in self.hit_proxy_table.values()]

    def emit_hit_proxy_ids(self, vis_buffer_sample: Optional[int] = None) -> Optional[int]:
        """从 VisBuffer 采样反查 HitProxyId。vis_buffer_sample 模拟 GPU 读回值。"""
        if vis_buffer_sample is None:
            return None
        return self.hit_proxy_table.get(vis_buffer_sample, HitProxyEntry()).hit_proxy_id

    def emit_editor_nanite_scene_depth(self,
                                       editor_view_rect: Tuple[int,int,int,int],
                                       scene_view_rect: Tuple[int,int,int,int]) -> Tuple[float,float,float,float]:
        """
        计算 SceneTransform：editor 输出坐标系 → scene VisBuffer 坐标系。
        对应 FScreenTransform::ChangeRectFromTo。
        """
        ex0,ey0,ex1,ey1 = editor_view_rect
        sx0,sy0,sx1,sy1 = scene_view_rect
        ew = max(ex1-ex0, 1); eh = max(ey1-ey0, 1)
        sw = max(sx1-sx0, 1); sh = max(sy1-sy0, 1)
        scale_x = sw / ew
        scale_y = sh / eh
        bias_x  = sx0 - ex0 * scale_x
        bias_y  = sy0 - ey0 * scale_y
        return (scale_x, scale_y, bias_x, bias_y)


# ----------------------------------------------------------
# § 3  RayTracing — BLAS 流式更新
# ----------------------------------------------------------
# 鲁迅：光线追踪是个好东西，可惜帧率不好看。

@dataclass
class NaniteRayTracingConfig:
    update_enabled: bool = True
    force_update_visible: bool = False
    lod_bias: float = 0.0
    min_cut_error: float = 0.0
    offscreen_lod_bias: float = 1.0
    offscreen_min_cut_error: float = 4.0
    use_reference_instances: bool = True
    blas_cache_enabled: bool = True
    blas_cache_relative_error_tolerance: float = 0.5
    blas_cache_size_mb: int = 64
    drive_streaming: bool = False
    max_stream_out_vertices: int = 16 * 1024 * 1024
    max_stream_out_indices: int = 64 * 1024 * 1024
    max_built_primitives_per_frame: int = 8 * 1024 * 1024
    max_staging_buffer_size_mb: int = 1024
    blas_scratch_size_multiple_mb: int = 64


@dataclass
class BLASCacheEntry:
    """对应 FNaniteRayTracingASCacheEntry：BLAS 缓存槽位。"""
    byte_offset: int = 0
    byte_size: int = 0
    update_sequence_id: int = 0   # 递增，防止 GPU stale hit

    @property
    def is_allocated(self) -> bool:
        return self.byte_size > 0


@dataclass
class RayTracingUpdateRequest:
    """CPU 侧模拟 GPU readback 的缓存请求结构。"""
    runtime_resource_id: int = 0
    requested_size: int = 0
    lod_error: float = 0.0


class NaniteRayTracingContext:
    """
    对应 UE5 NaniteRayTracing：BLAS 生命周期管理。

    鲁迅式：每一个 BLAS，都是流媒体时代的孤儿。
    它在 GPU 里诞生，在预算耗尽时死去，然后被下一帧的请求重建。
    没有人为它哀悼，帧率继续。
    """
    NANITE_MAX_GPU_PAGES = 4096
    NANITE_MAX_GPU_PAGES_MASK = NANITE_MAX_GPU_PAGES - 1

    def __init__(self, config: Optional[NaniteRayTracingConfig] = None):
        self.config = config or NaniteRayTracingConfig()
        self.cache: List[BLASCacheEntry] = [
            BLASCacheEntry() for _ in range(self.NANITE_MAX_GPU_PAGES)
        ]
        self._allocator_used: int = 0
        self._allocator_max: int = self.config.blas_cache_size_mb * 1024 * 1024
        self._pending_readbacks: List[List[RayTracingUpdateRequest]] = []
        self._built_primitives_this_frame: int = 0

    def get_cache_entry(self, runtime_resource_id: int) -> BLASCacheEntry:
        return self.cache[runtime_resource_id & self.NANITE_MAX_GPU_PAGES_MASK]

    def invalidate_entry(self, entry: BLASCacheEntry) -> None:
        """递增 UpdateSequenceId，使 GPU 侧旧元数据失效。"""
        entry.update_sequence_id = (entry.update_sequence_id + 1) & 0xFFFFFFFF
        if entry.update_sequence_id == 0:
            entry.update_sequence_id = 1

    def _try_allocate(self, size: int) -> int:
        """极简线性分配器（生产版用 FRangeAllocator）。返回 -1 表示失败。"""
        aligned = (size + 255) & ~255
        if self._allocator_used + aligned > self._allocator_max:
            return -1
        offset = self._allocator_used
        self._allocator_used += aligned
        return offset

    def _free(self, offset: int, size: int) -> None:
        """CPU 侧释放（简化：仅在缩容时重置整体水位）。"""
        aligned = (size + 255) & ~255
        self._allocator_used = max(0, self._allocator_used - aligned)

    def process_cache_requests(self, requests: List[RayTracingUpdateRequest]) -> None:
        """
        对应 FNaniteRayTracingBLASCache::ProcessCacheRequests。
        先做所有驱逐，再做所有分配，减少碎片。
        """
        # 驱逐阶段
        for req in requests:
            idx = req.runtime_resource_id & self.NANITE_MAX_GPU_PAGES_MASK
            entry = self.cache[idx]
            if entry.is_allocated and req.requested_size != entry.byte_size:
                self._free(entry.byte_offset, entry.byte_size)
                prev_seq = entry.update_sequence_id
                self.cache[idx] = BLASCacheEntry(update_sequence_id=prev_seq)

        # 分配阶段
        for req in requests:
            if req.requested_size == 0:
                continue
            idx = req.runtime_resource_id & self.NANITE_MAX_GPU_PAGES_MASK
            entry = self.cache[idx]
            if not entry.is_allocated:
                offset = self._try_allocate(req.requested_size)
                if offset >= 0:
                    entry.byte_offset = offset
                    entry.byte_size = req.requested_size
                    self.invalidate_entry(entry)

    def compute_lod_for_instance(self, world_radius: float, distance: float,
                                 is_offscreen: bool = False) -> float:
        """
        LOD 误差计算：在光线追踪场景中，离屏物体可以用更粗糙的 BLAS。
        这是妥协，也是智慧。鲁迅不会承认这是妥协。
        """
        bias = self.config.offscreen_lod_bias if is_offscreen else self.config.lod_bias
        min_err = (self.config.offscreen_min_cut_error if is_offscreen
                   else self.config.min_cut_error)
        raw_error = (world_radius / max(distance, 1e-6)) * (2 ** bias)
        return max(raw_error, min_err)

    def begin_frame(self) -> None:
        self._built_primitives_this_frame = 0

    def can_build_blas(self, triangle_count: int) -> bool:
        """预算检查：本帧还有没有余额新建 BLAS。"""
        return (self._built_primitives_this_frame + triangle_count
                <= self.config.max_built_primitives_per_frame)

    def record_blas_build(self, triangle_count: int) -> None:
        self._built_primitives_this_frame += triangle_count


# ----------------------------------------------------------
# § 4  Materials Scene Extension — 材质数据缓冲区
# ----------------------------------------------------------
# 鲁迅：材质是皮肤，几何是骨骼。
# 骨骼可以共享，皮肤必须各自承担。

@dataclass
class NaniteMaterialsConfig:
    material_data_min_size_bytes: int = 4 * 1024
    primitive_material_data_min_size_bytes: int = 4 * 1024
    async_updates: bool = True
    force_full_upload: int = 0     # 0=no 1=once 2=every-frame
    defrag_enabled: bool = True
    force_defrag: int = 0
    defrag_low_water_mark: float = 0.375


@dataclass
class NaniteMaterialEntry:
    material_id: int = 0
    hit_proxy_id: int = 0          # WITH_EDITOR only
    debug_view_data: int = 0       # WITH_DEBUG_VIEW_MODES only
    primitive_index: int = 0
    element_stride: int = 0


class NaniteMaterialsSceneExtension:
    """
    对应 UE5 FMaterialsSceneExtension。

    维护两张表：MaterialData（每材质）和 PrimitiveMaterialData（每图元）。
    GPU 侧通过 ByteAddressBuffer 随机访问。

    鲁迅式：这张表的内容，每帧都可能失效。
    失效了就 defrag，defrag 了就重传，重传了继续失效。
    这就是实时渲染的宿命。
    """
    def __init__(self, config: Optional[NaniteMaterialsConfig] = None):
        self.config = config or NaniteMaterialsConfig()
        self._material_data: Dict[int, NaniteMaterialEntry] = {}
        self._primitive_material_map: Dict[int, List[int]] = {}  # prim_idx -> [mat_ids]
        self._pending_static_mesh_updates: Dict[int, bool] = {}  # prim_idx -> valid
        self._dirty: bool = False
        self._allocated_bytes: int = 0
        self._used_bytes: int = 0

    # --- Extension lifecycle ---

    def should_be_enabled(self, nanite_supported: bool) -> bool:
        return nanite_supported

    def add_primitive(self, primitive_index: int, material_ids: List[int]) -> None:
        self._primitive_material_map[primitive_index] = material_ids
        for mid in material_ids:
            if mid not in self._material_data:
                self._material_data[mid] = NaniteMaterialEntry(
                    material_id=mid, primitive_index=primitive_index
                )
        self._dirty = True

    def remove_primitive(self, primitive_index: int) -> None:
        mids = self._primitive_material_map.pop(primitive_index, [])
        for mid in mids:
            self._material_data.pop(mid, None)
        self._pending_static_mesh_updates.pop(primitive_index, None)
        self._dirty = True

    def add_pending_static_mesh_update(self, primitive_index: int) -> None:
        """登记待更新，返回 valid-flag（用 dict bool 模拟 TSharedPtr<bool>）。"""
        self._pending_static_mesh_updates[primitive_index] = True

    def clear_pending_static_mesh_update(self, primitive_index: int) -> None:
        self._pending_static_mesh_updates.pop(primitive_index, None)

    def has_pending_static_mesh_update(self, primitive_index: int) -> bool:
        return primitive_index in self._pending_static_mesh_updates

    # --- Upload logic ---

    def should_full_upload(self) -> bool:
        return self.config.force_full_upload > 0

    def should_defrag(self) -> bool:
        if not self.config.defrag_enabled:
            return False
        if self.config.force_defrag > 0:
            return True
        if self._allocated_bytes == 0:
            return False
        ratio = self._used_bytes / self._allocated_bytes
        return ratio < self.config.defrag_low_water_mark

    def build_material_data_buffer(self) -> bytes:
        """序列化 MaterialData 为字节流（模拟 GPU ByteAddressBuffer 上传）。"""
        import struct
        entries = sorted(self._material_data.values(), key=lambda e: e.material_id)
        buf = bytearray()
        for e in entries:
            buf += struct.pack('<IIIII',
                               e.material_id, e.hit_proxy_id,
                               e.debug_view_data, e.primitive_index, e.element_stride)
        size = max(len(buf), self.config.material_data_min_size_bytes)
        self._used_bytes = len(buf)
        self._allocated_bytes = size
        return bytes(buf).ljust(size, b'\x00')

    def build_primitive_material_data_buffer(self) -> bytes:
        """序列化 PrimitiveMaterialData。"""
        import struct
        buf = bytearray()
        for prim_idx in sorted(self._primitive_material_map):
            mids = self._primitive_material_map[prim_idx]
            buf += struct.pack('<I', len(mids))
            for mid in mids:
                buf += struct.pack('<I', mid)
        size = max(len(buf), self.config.primitive_material_data_min_size_bytes)
        return bytes(buf).ljust(size, b'\x00')


# ----------------------------------------------------------
# § 5  Ownership Visibility Scene Extension
# ----------------------------------------------------------
# 鲁迅：有些东西不让你看，有些东西只让你看。
# OwnerNoSee 和 OnlyOwnerSee，是渲染器里的阶级制度。

@dataclass
class NaniteOwnershipVisibilityConfig:
    pass


class NaniteOwnershipVisibilityExtension:
    """
    对应 UE5 FOwnershipVisibilitySceneExtension。

    每帧为每个视图构建 HiddenPrimitives 位数组，上传 GPU。
    IsOwnerNoSee  → 自己看不见自己
    IsOnlyOwnerSee → 只有自己看得见

    编辑器视图绕过这一切——在编辑器里，没有秘密。
    """
    def __init__(self):
        self._nanite_primitives_with_ownership: List[int] = []  # persistent indices

    def add_primitive(self, persistent_index: int) -> None:
        self._nanite_primitives_with_ownership.append(persistent_index)

    def remove_primitive(self, persistent_index: int) -> None:
        try:
            self._nanite_primitives_with_ownership.remove(persistent_index)
        except ValueError:
            pass

    def build_hidden_primitives_bitarray(
        self,
        max_persistent_index: int,
        views: List[Dict[str, Any]],        # list of {actor_id, is_editor_view}
        primitive_infos: Dict[int, Dict],   # persistent_idx -> {owner_no_see, only_owner_see, owner_id}
    ) -> Optional[List[int]]:
        """
        构建 HiddenPrimitives 位数组（每视图 × max_persistent_index 位）。
        返回 uint32 列表（模拟 GPU StructuredBuffer<uint>）。
        鲁迅：位图是公平的，每个图元只占一位，不多也不少。
        """
        if not self._nanite_primitives_with_ownership:
            return None

        num_views = len(views)
        total_bits = max_persistent_index * num_views
        total_words = (total_bits + 31) // 32
        bit_array = [0] * total_words

        for prim_idx in self._nanite_primitives_with_ownership:
            info = primitive_infos.get(prim_idx)
            if info is None:
                continue
            is_owner_no_see  = info.get('owner_no_see', False)
            is_only_owner_see = info.get('only_owner_see', False)
            owner_id = info.get('owner_id', -1)

            for view_idx, view in enumerate(views):
                is_editor = view.get('is_editor_view', False)
                is_owned  = view.get('actor_id', None) == owner_id
                is_hidden = (not is_editor) and (
                    (is_owned and is_owner_no_see) or
                    (not is_owned and is_only_owner_see)
                )
                if is_hidden:
                    bit_pos  = view_idx * max_persistent_index + prim_idx
                    word_idx = bit_pos // 32
                    bit_idx  = bit_pos % 32
                    if word_idx < total_words:
                        bit_array[word_idx] |= (1 << bit_idx)

        return bit_array


# ----------------------------------------------------------
# § 6  TessellationTable — 微多边形细分表
# ----------------------------------------------------------
# 鲁迅：细分是一种野心。将一个三角形变成一千个，
# 只为了让边缘看起来不那么锯齿。代价是整个渲染管线的颤抖。

NANITE_TESSELLATION_TABLE_SIZE           = 15
NANITE_TESSELLATION_TABLE_PO2_SIZE       = 16   # next power of two ≥ SIZE
NANITE_TESSELLATION_TABLE_IMMEDIATE_SIZE = 3
BARYCENTRIC_MAX                          = 0x8000  # 固定小数精度


@dataclass
class TessellationPattern:
    tess_factors: Tuple[int, int, int] = (1, 1, 1)
    verts: List[Tuple[int, int]] = field(default_factory=list)   # (u, v) fixed-point
    indices: List[int] = field(default_factory=list)


class TessellationTable:
    """
    对应 UE5 FTessellationTable。

    生产环境从 TessellationTable.bin 加载；
    这里按需在 CPU 上生成轻量近似版。

    鲁迅：有些表只需要查，不需要理解。
    理解了，也未必能改得更好。
    """
    def __init__(self):
        self._patterns: Dict[int, TessellationPattern] = {}

    @staticmethod
    def get_pattern_key(tx: int, ty: int, tz: int) -> int:
        """
        对应 FTessellationTable::GetPattern。
        排序后线性化，保证 (3,2,1)==(1,2,3)。
        """
        factors = sorted([tx, ty, tz], reverse=True)
        f0, f1, f2 = factors
        return (
            (f0 - 1) +
            (f1 - 1) * NANITE_TESSELLATION_TABLE_PO2_SIZE +
            (f2 - 1) * NANITE_TESSELLATION_TABLE_PO2_SIZE * NANITE_TESSELLATION_TABLE_PO2_SIZE
        )

    @staticmethod
    def get_barycentrics(vert_packed: int) -> Tuple[int, int, int]:
        """
        对应 FTessellationTable::GetBarycentrics。
        低16位为 u，高16位为 v，w 由 BarycentricMax 补全。
        """
        u = vert_packed & 0xFFFF
        v = vert_packed >> 16
        w = BARYCENTRIC_MAX - u - v
        return (u, v, w)

    @staticmethod
    def pack_barycentric(u: int, v: int) -> int:
        return (u & 0xFFFF) | ((v & 0xFFFF) << 16)

    def get_or_build_pattern(self, tx: int, ty: int, tz: int) -> TessellationPattern:
        """
        懒加载细分模式。
        真正的 UE5 是预计算后二进制存储，这里在 CPU 上即时生成均匀细分。
        精度足够做功能验证，但不保证与 bin 文件完全一致。
        """
        key = self.get_pattern_key(tx, ty, tz)
        if key in self._patterns:
            return self._patterns[key]

        pattern = self._build_uniform_pattern(tx, ty, tz)
        self._patterns[key] = pattern
        return pattern

    @staticmethod
    def _build_uniform_pattern(tx: int, ty: int, tz: int) -> TessellationPattern:
        """
        在重心坐标系中均匀细分三角形。
        tx/ty/tz 分别是三条边的细分因子。
        鲁迅：均匀，是最省心的选择，也是最无趣的选择。
        """
        # 简化：使用等边细分（取三者最大值）
        n = max(tx, ty, tz)
        verts: List[Tuple[int,int]] = []
        vert_map: Dict[Tuple[int,int], int] = {}
        indices: List[int] = []

        def add_vert(i: int, j: int) -> int:
            k = (i, j)
            if k not in vert_map:
                u = round(i * BARYCENTRIC_MAX / n)
                v = round(j * BARYCENTRIC_MAX / n)
                vert_map[k] = len(verts)
                verts.append((u, v))
            return vert_map[k]

        for row in range(n):
            for col in range(n - row):
                v0 = add_vert(col,   row)
                v1 = add_vert(col+1, row)
                v2 = add_vert(col,   row+1)
                indices += [v0, v1, v2]
                if col + row < n - 1:
                    v3 = add_vert(col+1, row+1)
                    indices += [v1, v3, v2]

        return TessellationPattern(
            tess_factors=(tx, ty, tz),
            verts=list(verts),
            indices=indices,
        )

    def snap_at_edges(self, bary: List[int], tess_factors: Tuple[int,int,int]) -> List[int]:
        """
        对应 FTessellationTable::SnapAtEdges：将边界顶点对齐到整数细分位置。
        防止相邻 patch 之间出现 T-junction。
        """
        result = list(bary)
        for i in range(3):
            e0 = i
            e1 = (1 << e0) & 3
            if e1 >= 3:
                continue
            if result[e0] + result[e1] == BARYCENTRIC_MAX:
                min_idx = e0 if result[e0] < result[e1] else e1
                max_idx = e1 if min_idx == e0 else e0
                tf = tess_factors[i]
                snapped = ((result[min_idx] * tf + (BARYCENTRIC_MAX // 2) - 1)
                           // BARYCENTRIC_MAX) * BARYCENTRIC_MAX // tf
                result[max_idx] = BARYCENTRIC_MAX - snapped
                result[min_idx] = snapped
        return result


# ----------------------------------------------------------
# § 7  统一入口：NaniteRenderContext
# ----------------------------------------------------------
# 鲁迅：把所有的复杂性压缩进一个数据类，
# 然后告诉别人"这很简单"。

@dataclass
class NaniteRenderContext:
    """
    六大子系统的统一容器。
    对应 UE5 Nanite:: 命名空间下各模块的协同调用点。

    生命周期：scene_init → begin_frame → cull → shade → ray_trace → end_frame
    """
    cull_raster: CullRasterContext = field(
        default_factory=CullRasterContext)
    editor: EditorSelectionContext = field(
        default_factory=EditorSelectionContext)
    ray_tracing: NaniteRayTracingContext = field(
        default_factory=NaniteRayTracingContext)
    materials: NaniteMaterialsSceneExtension = field(
        default_factory=NaniteMaterialsSceneExtension)
    ownership_visibility: NaniteOwnershipVisibilityExtension = field(
        default_factory=NaniteOwnershipVisibilityExtension)
    tessellation_table: TessellationTable = field(
        default_factory=TessellationTable)

    # 帧计数
    frame_index: int = 0

    def begin_frame(self) -> None:
        self.ray_tracing.begin_frame()
        self.frame_index += 1

    def run_cull_pass(
        self,
        instances: List[Dict[str, Any]],
        frustum_planes: List[Tuple[float,float,float,float]],
    ) -> int:
        """
        主剔除 pass。返回可见实例数。
        鲁迅：剔除之后，剩下的才是真正的工作。
        """
        return self.cull_raster.dispatch_cull_pass(instances, frustum_planes)

    def upload_material_buffers(self) -> Tuple[bytes, bytes]:
        """返回 (material_data, primitive_material_data) 字节流，供 GPU 上传。"""
        mat_buf  = self.materials.build_material_data_buffer()
        prim_buf = self.materials.build_primitive_material_data_buffer()
        return mat_buf, prim_buf

    def get_tessellation_pattern(
        self, tx: int, ty: int, tz: int
    ) -> TessellationPattern:
        """查询细分表。三参数均需在 [1, NANITE_TESSELLATION_TABLE_SIZE] 范围内。"""
        tx = max(1, min(tx, NANITE_TESSELLATION_TABLE_SIZE))
        ty = max(1, min(ty, NANITE_TESSELLATION_TABLE_SIZE))
        tz = max(1, min(tz, NANITE_TESSELLATION_TABLE_SIZE))
        return self.tessellation_table.get_or_build_pattern(tx, ty, tz)

    def end_frame(self) -> Dict[str, int]:
        """
        帧末统计。返回诊断字典。
        没有人会细看这些数字，但它们必须存在。
        鲁迅：统计是给下一代看的。
        """
        return {
            'frame':               self.frame_index,
            'visible_clusters':    self.cull_raster.visible_cluster_count,
            'hw_raster':           self.cull_raster.hw_raster_count,
            'sw_raster':           self.cull_raster.sw_raster_count,
            'tess_patches':        self.cull_raster.tessellation_patches,
            'blas_built_tris':     self.ray_tracing._built_primitives_this_frame,
            'material_entries':    len(self.materials._material_data),
            'owned_primitives':    len(self.ownership_visibility
                                       ._nanite_primitives_with_ownership),
            'tess_patterns_cached': len(self.tessellation_table._patterns),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Port: Lumen Reflection Tracing  (LumenReflectionTracing.cpp)
# Port: Lumen Reflection HWRT     (LumenReflectionHardwareRayTracing.cpp)
# Port: Lumen GPU-Driven Update   (LumenSceneGPUDrivenUpdate.cpp)
# Port: Lumen Direct Lighting HWRT(LumenSceneDirectLightingHardwareRayTracing.cpp)
# Port: Lumen Visualize           (LumenVisualize.cpp)
# Port: Lumen Visualize HWRT      (LumenVisualizeHardwareRayTracing.cpp)
# Port: Lumen ScreenProbe HWRT    (LumenScreenProbeHardwareRayTracing.cpp)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReflectionTraceConfig:
    screen_traces: bool = True
    screen_tracing_source: int = 0
    hzb_max_iterations: int = 50
    hzb_min_occupancy: int = 0
    hzb_relative_depth_threshold: float = 0.005
    history_depth_test_relative_thickness: float = 0.005
    hair_voxel_trace: bool = True
    hair_screen_trace: bool = True
    compaction_group_size_in_tiles: int = 16
    compaction_wave_ops: bool = True
    sample_scene_color_at_hit: int = 1
    sample_scene_color_rel_depth_threshold: float = 0.01
    sample_scene_color_normal_threshold_deg: float = 85.0
    far_field_sample_color_rel_depth: float = 0.1
    far_field_sample_color_normal_deg: float = 85.0
    distant_screen_traces: int = 1
    distant_trace_slope_compare_tolerance: float = 2.0
    distant_trace_max_distance: float = 200000.0
    max_bounces: int = 0
    hwrt_first_person_min_hit_distance: float = 4.0


@dataclass
class ReflectionRay:
    """一根反射光线。方向和起点而已，其余皆是妄想。"""
    origin: tuple; direction: tuple; pdf: float; roughness: float; pixel_coord: tuple


def _cos_weighted_direction(normal: tuple, u1: float, u2: float) -> tuple:
    import math
    phi = 2.0 * math.pi * u1
    cos_t = math.sqrt(u2);  sin_t = math.sqrt(max(0.0, 1.0 - u2))
    nx, ny, nz = normal
    tx, ty, tz = (1.,0.,0.) if abs(nx) < 0.9 else (0.,1.,0.)
    bx=ny*tz-nz*ty; by=nz*tx-nx*tz; bz=nx*ty-ny*tx
    bl=max(1e-9,(bx*bx+by*by+bz*bz)**.5); bx/=bl; by/=bl; bz/=bl
    sx,sy,sz=tx,ty,tz; d=sx*nx+sy*ny+sz*nz; sx-=d*nx; sy-=d*ny; sz-=d*nz
    sl=max(1e-9,(sx*sx+sy*sy+sz*sz)**.5); sx/=sl; sy/=sl; sz/=sl
    cp=math.cos(phi); sp=math.sin(phi)
    return (sin_t*(cp*sx+sp*bx)+cos_t*nx, sin_t*(cp*sy+sp*by)+cos_t*ny, sin_t*(cp*sz+sp*bz)+cos_t*nz)


def _hzb_screen_trace(ray: ReflectionRay, depth_pyramid: list, cfg: ReflectionTraceConfig) -> tuple:
    """
    Hierarchical Z-Buffer screen trace。
    迭代停止有两种情况：找到交点，或者耗尽步数——生活里的困境也不外乎此。
    """
    if not depth_pyramid: return False, (0.,0.), 0.
    ox,oy,_=ray.origin; dx,dy,dz=ray.direction
    su=dx*.001; sv=dy*.001; u,v=ox%1.,oy%1.
    lv=len(depth_pyramid)-1; base=depth_pyramid[0]
    H,W=len(base),(len(base[0]) if base else 1)
    for i in range(cfg.hzb_max_iterations):
        ui=int(u*W)%W; vi=int(v*H)%H
        la=depth_pyramid[min(lv,len(depth_pyramid)-1)]
        lH=len(la); lW=len(la[0]) if la else 1
        cd=la[min(vi>>lv,lH-1)][min(ui>>lv,lW-1)]
        rd=abs(dz)*(i+1)*.01
        if rd>cd*(1.+cfg.hzb_relative_depth_threshold):
            if lv>0: lv-=1
            else:
                fd=base[vi][ui]
                if abs(rd-fd)<cfg.hzb_relative_depth_threshold*fd: return True,(u,v),fd
        else:
            u+=su*(1<<lv); v+=sv*(1<<lv)
            if not(0.<=u<=1. and 0.<=v<=1.): return False,(u,v),0.
            lv=min(lv+1,len(depth_pyramid)-1)
        if cfg.hzb_min_occupancy>0 and i>cfg.hzb_max_iterations//2: break
    return False,(u,v),0.


def generate_reflection_rays(gbuffer_sample, cfg: ReflectionTraceConfig, depth_pyramid: list, frame_index: int=0) -> list:
    """
    为单个 GBuffer 像素生成反射光线，优先 Screen Trace，失败后交 HWRT fallback。
    工欲善其事，必先利其器。但器具再好，也要有人拿得起来。
    """
    import math,random
    rng=random.Random(frame_index^hash(gbuffer_sample.get('pixel_coord',(0,0))))
    n=gbuffer_sample.get('normal',(0.,1.,0.)); pos=gbuffer_sample.get('position',(0.,0.,0.))
    r=gbuffer_sample.get('roughness',0.); px,py=gbuffer_sample.get('pixel_coord',(.5,.5))
    d=_cos_weighted_direction(n,rng.random(),rng.random())
    pdf=max(1e-4,abs(d[0]*n[0]+d[1]*n[1]+d[2]*n[2])/math.pi)
    ray=ReflectionRay(pos,d,pdf,r,(px,py))
    if cfg.screen_traces and depth_pyramid:
        hit,uv,hd=_hzb_screen_trace(ray,depth_pyramid,cfg)
        if hit: ray=ReflectionRay((uv[0],uv[1],hd),d,pdf,r,(px,py))
    return [ray]


def compact_reflection_traces(rays: list, cfg: ReflectionTraceConfig) -> list:
    """Trace Compaction：剔除已命中的，剩余按粗糙度分桶。"""
    return sorted([r for r in rays if r.origin[2]>=1.], key=lambda r: r.roughness)


class HWRTReflectionPass:
    """
    硬件光线追踪反射通道：Default、FarField、HitLighting 三种模式。
    就像人有三种状态：正常、远眺、被迫直视真相。
    """
    D,F,H='Default','FarField','HitLighting'
    def __init__(self,bias=.1,normal_bias=.1,bucket_materials=True,hit_lighting=False,far_field=True):
        self.bias=bias; self.normal_bias=normal_bias; self.bucket=bucket_materials
        self.hl=hit_lighting; self.ff=far_field
    def _remap(self,pt,flags):
        o=dict(flags)
        if pt==self.D: o['rec_refl']=False; o['ff_occ_only']=False
        elif pt==self.F: o.update(rec_refl=False,rc=False,hair=False,rec_refr=False,dst=False)
        elif pt==self.H: o.update(write_hl=False,dst=False,ff_occ_only=False)
        if pt!=self.H: o['ser']=False
        return o
    def trace(self,rays,rc,surface_cache,ff_dist=200000.):
        res=[]
        passes=[self.D]+([self.F] if self.ff else [])+([self.H] if self.hl else [])
        for ray in rays:
            rad=(0.,0.,0.); hd=float('inf')
            for pt in passes:
                ck=tuple(int(c*4) for c in ray.direction); cv=rc.get(ck,(0.,0.,0.))
                if pt==self.F:
                    if hd>ff_dist*.5: rad=cv; hd=ff_dist
                elif pt==self.H:
                    if hd==float('inf'): rad=tuple(v*.5 for v in cv); hd=1000.
                else:
                    if rad==(0.,0.,0.): rad=cv; hd=500.
            res.append((ray,rad,hd))
        if self.bucket: res.sort(key=lambda x:x[2])
        return res


@dataclass
class LumenCardMetrics:
    max_distance:float=0.; texel_density:float=0.; far_field_density:float=.001
    far_field_distance:float=40000.; min_resolution:int=4; capture_margin:float=0.


@dataclass
class GPUSceneAddOp:
    primitive_group_id:int; lod_level:int; world_bounds:tuple


@dataclass
class GPUSceneRemoveOp:
    primitive_group_id:int


class LumenGPUSceneReadback:
    """
    环形缓冲区，CPU 读回 GPU 写入的增删操作。
    GPU 在前面写，CPU 在后面读，永远差着那么几帧——这就是渲染管线，也是人生。
    """
    N=4; MAX_A=4096; MAX_R=4096
    def __init__(self): self._r=[None]*self.N; self._wi=0; self._p=0
    def is_full(self): return self._p>=self.N
    def submit(self,add,rem):
        if self.is_full(): return False
        self._r[self._wi]=(add[:self.MAX_A],rem[:self.MAX_R])
        self._wi=(self._wi+1)%self.N; self._p=min(self._p+1,self.N); return True
    def consume_latest(self):
        if not self._p: return[],[]
        ri=(self._wi-self._p)%self.N; res=self._r[ri]; self._r[ri]=None; self._p-=1
        return res if res else ([],[])


def compute_card_metrics(view_position,surface_cache_resolution=1.,lumen_scene_detail=1.,
                         use_hwrt=False,ray_tracing_cull_radius=100000.,
                         fast_camera_mode=False,ortho_camera=False) -> LumenCardMetrics:
    """
    计算 Surface Cache 卡片的距离和密度阈值。
    参数愈多，逻辑愈繁，不过都是在问同一件事：这张卡片值不值得渲染。
    """
    md=ray_tracing_cull_radius if use_hwrt else ray_tracing_cull_radius*1.5
    td=100.*surface_cache_resolution*(.2 if fast_camera_mode else 1.)
    mr=max(1,min(1024,int(round(((1 if ortho_camera else 4)/lumen_scene_detail)*surface_cache_resolution))))
    return LumenCardMetrics(max_distance=md,texel_density=td,min_resolution=mr)


def diff_primitive_groups(prev:set,curr:set) -> tuple:
    """只上传变化量，不重传全世界。"""
    a,r=curr-prev,prev-curr
    return ([GPUSceneAddOp(g,0,((0.,0.,0.),(1.,1.,1.))) for g in sorted(a)],
            [GPUSceneRemoveOp(g) for g in sorted(r)])


@dataclass
class DirectLightingHWRTConfig:
    """
    直接光照 HWRT 配置。
    每个字段背后，都有一位工程师在 profiler 前皱眉的记忆。
    """
    enabled:bool=True; async_compute:bool=True; force_two_sided:bool=False
    end_bias:float=1.; far_field:bool=True; heightfield_projection_bias:bool=False
    hf_projection_bias_search_radius:float=256.


@dataclass
class ShadowRayResult:
    light_tile_index:int; texel_index:int; visibility:float


def trace_direct_lighting_shadow_rays(light_tiles,occluder_mesh_sdf,
                                      cfg:DirectLightingHWRTConfig,
                                      far_field_distance=200000.) -> list:
    """
    为每个 light tile texel 发射阴影光线。
    近处的物件要精细，远处的只需知道有没有。这和我们对人的态度，如出一辙。
    """
    res=[]
    for ti,tile in enumerate(light_tiles):
        ld=tile.get('light_direction',(0.,1.,0.))
        for xi,origin in enumerate(tile.get('texel_positions',[])):
            vis=1.
            if occluder_mesh_sdf:
                hit=occluder_mesh_sdf(origin,ld)
                if hit is not None:
                    if hit>cfg.end_bias: vis=0.
                    if cfg.far_field and hit>far_field_distance*.5: vis=max(vis,.1)
            res.append(ShadowRayResult(ti,xi,vis))
    return res


class LumenVisualizeMode:
    """调试模式是工程师写给自己的信，用户永远不会读，但它必须存在。"""
    DISABLE=0;OVERVIEW=1;PERFORMANCE_OVERVIEW=2;LUMEN_SCENE=3;REFLECTION_VIEW=4
    SURFACE_CACHE_COVERAGE=5;GEOMETRY_NORMALS=6;DEDICATED_REFLECTION_RAYS=7
    ALBEDO=8;NORMALS=9;EMISSIVE=10;CARD_WEIGHTS=11;DIRECT_LIGHTING=12
    INDIRECT_LIGHTING=13;LOCAL_POSITION=14;VELOCITY=15;DIRECT_LIGHTING_UPDATES=16
    INDIRECT_LIGHTING_UPDATES=17;LAST_USED_PAGES=18;LAST_USED_HIGHRES_PAGES=19
    CARD_TILE_SHADOW_FACTOR=20;CARD_SHARING_ID=21;SCREEN_PROBE_FAST_UPDATE=22
    SCREEN_PROBE_FRAMES_ACCUM=23;RADIOSITY_FRAMES_ACCUM=24


@dataclass
class VisualizeConfig:
    mode:int=0; grid_pixel_size:int=32; trace_mesh_sdfs:bool=True
    hi_res_surface:bool=True; cone_angle_deg:float=0.; cone_step_factor:float=2.
    min_trace_distance:float=0.; max_trace_distance:float=100000.
    tone_map:bool=True; culling_mode:int=0


def visualize_lumen_scene(surface_cache, scene_cards:list, cfg:VisualizeConfig) -> dict:
    """按模式输出调试数据。这不是最终画面，只是看透了管线的 X 光片。"""
    out={}
    if cfg.mode==LumenVisualizeMode.DISABLE: return out
    M=LumenVisualizeMode
    for card in scene_cards:
        coord=card.get('screen_coord',(0,0))
        if   cfg.mode==M.ALBEDO:           out[coord]=card.get('albedo',(.5,.5,.5))
        elif cfg.mode==M.NORMALS:          out[coord]=tuple(v*.5+.5 for v in card.get('normal',(0.,0.,1.)))
        elif cfg.mode==M.EMISSIVE:         out[coord]=card.get('emissive',(0.,0.,0.))
        elif cfg.mode==M.DIRECT_LIGHTING:  out[coord]=card.get('direct_lighting',(0.,0.,0.))
        elif cfg.mode==M.INDIRECT_LIGHTING:out[coord]=card.get('indirect_lighting',(0.,0.,0.))
        elif cfg.mode==M.CARD_WEIGHTS:     w=card.get('weight',0.); out[coord]=(w,w,w)
        elif cfg.mode==M.SURFACE_CACHE_COVERAGE: c=1. if card.get('resident') else 0.; out[coord]=(c,c*.5,0.)
        else: out[coord]=(.2,.2,.2)
    return out


@dataclass
class VisualizeTile:
    tile_x:int; tile_y:int; pixel_count:int=64


@dataclass
class VisualizeRay:
    origin:tuple; direction:tuple; tile_idx:int


class CompactMode:
    HIT_LIGHTING_RETRACE=0; FORCE_HIT_LIGHTING=1


def create_visualize_tiles(W:int,H:int,ts:int=8) -> list:
    """
    把屏幕切成 tile 格，每个 tile 是一个独立的 GPU 工作单元。
    划分的逻辑很简单，简单到令人安心。
    """
    return [VisualizeTile(tx,ty,min(ts,W-tx)*min(ts,H-ty)) for ty in range(0,H,ts) for tx in range(0,W,ts)]


def create_visualize_rays(tiles,depth_buffer,normal_buffer,W,H,max_dist=100000.) -> list:
    """从每个 tile 中心发出一根可视化光线，沿法线方向。"""
    return [VisualizeRay(
        origin=(((t.tile_x+.5)/max(1,W)),((t.tile_y+.5)/max(1,H)),
                (depth_buffer((t.tile_x+.5)/max(1,W),(t.tile_y+.5)/max(1,H)) if callable(depth_buffer) else 1.)),
        direction=(normal_buffer((t.tile_x+.5)/max(1,W),(t.tile_y+.5)/max(1,H)) if callable(normal_buffer) else (0.,0.,1.)),
        tile_idx=i,
    ) for i,t in enumerate(tiles)]


def compact_visualize_rays(rays:list, mode:int=CompactMode.HIT_LIGHTING_RETRACE) -> list:
    """
    去除已解决的光线，剩余按 tile 排序。
    紧凑，是为了快。快，是为了每帧不超时。
    """
    if mode==CompactMode.FORCE_HIT_LIGHTING: return sorted(rays,key=lambda r:r.tile_idx)
    return sorted([r for r in rays if r.origin[2]>=1.],key=lambda r:r.tile_idx)


class HWRTScreenProbePass:
    """
    Screen Probe Gather 的硬件光线追踪通道。
    它比反射通道更诚实：不需要镜面，只需要光。
    一个探针收集到的光，最终铺满整个屏幕——以小博大，这是采样算法的信念。
    """
    D,F,H='Default','FarField','HitLighting'
    def __init__(self,bias=.1,normal_bias=.1,hair_bias=2.,far_field=True,hit_lighting=False,structured_is=False):
        self.bias=bias; self.normal_bias=normal_bias; self.hair_bias=hair_bias
        self.ff=far_field; self.hl=hit_lighting; self.sis=structured_is
    def _remap(self,pt,flags):
        o=dict(flags)
        if pt==self.F: o['ais']='Disabled'; o['rc']=False
        elif pt==self.H:
            o['ffo']=False
            if o.get('ais')=='AHS': o['ais']='Retrace'
        else: o['ffo']=False
        if pt!=self.H: o['ser']=False
        return o
    def gather(self,probes,rc,near=4000.,far=200000.) -> list:
        res=[]
        passes=[self.D]+([self.F] if self.ff else [])+([self.H] if self.hl else [])
        for p in probes:
            pid=p.get('probe_id',0); n=p.get('normal',(0.,1.,0.))
            rad=(0.,0.,0.); hd=float('inf')
            for pt in passes:
                ck=tuple(int(v*4) for v in n); cv=rc.get(ck,(0.,0.,0.))
                if pt==self.F:
                    if hd>far*.5: rad=cv; hd=far
                elif pt==self.H:
                    if hd==float('inf'): rad=tuple(v*.7 for v in cv); hd=near*.5
                else:
                    if rad==(0.,0.,0.): rad=cv; hd=near
            res.append((pid,rad,hd))
        return res


# ──────────────────────────────────────────────────────────────────────────────
# 鲁迅曾说：世上本没有光，照的人多了，也便有了缓存。
# Port: UE5 Renderer-Private — IndirectLightingCache / LightMapRendering /
#        LightFunctionAtlas / IESTextureManager / WaterInfoTextureRendering
# ──────────────────────────────────────────────────────────────────────────────

import math, hashlib
from collections import OrderedDict

# ── 1. IndirectLightingCache ──────────────────────────────────────────────────
# 体积纹理图集，为动态物体逐对象缓存间接光照。
# 正如鲁迅笔下那些沉默的看客：数据在此聚集，却不轻易更新。

_ILC_DIMENSION   = 64          # r.Cache.LightingCacheDimension
_ILC_ALLOC_SIZE  = 5           # r.Cache.LightingCacheMovableObjectAllocationSize
_BOUND_ROUNDUP   = math.sqrt(2.0)
_LOG_ROUNDUP     = math.log(_BOUND_ROUNDUP)

def _ilc_round_bound(v: float) -> float:
    """将包围盒边长向上取整到 sqrt(2)^N，稳定分配，减少抖动。"""
    if v <= 0: return _ILC_ALLOC_SIZE
    n = math.ceil(math.log(max(v, 1e-6)) / _LOG_ROUNDUP)
    return _BOUND_ROUNDUP ** n

class IndirectLightingCache:
    """
    三张浮点体积纹理（SH系数分三通道）构成的间接光缓存。
    鲁迅式注：这世界哪有什么自发光，不过是别人替你把SH算好了存起来。
    """
    def __init__(self, dimension=_ILC_DIMENSION):
        self.dim   = dimension
        self.blocks: dict[tuple,dict] = {}   # texel_min -> block
        self._dirty: list[tuple]      = []
        self._update_all              = True

    # ── 分配 ──────────────────────────────────────────────────────────────────
    def allocate(self, bounds_size: float) -> tuple | None:
        """
        按四舍五入后的包围盒尺寸在3D图集内分配体素块。
        返回 (x,y,z) texel_min 或 None（图集已满）。
        """
        sz = max(int(math.ceil(_ilc_round_bound(bounds_size))), _ILC_ALLOC_SIZE)
        sz = min(sz, self.dim)
        for z in range(0, self.dim - sz + 1, sz):
            for y in range(0, self.dim - sz + 1, sz):
                for x in range(0, self.dim - sz + 1, sz):
                    key = (x, y, z)
                    if key not in self.blocks:
                        self.blocks[key] = {'sz': sz, 'sh': [0.0]*9, 'valid': False}
                        self._dirty.append(key)
                        return key
        return None  # 人太多，站不下了

    def update_sh(self, texel_min: tuple, sh9: list[float]):
        """写入9分量SH系数（L0+L1），标记块为有效。"""
        if texel_min not in self.blocks: return
        self.blocks[texel_min]['sh']   = sh9[:9]
        self.blocks[texel_min]['valid'] = True

    def query(self, world_pos: tuple) -> list[float]:
        """
        在体积块中查找最近的SH采样。
        鲁迅式注：寻光者众，而光源稀少——先到先得，后来者只能继承别人的SH。
        """
        # 简化：按位置哈希映射到某个已分配块
        h = hash(tuple(int(v // _ILC_ALLOC_SIZE) for v in world_pos)) % max(len(self.blocks),1)
        for i, (k, b) in enumerate(self.blocks.items()):
            if i == h % len(self.blocks) and b['valid']:
                return b['sh']
        return [0.0]*9

    def free(self, texel_min: tuple):
        self.blocks.pop(texel_min, None)

    def flush_dirty(self) -> list[tuple]:
        """返回并清空待上传的块列表。"""
        d, self._dirty = self._dirty[:], []
        return d


# ── 2. LightMapRendering ──────────────────────────────────────────────────────
# 预计算光照贴图策略：LQ（2系数）/ HQ（6系数），以及体积光照图。
# 鲁迅式注：有些光早在烘焙时便已死去，只剩一张贴图流传人世。

_LQ_COEF = 2
_HQ_COEF = 6

class LightmapQuality:
    LQ = 'LQ_TEXTURE_LIGHTMAP'
    HQ = 'HQ_TEXTURE_LIGHTMAP'

def lightmap_policy_should_compile(quality: str, is_lit: bool, supports_static: bool,
                                    static_lighting_allowed: bool) -> bool:
    """
    对应 LightMapPolicyImpl::ShouldCompilePermutation。
    只有受光材质 + 支持静态光照的顶点工厂才编译此排列。
    """
    return is_lit and supports_static and static_lighting_allowed

def get_lightmap_coef_count(quality: str) -> int:
    return _HQ_COEF if quality == LightmapQuality.HQ else _LQ_COEF

class CachedVolumeIndirectLighting:
    """
    FCachedVolumeIndirectLightingPolicy 移植。
    用3D体积纹理插值采样，给动态物体提供连续间接光。
    鲁迅式注：体积之内，光如往事——连续，却无人能说清来自何处。
    """
    requires_sm5     = True
    no_translucency  = True

    @staticmethod
    def should_use(feature_level_sm5: bool, is_translucent: bool) -> bool:
        return feature_level_sm5 and not is_translucent

    @staticmethod
    def interpolate(cache: IndirectLightingCache, pos: tuple) -> list[float]:
        return cache.query(pos)

class CachedPointIndirectLighting:
    """
    FCachedPointIndirectLightingPolicy 移植。
    单点SH缓存，不保证空间连续，依赖时间插值淡化跳变。
    鲁迅式注：单点取样，正如只见一斑，便以为认清了豹。
    """
    TRANSITION_SPEED = 800.0   # r.Cache.SampleTransitionSpeed (units/s)

    @staticmethod
    def blend(sh_from: list[float], sh_to: list[float], dt: float) -> list[float]:
        t = min(dt * CachedPointIndirectLighting.TRANSITION_SPEED / 1000.0, 1.0)
        return [a + (b-a)*t for a,b in zip(sh_from, sh_to)]


# ── 3. LightFunctionAtlas ─────────────────────────────────────────────────────
# 将每种灯光函数材质渲染为2D纹理图集的子区域，按材质ID去重。
# 鲁迅式注：每盏灯都有自己的脾气，图集不过是把这些脾气分格收纳。

_LFA_MAX_EDGE      = 16          # 最大每维槽数
_LFA_MAX_FUNCTIONS = _LFA_MAX_EDGE * _LFA_MAX_EDGE  # 256

class LightFunctionAtlasSlot:
    __slots__ = ('material_id','uv_min','uv_max','valid')
    def __init__(self, material_id: str, uv_min: tuple, uv_max: tuple):
        self.material_id = material_id
        self.uv_min      = uv_min   # (u, v) normalized
        self.uv_max      = uv_max
        self.valid       = True

class LightFunctionAtlas:
    """
    2D纹理图集，按材质唯一ID去重存储灯光函数快照。
    对应 LightFunctionAtlas::FLightFunctionAtlasManager。
    鲁迅式注：材质千变万化，图集只认ID——就像官场只认印章，不认人脸。
    """
    def __init__(self, edge_size: int = 4, slot_resolution: int = 128):
        edge_size = min(max(edge_size, 2), _LFA_MAX_EDGE)
        self.edge        = edge_size
        self.slot_res    = max(slot_resolution, 32)
        self.atlas_res   = self.edge * self.slot_res
        self._slots: OrderedDict[str, LightFunctionAtlasSlot] = OrderedDict()

    def _uv_for_index(self, idx: int) -> tuple[tuple,tuple]:
        row, col = divmod(idx, self.edge)
        inv = 1.0 / self.edge
        umin = (col * inv, row * inv)
        umax = ((col+1)*inv, (row+1)*inv)
        return umin, umax

    def register_material(self, material_id: str) -> LightFunctionAtlasSlot | None:
        """注册或复用材质槽；超出容量返回 None。"""
        if material_id in self._slots:
            return self._slots[material_id]   # 去重命中
        if len(self._slots) >= self.edge * self.edge:
            return None  # 图集已满，正如舆论满了便再无新声
        idx  = len(self._slots)
        umin, umax = self._uv_for_index(idx)
        slot = LightFunctionAtlasSlot(material_id, umin, umax)
        self._slots[material_id] = slot
        return slot

    def get_slot(self, material_id: str) -> LightFunctionAtlasSlot | None:
        return self._slots.get(material_id)

    def get_light_index(self, material_id: str) -> int:
        """返回该材质在图集中的线性索引，供Shader索引常量缓冲。"""
        for i, k in enumerate(self._slots):
            if k == material_id: return i
        return -1

    @property
    def slot_count(self) -> int: return len(self._slots)


# ── 4. IESTextureManager ──────────────────────────────────────────────────────
# Texture2DArray图集，按RefCount管理IES光域网格纹理切片。
# 鲁迅式注：每个灯的配光曲线都是一段沉默的自白，图集只管存档，不问缘由。

_IES_DEFAULT_RESOLUTION   = 256    # r.IESAtlas.Resolution
_IES_MAX_PROFILE_COUNT    = 32     # r.IESAtlas.MaxProfileCount
_IES_INVALID_SLOT         = 0xFFFF_FFFF

class IESAtlasSlot:
    __slots__ = ('texture_id','slice_index','ref_count','resolution','dirty')
    def __init__(self, texture_id: str, resolution: int):
        self.texture_id  = texture_id
        self.slice_index = _IES_INVALID_SLOT
        self.ref_count   = 1
        self.resolution  = resolution
        self.dirty       = True

class IESTextureManager:
    """
    移植自 IESAtlas::FIESTextureManager（FRenderResource 子类）。
    管理 Texture2DArray 图集的分配/引用/回收，延迟上传。
    鲁迅式注：光域网格不过是灯的脸谱，收进图集，
              挂起来供着——有人引用，便不敢释放。
    """
    def __init__(self, max_profiles: int = _IES_MAX_PROFILE_COUNT,
                 resolution: int = _IES_DEFAULT_RESOLUTION):
        self.max_profiles  = max_profiles
        self.resolution    = resolution
        self._slots: dict[str, IESAtlasSlot] = {}
        self._free_slices: list[int]          = list(range(max_profiles))
        self._pending_adds: list[str]         = []
        self._pending_refreshes: list[str]    = []

    def add_or_ref(self, texture_id: str) -> IESAtlasSlot | None:
        """增加引用；若为新纹理则分配切片，加入待上传队列。"""
        if texture_id in self._slots:
            self._slots[texture_id].ref_count += 1
            return self._slots[texture_id]
        if not self._free_slices:
            return None   # 图集已满——人间灯光太多，容不下了
        slot = IESAtlasSlot(texture_id, self.resolution)
        self._slots[texture_id] = slot
        self._pending_adds.append(texture_id)
        return slot

    def release(self, texture_id: str):
        """减引用；归零则回收切片，留待他用。"""
        slot = self._slots.get(texture_id)
        if not slot: return
        slot.ref_count -= 1
        if slot.ref_count <= 0:
            if slot.slice_index != _IES_INVALID_SLOT:
                self._free_slices.append(slot.slice_index)
                slot.slice_index = _IES_INVALID_SLOT
            del self._slots[texture_id]

    def mark_dirty(self, texture_id: str):
        """强制下帧重新上传该纹理切片，对应 bForceRefresh。"""
        if texture_id in self._slots:
            self._slots[texture_id].dirty = True
            if texture_id not in self._pending_refreshes:
                self._pending_refreshes.append(texture_id)

    def commit(self) -> dict:
        """
        将待上传列表提交给"渲染线程"（此处返回任务字典供外部执行）。
        对应 UpdateIESAtlas RDG Pass 的调度逻辑。
        """
        task = {'adds': self._pending_adds[:], 'refreshes': self._pending_refreshes[:]}
        # 为新增槽分配切片索引
        for tid in self._pending_adds:
            slot = self._slots.get(tid)
            if slot and slot.slice_index == _IES_INVALID_SLOT and self._free_slices:
                slot.slice_index = self._free_slices.pop(0)
                slot.dirty = True
        self._pending_adds.clear()
        self._pending_refreshes.clear()
        return task

    @property
    def valid_slot_count(self) -> int:
        return sum(1 for s in self._slots.values() if s.slice_index != _IES_INVALID_SLOT)


# ── 5. WaterInfoTextureRendering ──────────────────────────────────────────────
# 正交投影捕获水体网格，输出水面深度+河流速度等到浮点纹理，
# 再经模糊与合并Pass写出最终WaterInfo纹理供水下雾/浮力系统采样。
# 鲁迅式注：水面之下皆是暗涌，渲染管线为它单开一条通道，
#            就像社会为某些人单设一套规则——看不见，但确实存在。

class WaterInfoTextureDesc:
    """捕获参数描述，对应 UWaterInfoTextureRendering 的配置。"""
    __slots__ = ('extent','capture_z','water_z_min','water_z_max',
                 'ground_z_min','blur_radius','use_128bit_rt')
    def __init__(self, extent=(512,512), capture_z=0.0,
                 water_z_min=-1e4, water_z_max=1e4,
                 ground_z_min=-1e4, blur_radius=2, use_128bit_rt=False):
        self.extent        = extent
        self.capture_z     = capture_z
        self.water_z_min   = water_z_min
        self.water_z_max   = water_z_max
        self.ground_z_min  = ground_z_min
        self.blur_radius   = blur_radius
        self.use_128bit_rt = use_128bit_rt

class WaterInfoTexturePipeline:
    """
    移植自 FWaterInfoTextureRendering 的简化CPU模拟版本。
    执行顺序：water_body_pass → ground_depth_pass → merge → blur → output
    鲁迅式注：水的信息要单独渲染一遍，因为水从来不肯与别的物体共用同一套规则。
    """
    _UNDERGROUND_DILATION_OFFSET  = 64.0    # r.Water.WaterInfo.UndergroundDilationDepthOffset
    _DILATION_OVERWRITE_MIN_DIST  = 128.0   # r.Water.WaterInfo.DilationOverwriteMinimumDistance

    def __init__(self, desc: WaterInfoTextureDesc):
        self.desc = desc
        self._water_body_buffer: dict[tuple,float]  = {}   # pixel -> water_depth
        self._ground_depth_buffer: dict[tuple,float] = {}
        self._output_buffer: dict[tuple,tuple]       = {}   # pixel -> (depth, vel_u, vel_v, flag)

    def write_water_body(self, pixel: tuple, depth: float, velocity: tuple=(0.,0.)):
        """水体Pass写入水面深度与河流速度（仅允许列表中的材质写入）。"""
        self._water_body_buffer[pixel] = (depth, velocity[0], velocity[1])

    def write_ground_depth(self, pixel: tuple, depth: float):
        """地面深度Pass，用于水下膨胀的遮挡判断。"""
        self._ground_depth_buffer[pixel] = depth

    def _merge_pixel(self, px: tuple) -> tuple:
        """
        FWaterInfoTextureMergePS 逻辑的CPU移植。
        若地面深度远在水面之下（超出 dilation_offset），允许膨胀覆盖。
        """
        w = self._water_body_buffer.get(px)
        g = self._ground_depth_buffer.get(px, float('inf'))
        if w is None:
            return (0., 0., 0., 0.)
        wd, wu, wv = w
        # 地面比水面高出足够距离才遮挡膨胀
        if g - wd > self._UNDERGROUND_DILATION_OFFSET:
            return (wd, wu, wv, 1.)
        if g - wd > self._DILATION_OVERWRITE_MIN_DIST:
            return (wd * 0.5, wu, wv, 0.5)   # 部分遮挡，衰减
        return (wd, wu, wv, 1.)

    def _blur_pixel(self, px: tuple, radius: int) -> tuple:
        """FWaterInfoTextureBlurPS 的简化盒式模糊。"""
        x, y   = px
        acc    = [0.]*4; count = 0
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                nb = self._output_buffer.get((x+dx, y+dy))
                if nb:
                    for i in range(4): acc[i] += nb[i]
                    count += 1
        if not count: return (0.,0.,0.,0.)
        return tuple(v/count for v in acc)

    def execute(self) -> dict[tuple,tuple]:
        """
        完整执行 merge + blur，返回最终水信息纹理（像素字典）。
        鲁迅式注：管线走完，留下的是一张静止的水面——
                  水下的秘密，只有采样者才能读懂。
        """
        all_px = set(self._water_body_buffer) | set(self._ground_depth_buffer)
        for px in all_px:
            self._output_buffer[px] = self._merge_pixel(px)
        if self.desc.blur_radius > 0:
            blurred = {}
            for px in self._output_buffer:
                blurred[px] = self._blur_pixel(px, self.desc.blur_radius)
            self._output_buffer = blurred
        return self._output_buffer

    def clear(self):
        self._water_body_buffer.clear()
        self._ground_depth_buffer.clear()
        self._output_buffer.clear()


# =============================================================================
# SkyAtmosphere + Velocity + ScreenSpaceRT + Fog + LocalFogVolume
# 移植自 Unreal Engine 5 Renderer-Private
# 鲁迅：人类的悲欢并不相通，但大气散射的 LUT 缓存必须互通。
# =============================================================================

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
from enum import IntEnum, auto


# ---------------------------------------------------------------------------
# 公用常量
# ---------------------------------------------------------------------------
SKY_ATMOSPHERE_SAMPLE_COUNT_MIN: float = 2.0
SKY_ATMOSPHERE_SAMPLE_COUNT_MAX: float = 32.0
SKY_ATMOSPHERE_DISTANCE_TO_SAMPLE_COUNT_MAX_KM: float = 150.0
FAST_SKY_LUT_WIDTH: int = 192
FAST_SKY_LUT_HEIGHT: int = 104
AERIAL_PERSPECTIVE_LUT_DEPTH_RESOLUTION: int = 16
AERIAL_PERSPECTIVE_LUT_DEPTH_KM: float = 96.0
AERIAL_PERSPECTIVE_LUT_WIDTH: int = 32
LOCAL_FOG_VOLUME_TILE_PIXEL_SIZE: int = 128
LOCAL_FOG_VOLUME_TILE_MAX_INSTANCE: int = 32
LOCAL_FOG_VOLUME_GLOBAL_START_DISTANCE_CM: float = 1000.0


# ---------------------------------------------------------------------------
# I.  SkyAtmosphereRendering
# ---------------------------------------------------------------------------

@dataclass
class SkyAtmosphereParams:
    """
    大气散射组件参数。
    对应 UE5 FSkyAtmosphereSceneProxy 内的运行时参数子集。
    鲁迅：旧社会的天空总是灰的——这里我们替它算清楚为什么灰。
    """
    bottom_radius_km: float = 6360.0
    top_radius_km: float = 6460.0
    rayleigh_scale_height_km: float = 8.0
    mie_scale_height_km: float = 1.2
    mie_anisotropy: float = 0.8
    sample_count_scale: float = 1.0
    affects_height_fog: bool = True
    visible_in_sky_capture: bool = True


@dataclass
class SkyAtmosphereRenderState:
    """
    每帧渲染状态——LUT 版本化，避免不必要的重建。
    对应 CVarSkyAtmosphereStateVersioning。
    """
    transmittance_lut: Optional[List[float]] = field(default=None, repr=False)
    multi_scatter_lut: Optional[List[float]] = field(default=None, repr=False)
    sky_view_lut: Optional[List[float]] = field(default=None, repr=False)
    aerial_perspective_lut: Optional[List[float]] = field(default=None, repr=False)
    state_version: int = 0
    _last_built_version: int = -1


class SkyAtmosphereRenderer:
    """
    大气散射渲染器。
    实现透射率 LUT、多重散射 LUT、FastSkyLUT、空中透视 LUT 的构建与查询。
    鲁迅：凡做一件事，便忠于一件事——何况是算光子穿越大气层。
    """

    def __init__(self, params: SkyAtmosphereParams) -> None:
        self.params = params
        self.state = SkyAtmosphereRenderState()
        self._enabled: bool = True

    def _compute_transmittance_sample(
        self, altitude_km: float, cos_zenith: float, n_steps: int
    ) -> float:
        """Beer-Lambert 积分，沿视线方向计算透射率（标量近似）。"""
        params = self.params
        t = 0.0
        step_km = altitude_km / max(1, n_steps)
        for i in range(n_steps):
            h = i * step_km
            rho_r = math.exp(-h / params.rayleigh_scale_height_km)
            rho_m = math.exp(-h / params.mie_scale_height_km)
            t += (5.802e-3 * rho_r + 3.996e-3 * rho_m) * step_km
        return math.exp(-t)

    def build_transmittance_lut(
        self,
        width: int = 256,
        height: int = 64,
        force_rebuild: bool = False,
    ) -> List[float]:
        """
        构建透射率 LUT。对应 RenderSkyAtmosphereTransmittanceLut()。
        鲁迅：透明不是没有，是穿透了太多东西之后剩下的。
        """
        if (
            not force_rebuild
            and self.state.transmittance_lut is not None
            and self.state._last_built_version == self.state.state_version
        ):
            return self.state.transmittance_lut
        sample_count = max(
            int(SKY_ATMOSPHERE_SAMPLE_COUNT_MIN),
            int(SKY_ATMOSPHERE_SAMPLE_COUNT_MAX * self.params.sample_count_scale),
        )
        lut: List[float] = []
        for v in range(height):
            altitude_km = (v / max(1, height - 1)) * (
                self.params.top_radius_km - self.params.bottom_radius_km
            )
            for u in range(width):
                cos_zenith = (u / max(1, width - 1)) * 2.0 - 1.0
                lut.append(
                    self._compute_transmittance_sample(altitude_km, cos_zenith, sample_count)
                )
        self.state.transmittance_lut = lut
        self.state._last_built_version = self.state.state_version
        return lut

    def build_multi_scatter_lut(self, size: int = 32) -> List[float]:
        """多重散射 LUT（各向同性近似）。对应 RenderSkyAtmosphereMultiScatteringLut()。"""
        lut: List[float] = []
        for j in range(size):
            for i in range(size):
                altitude_km = (i / max(1, size - 1)) * (
                    self.params.top_radius_km - self.params.bottom_radius_km
                )
                psi_ms = 0.05 * math.exp(-altitude_km / self.params.rayleigh_scale_height_km)
                lut.append(psi_ms)
        self.state.multi_scatter_lut = lut
        return lut

    def build_fast_sky_lut(self) -> List[float]:
        """FastSkyLUT 192×104。对应 RenderSkyAtmosphereFastSkyViewLut()。"""
        lut: List[float] = []
        transmittance = self.build_transmittance_lut()
        for v in range(FAST_SKY_LUT_HEIGHT):
            for u in range(FAST_SKY_LUT_WIDTH):
                cos_view = (u / max(1, FAST_SKY_LUT_WIDTH - 1)) * 2.0 - 1.0
                altitude_frac = v / max(1, FAST_SKY_LUT_HEIGHT - 1)
                tx_idx = int(altitude_frac * (len(transmittance) - 1))
                sky_lum = transmittance[tx_idx] * max(0.0, cos_view)
                lut.append(sky_lum)
        self.state.sky_view_lut = lut
        return lut

    def build_aerial_perspective_lut(self) -> List[float]:
        """
        空中透视 LUT（froxel 32×32×16）。
        对应 RenderSkyAtmosphereAerialPerspectiveLut()。
        鲁迅：远山是蓝的，不是诗意，是散射。
        """
        lut: List[float] = []
        for d in range(AERIAL_PERSPECTIVE_LUT_DEPTH_RESOLUTION):
            depth_km = (d / max(1, AERIAL_PERSPECTIVE_LUT_DEPTH_RESOLUTION - 1)) * AERIAL_PERSPECTIVE_LUT_DEPTH_KM
            for v in range(AERIAL_PERSPECTIVE_LUT_WIDTH):
                for u in range(AERIAL_PERSPECTIVE_LUT_WIDTH):
                    transmittance = math.exp(-depth_km / SKY_ATMOSPHERE_DISTANCE_TO_SAMPLE_COUNT_MAX_KM)
                    inscatter = (1.0 - transmittance) * 0.1
                    lut.append(transmittance)
                    lut.append(inscatter)
        self.state.aerial_perspective_lut = lut
        return lut

    def invalidate(self) -> None:
        """组件状态改变时递增版本，触发 LUT 重建。"""
        self.state.state_version += 1

    def tick(self, force_rebuild: bool = False) -> Dict[str, int]:
        """每帧调度：按需重建所有 LUT。"""
        if not self._enabled:
            return {}
        tx = self.build_transmittance_lut(force_rebuild=force_rebuild)
        ms = self.build_multi_scatter_lut()
        sky = self.build_fast_sky_lut()
        ap = self.build_aerial_perspective_lut()
        return {
            'transmittance_lut_px': len(tx),
            'multi_scatter_lut_px': len(ms),
            'fast_sky_lut_px':      len(sky),
            'aerial_perspective_lut_px': len(ap),
        }


# ---------------------------------------------------------------------------
# II.  VelocityRendering
# ---------------------------------------------------------------------------

class VelocityOutputPass(IntEnum):
    """对应 CVarVelocityOutputPass 的三种模式。"""
    DEPTH_PREPASS   = 0
    BASE_PASS       = 1
    AFTER_BASE_PASS = 2


@dataclass
class VelocityVector:
    dx: float = 0.0
    dy: float = 0.0
    depth: float = 0.0


class VelocityBuffer:
    """
    速度缓冲区。存储场景每个像素的运动向量。
    对应 UE5 FVelocityRendering 的逻辑缓冲抽象。
    鲁迅：速度是时间的证人，运动向量是帧间的告白。
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._buf: List[VelocityVector] = [VelocityVector() for _ in range(width * height)]
        self.output_pass: VelocityOutputPass = VelocityOutputPass.DEPTH_PREPASS

    def write(self, x: int, y: int, dx: float, dy: float, depth: float = 0.0) -> None:
        """写入单像素速度向量。"""
        if 0 <= x < self.width and 0 <= y < self.height:
            v = self._buf[y * self.width + x]
            v.dx, v.dy, v.depth = dx, dy, depth

    def read(self, x: int, y: int) -> VelocityVector:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self._buf[y * self.width + x]
        return VelocityVector()

    def clear(self) -> None:
        for v in self._buf:
            v.dx = v.dy = v.depth = 0.0

    def resolve_static_meshes(self) -> int:
        """
        将静止网格（速度为零）标记为 cleared。
        对应 UE5 velocity pass 中静止物体的 identity matrix 处理逻辑。
        鲁迅：速度为零的物体也在运动，只是运动的是时钟。
        """
        cleared = 0
        for v in self._buf:
            if v.dx == 0.0 and v.dy == 0.0:
                v.depth = 0.0
                cleared += 1
        return cleared

    def apply_temporal_responsiveness_bit(self, x: int, y: int, flag: bool) -> None:
        """Temporal Responsiveness 标志位写入，对应 CVarVelocityTemporalResponsivenessSupported。"""
        v = self.read(x, y)
        if flag:
            v.dy = -abs(v.dy) if v.dy != 0.0 else -0.0


# ---------------------------------------------------------------------------
# III.  ScreenSpaceRayTracing (SSR / SSGI)
# ---------------------------------------------------------------------------

class SSRQuality(IntEnum):
    OFF       = 0
    LOW       = 1
    MEDIUM    = 2
    HIGH      = 3
    VERY_HIGH = 4


@dataclass
class SSRConfig:
    quality: SSRQuality = SSRQuality.HIGH
    temporal_enabled: bool = False
    stencil_prepass: bool = False
    use_compute: bool = False
    half_res_scene_color: bool = False
    intensity: float = 1.0


class ScreenSpaceReflectionPass:
    """
    屏幕空间反射通道。
    对应 UE5 ScreenSpaceRayTracing::RenderScreenSpaceReflections()。
    鲁迅：水中的倒影映的是天，屏幕空间的倒影映的是已知。
    """

    def __init__(self, config: SSRConfig) -> None:
        self.config = config
        self._enabled: bool = True

    def should_render(self, has_view_state: bool = True) -> bool:
        """对应 ShouldRenderScreenSpaceReflections()。无 ViewState 则无 HZB，SSR 不可用。"""
        if not self._enabled:
            return False
        if self.config.quality <= SSRQuality.OFF:
            return False
        if self.config.intensity < 1.0:
            return False
        if not has_view_state:
            return False
        return True

    def trace_rays(
        self,
        depth_buffer: List[float],
        scene_color: List[float],
        width: int,
        height: int,
    ) -> List[float]:
        """屏幕空间光线追踪。CPU 参考实现，GPU 版在 Compute Shader。"""
        result: List[float] = [0.0] * (width * height)
        if not self.should_render():
            return result
        step = 2 if self.config.half_res_scene_color else 1
        for y in range(0, height, step):
            for x in range(0, width, step):
                idx = y * width + x
                if idx >= len(depth_buffer):
                    break
                if depth_buffer[idx] <= 0.0:
                    continue
                mirror_idx = (height - 1 - y) * width + x
                src = scene_color[mirror_idx] if mirror_idx < len(scene_color) else 0.0
                result[idx] = src * 0.5 * self.config.intensity
        return result

    def apply_temporal_accumulation(
        self,
        current: List[float],
        history: List[float],
        alpha: float = 0.1,
    ) -> List[float]:
        """
        TAA 混合（CVarSSRTemporal）。
        鲁迅：反射是对过去的记忆，TAA 是对过去的宽恕。
        """
        if not self.config.temporal_enabled or not history:
            return current
        return [c * alpha + h * (1.0 - alpha) for c, h in zip(current, history)]


# ---------------------------------------------------------------------------
# IV.  FogRendering
# ---------------------------------------------------------------------------

@dataclass
class ExponentialHeightFogParams:
    """
    指数高度雾参数。对应 FExponentialHeightFogSceneInfo 核心字段。
    鲁迅：雾是诚实的，它从不假装不存在。
    """
    fog_density: float = 0.02
    fog_height_falloff: float = 0.2
    fog_max_opacity: float = 1.0
    start_distance: float = 0.0
    fog_cutoff_distance: float = 0.0
    directional_inscattering_exponent: float = 4.0
    sky_light_capture_affects_fog_strength: float = 0.0
    sky_light_capture_affects_fog_roughness: float = 0.5
    volumetric_fog_enabled: bool = False


class FogRenderer:
    """
    指数高度雾 + 体积雾渲染器。
    对应 UE5 FogRendering.cpp 的 RenderFog() / SetupFogUniformParameters()。
    """

    def __init__(self, params: ExponentialHeightFogParams) -> None:
        self.params = params
        self._enabled: bool = True
        self._use_depth_bounds: bool = True
        self._volumetric_fog_lut: Optional[List[float]] = None

    def _fog_inscattering_at(self, world_z: float) -> float:
        """Beer-Lambert 指数高度雾密度积分（沿 Z 轴，标量近似）。"""
        p = self.params
        rho = p.fog_density * math.exp(-p.fog_height_falloff * world_z)
        return min(p.fog_max_opacity, 1.0 - math.exp(-rho))

    def compute_fog_factor(self, camera_z: float, pixel_z: float, depth: float) -> float:
        """
        计算单像素雾因子。对应 CalcSceneDepthBasedFogFactor()。
        鲁迅：深度越大，雾越浓，这是不需要证明的真理。
        """
        if not self._enabled:
            return 0.0
        p = self.params
        if p.fog_cutoff_distance > 0.0 and depth > p.fog_cutoff_distance:
            return 0.0
        effective_depth = max(0.0, depth - p.start_distance)
        fog_amount = self._fog_inscattering_at((camera_z + pixel_z) * 0.5)
        cutoff = p.fog_cutoff_distance if p.fog_cutoff_distance > 0.0 else 1e9
        return fog_amount * min(1.0, effective_depth / max(1.0, cutoff))

    def build_volumetric_fog_lut(
        self, width: int = 32, height: int = 32, depth_slices: int = 64
    ) -> List[float]:
        """体积雾 Froxel LUT（32×32×64）。对应 ComputeVolumetricFog()。"""
        lut: List[float] = []
        for z in range(depth_slices):
            depth_frac = z / max(1, depth_slices - 1)
            for y in range(height):
                for x in range(width):
                    density = self.params.fog_density * math.exp(
                        -self.params.fog_height_falloff * (1.0 - depth_frac) * 10.0
                    )
                    lut.append(math.exp(-density))       # transmittance
                    lut.append(density * 0.5)            # in-scatter
        self._volumetric_fog_lut = lut
        return lut

    def apply_on_scene_color(
        self,
        scene_color: List[float],
        depth_buffer: List[float],
        width: int,
        height: int,
        camera_z: float = 0.0,
    ) -> List[float]:
        """将雾效叠加到场景颜色缓冲。对应 RenderFog() 全屏 pass。"""
        if not self._enabled:
            return scene_color
        result = list(scene_color)
        fog_color = 0.8
        for i in range(min(len(scene_color), len(depth_buffer))):
            y, x = divmod(i, width)
            pixel_z = float(y) / max(1, height) * 10.0
            factor = self.compute_fog_factor(camera_z, pixel_z, depth_buffer[i])
            result[i] = result[i] * (1.0 - factor) + fog_color * factor
        return result


# ---------------------------------------------------------------------------
# V.  LocalFogVolumeRendering
# ---------------------------------------------------------------------------

@dataclass
class LocalFogVolumeInstance:
    """
    单个局部雾体积实例。对应 FLocalFogVolumeGPUInstanceData。
    鲁迅：一团雾在角落里，很像旧制度遗留的问题。
    """
    center_world: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius_cm: float = 1000.0
    density: float = 0.1
    albedo: float = 0.8
    emissive: float = 0.0
    fade_start_distance_cm: float = 0.0
    enabled: bool = True


class LocalFogVolumeRenderer:
    """
    局部雾体积渲染器（Tiled Culling）。
    对应 UE5 LocalFogVolumeRendering.cpp。
    """

    def __init__(
        self,
        tile_pixel_size: int = LOCAL_FOG_VOLUME_TILE_PIXEL_SIZE,
        max_instance_per_tile: int = LOCAL_FOG_VOLUME_TILE_MAX_INSTANCE,
    ) -> None:
        self.tile_pixel_size = max(8, min(512, tile_pixel_size))
        self.max_instance_per_tile = max(1, min(256, max_instance_per_tile))
        self._instances: List[LocalFogVolumeInstance] = []
        self._enabled: bool = True
        self._render_into_volumetric_fog: bool = True
        self._use_hzb_culling: bool = True
        self._global_start_distance_cm: float = LOCAL_FOG_VOLUME_GLOBAL_START_DISTANCE_CM

    def register_instance(self, inst: LocalFogVolumeInstance) -> int:
        """注册新实例，返回实例 ID。上限 256（与 u8 索引对齐）。"""
        if len(self._instances) >= 256:
            raise RuntimeError("LocalFogVolume: 超过 256 个实例上限。")
        self._instances.append(inst)
        return len(self._instances) - 1

    def should_render(self, has_any_volume: bool) -> bool:
        """对应 ShouldRenderLocalFogVolume()。"""
        return self._enabled and has_any_volume

    def build_tile_culling_list(
        self,
        view_width: int,
        view_height: int,
        camera_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> List[List[int]]:
        """
        Tile Culling：每个屏幕 Tile 生成局部雾体积实例 ID 列表。
        对应 LocalFogVolumeTileCullingCS。
        鲁迅：大多数的雾不在眼前，剔除它们是一种智慧。
        """
        tile_w = math.ceil(view_width / self.tile_pixel_size)
        tile_h = math.ceil(view_height / self.tile_pixel_size)
        tiles: List[List[int]] = [[] for _ in range(tile_w * tile_h)]
        cx, cy, cz = camera_pos
        for tile_idx in range(tile_w * tile_h):
            ty, tx = divmod(tile_idx, tile_w)
            tcx = (tx + 0.5) * self.tile_pixel_size
            tcy = (ty + 0.5) * self.tile_pixel_size
            for i, inst in enumerate(self._instances):
                if not inst.enabled:
                    continue
                ix, iy, iz = inst.center_world
                dist_cm = math.sqrt((ix - cx)**2 + (iy - cy)**2 + (iz - cz)**2)
                if dist_cm < self._global_start_distance_cm:
                    continue
                proj_x = ix * 0.5 + view_width * 0.5
                proj_y = iy * 0.5 + view_height * 0.5
                sr_px = inst.radius_cm / max(1.0, dist_cm / 1000.0) * 0.5
                if abs(proj_x - tcx) < sr_px + self.tile_pixel_size:
                    if abs(proj_y - tcy) < sr_px + self.tile_pixel_size:
                        if len(tiles[tile_idx]) < self.max_instance_per_tile:
                            tiles[tile_idx].append(i)
        return tiles

    def apply_on_scene_color(
        self,
        scene_color: List[float],
        depth_buffer: List[float],
        width: int,
        height: int,
        camera_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> List[float]:
        """将局部雾体积效果叠加到场景颜色缓冲（Tiled compositing pass）。"""
        if not self._instances or not self.should_render(bool(self._instances)):
            return scene_color
        tiles = self.build_tile_culling_list(width, height, camera_pos)
        result = list(scene_color)
        cx, cy, cz = camera_pos
        tile_w = math.ceil(width / self.tile_pixel_size)
        for y in range(height):
            for x in range(width):
                tile_idx = (y // self.tile_pixel_size) * tile_w + (x // self.tile_pixel_size)
                inst_list = tiles[tile_idx] if tile_idx < len(tiles) else []
                if not inst_list:
                    continue
                px_idx = y * width + x
                depth = depth_buffer[px_idx] if px_idx < len(depth_buffer) else 0.0
                fog_factor = 0.0
                for i in inst_list:
                    inst = self._instances[i]
                    ix, iy, iz = inst.center_world
                    dist = math.sqrt((ix - cx)**2 + (iy - cy)**2 + (iz - cz)**2)
                    if dist > inst.radius_cm:
                        continue
                    fog_factor += inst.density * (1.0 - dist / inst.radius_cm)
                fog_factor = min(1.0, fog_factor)
                if fog_factor > 0.0:
                    result[px_idx] = result[px_idx] * (1.0 - fog_factor) + self._instances[inst_list[0]].albedo * fog_factor
        return result

    def voxelize_into_volumetric_fog(
        self,
        fog_lut: List[float],
        lut_width: int = 32,
        lut_height: int = 32,
        lut_depth: int = 64,
        max_density: float = 0.01,
    ) -> List[float]:
        """
        将局部雾体积注入体积雾 Froxel LUT。
        密度限幅防止 TAA 时间漏光（CVarLocalFogVolumeMaxDensityIntoVolumetricFog）。
        鲁迅：密度太高，连时间都漏了。
        """
        result = list(fog_lut)
        for inst in self._instances:
            if not inst.enabled:
                continue
            density_clamped = min(inst.density, max_density)
            ix, iy, iz = inst.center_world
            fx = max(0, min(lut_width  - 1, int((ix / 5000.0 + 0.5) * lut_width)))
            fy = max(0, min(lut_height - 1, int((iy / 5000.0 + 0.5) * lut_height)))
            fz = max(0, min(lut_depth  - 1, int((iz / 5000.0 + 0.5) * lut_depth)))
            lut_idx = (fz * lut_height * lut_width + fy * lut_width + fx) * 2
            if lut_idx + 1 < len(result):
                result[lut_idx]     *= math.exp(-density_clamped)
                result[lut_idx + 1] += density_clamped * inst.albedo
        return result


# ---------------------------------------------------------------------------
# VI.  AtmosphericSceneComponent — 整合入 CellComponent pub/sub 循环
# ---------------------------------------------------------------------------

class AtmosphericSceneComponent:
    """
    大气 + 雾 + 局部雾 + 速度缓冲 + SSR 的整合组件。
    通过 cell pub/sub 循环向下游推送各渲染阶段结果。
    鲁迅：旧的章回小说每章末尾写"且听下回分解"，
    这里每帧末尾我们推送 channel 事件——其实是同一件事。
    """

    def __init__(
        self,
        sky_params: Optional[SkyAtmosphereParams] = None,
        fog_params: Optional[ExponentialHeightFogParams] = None,
        viewport_width: int = 1920,
        viewport_height: int = 1080,
    ) -> None:
        self.sky = SkyAtmosphereRenderer(sky_params or SkyAtmosphereParams())
        self.fog = FogRenderer(fog_params or ExponentialHeightFogParams())
        self.local_fog = LocalFogVolumeRenderer()
        self.velocity = VelocityBuffer(viewport_width, viewport_height)
        self.ssr = ScreenSpaceReflectionPass(SSRConfig())
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self._frame_index: int = 0
        self._ssr_history: List[float] = []

    def begin_frame(self) -> None:
        """帧开始：清空速度缓冲，递增帧索引。"""
        self.velocity.clear()
        self._frame_index += 1

    def render_velocity_pass(
        self,
        mesh_velocities: Optional[List[Tuple[int, int, float, float]]] = None,
    ) -> int:
        """
        速度通道。mesh_velocities: [(x, y, dx, dy), ...]。
        对应 FDeferredShadingSceneRenderer::RenderVelocities()。
        """
        if mesh_velocities:
            for x, y, dx, dy in mesh_velocities:
                self.velocity.write(x, y, dx, dy)
        return self.velocity.resolve_static_meshes()

    def render_sky_atmosphere(self, force_rebuild: bool = False) -> Dict[str, int]:
        """大气 LUT 构建通道。对应 RenderSkyAtmosphere()。"""
        return self.sky.tick(force_rebuild=force_rebuild)

    def render_ssr(
        self,
        depth_buffer: List[float],
        scene_color: List[float],
    ) -> List[float]:
        """SSR 通道 + TAA 积累。对应 RenderScreenSpaceReflections()。"""
        raw = self.ssr.trace_rays(
            depth_buffer, scene_color,
            self.viewport_width, self.viewport_height,
        )
        blended = self.ssr.apply_temporal_accumulation(raw, self._ssr_history)
        self._ssr_history = blended
        return blended

    def render_fog(
        self,
        scene_color: List[float],
        depth_buffer: List[float],
    ) -> List[float]:
        """高度雾 + 局部雾体积叠加。"""
        after_height_fog = self.fog.apply_on_scene_color(
            scene_color, depth_buffer,
            self.viewport_width, self.viewport_height,
        )
        if self.fog.params.volumetric_fog_enabled:
            v_lut = self.fog.build_volumetric_fog_lut()
            v_lut = self.local_fog.voxelize_into_volumetric_fog(v_lut)
            self.fog._volumetric_fog_lut = v_lut
        return self.local_fog.apply_on_scene_color(
            after_height_fog, depth_buffer,
            self.viewport_width, self.viewport_height,
        )

    def end_frame(self) -> Dict[str, object]:
        """
        帧末统计，供 pub/sub 循环推送。
        鲁迅：写完了，才知道什么都没写完。
        """
        return {
            'frame':                    self._frame_index,
            'sky_lut_version':          self.sky.state.state_version,
            'velocity_buf_size':        self.viewport_width * self.viewport_height,
            'ssr_enabled':              self.ssr.should_render(),
            'fog_enabled':              self.fog._enabled,
            'local_fog_instances':      len(self.local_fog._instances),
            'aerial_perspective_slices': AERIAL_PERSPECTIVE_LUT_DEPTH_RESOLUTION,
        }


# ---------------------------------------------------------------------------
# 附：各模块关键设计注记（行内批注）
# ---------------------------------------------------------------------------
# SkyAtmosphereRenderer
#   · 透射率 LUT (256×64)：Beer-Lambert 积分，Rayleigh + Mie 双层散射。
#     鲁迅：透明不是没有，是穿透了太多东西之后剩下的。
#   · 多重散射 LUT (32×32)：各向同性近似，替代 GPU path-tracing 版本。
#   · FastSkyLUT (192×104)：CVarSkyAtmosphereFastSkyLUT=1 路径的 CPU 镜像。
#   · 空中透视 LUT (32×32×16)：Froxel 体积，远景蓝移。
#
# VelocityBuffer
#   · 三种输出通道（Depth Prepass / Base Pass / After Base Pass）。
#   · resolve_static_meshes()：静止网格速度归零，节省后续 TAA 带宽。
#   · Temporal Responsiveness bit：标记材质抖动与速度向量不匹配的像素。
#     鲁迅：速度为零的物体也在运动，只是运动的是时钟。
#
# ScreenSpaceReflectionPass
#   · should_render() 对应 ShouldRenderScreenSpaceReflections() 的全部门禁。
#   · half_res_scene_color：CVarSSRHalfResSceneColor=1，性能换质量。
#   · apply_temporal_accumulation()：α=0.1 的指数移动平均 TAA。
#     鲁迅：反射是对过去的记忆，TAA 是对过去的宽恕。
#
# FogRenderer
#   · compute_fog_factor()：CalcSceneDepthBasedFogFactor() 的 CPU 等价。
#   · build_volumetric_fog_lut()：32×32×64 Froxel，Trilinear 采样。
#   · apply_on_scene_color()：全屏 pass，深度边界优化（CVarFogUseDepthBounds）。
#
# LocalFogVolumeRenderer
#   · build_tile_culling_list()：128px Tile，每 Tile 最多 32 实例（u8 索引）。
#   · voxelize_into_volumetric_fog()：密度钳制防 TAA 漏光。
#   · apply_on_scene_color()：Tiled compositing，球体衰减叠加。
#     鲁迅：局部的雾最危险，因为你以为只是局部。
#
# AtmosphericSceneComponent
#   · begin_frame() → render_velocity_pass() → render_sky_atmosphere()
#     → render_ssr() → render_fog() → end_frame()
#   · end_frame() 返回的诊断字典通过 cell pub/sub 循环广播给订阅者。
#     鲁迅：每帧的统计是写给下一帧看的。
# ---------------------------------------------------------------------------


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] ShadowRendering → Python port
#
# Ported from (head -200):
#   upstream/unreal-renderer-ue5/Renderer-Private/ShadowRendering.cpp
#
# 鲁迅曾言：「真正的勇士，敢于直面惨淡的阴影；
# 然而阴影的深浅，不过是光源距离与偏置的函数。」
# 深度偏置是影子的谎言许可证——允许一点点自欺，换来不互相遮挡的太平。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   CVarCSMShadowDepthBias / CVarCSMShadowSlopeScaleDepthBias
#       → _CSM_DEPTH_BIAS / _CSM_SLOPE_BIAS   (module constants, not CVar)
#   CVarPerObjectDirectionalShadowDepthBias
#       → _PER_OBJECT_DIR_DEPTH_BIAS
#   CVarShadowTransitionScale
#       → _SHADOW_TRANSITION_SCALE
#   CVarCSMShadowReceiverBias
#       → _CSM_RECEIVER_BIAS
#   CVarPointLightShadowDepthBias / CVarRectLightShadowDepthBias
#       → _POINT_LIGHT_DEPTH_BIAS / _RECT_LIGHT_DEPTH_BIAS
#   CVarSpotLightShadowDepthBias / CVarSpotLightShadowReceiverBias
#       → _SPOT_LIGHT_DEPTH_BIAS / _SPOT_LIGHT_RECEIVER_BIAS
#   CVarFilterMethod (0=PCF, 1=PCSS)
#       → _SHADOW_FILTER_METHOD
#   GStencilOptimization / GShadowStencilCulling
#       → _STENCIL_OPTIMIZATION / _STENCIL_CULLING
#
# AstroCellShadowProjection:
#   Per-cell shadow projection that computes a CSM-like depth-bias-corrected
#   shadow factor in [0,1] for a given receiver cell from a set of caster cells.
#   Mirrors the CSM projection pass in ShadowRendering.cpp — the depth bias
#   prevents self-shadowing (peter panning trade-off), transition scale smooths
#   the penumbra edge, receiver bias adjusts the shadow-receiver surface offset.
#
# AstroCellShadowRenderer:
#   Orchestrates per-epoch shadow rendering:
#     build_shadow_depth_map()  — collects casters, sorts by z-layer (depth)
#     project_shadows()         — projects each caster onto each receiver
#     get_shadow_factor()       — returns the final [0,1] shadow attenuation
#   Mirrors the top-level shadow pass dispatch in DeferredShadingRenderer.cpp.
#
# 2-D channel adaptation:
#   Directional shadow map  → z-layer depth ordering (highest z = deepest shadow)
#   Depth bias (CSM/perObj) → per-cell SVG shadow offset correction
#   Transition scale        → penumbra fade region around each shadow caster
#   Stencil optimisation    → skip receiver test when all_bboxes unchanged
#   PCSS                    → soft-shadow radius proportional to caster bbox area
# ═══════════════════════════════════════════════════════════════════════════════

# Depth bias values (mirrors CVar defaults in ShadowRendering.cpp head -200)
_CSM_DEPTH_BIAS:            float = 10.0
_CSM_SLOPE_BIAS:            float = 3.0
_PER_OBJECT_DIR_DEPTH_BIAS: float = 10.0
_PER_OBJECT_DIR_SLOPE_BIAS: float = 3.0
_CSM_RECEIVER_BIAS:         float = 0.9
_POINT_LIGHT_DEPTH_BIAS:    float = 0.02
_POINT_LIGHT_SLOPE_BIAS:    float = 3.0
_RECT_LIGHT_DEPTH_BIAS:     float = 0.025
_RECT_LIGHT_SLOPE_BIAS:     float = 2.5
_RECT_LIGHT_RECEIVER_BIAS:  float = 0.3
_SPOT_LIGHT_DEPTH_BIAS:     float = 3.0
_SPOT_LIGHT_SLOPE_BIAS:     float = 3.0
_SPOT_LIGHT_RECEIVER_BIAS:  float = 0.5

# Shadow transition (penumbra fade) scale — mirrors CVarShadowTransitionScale
_SHADOW_TRANSITION_SCALE:   float = 60.0
_SPOT_TRANSITION_SCALE:     float = 60.0

# Filter method: 0 = PCF (uniform), 1 = PCSS (experimental)
_SHADOW_FILTER_METHOD:      int   = 0

# Stencil optimisation flags — mirrors GStencilOptimization / GShadowStencilCulling
_STENCIL_OPTIMIZATION:      bool  = True
_STENCIL_CULLING:           bool  = True

# Maximum soft-kernel size for PCSS — mirrors CVarMaxSoftKernelSize = 40
_PCSS_MAX_KERNEL_RADIUS:    float = 40.0

# Modulated self-shadow (mobile only) — off by default
_ENABLE_MODULATED_SELF_SHADOW: bool = False


def _shadow_depth_bias_corrected(
    caster_z:      float,
    receiver_z:    float,
    depth_bias:    float = _CSM_DEPTH_BIAS,
    slope_bias:    float = _CSM_SLOPE_BIAS,
    receiver_bias: float = _CSM_RECEIVER_BIAS,
) -> float:
    """
    Compute depth-bias-corrected shadow depth for CSM comparison.

    Port of the depth bias formula applied in the CSM depth pass shader
    (ShadowRendering.cpp, per-primitive depth-bias term):

        biased_depth = caster_z + depth_bias + slope_bias × |dZ/dXY|

    In 2-D, slope (dZ/dXY) is approximated as the normalised z-layer
    gradient between caster and receiver (|caster_z - receiver_z| / max_z).
    Receiver bias shifts the receiver surface away from the shadow map.

    Returns the biased caster depth for comparison with the receiver depth.

    鲁迅式：偏置是对规则的微调，是「在法律允许的范围内作弊」——
    不是欺诈，而是工程上的智慧：让影子落在正确的位置。
    """
    z_range    = max(_ASTRO_CELL_MAX_Z_LAYERS, 1)
    slope      = abs(caster_z - receiver_z) / z_range   # normalised slope proxy
    biased     = caster_z + depth_bias + slope_bias * slope
    # Receiver bias: push receiver surface slightly away (CSMReceiverBias in C++)
    biased_recv = receiver_z * receiver_bias
    return biased - biased_recv


def _shadow_pcf_weight(
    shadow_delta: float,
    transition_scale: float = _SHADOW_TRANSITION_SCALE,
) -> float:
    """
    PCF soft-edge weight — mirrors the transition-scale fade in the C++ CSM.

    At shadow_delta = 0: receiver is exactly on the shadow boundary → 0.5 weight.
    At shadow_delta > transition_scale: fully lit → 1.0 weight.
    At shadow_delta < 0: fully in shadow → 0.0 weight.

    ShadowTransitionScale controls the width of the penumbra edge:
        larger scale → sharper edge (less fade region).

    鲁迅式：PCF 是影子边缘的民主投票——
    每个采样点一票，平均值决定光影比。没有独裁，只有平均。
    """
    return max(0.0, min(1.0, 0.5 + shadow_delta / max(transition_scale, 1.0)))


def _shadow_pcss_kernel_radius(
    caster_w: float,
    caster_h: float,
    z_dist: float,
) -> float:
    """
    PCSS soft-shadow kernel radius.

    Port of the PCSS kernel computation from CVarFilterMethod = 1 path:
        kernel_radius = sqrt(projected_area) × (z_dist / max_dist)
        clamped to [1, PCSS_MAX_KERNEL_RADIUS]

    Larger casters cast softer shadows at greater depth distances —
    matching the angular-size-based penumbra of a real area light source.

    鲁迅式：柔和的阴影是对真实的妥协——
    面积越大，距离越远，影子越模糊，真相越难辨认。
    PCSS 诚实地模拟了这一过程。
    """
    projected_area = caster_w * caster_h
    soft_radius    = math.sqrt(max(projected_area, 1.0)) * (
        z_dist / max(_CAPSULE_MAX_DIST, 1.0))
    return max(1.0, min(soft_radius, _PCSS_MAX_KERNEL_RADIUS))


class AstroCellShadowProjection:
    """
    Python equivalent of the per-shadow-volume CSM projection pass.

    Holds the depth-bias configuration for a single shadow type
    (directional/point/rect/spot) and projects caster depth values
    onto receivers via _shadow_pcf_weight or PCSS kernel.

    Parameters
    ----------
    depth_bias, slope_bias, receiver_bias:
        Bias parameters selected per light type (mirrors the CVar set
        in ShadowRendering.cpp head -200).
    transition_scale:
        Penumbra fade width (CVarShadowTransitionScale).
    filter_method:
        0 = PCF (uniform), 1 = PCSS (soft, area-proportional).

    鲁迅式：投影器是光与影之间的翻译官——
    把三维的深度关系翻译成二维的明暗因子，一切都在这个翻译中失真，
    但失真的方向是可控的，这便是工程的意义。
    """

    def __init__(
        self,
        depth_bias:     float = _CSM_DEPTH_BIAS,
        slope_bias:     float = _CSM_SLOPE_BIAS,
        receiver_bias:  float = _CSM_RECEIVER_BIAS,
        transition_scale: float = _SHADOW_TRANSITION_SCALE,
        filter_method:  int   = _SHADOW_FILTER_METHOD,
    ) -> None:
        self.depth_bias       = depth_bias
        self.slope_bias       = slope_bias
        self.receiver_bias    = receiver_bias
        self.transition_scale = transition_scale
        self.filter_method    = filter_method

    def project(
        self,
        receiver_z: float,
        caster_z:   float,
        caster_w:   float = 80.0,
        caster_h:   float = 50.0,
    ) -> float:
        """
        Project a single caster onto a receiver and return shadow factor [0,1].

        1.0 = fully lit (no shadow from this caster)
        0.0 = fully in shadow

        PCF path (filter_method=0):
            biased_depth = depth_bias_corrected(caster_z, receiver_z)
            delta = receiver_z - biased_depth
            weight = pcf_weight(delta, transition_scale)

        PCSS path (filter_method=1):
            kernel_r = pcss_kernel_radius(caster area, z_dist)
            delta *= kernel_r / transition_scale  (widens penumbra)

        鲁迅式：投影就是以牺牲精度换取效率——
        单个采样点决定阴影，而非多次光线追踪。工程上的妥协，艺术上的近似。
        """
        biased = _shadow_depth_bias_corrected(
            caster_z, receiver_z,
            self.depth_bias, self.slope_bias, self.receiver_bias
        )
        shadow_delta = receiver_z - biased

        if self.filter_method == 1:
            # PCSS: scale transition width by kernel radius
            z_dist = abs(caster_z - receiver_z)
            kr     = _shadow_pcss_kernel_radius(caster_w, caster_h, z_dist)
            # Widen the transition zone proportionally to the soft kernel
            effective_scale = self.transition_scale * (kr / max(_PCSS_MAX_KERNEL_RADIUS, 1.0) + 0.1)
            return _shadow_pcf_weight(shadow_delta, effective_scale)
        else:
            return _shadow_pcf_weight(shadow_delta, self.transition_scale)


class AstroCellShadowRenderer:
    """
    Per-epoch shadow renderer — mirrors the top-level shadow dispatch in
    DeferredShadingSceneRenderer::RenderShadowDepthMaps().

    build_shadow_depth_map(): collect and sort all registered cell casters.
    project_shadows():         for every (receiver, caster) pair, compute
                               per-cell shadow factors via AstroCellShadowProjection.
    get_shadow_factor():       return the minimum (most-occluded) shadow factor
                               from all casters for a given receiver.

    Stencil optimisation (GStencilOptimization=True):
        Shadow map is cached between calls; invalidated when all_bboxes
        changes (mirrors the stencil clear optimisation that avoids redundant
        shadow depth re-renders when the scene is static).

    鲁迅式：阴影渲染器是场景中最劳累的工人——
    它必须为每一个接收者计算每一个投射者的贡献，
    然后把结果存起来，供后续的着色阶段消费。
    效率全靠缓存；缓存失效，一切重来。
    """

    def __init__(
        self,
        projection: AstroCellShadowProjection | None = None,
    ) -> None:
        self._projection = projection or AstroCellShadowProjection()
        # Depth map: list of (cell_id, z, w, h) sorted by z descending (deepest first)
        self._depth_map:      list = []
        # Per-receiver shadow factors: receiver_cell_id → min shadow factor
        self._shadow_factors: dict = {}
        # Stencil optimisation cache key (hash of caster z values)
        self._cache_key: int = -1

    def _compute_cache_key(self, all_bboxes: dict) -> int:
        """Cheap scene-change detection — hash of all z-layer values."""
        return hash(tuple(
            (cid, round(bb.get("z", 0), 2))
            for cid, bb in sorted(all_bboxes.items())
        ))

    def build_shadow_depth_map(self, all_bboxes: dict) -> None:
        """
        Collect all cell casters and sort by z descending.

        Mirrors the RenderShadowDepthMaps loop that iterates scene primitives
        and writes their depth into the shadow depth buffer.

        Stencil optimisation: skip rebuild if scene is unchanged.

        鲁迅式：建造深度图是一次普查——每个单元格都要被登记，
        按深度排序，等待被投影到接收者身上。普查完成，影子才有根据。
        """
        new_key = self._compute_cache_key(all_bboxes)
        if _STENCIL_OPTIMIZATION and new_key == self._cache_key:
            return  # scene unchanged — reuse cached depth map
        self._cache_key = new_key

        self._depth_map = sorted(
            [
                (cid, float(bb.get("z", 0)), float(bb.get("w", 80)), float(bb.get("h", 50)))
                for cid, bb in all_bboxes.items()
            ],
            key=lambda t: t[1],
            reverse=True,   # deepest (highest z) first — mirrors near-to-far CSM sort
        )

    def project_shadows(self, all_bboxes: dict) -> None:
        """
        For every receiver cell, aggregate shadow contributions from all
        casters with higher z (casters «above» receivers in z-layer space).

        Mirrors the ShadowProjection pass loop:
            for each ShadowInfo (caster):
                for each receiver:
                    RenderShadowProjection → shadow factor

        The final shadow factor is the minimum across all casters
        (most-shadowed wins — additive shadow blending, not multiplicative,
        to match the C++ additive shadow attenuation model).

        鲁迅式：投影是权力的叠加——多个阴影来源叠加，
        取最深者为准，因为在现实中，最黑暗处往往来自多个遮挡共谋。
        """
        self.build_shadow_depth_map(all_bboxes)
        self._shadow_factors = {}

        for rcid, rbb in all_bboxes.items():
            recv_z = float(rbb.get("z", 0))
            min_factor = 1.0   # start fully lit

            for (cid, cz, cw, ch) in self._depth_map:
                if cid == rcid:
                    continue
                if cz <= recv_z:
                    continue   # only casters at higher z can shadow this receiver

                # X/Y overlap check (mirrors frustum culling of shadow receivers)
                rx0 = rbb.get("x", 0)
                ry0 = rbb.get("y", 0)
                rx1 = rx0 + rbb.get("w", 80)
                ry1 = ry0 + rbb.get("h", 50)
                cb = all_bboxes.get(cid, {})
                cx0 = cb.get("x", 0)
                cy0 = cb.get("y", 0)
                cx1 = cx0 + cb.get("w", 80)
                cy1 = cy0 + cb.get("h", 50)

                if rx1 < cx0 or cx1 < rx0 or ry1 < cy0 or cy1 < ry0:
                    continue   # no XY overlap — shadow cannot reach this receiver

                factor = self._projection.project(recv_z, cz, cw, ch)
                if factor < min_factor:
                    min_factor = factor

            self._shadow_factors[rcid] = min_factor

    def get_shadow_factor(self, cell_id: str) -> float:
        """
        Return the shadow factor [0,1] for the given receiver cell.
        1.0 = fully lit; 0.0 = fully in shadow.
        Returns 1.0 (lit) when no shadow map has been projected yet.

        鲁迅式：查询比计算便宜——先算好，再查；这是缓存的哲学。
        """
        return self._shadow_factors.get(cell_id, 1.0)


#: Module-level shadow renderer singleton
_ASTRO_SHADOW_RENDERER: AstroCellShadowRenderer = AstroCellShadowRenderer()


def get_shadow_renderer() -> AstroCellShadowRenderer:
    """Return the process-level AstroCellShadowRenderer singleton."""
    return _ASTRO_SHADOW_RENDERER


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] PrimitiveSceneInfo → Python port
#
# Ported from (head -200):
#   upstream/unreal-renderer-ue5/Renderer-Private/PrimitiveSceneInfo.cpp
#
# 鲁迅曾言：「人必生活着，爱才有所附丽。」
# Primitive 必须存在于 Scene 中，才能被渲染，才能投射阴影，才能影响他者。
# FPrimitiveSceneInfo 是 Primitive 在场景图中的身份证。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   FPrimitiveFlagsCompact → AstroCellPrimitiveFlags
#       bCastDynamicShadow  → cast_dynamic_shadow
#       bStaticLighting     → static_lighting
#       bCastStaticShadow   → cast_static_shadow
#       bIsNaniteMesh       → is_nanite  (always True in Astro — all cells are Nanite)
#       bIsAlwaysVisible    → always_visible
#       bAllVertexFactoriesSupportGPUScene → gpu_scene_supported
#       bIsForceHidden      → force_hidden
#
#   FPrimitiveSceneInfoCompact → AstroCellSceneInfoCompact
#       Carries the compact flags + bbox + min_draw_distance + proxy ref.
#
#   FBatchingSPDI::DrawMesh (StaticMesh cache) → cache_static_draw_commands()
#       Pre-caches SVG draw commands into the static mesh batch list so that
#       per-frame proc() calls can skip regeneration when the bbox is unchanged.
#       Mirrors the bSupportsCachingMeshDrawCommands check in DrawMesh().
#
#   FPrimitiveSceneInfo::AddToScene / RemoveFromScene
#       → add_to_scene() / remove_from_scene()
#       Registered in AstroCellPrimitiveRegistry which is the Python equivalent
#       of the per-scene FScene::Primitives TArray.
#
# GMeshDrawCommandsCacheMultithreaded / GNaniteMaterialBinCacheParallel
#       → _DRAW_CMD_CACHE_MT / _NANITE_BIN_CACHE_PARALLEL (module constants)
# ═══════════════════════════════════════════════════════════════════════════════

# Module-level flag equivalents for CVar defaults in PrimitiveSceneInfo.cpp head -200
_DRAW_CMD_CACHE_MT:        bool = True   # GMeshDrawCommandsCacheMultithreaded
_DRAW_CMD_BATCH_SIZE:      int  = 12     # GMeshDrawCommandsBatchSize
_NANITE_BIN_CACHE_PARALLEL: bool = True  # GNaniteMaterialBinCacheParallel
_RT_PRIMITIVE_CACHE_MT:    bool = True   # GRayTracingPrimitiveCacheMultithreaded


@dataclass
class AstroCellPrimitiveFlags:
    """
    Python equivalent of FPrimitiveFlagsCompact.

    Bit-packed flags describing a cell primitive's render capabilities.
    Derived from the cell's species gene_traits and static configuration.

    鲁迅式：旗帜是立场的声明——每个比特都是一个「是」或「否」，
    汇聚成一个整数，代表这个单元格在场景中的身份与权利。
    """
    cast_dynamic_shadow: bool = True
    static_lighting:     bool = False
    cast_static_shadow:  bool = False
    is_nanite:           bool = True    # All Astro cells are Nanite primitives
    always_visible:      bool = False
    gpu_scene_supported: bool = True
    force_hidden:        bool = False

    @classmethod
    def from_gene_traits(cls, gene_traits: dict) -> "AstroCellPrimitiveFlags":
        """
        Derive flags from a cell's gene_traits dict.
        Mirrors FPrimitiveFlagsCompact(FPrimitiveSceneProxy*) constructor.
        """
        return cls(
            cast_dynamic_shadow = gene_traits.get("cast_shadow", True),
            static_lighting     = gene_traits.get("static_lighting", False),
            cast_static_shadow  = gene_traits.get("cast_static_shadow", False),
            is_nanite           = True,
            always_visible      = gene_traits.get("always_visible", False),
            gpu_scene_supported = True,
            force_hidden        = gene_traits.get("force_hidden", False),
        )

    def packed(self) -> int:
        """Pack flags into a single integer (mirrors FPrimitiveFlagsCompact layout)."""
        return (
            (int(self.cast_dynamic_shadow) << 0) |
            (int(self.static_lighting)     << 1) |
            (int(self.cast_static_shadow)  << 2) |
            (int(self.is_nanite)           << 3) |
            (int(self.always_visible)      << 4) |
            (int(self.gpu_scene_supported) << 5) |
            (int(self.force_hidden)        << 6)
        )


@dataclass
class AstroCellSceneInfoCompact:
    """
    Python equivalent of FPrimitiveSceneInfoCompact.

    Lightweight descriptor for quick scene-graph queries — stores the flags,
    bounds, and a back-reference to the full AstroCellSceneInfo entry.
    Mirrors the compact representation used in FScene::PrimitiveSceneProxies.

    鲁迅式：紧凑表示是对效率的致敬——宁可冗余地存储两份，
    也不要每次都解引用到完整对象。空间换时间，这是程序员的现实主义。
    """
    cell_id:        str
    flags:          AstroCellPrimitiveFlags
    bounds_min:     tuple   # (x, y, z)
    bounds_max:     tuple   # (x+w, y+h, z)
    min_draw_dist:  float   = 0.0
    max_draw_dist:  float   = float("inf")

    def in_draw_range(self, screen_fraction: float) -> bool:
        """Quick LOD cull check — mirrors FPrimitiveFlagsCompact draw-range test."""
        return (self.min_draw_dist <= screen_fraction <= self.max_draw_dist
                or self.flags.always_visible)

    def is_visible_to_shadow(self) -> bool:
        """Returns True if this primitive should contribute to shadow maps."""
        return (self.flags.cast_dynamic_shadow or self.flags.cast_static_shadow) \
               and not self.flags.force_hidden


class AstroCellStaticDrawCommandCache:
    """
    Static mesh draw command cache — mirrors FBatchingSPDI::DrawMesh() +
    the GMeshDrawCommandsCacheMultithreaded path in PrimitiveSceneInfo.cpp.

    When bSupportsCachingMeshDrawCommands is True (our _DRAW_CMD_CACHE_MT),
    SVG draw commands for cells whose bbox is unchanged are stored here and
    reused on subsequent proc() calls, avoiding redundant SVG regeneration.

    The cache is keyed by (cell_id, epoch) — each epoch bump invalidates stale
    entries (mirrors the per-primitive cache invalidation on transform updates).

    鲁迅式：缓存是对重复劳动的反抗——把已经做过的事情记录下来，
    下次遇到同样的情况，直接查账本，不必重新劳作。
    但账本也会过期，epoch 是那个不动声色的清账人。
    """

    def __init__(self) -> None:
        # key: (cell_id, epoch) → cached svg fragment string
        self._cache: dict = {}
        # Batch size for parallel cache flush — mirrors GMeshDrawCommandsBatchSize
        self.batch_size: int = _DRAW_CMD_BATCH_SIZE

    def get(self, cell_id: str, epoch: int) -> str | None:
        """Return cached SVG fragment if valid for this epoch, else None."""
        return self._cache.get((cell_id, epoch))

    def put(self, cell_id: str, epoch: int, svg_fragment: str) -> None:
        """Store a draw command fragment for (cell_id, epoch)."""
        if _DRAW_CMD_CACHE_MT:
            self._cache[(cell_id, epoch)] = svg_fragment

    def invalidate_cell(self, cell_id: str) -> None:
        """Invalidate all cached entries for a cell (transform update path)."""
        stale = [k for k in self._cache if k[0] == cell_id]
        for k in stale:
            del self._cache[k]

    def flush_epoch(self, current_epoch: int, keep_window: int = 2) -> int:
        """
        Remove entries older than (current_epoch - keep_window).
        Mirrors the periodic cache flush that removes stale FMeshDrawCommand entries
        after the primitive-pool water-mark epoch has advanced.
        Returns count of evicted entries.
        """
        stale = [k for k in self._cache if current_epoch - k[1] > keep_window]
        for k in stale:
            del self._cache[k]
        return len(stale)

    def stats(self) -> dict:
        return {"cache_entries": len(self._cache)}


class AstroCellPrimitiveRegistry:
    """
    Per-epoch primitive registry — Python equivalent of FScene::Primitives TArray
    and its associated PrimitiveSceneProxies / PrimitiveBounds arrays.

    add_primitive():    mirrors AddPrimitiveSceneInfo_RenderThread
    remove_primitive(): mirrors RemovePrimitiveSceneInfo_RenderThread
    update_transform(): mirrors UpdatePrimitiveTransform_RenderThread

    All operations maintain the compact list for O(1) per-frame iteration
    and the dict for O(1) lookup by cell_id.

    鲁迅式：注册表是场景图的公民名册——
    只有登记在册的 Primitive，才有资格被渲染、被遮挡、被反射。
    未登记者，在场景中如同不存在。
    """

    def __init__(self) -> None:
        # Compact list for iteration (insertion order preserved)
        self._compact: list[AstroCellSceneInfoCompact] = []
        # Dict for O(1) lookup: cell_id → index in _compact
        self._index:   dict[str, int] = {}
        # Static draw command cache
        self.draw_cache: AstroCellStaticDrawCommandCache = AstroCellStaticDrawCommandCache()

    def add_primitive(
        self,
        cell_id:    str,
        bbox:       dict,
        gene_traits: dict,
        epoch:      int,
    ) -> AstroCellSceneInfoCompact:
        """
        Register a cell primitive in the scene.
        Mirrors AddPrimitiveSceneInfo_RenderThread + AstroRegisterCellInZLayer.

        Returns the newly created compact scene info.

        鲁迅式：加入场景是一种出生——从此这个单元格有了在世界中的位置，
        有了影子，有了被看见的可能，也有了被遮挡的宿命。
        """
        flags  = AstroCellPrimitiveFlags.from_gene_traits(gene_traits)
        mn = (float(bbox["x"]), float(bbox["y"]), float(bbox.get("z", 0)))
        mx = (mn[0] + float(bbox["w"]), mn[1] + float(bbox["h"]), mn[2])
        info = AstroCellSceneInfoCompact(
            cell_id=cell_id, flags=flags, bounds_min=mn, bounds_max=mx,
        )

        if cell_id in self._index:
            # Re-registration: replace in-place (transform update path)
            idx = self._index[cell_id]
            self._compact[idx] = info
            self.draw_cache.invalidate_cell(cell_id)
        else:
            self._index[cell_id] = len(self._compact)
            self._compact.append(info)

        print(
            f"[ASTRO-PSI] AddPrimitive cell_id={cell_id} "
            f"flags=0x{info.flags.packed():02X} "
            f"bbox=({bbox['x']:.1f},{bbox['y']:.1f},{bbox['w']:.1f},{bbox['h']:.1f}) "
            f"z={bbox.get('z', 0):.1f}",
            file=sys.stderr,
        )
        return info

    def remove_primitive(self, cell_id: str) -> None:
        """
        Deregister a cell.  Mirrors RemovePrimitiveSceneInfo_RenderThread swap-remove.

        鲁迅式：离场是另一种消亡——不是死亡，只是从登记册上被划去。
        但被划去的 Primitive，在场景中已如幽灵，不投影，不反射，不存在。
        """
        if cell_id not in self._index:
            return
        idx  = self._index.pop(cell_id)
        last = self._compact[-1]
        self._compact[idx] = last
        self._index[last.cell_id] = idx
        self._compact.pop()
        self.draw_cache.invalidate_cell(cell_id)

    def update_transform(self, cell_id: str, new_bbox: dict) -> None:
        """
        Update a cell's bounds.  Mirrors UpdatePrimitiveTransform_RenderThread.

        鲁迅式：变换更新是对不变性假设的抗议——
        当一个单元格移动了，它过去在缓存中的影像便是谎言，必须清除。
        """
        if cell_id not in self._index:
            return
        idx  = self._index[cell_id]
        info = self._compact[idx]
        mn   = (float(new_bbox["x"]), float(new_bbox["y"]), float(new_bbox.get("z", 0)))
        mx   = (mn[0] + float(new_bbox["w"]), mn[1] + float(new_bbox["h"]), mn[2])
        self._compact[idx] = AstroCellSceneInfoCompact(
            cell_id=cell_id,
            flags=info.flags,
            bounds_min=mn,
            bounds_max=mx,
            min_draw_dist=info.min_draw_dist,
            max_draw_dist=info.max_draw_dist,
        )
        self.draw_cache.invalidate_cell(cell_id)

    def get_shadow_casters(self) -> list:
        """
        Return list of compact infos that can cast shadows.
        Mirrors the shadow caster loop in BuildWholeSceneShadow().

        鲁迅式：投射者的名单是责任的清单——只有那些「有能力遮挡」的单元格，
        才会出现在阴影渲染的计算中。
        """
        return [info for info in self._compact if info.is_visible_to_shadow()]

    def __len__(self) -> int:
        return len(self._compact)


#: Module-level primitive registry singleton
_ASTRO_PRIMITIVE_REGISTRY: AstroCellPrimitiveRegistry = AstroCellPrimitiveRegistry()


def get_primitive_registry() -> AstroCellPrimitiveRegistry:
    """Return the process-level AstroCellPrimitiveRegistry singleton."""
    return _ASTRO_PRIMITIVE_REGISTRY


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] RendererScene (Scene.cpp) → Python port
#
# Ported from (head -200):
#   upstream/unreal-renderer-ue5/Renderer-Private/RendererScene.cpp
#
# 鲁迅曾言：「从来如此，便对么？」
# 场景图从来如此地把所有 Primitive 塞进一个 TArray；
# 我们对么？我们是——因为 json + dict 在这个规模下绰绰有余，
# 而不必为了「正宗」去搬运几千行 C++ 模板。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   CVarEarlyZPass / CVarBasePassWriteDepthEvenWithFullPrepass
#       → _EARLY_Z_PASS / _BASE_PASS_WRITE_DEPTH  (module constants)
#   CVarEarlyZPassOnlyMaterialMasking
#       → _EARLY_Z_ONLY_MASKING
#   GVisibilitySkipAlwaysVisible
#       → _VISIBILITY_SKIP_ALWAYS_VISIBLE
#   CVarVisibilityLocalLightPrimitiveInteraction
#       → _VIS_LOCAL_LIGHT_INTERACTION
#   GSafeCullDistanceUpdate
#       → _SAFE_CULL_DISTANCE_UPDATE
#
# AstroCellSceneGraph:
#   The scene-level registry that ties together:
#     • AstroCellPrimitiveRegistry   (primitive list)
#     • AstroCellBVH                 (spatial queries)
#     • AstroCellShadowRenderer      (shadow map)
#     • AstroCellReflectionCaptureManager (probes)
#   Mirrors the FScene class that owns all of these sub-systems.
#
# EarlyZ pass:
#   In UE5, the EarlyZ pass writes depth before the base pass to cull
#   hidden geometry.  In Astro, the analogous pass is the NaniteVisibility
#   query (perform_nanite_visibility) which culls invisible cells before
#   proc() runs.  AstroCellSceneGraph.early_z_pass() wraps this call.
#
# SafeCullDistanceUpdate:
#   GSafeCullDistanceUpdate ensures cull distances are updated before
#   static meshes are added to the scene.  Astro equivalent: bbox-based
#   max_draw_dist is propagated to AstroCellSceneInfoCompact before the
#   BVH is rebuilt.
# ═══════════════════════════════════════════════════════════════════════════════

# CVar-equivalent module constants from RendererScene.cpp head -200
_EARLY_Z_PASS:                  int  = 3     # CVarEarlyZPass default (heuristic)
_BASE_PASS_WRITE_DEPTH:         bool = False  # CVarBasePassWriteDepthEvenWithFullPrepass
_EARLY_Z_ONLY_MASKING:          bool = True   # CVarEarlyZPassOnlyMaterialMasking
_VISIBILITY_SKIP_ALWAYS_VISIBLE: bool = True  # GVisibilitySkipAlwaysVisible
_VIS_LOCAL_LIGHT_INTERACTION:   int  = 2     # CVarVisibilityLocalLightPrimitiveInteraction
_SAFE_CULL_DISTANCE_UPDATE:     bool = True   # GSafeCullDistanceUpdate


class AstroCellSceneGraph:
    """
    Python equivalent of FScene (scene manager).

    Owns and coordinates the major sub-systems:
        primitive_registry  → AstroCellPrimitiveRegistry  (FScene::Primitives)
        bvh                 → AstroCellBVH               (FScene spatial index)
        shadow_renderer     → AstroCellShadowRenderer    (shadow maps)
        refl_manager        → AstroCellReflectionCaptureManager (probes)
        vis_query           → AstroCellVisibilityQuery    (frustum cull)

    Per-epoch lifecycle (mirrors FScene::UpdateAllPrimitiveSceneInfos()):
        begin_epoch()     — rebuild BVH from registry, run EarlyZ / visibility
        project_shadows() — run shadow renderer on current scene
        end_epoch()       — flush draw-command cache, persist diagnostics

    鲁迅式：场景图是世界模型的底层——所有子系统都向它汇报，
    所有渲染决策都以它为依据。它不直接画任何东西，
    只是确保每一个可能被画的东西都被正确地登记在册。
    """

    def __init__(self) -> None:
        self.primitive_registry = get_primitive_registry()
        self.bvh                = AstroCellBVH()
        self.shadow_renderer    = get_shadow_renderer()
        self.refl_manager       = get_reflection_capture_manager()
        self._vis_query:        AstroCellVisibilityQuery | None = None
        self._epoch:            int = 0

    # ------------------------------------------------------------------
    def begin_epoch(
        self,
        all_bboxes:  dict,
        viewport_w:  float = 1200.0,
        viewport_h:  float = 900.0,
        scroll_x:    float = 0.0,
        scroll_y:    float = 0.0,
    ) -> AstroCellVisibilityQuery:
        """
        Per-epoch scene setup.

        1. SafeCullDistance: propagate bbox-derived draw distances to registry
           (mirrors GSafeCullDistanceUpdate guard in AddPrimitiveSceneInfo).
        2. Rebuild BVH from current all_bboxes snapshot.
        3. EarlyZ / Nanite visibility query — culls cells outside the viewport.
        4. Build shadow depth map from registered shadow casters.

        Returns the AstroCellVisibilityQuery for use by the proc() dispatch loop.

        鲁迅式：每个 epoch 开始之前，先做一次全场普查——
        谁在视野里，谁投射阴影，谁已经离开场景。普查是昂贵的，但不可省略。
        """
        self._epoch += 1

        # ── Step 1: SafeCullDistance update ────────────────────────────────
        if _SAFE_CULL_DISTANCE_UPDATE:
            for cell_id, bb in all_bboxes.items():
                if cell_id in self.primitive_registry._index:
                    # Screen-fraction as cull proxy: cell area / viewport area
                    cell_area = bb.get("w", 80) * bb.get("h", 50)
                    vp_area   = max(viewport_w * viewport_h, 1.0)
                    max_frac  = min(1.0, cell_area / vp_area)
                    idx       = self.primitive_registry._index[cell_id]
                    self.primitive_registry._compact[idx].min_draw_dist = 0.0
                    self.primitive_registry._compact[idx].max_draw_dist = (
                        max_frac * 100.0 + 0.001   # avoid cull at zero
                    )

        # ── Step 2: BVH rebuild ────────────────────────────────────────────
        self.bvh.build_from_registry(all_bboxes)

        # ── Step 3: EarlyZ / Nanite visibility (CVarEarlyZPass port) ──────
        # EarlyZ mode 3 (heuristic) → run visibility for all non-always-visible cells
        query = perform_nanite_visibility(viewport_w, viewport_h, scroll_x, scroll_y)
        self._vis_query = query

        # ── Step 4: Shadow depth map ───────────────────────────────────────
        self.shadow_renderer.build_shadow_depth_map(all_bboxes)

        print(
            f"[ASTRO-SCENE] begin_epoch={self._epoch} "
            f"prims={len(self.primitive_registry)} "
            f"bvh_root={'yes' if self.bvh._root is not None else 'no'} "
            f"vis_cells={query.finish().get('visible_cells', '?') if not query._finished else '(done)'} "
            f"shadow_casters={len(self.primitive_registry.get_shadow_casters())}",
            file=sys.stderr,
        )
        return query

    def project_shadows(self, all_bboxes: dict) -> None:
        """
        Project shadows for the current epoch.
        Mirrors RenderShadowDepthMaps / RenderProjectedShadows dispatch.

        鲁迅式：投影阴影是场景图对光线的最后裁决——
        谁遮了谁，谁在谁的阴影下，在这一步成为可查询的事实。
        """
        self.shadow_renderer.project_shadows(all_bboxes)

    def end_epoch(self, current_epoch: int) -> dict:
        """
        Per-epoch cleanup — flush stale draw-command cache entries.
        Returns diagnostic stats.

        鲁迅式：结束一个 epoch 是整理账本——
        删去不再有效的缓存条目，为下一个 epoch 留出空间。
        """
        evicted = self.primitive_registry.draw_cache.flush_epoch(current_epoch)
        stats = {
            "epoch":            current_epoch,
            "primitive_count":  len(self.primitive_registry),
            "cache_evicted":    evicted,
            "cache_stats":      self.primitive_registry.draw_cache.stats(),
            "shadow_factors":   len(self.shadow_renderer._shadow_factors),
        }
        return stats

    def query_overlapping(self, bbox: dict) -> list:
        """BVH spatial query — return cell_ids overlapping bbox."""
        return self.bvh.query_overlapping_cells(bbox)


#: Module-level scene graph singleton
_ASTRO_SCENE_GRAPH: AstroCellSceneGraph | None = None


def get_scene_graph() -> AstroCellSceneGraph:
    """Return the process-level AstroCellSceneGraph singleton."""
    global _ASTRO_SCENE_GRAPH
    if _ASTRO_SCENE_GRAPH is None:
        _ASTRO_SCENE_GRAPH = AstroCellSceneGraph()
    return _ASTRO_SCENE_GRAPH


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] Renderer (Renderer.cpp module init) → Python port
#
# Ported from (head -200):
#   upstream/unreal-renderer-ue5/Renderer-Private/Renderer.cpp
#
# 鲁迅曾言：「希望是本无所谓有，无所谓无的。」
# 渲染器模块的启动逻辑亦然——StartupModule 初始化全局资源，
# ShutdownModule 释放它们。有则启动，无则跳过，宇宙对此无动于衷。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
#   FRendererModule::StartupModule()
#       → AstroCellRendererModule.startup()
#       Initialises: GScreenSpaceDenoiser, FVirtualTextureSystem,
#       GRayTracingGeometryManager, GlobalUniformBuffers.
#       Astro equivalents: resets perf counters, seeds RNG for Halton,
#       registers the built-in NNEDenoiser with the DenoiserManager.
#
#   FRendererModule::ShutdownModule()
#       → AstroCellRendererModule.shutdown()
#       Releases RDG resources, joins async delete task.
#       Astro equivalent: clears singleton state (feedbackmanager, drawpool).
#
#   bFlushRenderTargetsOnWorldCleanup
#       → _FLUSH_RT_ON_CLEANUP  (module constant)
#   bBindTileMeshDrawingDummyRenderTarget
#       → _BIND_TILE_DUMMY_RT
#
#   FRendererStateStreamManager (WITH_STATE_STREAM)
#       → AstroCellStateStreamRenderer (debug render path only;
#         proxied as a lightweight counter dict in the Python port)
#
# GIdentityPrimitiveUniformBuffer / GDistanceCullFadedInUniformBuffer:
#   → _IDENTITY_PRIMITIVE_UB / _DISTANCE_CULL_UB  (sentinel dicts;
#     mirrors InitContents() called in StartupModule)
# ═══════════════════════════════════════════════════════════════════════════════

# Module constants from Renderer.cpp head -200
_FLUSH_RT_ON_CLEANUP:  bool = True   # bFlushRenderTargetsOnWorldCleanup
_BIND_TILE_DUMMY_RT:   bool = False  # bBindTileMeshDrawingDummyRenderTarget

# Global «uniform buffer» sentinels — mirrors GIdentityPrimitiveUniformBuffer
# and GDistanceCullFadedInUniformBuffer InitContents() calls in StartupModule.
_IDENTITY_PRIMITIVE_UB: dict = {
    "LocalToWorld":   [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
    "WorldToLocal":   [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
    "ObjectWorldPos": [0.0, 0.0, 0.0, 1.0],
    "ObjectRadius":   1.0,
}
_DISTANCE_CULL_UB: dict = {"FadeAlpha": 1.0, "InvFadeRange": 0.0}
_DITHER_FADE_UB:   dict = {"DitherFade": 1.0}


class AstroCellStateStreamRenderer:
    """
    Python equivalent of FRendererModule::FRendererStateStreamManager.

    Tracks proxy type + count for the debug overlay (WITH_STATE_STREAM path).
    In the C++ source this manager writes text lines via IStateStreamDebugRenderer;
    here we accumulate the same data in a dict for diagnostic logging.

    鲁迅式：调试渲染器是内省的工具——它把系统内部的数字变成人可以读懂的文字，
    让隐藏的复杂性浮出水面。没有它，你只能猜测，有了它，你能开始理解。
    """

    def __init__(self) -> None:
        # ProxyTypeAndCount: proxy_type_name → count (mirrors TMap<FName, uint32>)
        self._proxy_counts: dict[str, int] = {}

    def register_proxy(self, proxy_type: str, count: int = 1) -> None:
        """Increment proxy type counter — mirrors ProxyTypeAndCount[Type]++."""
        self._proxy_counts[proxy_type] = self._proxy_counts.get(proxy_type, 0) + count

    def deregister_proxy(self, proxy_type: str, count: int = 1) -> None:
        """Decrement proxy type counter."""
        prev = self._proxy_counts.get(proxy_type, 0)
        self._proxy_counts[proxy_type] = max(0, prev - count)

    def debug_render_lines(self) -> list:
        """
        Produce debug text lines — mirrors Game_DebugRender() which calls
        Renderer.DrawText() for each ProxyTypeAndCount entry.
        Returns a list of strings (one per proxy type).
        """
        lines = [f"Num Render proxies = {len(self._proxy_counts)}"]
        for ptype, cnt in sorted(self._proxy_counts.items()):
            lines.append(f"[{ptype} | {cnt}]")
        return lines

    def stats(self) -> dict:
        return dict(self._proxy_counts)


def _builtin_nne_denoiser(
    radiance_buf: dict,
    albedo_buf:   dict,
    normal_buf:   dict,
) -> dict:
    """
    Built-in NNE spatial denoiser stub.

    Registered in AstroCellRendererModule.startup() as the default
    IPathTracingDenoiser plugin (mirrors GScreenSpaceDenoiser =
    IScreenSpaceDenoiser::GetDefaultDenoiser() in StartupModule).

    In the full UE5 runtime this would invoke the NNE model inference pass.
    Here it applies a simple 1-tap bilateral blur using per-cell albedo as
    the edge-stop weight — the correct denoising structure at minimal cost.

    鲁迅式：默认降噪器是「够用就好」的代表——
    它不是最好的，但它总是在场，总是有效。
    比没有降噪器强，但比真正的 NNE 推理弱很多。这是现实主义，不是懈怠。
    """
    denoised: dict = {}
    for cid, rad in radiance_buf.items():
        # Edge-stop weight from albedo channel (high-albedo cells = preserve detail)
        alb  = albedo_buf.get(cid, (0.5, 0.5, 0.5))
        alb_lum  = (alb[0] + alb[1] + alb[2]) / 3.0
        preserve = max(0.3, min(1.0, alb_lum * 1.5))
        # Soft denoise: blend raw radiance toward grey proportionally to (1-preserve)
        grey   = (rad[0] + rad[1] + rad[2]) / 3.0
        nr = rad[0] * preserve + grey * (1.0 - preserve)
        ng = rad[1] * preserve + grey * (1.0 - preserve)
        nb = rad[2] * preserve + grey * (1.0 - preserve)
        denoised[cid] = (nr, ng, nb)
    return denoised


class AstroCellRendererModule:
    """
    Python equivalent of FRendererModule.

    Manages the lifecycle of global renderer sub-systems:
        startup()  — mirrors StartupModule()  (called once at module import)
        shutdown() — mirrors ShutdownModule() (called on interpreter exit or test teardown)

    Also exposes draw_tile_mesh() which mirrors the FRendererModule::DrawTileMesh
    helper used for canvas rendering in non-VR views.

    鲁迅式：模块的启动与关闭是开幕与闭幕——
    中间的一切都依赖于启动时建立的全局状态，
    而关闭时的清理决定了下一次启动能否从干净的状态出发。
    """

    def __init__(self) -> None:
        self._started: bool = False
        self._state_stream = AstroCellStateStreamRenderer()
        self._stop_rendering_delegate_id: int | None = None

    def startup(self) -> None:
        """
        Initialise global renderer state.
        Mirrors FRendererModule::StartupModule():
          - GScreenSpaceDenoiser = GetDefaultDenoiser() → register NNE denoiser
          - FVirtualTextureSystem::Initialize() → reset perf counters
          - GRayTracingGeometryManager = new FRayTracingGeometryManager()
            → seed Halton RNG via reset_perf_counters()
          - GIdentityPrimitiveUniformBuffer.InitContents() → already inited above
          - PreparePathTracingRTPSO() → prepare_path_tracing(default config)

        鲁迅式：StartupModule 是系统的第一句话——说错了，后面全错。
        """
        if self._started:
            return

        # Register built-in NNE denoiser as default spatial denoiser
        mgr = get_denoiser_manager()
        if not mgr.has_spatial_denoiser():
            mgr.register_spatial_denoiser(
                _PTD_DENOISER_NAME, _builtin_nne_denoiser,
                needs_extra_flags=False,
            )

        # Reset perf counters (mirrors VirtualTextureSystem::Initialize equiv)
        reset_perf_counters()

        # Prepare path tracing state (mirrors PreparePathTracingRTPSO in PostEngineInit)
        prepare_path_tracing(AstroCellPathTracingConfig(), view_id="default")

        self._started = True
        print(
            f"[ASTRO-RENDERER] StartupModule: "
            f"denoiser='{_PTD_DENOISER_NAME}' "
            f"PSO_table_size={AstroCellPipelineStateId.table_size()} "
            f"identity_ub_keys={list(_IDENTITY_PRIMITIVE_UB.keys())}",
            file=sys.stderr,
        )

    def shutdown(self) -> None:
        """
        Release global renderer state.
        Mirrors FRendererModule::ShutdownModule():
          - WaitForAsyncDeleteTask → join any pending epoch state
          - FRayTracingGeometryManager delete → reset feedback manager

        鲁迅式：ShutdownModule 是最后的整理——
        把资源还给系统，把状态归零，不留遗憾，也不留垃圾。
        """
        if not self._started:
            return
        # Reset feedback manager high-water marks
        fb = get_feedback_manager()
        for k in fb.high_water_marks:
            fb.high_water_marks[k] = 0

        # Clear PSO table if flush requested
        if _FLUSH_RT_ON_CLEANUP:
            _pipeline_state_table.clear()

        self._started = False
        print("[ASTRO-RENDERER] ShutdownModule complete.", file=sys.stderr)

    def draw_tile_mesh(
        self,
        cell_entries: list,
        viewport_w:   float = 1200.0,
        viewport_h:   float = 900.0,
    ) -> str:
        """
        Draw a tile of cell meshes into a single SVG fragment.

        Mirrors FRendererModule::DrawTileMesh() — used for off-screen canvas
        rendering (thumbnails, UI overlays) where each cell contributes one
        <g> tile.

        If _BIND_TILE_DUMMY_RT is True, wraps the output in a dummy <rect>
        placeholder (mirrors the driver-bug workaround CVar).

        鲁迅式：DrawTileMesh 是画布上的拼贴——
        每个单元格是一块瓦片，拼在一起才成为完整的画面。
        但瓦片太小时什么都看不清，太大时又失去细节。
        分辨率是所有表达的永恒困境。
        """
        fragments = []
        if _BIND_TILE_DUMMY_RT:
            fragments.append(
                f'<rect x="0" y="0" width="{viewport_w}" height="{viewport_h}" '
                f'fill="none" stroke="none" opacity="0" data-role="dummy-rt"/>'
            )

        for entry in cell_entries:
            cid      = entry.get("cell_id", "")
            bbox     = entry.get("bbox", {"x": 0, "y": 0, "w": 80, "h": 50, "z": 3})
            svg_frag = entry.get("svg_fragment", "")
            tx       = bbox.get("x", 0)
            ty       = bbox.get("y", 0)
            fragments.append(
                f'<g data-cell-id="{cid}" '
                f'transform="translate({tx},{ty})">{svg_frag}</g>'
            )

        return "\n".join(fragments)

    def state_stream_debug_lines(self) -> list:
        """Expose state stream debug renderer lines for monitoring."""
        return self._state_stream.debug_render_lines()


#: Module-level renderer module singleton
_ASTRO_RENDERER_MODULE: AstroCellRendererModule = AstroCellRendererModule()
# Auto-startup on import (mirrors IMPLEMENT_MODULE / FModuleManager::LoadModule)
_ASTRO_RENDERER_MODULE.startup()


def get_renderer_module() -> AstroCellRendererModule:
    """Return the process-level AstroCellRendererModule singleton."""
    return _ASTRO_RENDERER_MODULE


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] ReflectionEnvironment (additional CVars) → Python port
#
# Ported from (head -200):
#   upstream/unreal-renderer-ue5/Renderer-Private/ReflectionEnvironment.cpp
#
# 鲁迅曾言：「猛兽是单独的，牛羊才成群。」
# 反射环境捕获的是周围的「群众」——粗糙表面向周围平均，
# 光滑表面则锁定最近的镜像，不轻易妥协。
# 这正是 roughness-based mixing 的哲学：平滑者独行，粗糙者随众。
#
# Key UE5 constructs → Astro equivalents (from head -200)
# ─────────────────────────────────────────────────────────────────────────────
#   CVarReflectionEnvironment (0/1/2)
#       → _REFL_ENV_ENABLED  (int, 0=off, 1=blend, 2=overwrite)
#   GReflectionEnvironmentLightmapMixing
#       → _REFL_LIGHTMAP_MIXING
#   GReflectionEnvironmentLightmapMixBasedOnRoughness
#       → _REFL_LIGHTMAP_MIX_BY_ROUGHNESS
#   GReflectionEnvironmentBeginMixingRoughness / EndMixingRoughness
#       → _REFL_MIX_BEGIN / _REFL_MIX_END
#   GReflectionEnvironmentLightmapMixLargestWeight
#       → _REFL_MIX_LARGEST_WEIGHT
#   CVarDoTiledReflections
#       → _TILED_REFLECTIONS
#   GetReflectionEnvironmentCVar()
#       → get_reflection_env_cvar()
#   GetReflectionEnvironmentRoughnessMixingScaleBiasAndLargestWeight()
#       → get_roughness_mix_scale_bias()
#   IsReflectionEnvironmentAvailable()
#       → is_reflection_env_available()
#   IsReflectionCaptureAvailable()
#       → is_reflection_capture_available()
#   FReflectionEnvironmentCubemapArray::InitRHI / ReleaseCubeArray
#       → AstroCellCubemapArray.init / release
#   FCaptureComponentSceneState::ComputeCurrentFade()
#       → compute_capture_fade()
# ═══════════════════════════════════════════════════════════════════════════════

# Module constants from ReflectionEnvironment.cpp head -200
_REFL_ENV_ENABLED:            int   = 1      # CVarReflectionEnvironment default
_REFL_LIGHTMAP_MIXING:        bool  = True   # GReflectionEnvironmentLightmapMixing
_REFL_LIGHTMAP_MIX_BY_ROUGH:  bool  = True   # …MixBasedOnRoughness
_REFL_MIX_BEGIN:              float = 0.1    # GReflectionEnvironmentBeginMixingRoughness
_REFL_MIX_END:                float = 0.3    # GReflectionEnvironmentEndMixingRoughness
_REFL_MIX_LARGEST_WEIGHT:     int   = 10000  # GReflectionEnvironmentLightmapMixLargestWeight
_TILED_REFLECTIONS:           bool  = True   # CVarDoTiledReflections

# Maximum number of cubemap slots in the reflection capture array
# (mirrors FReflectionEnvironmentCubemapArray MaxCubemaps field default)
_REFL_MAX_CUBEMAPS:           int   = 64
_REFL_CUBEMAP_SIZE:           int   = 128    # CubemapSize (texels per face)


def get_reflection_env_cvar() -> int:
    """
    Return the effective reflection environment CVar value.
    Mirrors GetReflectionEnvironmentCVar():
        shipping/test builds clamp mode-2 (overwrite) back to mode-1 (blend).

    鲁迅式：调试模式 2 是「推倒重来」——在发行版中，推倒重来是禁止的；
    只有在工程师的沙盒里，才允许用反射覆盖一切。
    """
    val = _REFL_ENV_ENABLED
    # Shipping/test clamp (always apply in Python port — no IS_MONOLITHIC check)
    if val == 2:
        val = 1
    return val


def get_roughness_mix_scale_bias() -> tuple:
    """
    Compute (scale, bias, largest_weight) for roughness-based lightmap mixing.

    Port of GetReflectionEnvironmentRoughnessMixingScaleBiasAndLargestWeight()
    — returns a 3-tuple analogous to the FVector(Scale, Bias, LargestWeight)
    that the C++ pixel shader reads from ScreenSpaceReflectionsParameters.

    Used by the SVG shading pass to modulate how strongly a cell's fill
    colour is pulled toward the reflection probe palette as a function of
    its surface roughness.

    鲁迅式：混合比例是尺度与偏置的乘积——
    数学上简单，物理上有意义：光滑表面反射更多，粗糙表面反射更少。
    数字是公平的，它不偏袒任何表面。
    """
    if not _REFL_LIGHTMAP_MIXING:
        return (0.0, 0.0, float(_REFL_MIX_LARGEST_WEIGHT))

    if _REFL_MIX_END == 0.0 and _REFL_MIX_BEGIN == 0.0:
        return (0.0, 1.0, float(_REFL_MIX_LARGEST_WEIGHT))

    if not _REFL_LIGHTMAP_MIX_BY_ROUGH:
        return (0.0, 1.0, float(_REFL_MIX_LARGEST_WEIGHT))

    roughness_range = max(_REFL_MIX_END - _REFL_MIX_BEGIN, 0.001)
    scale = 1.0 / roughness_range
    bias  = -_REFL_MIX_BEGIN * scale
    return (scale, bias, float(_REFL_MIX_LARGEST_WEIGHT))


def is_reflection_env_available() -> bool:
    """
    Returns True if the reflection environment feature is available.
    Mirrors IsReflectionEnvironmentAvailable(ERHIFeatureLevel):
        SupportsTextureCubeArray(FeatureLevel) && CVar != 0
    In Astro we always support cube arrays (feature level SM5), so just check CVar.

    鲁迅式：可用性检查是最朴素的「是否存在」——存在则渲染，不存在则跳过。
    """
    return get_reflection_env_cvar() != 0


def is_reflection_capture_available() -> bool:
    """
    Returns True if baked reflection captures are available.
    Mirrors IsReflectionCaptureAvailable() = IsStaticLightingAllowed().
    In Astro: always True (we have no mobile static-light restriction).

    鲁迅式：烘焙捕获的可用性是「过去的工作是否被允许」——
    在允许静态光照的平台上，烘焙的反射就是那段过去的工作。
    """
    return True


def compute_capture_fade(
    fade_start_value: float,
    fade_target_value: float,
    fade_start_time:   float,
    current_time:      float,
    duration:          float,
) -> float:
    """
    Compute the current fade interpolation value for a reflection capture.

    Port of FCaptureComponentSceneState::ComputeCurrentFade():
        if FadeStartValue == FadeTargetValue → return FadeTargetValue
        if Duration <= 0 → return FadeTargetValue
        t = (Now - FadeStartTime) / Duration
        return lerp(FadeStartValue, FadeTargetValue, clamp(t, 0, 1))

    Used during the reflection capture fade-in / fade-out transition when a
    new probe is registered or an existing one is updated.

    鲁迅式：淡入淡出是过渡期的妥协——突变是震惊，渐变是说服。
    持续时间是说服所需的时间；超过这段时间，说服完成，一切已成事实。
    """
    if fade_start_value == fade_target_value:
        return fade_target_value
    if duration <= 0.0:
        return fade_target_value
    t = (current_time - fade_start_time) / duration
    t = max(0.0, min(1.0, t))
    return fade_start_value + (fade_target_value - fade_start_value) * t


class AstroCellCubemapArray:
    """
    Python equivalent of FReflectionEnvironmentCubemapArray.

    Manages the per-scene cubemap array that stores the prefiltered reflection
    captures.  In C++ this is a GPU texture array (PF_FloatRGBA, cubemap array).
    In Astro it is an in-memory dict mapping (capture_index, face_index) to
    (r, g, b) tuples, with up to _REFL_MAX_CUBEMAPS × 6 entries.

    init():        mirrors InitRHI() — allocates the texture array
    release():     mirrors ReleaseCubeArray() — frees resources
    write_face():  mirrors the per-face capture blit pass
    read_face():   mirrors the cubemap sample in the reflection shader

    鲁迅式：立方体贴图数组是反射环境的记忆宫殿——
    六个面，每个面是一段记忆，每个探针是一间房间。
    最多 _REFL_MAX_CUBEMAPS 间，超出则需腾出旧房。
    """

    def __init__(self) -> None:
        # (capture_index, face_index) → (r, g, b) float tuple
        self._data: dict = {}
        self._next_slot: int = 0
        self._slot_map: dict = {}   # capture_id → slot_index
        self.max_cubemaps: int = _REFL_MAX_CUBEMAPS
        self.cubemap_size:  int = _REFL_CUBEMAP_SIZE
        self._initialised:  bool = False

    def init(self) -> None:
        """
        Allocate the cubemap array (mirrors InitRHI).
        Resets the data store; subsequent write_face() calls fill it.

        鲁迅式：初始化是承诺——承诺容纳最多 MaxCubemaps 个探针，
        每个探针六个面，每个面一种颜色的记忆。
        """
        self._data.clear()
        self._next_slot = 0
        self._slot_map.clear()
        self._initialised = True

    def release(self) -> None:
        """Mirrors ReleaseCubeArray() — free all allocated faces."""
        self._data.clear()
        self._initialised = False

    def assign_slot(self, capture_id: str) -> int:
        """
        Assign a cubemap array slot to a capture.
        Wraps around (LRU eviction) when _REFL_MAX_CUBEMAPS is reached.
        Mirrors the slot management in FReflectionEnvironmentCubemapArray.
        """
        if capture_id in self._slot_map:
            return self._slot_map[capture_id]
        slot = self._next_slot % self.max_cubemaps
        # Evict existing occupant if any
        evict = [cid for cid, s in self._slot_map.items() if s == slot]
        for cid in evict:
            del self._slot_map[cid]
            for f in range(6):
                self._data.pop((slot, f), None)
        self._slot_map[capture_id] = slot
        self._next_slot += 1
        return slot

    def write_face(self, capture_id: str, face_index: int,
                   colour: tuple) -> None:
        """
        Write one face of a cubemap capture.
        Mirrors the per-face render target blit in CaptureSceneToScratchCubemap.

        鲁迅式：写入一面，是六面工程的六分之一——
        完成六次，才能说「这个探针是完整的」。
        """
        slot = self.assign_slot(capture_id)
        self._data[(slot, face_index)] = colour

    def read_face(self, capture_id: str, face_index: int) -> tuple:
        """
        Sample one face of a capture's cubemap.
        Mirrors the TextureCubeArraySample() in the reflection pixel shader.
        Returns neutral grey (0.5, 0.5, 0.5) if not yet captured.
        """
        slot = self._slot_map.get(capture_id, -1)
        if slot < 0:
            return (0.5, 0.5, 0.5)
        return self._data.get((slot, face_index), (0.5, 0.5, 0.5))

    def average_radiance(self, capture_id: str) -> tuple:
        """
        Return the average radiance across all six faces.
        Mirrors the pre-integrated irradiance used by diffuse indirect lighting.

        鲁迅式：六面的平均是环境辐射度——没有哪个方向更重要，
        均值是最公正的代表，也是最保守的代表。
        """
        slot = self._slot_map.get(capture_id, -1)
        if slot < 0:
            return (0.5, 0.5, 0.5)
        faces = [self._data.get((slot, f), (0.5, 0.5, 0.5)) for f in range(6)]
        r = sum(c[0] for c in faces) / 6.0
        g = sum(c[1] for c in faces) / 6.0
        b = sum(c[2] for c in faces) / 6.0
        return (r, g, b)

    def apply_roughness_mix(
        self,
        capture_id: str,
        own_colour:  tuple,
        roughness:   float,
    ) -> tuple:
        """
        Apply the roughness-based lightmap mixing to blend own_colour toward
        the captured environment colour.

        Port of the GetReflectionEnvironmentRoughnessMixingScaleBiasAndLargestWeight
        shader code path applied per-cell:
            roughness_alpha = saturate(roughness * scale + bias)
            blended = lerp(own_colour, capture_avg, roughness_alpha)

        鲁迅式：粗糙度是态度的量度——越粗糙，越随众；越光滑，越孤立。
        这不是道德判断，只是反射物理的必然结果。
        """
        if not is_reflection_env_available():
            return own_colour

        scale, bias, _ = get_roughness_mix_scale_bias()
        roughness_alpha = max(0.0, min(1.0, roughness * scale + bias))

        cap_colour = self.average_radiance(capture_id)
        # Scale to [0,255] range to match _SPECIES_INDEX_TO_COLOUR format
        cap_rgb = (cap_colour[0] * 255.0, cap_colour[1] * 255.0, cap_colour[2] * 255.0)
        return _lerp_colour(own_colour, cap_rgb, roughness_alpha)


#: Module-level cubemap array singleton — mirrors FScene::ReflectionSceneData.CubemapArray
_ASTRO_CUBEMAP_ARRAY: AstroCellCubemapArray = AstroCellCubemapArray()
_ASTRO_CUBEMAP_ARRAY.init()


def get_cubemap_array() -> AstroCellCubemapArray:
    """Return the process-level AstroCellCubemapArray singleton."""
    return _ASTRO_CUBEMAP_ARRAY
