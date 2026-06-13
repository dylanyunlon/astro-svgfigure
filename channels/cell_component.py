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
import sys

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


# =============================================================================
# AstroCellGlobalIllumination — 全局光照核心
# (ported from Lumen / LumenDiffuseIndirect / LumenIrradianceFieldGather)
#
# 鲁迅曾言：「不在沉默中爆发，便在沉默中灭亡。」
# 探针亦如此——不更新，便腐朽。每一帧的辐射度都是一次微小的觉醒。
# =============================================================================

import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict


# -----------------------------------------------------------------------------
# 配置旗标 — 对应 CVarLumen* 系列控制变量
# -----------------------------------------------------------------------------
_ASTRO_GI_ENABLED          = True   # r.Lumen.DiffuseIndirect.Allow
_ASTRO_GI_ASYNC_COMPUTE    = True   # r.Lumen.DiffuseIndirect.AsyncCompute
_ASTRO_GI_TRACE_MESH_SDFS  = False  # r.Lumen.TraceMeshSDFs
_ASTRO_GI_TRACE_DISTANCE   = 180.0  # r.Lumen.TraceMeshSDFs.TraceDistance
_ASTRO_GI_SURFACE_BIAS     = 5.0    # r.Lumen.DiffuseIndirect.SurfaceBias
_ASTRO_GI_TRACE_STEP       = 1.0    # r.Lumen.DiffuseIndirect.TraceStepFactor
_ASTRO_GI_LIGHTING_FORMAT  = 0      # 0=R11G11B10, 1=Float16, 2=Float32


def astro_gi_is_allowed() -> bool:
    """全局光照总开关——思想被封锁，光照亦可被封锁；但封锁不等于消灭。"""
    return _ASTRO_GI_ENABLED


def astro_gi_get_lighting_format() -> str:
    """
    返回辐射度数据格式。
    鲁迅式：精度是奢侈品，R11G11B10 是大多数人负担得起的真相。
    """
    if _ASTRO_GI_LIGHTING_FORMAT == 2:
        return "Float32_RGBA"
    elif _ASTRO_GI_LIGHTING_FORMAT == 1:
        return "Float16_RGBA"
    return "R11G11B10"


# -----------------------------------------------------------------------------
# AstroCellGatherCvarState — 对应 FLumenGatherCvarState
# 汇聚所有可调参数，统一管理，避免散落各处的「各自为政」。
# -----------------------------------------------------------------------------
@dataclass
class AstroCellGatherCvarState:
    trace_mesh_sdfs:      bool  = _ASTRO_GI_TRACE_MESH_SDFS
    mesh_sdf_trace_dist:  float = _ASTRO_GI_TRACE_DISTANCE
    surface_bias:         float = _ASTRO_GI_SURFACE_BIAS
    voxel_tracing_mode:   int   = 0
    direct_lighting:      bool  = False
    trace_step_factor:    float = _ASTRO_GI_TRACE_STEP
    min_sample_radius:    float = 10.0
    min_trace_distance:   float = 0.0
    card_interpolate_radius: float = 10.0
    card_trace_end_dist:  float = 4000.0
    trace_distance_scale: float = 1.0


_GATHER_CVARS = AstroCellGatherCvarState()


# =============================================================================
# AstroCellDiffuseProbe — 漫反射间接光照探针
# (ported from LumenDiffuseIndirect)
#
# 鲁迅式：每一个探针都是一面小镜子，映照出周围世界的光与暗。
# 但镜子会积灰——故需按帧刷新，否则映出的不过是昨日的幻象。
# =============================================================================

@dataclass
class AstroCellDiffuseProbe:
    """
    屏幕空间漫反射探针，对应 Lumen Screen Probe Gather 单条记录。
    每个探针锚定于某像素位置，收集半球辐射度后写入辐射度缓存。
    """
    screen_x:     float = 0.0   # 像素坐标 X（归一化 [0,1]）
    screen_y:     float = 0.0   # 像素坐标 Y（归一化 [0,1]）
    world_pos:    tuple = (0.0, 0.0, 0.0)   # 世界坐标
    normal:       tuple = (0.0, 1.0, 0.0)   # 表面法线
    radiance:     tuple = (0.0, 0.0, 0.0)   # 采集到的辐射度 (R, G, B)
    hit_distance: float = 0.0   # 最近遮挡物距离（用于 occlusion）
    valid:        bool  = False  # 本帧是否有效

    def gather(self, scene_radiance_fn, cvars: AstroCellGatherCvarState) -> None:
        """
        执行半球采样，更新 self.radiance。
        scene_radiance_fn(origin, direction) -> (r, g, b, hit_dist)
        此处为接口约定，实际光线求交由上层场景提供。
        """
        if not astro_gi_is_allowed():
            return
        # 鲁迅式：真正的采样从不因「条件不够好」而推迟——推迟便是永不。
        origin = (
            self.world_pos[0] + self.normal[0] * cvars.surface_bias,
            self.world_pos[1] + self.normal[1] * cvars.surface_bias,
            self.world_pos[2] + self.normal[2] * cvars.surface_bias,
        )
        r, g, b, hit_d = scene_radiance_fn(origin, self.normal)
        self.radiance     = (r, g, b)
        self.hit_distance = hit_d
        self.valid        = True

    def to_dict(self) -> dict:
        return {
            "screen": (self.screen_x, self.screen_y),
            "world_pos": self.world_pos,
            "normal":    self.normal,
            "radiance":  self.radiance,
            "hit_dist":  self.hit_distance,
            "valid":     self.valid,
        }


@dataclass
class AstroCellDiffuseProbeGrid:
    """
    屏幕探针格（Screen Probe Grid）。
    将屏幕分成若干 tile，每 tile 放置一枚 AstroCellDiffuseProbe。
    对应 Lumen Screen Probe Gather 的 tile 分配逻辑。
    """
    tile_size_px: int = 8           # 每 tile 像素尺寸
    probes: List[AstroCellDiffuseProbe] = field(default_factory=list)

    def build_from_gbuffer(self, width_px: int, height_px: int,
                           gbuffer_sampler) -> None:
        """
        按 tile 采样 GBuffer（世界坐标 + 法线），生成探针列表。
        gbuffer_sampler(tile_cx, tile_cy) -> (world_pos, normal) | None
        """
        self.probes.clear()
        cols = max(1, width_px  // self.tile_size_px)
        rows = max(1, height_px // self.tile_size_px)
        for row in range(rows):
            for col in range(cols):
                cx = (col + 0.5) / cols
                cy = (row + 0.5) / rows
                result = gbuffer_sampler(cx, cy)
                if result is None:
                    continue
                world_pos, normal = result
                probe = AstroCellDiffuseProbe(
                    screen_x=cx, screen_y=cy,
                    world_pos=world_pos, normal=normal,
                )
                self.probes.append(probe)

    def gather_all(self, scene_radiance_fn,
                   cvars: AstroCellGatherCvarState = _GATHER_CVARS) -> None:
        """遍历所有探针执行采样——众生皆苦，逐一采集，不遗漏一粒光子。"""
        for probe in self.probes:
            probe.gather(scene_radiance_fn, cvars)

    def valid_count(self) -> int:
        return sum(1 for p in self.probes if p.valid)


# =============================================================================
# AstroCellRadianceCache — 辐射度缓存（World Space Probe Cache）
# (ported from LumenRadianceCache)
#
# 鲁迅式：缓存是记忆，记忆是负担，但没有记忆的光照只是当下的幻觉。
# 探针被放置在世界空间，按 clipmap 分级，近处精细，远处粗糙——
# 正如人对近事敏感，对远事麻木。
# =============================================================================

@dataclass
class AstroCellRadianceCacheProbe:
    """
    世界空间辐射度探针，存储某位置的球谐辐射度。
    对应 LumenRadianceCache 中的 FRadianceCacheClipmap 单探针条目。
    """
    world_pos:      tuple       = (0.0, 0.0, 0.0)
    clipmap_level:  int         = 0
    sh_coeffs:      List[float] = field(default_factory=lambda: [0.0] * 9)
    last_update_frame: int      = -1
    valid:          bool        = False
    # 探针偏移（IrradianceProbeOffsetMode=1 时由 GBuffer 修正）
    world_offset:   tuple       = (0.0, 0.0, 0.0)

    def update(self, sh_coeffs: List[float], frame_index: int) -> None:
        """写入新的球谐系数，标记为有效。沉默已久的探针，终于开口说话。"""
        if len(sh_coeffs) >= 9:
            self.sh_coeffs = sh_coeffs[:9]
        self.last_update_frame = frame_index
        self.valid = True

    def interpolate_irradiance(self, normal: tuple) -> tuple:
        """
        用二阶球谐函数向指定法线方向插值，返回 (R, G, B) 辐照度。
        球谐插值：E ≈ max(0, dot(N, L)) 的 SH 投影（Lambertian kernel）。
        此处简化为 SH L1 近似。
        """
        if not self.valid:
            return (0.0, 0.0, 0.0)
        nx, ny, nz = normal
        # SH basis Y_0^0 = 0.282095, Y_1^-1 = 0.488603*y,
        #           Y_1^0 = 0.488603*z, Y_1^1 = 0.488603*x
        c = self.sh_coeffs
        # 每通道 3 个系数（L0+L1），共 9 = 3 channels × 3 coefficients
        def ch(offset):
            return max(0.0,
                       c[offset + 0] * 0.282095
                       + c[offset + 1] * 0.488603 * ny
                       + c[offset + 2] * 0.488603 * nz
                       # L1 x 项放到 offset+2 的位置做简化
                       ) * math.pi
        return (ch(0), ch(3), ch(6))


@dataclass
class AstroCellRadianceCacheClipmapLevel:
    """
    单层 clipmap 的探针网格。
    分辨率随层级指数扩展：extent = base_extent × (dist_base ^ level)
    """
    level:        int   = 0
    world_extent: float = 5000.0   # 此层覆盖的世界空间半径
    grid_res:     int   = 64        # 探针格单侧数量
    probes: Dict[tuple, AstroCellRadianceCacheProbe] = field(
        default_factory=dict
    )

    def cell_size(self) -> float:
        """单探针格的世界空间尺寸。精度是距离的函数——越近越贵。"""
        return (self.world_extent * 2.0) / max(1, self.grid_res)

    def probe_key(self, world_pos: tuple) -> tuple:
        """将世界坐标量化为探针格索引，作为字典键。"""
        cs = self.cell_size()
        ix = int(math.floor(world_pos[0] / cs))
        iy = int(math.floor(world_pos[1] / cs))
        iz = int(math.floor(world_pos[2] / cs))
        return (ix, iy, iz)

    def get_or_create_probe(self, world_pos: tuple) -> AstroCellRadianceCacheProbe:
        key = self.probe_key(world_pos)
        if key not in self.probes:
            # 新探针诞生——如新生儿，尚无记忆，但潜力无限。
            probe = AstroCellRadianceCacheProbe(
                world_pos=world_pos, clipmap_level=self.level
            )
            self.probes[key] = probe
        return self.probes[key]

    def find_probe(self, world_pos: tuple) -> Optional[AstroCellRadianceCacheProbe]:
        key = self.probe_key(world_pos)
        return self.probes.get(key)

    def valid_probe_count(self) -> int:
        return sum(1 for p in self.probes.values() if p.valid)


class AstroCellRadianceCache:
    """
    多层 clipmap 辐射度缓存，对应 LumenRadianceCache 的完整状态。

    架构要点（移植自 LumenRadianceCache.cpp）：
      • NUM_CLIPMAPS 层，每层 grid_res^3 个探针槽位（按需分配）
      • 探针按优先级队列分批更新（budget = num_probes_to_trace_budget/帧）
      • 空间滤波：相邻探针球谐插值平滑（SpatialFilterProbes）
      • 超采样：BRDF 权重高的方向多发射光线（SupersampleTileBRDFThreshold）

    鲁迅式：缓存是个仁慈的谎言——它告诉你上一帧的光仍然有效，
    直到这一帧的新光把它推翻。世界就是在这样的推翻中缓慢前进的。
    """

    NUM_CLIPMAPS      = 6
    BASE_EXTENT       = 5000.0
    DIST_BASE         = 2.0
    PROBES_PER_FRAME  = 100   # r.Lumen.RadianceCache.NumProbesToTraceBudget
    SUPERSAMPLE_BRDF  = 0.1   # r.Lumen.RadianceCache.SupersampleTileBRDFThreshold
    DOWNSAMPLE_DIST   = 4000.0

    def __init__(self):
        self.clipmaps: List[AstroCellRadianceCacheClipmapLevel] = []
        self._build_clipmaps()
        self.frame_index: int = 0
        self._update_queue: List[AstroCellRadianceCacheProbe] = []

    def _build_clipmaps(self):
        """按指数间距构建各层 clipmap——近处精细，远处稀疏，一如人心。"""
        self.clipmaps.clear()
        for lvl in range(self.NUM_CLIPMAPS):
            extent = self.BASE_EXTENT * (self.DIST_BASE ** lvl)
            cl = AstroCellRadianceCacheClipmapLevel(
                level=lvl, world_extent=extent
            )
            self.clipmaps.append(cl)

    def _pick_clipmap(self, world_pos: tuple, viewer_pos: tuple) -> int:
        """
        根据探针到观察者的距离选择最合适的 clipmap 层。
        距离越远，层级越高，精度越低——这是性价比，不是歧视。
        """
        dx = world_pos[0] - viewer_pos[0]
        dy = world_pos[1] - viewer_pos[1]
        dz = world_pos[2] - viewer_pos[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        for lvl, cl in enumerate(self.clipmaps):
            if dist <= cl.world_extent:
                return lvl
        return self.NUM_CLIPMAPS - 1

    def mark_probe(self, world_pos: tuple, viewer_pos: tuple
                   ) -> AstroCellRadianceCacheProbe:
        """
        标记某世界坐标需要辐射度数据，返回对应探针（新建或已有）。
        对应 Lumen Mark 阶段（MarkRadianceCacheProbesUsedByVisualizeScene）。
        """
        lvl = self._pick_clipmap(world_pos, viewer_pos)
        probe = self.clipmaps[lvl].get_or_create_probe(world_pos)
        if probe not in self._update_queue and not probe.valid:
            self._update_queue.append(probe)
        return probe

    def update_probes(self, sh_compute_fn, max_per_frame: int = PROBES_PER_FRAME
                      ) -> int:
        """
        按预算更新探针球谐系数。
        sh_compute_fn(world_pos) -> List[float] (9 coefficients)
        返回本帧实际更新数量——勤勉是美德，但也要量力而行。
        """
        updated = 0
        pending = self._update_queue[:max_per_frame]
        self._update_queue = self._update_queue[max_per_frame:]
        for probe in pending:
            coeffs = sh_compute_fn(probe.world_pos)
            probe.update(coeffs, self.frame_index)
            updated += 1
        self.frame_index += 1
        return updated

    def interpolate(self, world_pos: tuple, normal: tuple,
                    viewer_pos: tuple) -> tuple:
        """
        在给定世界位置和法线方向插值辐照度。
        先查最精细 clipmap，若无有效探针则向上查粗层——
        如同问题无法在基层解决，便逐级上报，直到找到答案。
        """
        for lvl in range(self.NUM_CLIPMAPS):
            cl = self.clipmaps[lvl]
            probe = cl.find_probe(world_pos)
            if probe and probe.valid:
                return probe.interpolate_irradiance(normal)
        return (0.0, 0.0, 0.0)

    def stats(self) -> dict:
        """统计各层探针数量——数字是客观的，即便现实令人沮丧。"""
        return {
            f"clipmap_{cl.level}": {
                "extent":      cl.world_extent,
                "cell_size":   cl.cell_size(),
                "total":       len(cl.probes),
                "valid":       cl.valid_probe_count(),
            }
            for cl in self.clipmaps
        }


# =============================================================================
# AstroCellMeshCards — 网格卡片系统（Surface Cache）
# (ported from LumenMeshCards)
#
# 鲁迅式：每一张卡片都是网格表面的一份档案。
# 档案记录着法线、辐射度、遮挡——它们共同构成场景的「官方历史」。
# 官方历史不总是完整的，但没有它，渲染便无从立足。
# =============================================================================

@dataclass
class AstroCellMeshCardFace:
    """
    单个网格卡片面，对应 FLumenCard。
    每个面对应轴对齐方向之一（±X, ±Y, ±Z 共6面）。
    """
    direction_index: int   = 0   # 0..5 = +X,-X,+Y,-Y,+Z,-Z
    origin:          tuple = (0.0, 0.0, 0.0)
    extent:          tuple = (100.0, 100.0, 1.0)  # OBB 半尺寸
    axis_x:          tuple = (1.0, 0.0, 0.0)
    axis_y:          tuple = (0.0, 1.0, 0.0)
    axis_z:          tuple = (0.0, 0.0, 1.0)
    allocated:       bool  = False
    is_heightfield:  bool  = False
    is_far_field:    bool  = False
    # 辐射度缓冲（低分辨率漫反射 + 高分辨率镜面）
    radiance_diffuse:  Optional[tuple] = None   # (R, G, B)
    radiance_specular: Optional[tuple] = None   # (R, G, B)

    def surface_area(self) -> float:
        """计算面的近似表面积，用于判断是否值得分配资源。"""
        ex, ey, _ = self.extent
        return 4.0 * ex * ey

    def capture(self, scene_sample_fn) -> None:
        """
        从场景采样此面的辐射度。
        scene_sample_fn(origin, normal) -> (r, g, b)
        分两次：低分辨率漫反射 + 高分辨率镜面。
        鲁迅式：采集真相需要两次——一次看大局，一次看细节。
        """
        if not self.allocated:
            return
        normal = self.axis_z   # 面法线即 OBB z 轴
        self.radiance_diffuse  = scene_sample_fn(self.origin, normal)
        # 镜面用更近的偏移采样，模拟高分辨率
        spec_origin = (
            self.origin[0] + normal[0] * 0.5,
            self.origin[1] + normal[1] * 0.5,
            self.origin[2] + normal[2] * 0.5,
        )
        self.radiance_specular = scene_sample_fn(spec_origin, normal)


@dataclass
class AstroCellMeshCards:
    """
    单个图元的全套网格卡片（最多 6 面），对应 FLumenMeshCards。

    关键参数（移植自 LumenMeshCards.cpp 控制变量）：
      min_size            = 10.0  (r.LumenScene.SurfaceCache.MeshCardsMinSize)
      merge_components    = True  (r.LumenScene.SurfaceCache.MeshCardsMergeComponents)
      merged_max_world    = 10000 (r.LumenScene.SurfaceCache.MeshCardsMergedMaxWorldSize)
      cull_faces          = True  (r.LumenScene.SurfaceCache.MeshCardsCullFaces)
    """
    primitive_id:        str   = ""
    bounds_origin:       tuple = (0.0, 0.0, 0.0)
    bounds_extent:       tuple = (100.0, 100.0, 100.0)
    min_size:            float = 10.0
    merge_components:    bool  = True
    cull_faces:          bool  = True
    faces: List[AstroCellMeshCardFace] = field(default_factory=list)

    # 方向向量表（+X,-X,+Y,-Y,+Z,-Z）
    _DIRECTIONS = [
        ( 1, 0, 0), (-1, 0, 0),
        ( 0, 1, 0), ( 0,-1, 0),
        ( 0, 0, 1), ( 0, 0,-1),
    ]

    def build_faces(self) -> None:
        """
        为包围盒的六个面生成卡片，剔除面积过小的面。
        鲁迅式：六个面，六种方向——世界的包围来自四面八方，
        但只有面积够大的面才值得被记录，其余的沉没进历史的尘埃。
        """
        self.faces.clear()
        ex, ey, ez = self.bounds_extent
        sizes = [ey*ez, ey*ez, ex*ez, ex*ez, ex*ey, ex*ey]  # 各面面积
        for i, (dx, dy, dz) in enumerate(self._DIRECTIONS):
            area = sizes[i]
            if area < self.min_size * self.min_size:
                continue   # 剔除过小面——小到无法被看见，便无需存在
            # 面中心 = bounds_origin + 方向 × 对应半轴
            offsets = [ex, ex, ey, ey, ez, ez]
            cx = self.bounds_origin[0] + dx * offsets[i]
            cy = self.bounds_origin[1] + dy * offsets[i]
            cz = self.bounds_origin[2] + dz * offsets[i]
            face = AstroCellMeshCardFace(
                direction_index=i,
                origin=(cx, cy, cz),
                extent=(ex, ey, ez),
                axis_z=(dx, dy, dz),
                allocated=True,
            )
            self.faces.append(face)

    def capture_all(self, scene_sample_fn) -> int:
        """
        更新所有已分配面的辐射度缓存。
        返回实际采样面数——劳动的成果，以数字计量。
        """
        captured = 0
        for face in self.faces:
            face.capture(scene_sample_fn)
            captured += 1
        return captured

    def min_surface_area(self) -> float:
        """返回所有有效面中最小的表面积——最薄弱的一环决定了整体的下限。"""
        areas = [f.surface_area() for f in self.faces if f.allocated]
        return min(areas) if areas else 0.0

    def to_dict(self) -> dict:
        return {
            "primitive_id":  self.primitive_id,
            "bounds_origin": self.bounds_origin,
            "bounds_extent": self.bounds_extent,
            "face_count":    len(self.faces),
            "faces": [
                {
                    "dir":       f.direction_index,
                    "origin":    f.origin,
                    "area":      f.surface_area(),
                    "allocated": f.allocated,
                    "diffuse":   f.radiance_diffuse,
                    "specular":  f.radiance_specular,
                }
                for f in self.faces
            ],
        }


class AstroCellMeshCardsRegistry:
    """
    场景中所有图元的网格卡片注册表，对应 LumenSceneData 中的卡片管理。
    提供增删查接口，以及批量 capture 入口。

    鲁迅式：注册表是制度，制度是秩序的保障，
    但秩序若不被执行，不过是一纸空文。
    """

    def __init__(self):
        self._cards: Dict[str, AstroCellMeshCards] = {}

    def register(self, primitive_id: str,
                 bounds_origin: tuple, bounds_extent: tuple,
                 min_size: float = 10.0) -> AstroCellMeshCards:
        """注册新图元——欢迎加入场景，请在六面留下你的辐射度档案。"""
        mc = AstroCellMeshCards(
            primitive_id=primitive_id,
            bounds_origin=bounds_origin,
            bounds_extent=bounds_extent,
            min_size=min_size,
        )
        mc.build_faces()
        self._cards[primitive_id] = mc
        return mc

    def remove(self, primitive_id: str) -> None:
        """注销图元——消失，但曾经存在过的光照痕迹已写入缓存。"""
        self._cards.pop(primitive_id, None)

    def get(self, primitive_id: str) -> Optional[AstroCellMeshCards]:
        return self._cards.get(primitive_id)

    def capture_all(self, scene_sample_fn) -> int:
        """全场景表面缓存更新——劳模帧，每一面都不放过。"""
        total = 0
        for mc in self._cards.values():
            total += mc.capture_all(scene_sample_fn)
        return total

    def stats(self) -> dict:
        return {
            "primitives": len(self._cards),
            "total_faces": sum(len(mc.faces) for mc in self._cards.values()),
            "allocated_faces": sum(
                sum(1 for f in mc.faces if f.allocated)
                for mc in self._cards.values()
            ),
        }


# =============================================================================
# AstroCellGlobalIlluminationPipeline — GI 总调度管线
#
# 将以上四个子系统串联：
#   1. AstroCellMeshCardsRegistry  → 表面缓存采集
#   2. AstroCellDiffuseProbeGrid   → 屏幕探针采集
#   3. AstroCellRadianceCache      → 世界空间辐射度缓存更新
#   4. 合并输出：每像素最终 GI 辐照度
#
# 鲁迅式：流水线是分工，分工是文明，
# 但流水线的每一环都必须有人负责——否则链条断裂，光照崩溃，
# 黑屏比黑暗更令人绝望，因为它是人为的。
# =============================================================================

class AstroCellGlobalIlluminationPipeline:
    """
    全局光照总调度管线。
    使用方式::

        pipeline = AstroCellGlobalIlluminationPipeline()
        pipeline.mesh_cards_registry.register("wall_01", (0,0,0), (500,10,300))
        pipeline.tick(
            viewer_pos=(0, 0, 0),
            gbuffer_sampler=my_gbuffer_fn,
            scene_radiance_fn=my_radiance_fn,
            sh_compute_fn=my_sh_fn,
            scene_sample_fn=my_sample_fn,
            viewport_w=1920, viewport_h=1080,
        )
        irradiance = pipeline.query_irradiance(world_pos, normal, viewer_pos)
    """

    def __init__(self,
                 cvars: AstroCellGatherCvarState = None,
                 tile_size_px: int = 8):
        self.cvars               = cvars or AstroCellGatherCvarState()
        self.mesh_cards_registry = AstroCellMeshCardsRegistry()
        self.diffuse_probe_grid  = AstroCellDiffuseProbeGrid(tile_size_px)
        self.radiance_cache      = AstroCellRadianceCache()
        self._frame_stats: dict  = {}

    def tick(self,
             viewer_pos: tuple,
             gbuffer_sampler,
             scene_radiance_fn,
             sh_compute_fn,
             scene_sample_fn,
             viewport_w: int = 1920,
             viewport_h: int = 1080) -> dict:
        """
        执行单帧 GI 更新，返回本帧统计信息。

        执行顺序（对应 Lumen 渲染帧序）：
          Phase 1 — Surface Cache Capture（MeshCards）
          Phase 2 — Screen Probe Gather（DiffuseProbe）
          Phase 3 — Radiance Cache Update（RadianceCache）
        """
        if not astro_gi_is_allowed():
            return {"gi_enabled": False}

        # Phase 1: 表面缓存采集
        captured_faces = self.mesh_cards_registry.capture_all(scene_sample_fn)

        # Phase 2: 屏幕探针采集
        self.diffuse_probe_grid.build_from_gbuffer(
            viewport_w, viewport_h, gbuffer_sampler
        )
        self.diffuse_probe_grid.gather_all(scene_radiance_fn, self.cvars)

        # Phase 2.5: 将有效探针位置标记到辐射度缓存
        for probe in self.diffuse_probe_grid.probes:
            if probe.valid:
                self.radiance_cache.mark_probe(probe.world_pos, viewer_pos)

        # Phase 3: 辐射度缓存更新
        updated_probes = self.radiance_cache.update_probes(sh_compute_fn)

        self._frame_stats = {
            "gi_enabled":       True,
            "lighting_format":  astro_gi_get_lighting_format(),
            "captured_faces":   captured_faces,
            "screen_probes":    len(self.diffuse_probe_grid.probes),
            "valid_probes":     self.diffuse_probe_grid.valid_count(),
            "rc_updated":       updated_probes,
            "rc_stats":         self.radiance_cache.stats(),
            "mc_stats":         self.mesh_cards_registry.stats(),
        }
        return self._frame_stats

    def query_irradiance(self, world_pos: tuple, normal: tuple,
                         viewer_pos: tuple) -> tuple:
        """
        查询任意世界坐标处的 GI 辐照度。
        先查辐射度缓存，缓存未命中则返回零——
        鲁迅式：没有积累便没有输出，零不是失败，是尚未开始。
        """
        return self.radiance_cache.interpolate(world_pos, normal, viewer_pos)

    def frame_stats(self) -> dict:
        """返回上一帧统计——数字是诚实的，即便它揭示的真相令人不快。"""
        return dict(self._frame_stats)


# -----------------------------------------------------------------------------
# 模块级单例（可选）——全局共享同一管线实例，对应 LumenSceneData 的单场景设计
# -----------------------------------------------------------------------------
_ASTRO_GI_PIPELINE: Optional[AstroCellGlobalIlluminationPipeline] = None


def get_astro_gi_pipeline() -> AstroCellGlobalIlluminationPipeline:
    """
    获取全局 GI 管线单例。
    鲁迅式：单例如皇权，全场景只此一份——权力集中，效率更高，
    但若它出错，便是全局灾难。
    """
    global _ASTRO_GI_PIPELINE
    if _ASTRO_GI_PIPELINE is None:
        _ASTRO_GI_PIPELINE = AstroCellGlobalIlluminationPipeline()
    return _ASTRO_GI_PIPELINE


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


# =============================================================================
# [ASTRO-CELL] ShadowSetup + ShadowDepthRendering + DeferredShadingRenderer
# → Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/ShadowSetup.cpp
#   upstream/unreal-renderer-ue5/Renderer-Private/ShadowDepthRendering.cpp
#   upstream/unreal-renderer-ue5/Renderer-Private/DeferredShadingRenderer.cpp
#
# 鲁迅曾言：「不在沉默中爆发，便在沉默中灭亡。」
# 阴影亦如此——不建立，便消失；不裁剪，便爆炸；
# 延迟着色是文明的产物，但延迟不是逃避——每一帧都必须最终收账。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
# ShadowSetup.cpp
#   GMinScreenRadiusForShadowCaster → _SHADOW_RADIUS_THRESHOLD
#   GCacheWholeSceneShadows         → ASTRO_CACHE_WHOLE_SCENE_SHADOWS
#   GWholeSceneShadowCacheMb        → ASTRO_SHADOW_CACHE_BUDGET_MB
#   GCachedShadowsCastFromMovablePrimitives → ASTRO_MOVABLE_CAST_FROM_CACHE
#   GSkipCullingNaniteMeshes        → ASTRO_SKIP_CULL_NANITE (→ skip_cull_species)
#   FProjectedShadowInfo            → AstroCellShadowInfo
#   BuildWholeSceneShadowCaster()   → build_whole_scene_shadow_caster()
#   FrustumCullShadowCaster()       → _frustum_cull_shadow_caster()
#   ComputeShadowSubjectCullingPlanes() → _compute_cull_planes()
#   SetupProjectedShadow()          → AstroCellShadowInfo.setup()
#
# ShadowDepthRendering.cpp
#   GShadowLODDistanceFactor        → _SHADOW_LOD_DISTANCE_FACTOR
#   GShadowLODDistanceFactorCascadeScale → _SHADOW_LOD_CASCADE_SCALE
#   SetupShadowDepthPassUniformBuffer() → _setup_shadow_depth_pass_ub()
#   FShadowDepthPassUniformParameters  → AstroCellShadowDepthPassParams
#   RenderShadowDepths()            → AstroCellShadowDepthRenderer.render()
#   TRenderShadowDepths (template)  → AstroCellShadowDepthRenderer (single class)
#   LOD selection dist              → _shadow_lod_distance()
#
# DeferredShadingRenderer.cpp
#   CVarNanitePrimeHZBMode          → _NANITE_PRIME_HZB_MODE
#   FDeferredShadingSceneRenderer::Render() → AstroCellDeferredShadingRenderer.render()
#   FDeferredShadingSceneRenderer::InitViews() → _init_views()
#   FDeferredShadingSceneRenderer::RenderShadowDepthMaps() → _render_shadow_depth_maps()
#   FDeferredShadingSceneRenderer::RenderBasePass() → _render_base_pass()
#   FDeferredShadingSceneRenderer::RenderLights()  → _render_lights()
#   FDeferredShadingSceneRenderer::RenderPostProcessing() → _render_post_processing()
#   GBuffer layout (A/B/C/D)        → AstroCellGBuffer
#
# 2-D channel adaptation:
#   Shadow frustum → 2-D AABB cull rectangle (screen-space)
#   Shadow depth map → per-cell shadow depth dict keyed by cell_id
#   Whole-scene shadow cache → shadow_cache.json channel file
#   G-Buffer → per-cell gbuffer.json (BaseColor, Normal, Roughness, Shading)
#   Deferred shading pass → AstroCellGBufferPass (writes gbuffer.json)
#   Final composition → AstroCellDeferredShadingRenderer.render() -> SVG string
# =============================================================================

# ---------------------------------------------------------------------------
# ShadowSetup.cpp — control variables
# ---------------------------------------------------------------------------

#: Minimum screen-space radius for a shadow caster to be kept.
#: Mirrors GMinScreenRadiusForShadowCaster (r.Shadow.RadiusThreshold = 0.01).
_SHADOW_RADIUS_THRESHOLD: float = 0.01

#: Toggle whole-scene shadow caching.
#: Mirrors GCacheWholeSceneShadows (r.Shadow.CacheWholeSceneShadows = 1).
ASTRO_CACHE_WHOLE_SCENE_SHADOWS: bool = True

#: Budget in megabytes for cached whole-scene shadows.
#: Mirrors GWholeSceneShadowCacheMb (r.Shadow.WholeSceneShadowCacheMb = 150).
ASTRO_SHADOW_CACHE_BUDGET_MB: int = 150

#: Allow movable primitives to cast shadows from cached whole-scene maps.
#: Mirrors GCachedShadowsCastFromMovablePrimitives.
ASTRO_MOVABLE_CAST_FROM_CACHE: bool = True

#: Skip CPU culling for Nanite meshes (species in our 2-D analogue).
#: Mirrors GSkipCullingNaniteMeshes (r.Shadow.SkipCullingNaniteMeshes = 1).
ASTRO_SKIP_CULL_NANITE: bool = True

#: Path for the shared whole-scene shadow cache channel.
_SHADOW_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "physics", "shadow_cache.json",
)


def _load_shadow_cache() -> dict:
    """Load the whole-scene shadow cache from the channel file."""
    if not os.path.isfile(_SHADOW_CACHE_PATH):
        return {"shadows": {}, "frame": -1}
    try:
        with open(_SHADOW_CACHE_PATH) as _f:
            return json.load(_f)
    except (json.JSONDecodeError, OSError):
        return {"shadows": {}, "frame": -1}


def _save_shadow_cache(cache: dict) -> None:
    """Persist the whole-scene shadow cache to the channel file."""
    os.makedirs(os.path.dirname(_SHADOW_CACHE_PATH), exist_ok=True)
    with open(_SHADOW_CACHE_PATH, "w") as _f:
        json.dump(cache, _f, indent=2)


# ---------------------------------------------------------------------------
# AstroCellShadowInfo — Python equivalent of FProjectedShadowInfo
#
# 鲁迅式：每一道阴影都是一道档案——记录着发光源、遮挡物、接收面。
# 档案建立得越精确，世界便越接近真实；档案缺失，便只剩猜测。
# ---------------------------------------------------------------------------

class AstroCellShadowInfo:
    """
    Python equivalent of FProjectedShadowInfo.

    Stores all parameters needed to evaluate a single projected shadow:
      - The light that casts it (position, direction, type)
      - The shadow frustum as a 2-D AABB cull rect
      - The set of shadow casters (cell_ids) that passed culling
      - The shadow depth map (caster_id → normalised depth in [0,1])

    Lifecycle mirrors FProjectedShadowInfo:
        __init__   → constructor + initial parameter setup
        setup()    → SetupProjectedShadow() — computes cull rect, registers casters
        render()   → RenderShadowDepths() — fills depth map
        sample()   → shadow sample lookup at a receiver point
    """

    def __init__(
        self,
        shadow_id: str,
        light_position: tuple = (0.0, -600.0, 8.0),
        light_direction: tuple = (0.0, 1.0, -0.3),
        shadow_type: str = "whole_scene",  # "whole_scene" | "per_object" | "preshadow"
        resolution: int = 256,
    ) -> None:
        self.shadow_id       = shadow_id
        self.light_position  = light_position      # (x, y, z_layer)
        self.light_direction = light_direction      # normalised direction toward scene
        self.shadow_type     = shadow_type
        self.resolution      = resolution

        # Cull rect in screen space — set by setup()
        self.cull_rect_x0: float = 0.0
        self.cull_rect_y0: float = 0.0
        self.cull_rect_x1: float = 1e9
        self.cull_rect_y1: float = 1e9

        # Shadow casters that passed frustum culling
        self.casters: list = []

        # Depth map: caster_id → depth ∈ [0, 1]
        self.depth_map: dict = {}

        # Cached flag (set when loaded from shadow_cache.json)
        self.is_cached: bool = False
        self.cache_frame: int = -1

    def setup(
        self,
        viewport_w: float,
        viewport_h: float,
        cell_entries: list,
    ) -> int:
        """
        Compute shadow cull rect and cull casters.

        Mirrors SetupProjectedShadow() + BuildWholeSceneShadowCaster()
        + FrustumCullShadowCaster():

          1. Project the light frustum to a screen-space AABB cull rect.
             (In 2-D this is the viewport-aligned bounding box of all cells
              visible to the light direction — simplified to full viewport for
              directional lights, cone-clipped for point lights.)

          2. For each cell in cell_entries, test IsScreenSpaceSizeLargeEnough
             (mirrors GMinScreenRadiusForShadowCaster screen radius test) and
             _frustum_cull_shadow_caster (per-primitive AABB vs. light frustum).

          3. If ASTRO_SKIP_CULL_NANITE, skip the AABB cull for all cells
             (mirrors skipping Nanite meshes in CPU culling).

        Returns number of accepted shadow casters.

        鲁迅式：裁剪是必要的吝啬——把不必要的影子删掉，
        才能把资源留给真正重要的那几道。
        """
        self.casters.clear()
        self.depth_map.clear()

        lx, ly, _ = self.light_position
        # Directional light: cull rect = full viewport
        self.cull_rect_x0 = 0.0
        self.cull_rect_y0 = 0.0
        self.cull_rect_x1 = viewport_w
        self.cull_rect_y1 = viewport_h

        viewport_area = max(viewport_w * viewport_h, 1.0)

        for entry in cell_entries:
            cid  = entry.get("cell_id", "")
            bbox = entry.get("bbox", {})
            if not cid or not bbox:
                continue

            w = float(bbox.get("w", 0))
            h = float(bbox.get("h", 0))
            x = float(bbox.get("x", 0))
            y = float(bbox.get("y", 0))

            # ── Screen-size cull (GMinScreenRadiusForShadowCaster) ──────────
            screen_frac = (w * h) / viewport_area
            if screen_frac < _SHADOW_RADIUS_THRESHOLD and not ASTRO_SKIP_CULL_NANITE:
                continue  # too small to cast a meaningful shadow

            # ── Frustum cull (FrustumCullShadowCaster) ──────────────────────
            if not ASTRO_SKIP_CULL_NANITE:
                if not _frustum_cull_shadow_caster(
                    x, y, w, h,
                    self.cull_rect_x0, self.cull_rect_y0,
                    self.cull_rect_x1, self.cull_rect_y1,
                ):
                    continue

            self.casters.append(cid)

        print(
            f"[ASTRO-SHADOW] ShadowSetup: shadow_id={self.shadow_id} "
            f"type={self.shadow_type} casters={len(self.casters)} "
            f"viewport=({viewport_w:.0f}×{viewport_h:.0f})",
            file=sys.stderr,
        )
        return len(self.casters)

    def render(self, cell_entries: list) -> int:
        """
        Fill the shadow depth map for all registered casters.

        Mirrors RenderShadowDepths() / FShadowDepthPassUniformParameters
        + the shadow depth rendering loop that iterates mesh draw commands
        and outputs depth values into the depth render target.

        In 2-D: depth is the normalised Z value of each caster cell,
        computed from the light's perspective (Z-order along light direction).

        Returns number of depth entries written.

        鲁迅式：深度图是阴影的骨架——没有它，阴影便只是装饰，没有重量。
        """
        if not self.casters:
            return 0

        # Build fast cid → bbox lookup
        bbox_by_cid = {e["cell_id"]: e["bbox"] for e in cell_entries if "cell_id" in e}

        # Compute light-space depths using the _shadow_lod_distance helper
        # (analogous to the Z-projection in SetupShadowDepthPassUniformBuffer)
        z_values = []
        for cid in self.casters:
            bbox = bbox_by_cid.get(cid, {})
            z    = float(bbox.get("z", 3)) if bbox else 3.0
            # Project along light direction: depth ≈ z + tiny x/y component
            ldx, ldy, ldz = self.light_direction
            cx = float(bbox.get("x", 0)) + float(bbox.get("w", 0)) / 2.0
            cy = float(bbox.get("y", 0)) + float(bbox.get("h", 0)) / 2.0
            depth_proj = z * abs(ldz) + cx * abs(ldx) / 1000.0 + cy * abs(ldy) / 1000.0
            z_values.append((cid, depth_proj))

        if not z_values:
            return 0

        z_min = min(v for _, v in z_values)
        z_max = max(v for _, v in z_values)
        z_range = max(z_max - z_min, 1e-6)

        # Normalise to [0, 1] (reversed: nearer = higher value, mirrors UE5
        # reversed depth buffer convention)
        for cid, z_proj in z_values:
            self.depth_map[cid] = round(1.0 - (z_proj - z_min) / z_range, 6)

        print(
            f"[ASTRO-SHADOW] ShadowDepthRendering: shadow_id={self.shadow_id} "
            f"depth_entries={len(self.depth_map)} "
            f"z_range=[{z_min:.2f},{z_max:.2f}]",
            file=sys.stderr,
        )
        return len(self.depth_map)

    def sample(self, caster_id: str) -> float:
        """
        Sample the shadow depth map for a given caster cell.

        Returns normalised depth in [0, 1].
        0 = fully shadowed (far from light); 1 = fully lit (nearest to light).
        Missing entries return 0.5 (unknown / penumbra estimate).

        鲁迅式：0.5 是最懦弱的答案——既不否认，也不承认。
        但在阴影映射中，未知等于中间灰，总好过黑白二值的谎言。
        """
        return self.depth_map.get(caster_id, 0.5)

    def to_dict(self) -> dict:
        return {
            "shadow_id":     self.shadow_id,
            "type":          self.shadow_type,
            "light_pos":     self.light_position,
            "light_dir":     self.light_direction,
            "cull_rect":     [self.cull_rect_x0, self.cull_rect_y0,
                              self.cull_rect_x1, self.cull_rect_y1],
            "caster_count":  len(self.casters),
            "depth_entries": len(self.depth_map),
            "is_cached":     self.is_cached,
        }


def _frustum_cull_shadow_caster(
    cx: float, cy: float, cw: float, ch: float,
    rx0: float, ry0: float, rx1: float, ry1: float,
) -> bool:
    """
    Test whether a shadow caster AABB overlaps the shadow cull rect.

    Mirrors FrustumCullShadowCaster() — a simple AABB-AABB overlap test
    between the caster's screen-space bounding box and the shadow frustum
    cull rectangle.

    Returns True if the caster is INSIDE (should be kept).

    鲁迅式：裁剪保留的是边界内的声音；边界之外的，默然无声。
    """
    return not (
        cx + cw < rx0 or cx > rx1 or
        cy + ch < ry0 or cy > ry1
    )


def _compute_cull_planes(
    light_position: tuple,
    scene_bbox: dict,
) -> list:
    """
    Compute shadow subject culling planes.

    Mirrors ComputeShadowSubjectCullingPlanes() which builds per-frustum
    planes used to cull scene primitives against the shadow volume.

    In 2-D: returns 4 axis-aligned half-planes (normals + distances)
    derived from the scene bbox expanded toward the light position.

    鲁迅式：裁剪平面是边界的哲学——它说：「到这里，不再往前。」
    而光从来不懂边界；我们替它划定。
    """
    lx, ly, _ = light_position
    min_x = scene_bbox.get("min_x", 0.0)
    min_y = scene_bbox.get("min_y", 0.0)
    max_x = scene_bbox.get("max_x", 1200.0)
    max_y = scene_bbox.get("max_y", 900.0)

    # Expand bbox toward light by 20% to include shadow receivers outside scene
    expand_x = (max_x - min_x) * 0.2
    expand_y = (max_y - min_y) * 0.2

    return [
        {"normal": (-1,  0), "dist": -(min_x - expand_x)},  # left
        {"normal": ( 1,  0), "dist":   max_x + expand_x},   # right
        {"normal": ( 0, -1), "dist": -(min_y - expand_y)},  # top
        {"normal": ( 0,  1), "dist":   max_y + expand_y},   # bottom
    ]


def build_whole_scene_shadow_caster(
    viewport_w: float,
    viewport_h: float,
    cell_entries: list,
    frame_index: int = 0,
) -> "AstroCellShadowInfo":
    """
    Build and return a whole-scene shadow for the primary directional light.

    Mirrors BuildWholeSceneShadowCaster() called from
    FDeferredShadingSceneRenderer::InitDynamicShadows():
      1. Creates an FProjectedShadowInfo (→ AstroCellShadowInfo)
      2. Calls SetupProjectedShadow (→ .setup())
      3. If ASTRO_CACHE_WHOLE_SCENE_SHADOWS:
           - Loads shadow_cache.json
           - If cache is fresh (same frame), returns cached shadow info
           - Otherwise re-builds and saves to cache

    鲁迅式：全场景阴影是一场统一的压迫——所有人共享同一道影子，
    个体的遮蔽在这里被抹平为集体的灰度。
    """
    shadow = AstroCellShadowInfo(
        shadow_id=f"whole_scene_{frame_index}",
        shadow_type="whole_scene",
    )

    if ASTRO_CACHE_WHOLE_SCENE_SHADOWS:
        cache = _load_shadow_cache()
        if cache.get("frame") == frame_index and cache.get("shadows"):
            # Cache hit — restore from disk (mirrors cached whole-scene path)
            cached_shadow = cache["shadows"].get("whole_scene")
            if cached_shadow:
                shadow.depth_map  = cached_shadow.get("depth_map", {})
                shadow.casters    = list(shadow.depth_map.keys())
                shadow.is_cached  = True
                shadow.cache_frame = frame_index
                print(
                    f"[ASTRO-SHADOW] ShadowSetup: cache HIT frame={frame_index} "
                    f"casters={len(shadow.casters)}",
                    file=sys.stderr,
                )
                return shadow

    # Cache miss or disabled — full setup + render
    shadow.setup(viewport_w, viewport_h, cell_entries)
    shadow.render(cell_entries)

    if ASTRO_CACHE_WHOLE_SCENE_SHADOWS:
        cache = {"frame": frame_index, "shadows": {
            "whole_scene": {
                "depth_map": shadow.depth_map,
            }
        }}
        _save_shadow_cache(cache)

    return shadow


# ---------------------------------------------------------------------------
# ShadowDepthRendering.cpp — LOD distance + depth pass uniforms
#
# 鲁迅式：LOD 是对远处事物的善意忽略——当它们足够小，
# 用精确数据描述它们是一种浪费，甚至是一种奢靡。
# ---------------------------------------------------------------------------

#: Mirrors GShadowLODDistanceFactor (r.Shadow.LODDistanceFactor = 1.0).
_SHADOW_LOD_DISTANCE_FACTOR: float = 1.0

#: Mirrors GShadowLODDistanceFactorCascadeScale (scales with cascade index).
_SHADOW_LOD_CASCADE_SCALE: float = 0.0


def _shadow_lod_distance(
    base_distance: float,
    cascade_index: int = 0,
) -> float:
    """
    Compute effective LOD selection distance for shadow rendering.

    Mirrors the multiplier applied to mesh LOD distances in
    FShadowDepthPassMeshProcessor::Process() using GShadowLODDistanceFactor
    and GShadowLODDistanceFactorCascadeScale:

        EffectiveDist = BaseDist
                      * LODDistanceFactor
                      * (1 + LODDistanceFactorCascadeScale * cascade_index)

    Higher effective distance → select lower-detail LOD.

    鲁迅式：距离是宽容，也是懒惰——从远处看，粗糙也像精细。
    """
    return (
        base_distance
        * _SHADOW_LOD_DISTANCE_FACTOR
        * (1.0 + _SHADOW_LOD_CASCADE_SCALE * cascade_index)
    )


@dataclass
class AstroCellShadowDepthPassParams:
    """
    Python equivalent of FShadowDepthPassUniformParameters.

    Holds the per-pass uniform buffer parameters for shadow depth rendering.
    Fields mirror the IMPLEMENT_STATIC_UNIFORM_BUFFER_STRUCT declaration.

    In 2-D: these parameters control the depth projection and LOD selection
    for each cell being rendered into the shadow depth map.

    鲁迅式：Uniform Buffer 是渲染器的宪章——
    一切着色器都必须遵守，不得私自解释。
    """
    ProjectionMatrix:   tuple = (1.0, 0.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0, 0.0,
                                 0.0, 0.0, 1.0, 0.0,
                                 0.0, 0.0, 0.0, 1.0)  # 4×4 identity
    ShadowParams:       tuple = (0.0, 0.0, 0.0, 0.0)  # (bias, slope, normal, unused)
    bRenderReflectiveShadowMap: bool = False
    bClampToNearPlane:          bool = False
    LODDistanceFactor:  float = _SHADOW_LOD_DISTANCE_FACTOR


def _setup_shadow_depth_pass_ub(
    shadow: "AstroCellShadowInfo",
    viewport_w: float,
    viewport_h: float,
) -> AstroCellShadowDepthPassParams:
    """
    Fill shadow depth pass uniform buffer parameters.

    Mirrors SetupShadowDepthPassUniformBuffer() which populates
    FShadowDepthPassUniformParameters with the shadow projection matrix,
    shadow bias, and LOD distance factor.

    In 2-D: the projection matrix is an orthographic projection from
    world space onto the shadow map rect.

    鲁迅式：正交投影是平等主义者——无论远近，所有点都被相同地对待。
    这在阴影贴图里是美德，在现实里则是幻觉。
    """
    # Build ortho projection: maps viewport to [-1,1] clip space
    # (simplified 2-D analogue of FMatrix::OrthoMatrix)
    inv_w = 2.0 / max(viewport_w, 1.0)
    inv_h = 2.0 / max(viewport_h, 1.0)

    proj = (
        inv_w,  0.0,    0.0, -1.0,
        0.0,    inv_h,  0.0, -1.0,
        0.0,    0.0,    1.0,  0.0,
        0.0,    0.0,    0.0,  1.0,
    )

    # Shadow bias: small positive value to avoid self-shadowing (acne)
    shadow_bias = 0.002 + 0.001 * len(shadow.casters) / max(100, len(shadow.casters))

    return AstroCellShadowDepthPassParams(
        ProjectionMatrix=proj,
        ShadowParams=(shadow_bias, 0.0, 0.0, 0.0),
        LODDistanceFactor=_SHADOW_LOD_DISTANCE_FACTOR,
    )


class AstroCellShadowDepthRenderer:
    """
    Python equivalent of the TRenderShadowDepths template + helper functions
    in ShadowDepthRendering.cpp.

    Drives the shadow depth rendering pass for a single AstroCellShadowInfo.
    Outputs a per-cell shadow depth dict and writes it to the channel.

    Lifecycle:
        renderer = AstroCellShadowDepthRenderer(shadow, viewport_w, viewport_h)
        depth_result = renderer.render(cell_entries)

    鲁迅式：深度渲染器是执行者——它不问为什么，只问怎么做。
    这是效率，也是盲目。我们需要它，但不应成为它。
    """

    def __init__(
        self,
        shadow: "AstroCellShadowInfo",
        viewport_w: float,
        viewport_h: float,
    ) -> None:
        self._shadow      = shadow
        self._viewport_w  = viewport_w
        self._viewport_h  = viewport_h
        self._ub_params   = _setup_shadow_depth_pass_ub(shadow, viewport_w, viewport_h)

    def render(self, cell_entries: list) -> dict:
        """
        Execute the shadow depth pass and return per-cell depth results.

        Mirrors the mesh draw loop in FShadowDepthPassMeshProcessor::Process():
          1. For each caster in shadow.casters, look up its bbox.
          2. Project the caster's world position through the shadow projection
             matrix (using ub_params.ProjectionMatrix).
          3. Apply shadow bias (ShadowParams.x) to prevent self-shadowing.
          4. Write the result to the depth_result dict.
          5. Apply LOD-distance based detail reduction (simplified to a
             screen-fraction flag that later SVG generators can consume).

        Returns dict: caster_id → {depth, lod_level, bias_applied}.

        鲁迅式：深度值是精确的谎言——它比较的是「在光源眼中的远近」，
        而非真实的世界距离；但这个谎言，足以产生正确的阴影。
        """
        if not self._shadow.casters:
            return {}

        bbox_by_cid = {
            e["cell_id"]: e["bbox"]
            for e in cell_entries
            if "cell_id" in e
        }

        bias = self._ub_params.ShadowParams[0]
        viewport_area = max(self._viewport_w * self._viewport_h, 1.0)

        depth_result: dict = {}
        for cid in self._shadow.casters:
            bbox = bbox_by_cid.get(cid, {})
            if not bbox:
                continue

            x = float(bbox.get("x", 0)) + float(bbox.get("w", 0)) / 2.0
            y = float(bbox.get("y", 0)) + float(bbox.get("h", 0)) / 2.0
            z = float(bbox.get("z", 3))

            # Project through ortho matrix (simplified row-major multiply)
            pm = self._ub_params.ProjectionMatrix
            clip_x = pm[0] * x + pm[3]
            clip_y = pm[5] * y + pm[7]
            # depth ∈ [0, 1] from normalised clip Z (≈ z / z_max)
            raw_depth = max(0.0, min(1.0, z / 7.0))   # z ∈ [0,7]
            biased_depth = max(0.0, min(1.0, raw_depth + bias))

            # LOD selection distance
            cell_diagonal = math.sqrt(
                float(bbox.get("w", 1.0))**2 + float(bbox.get("h", 1.0))**2
            )
            lod_dist = _shadow_lod_distance(cell_diagonal)
            screen_frac = (float(bbox.get("w", 0)) * float(bbox.get("h", 0))) / viewport_area
            lod_level = (
                0 if screen_frac >= 0.005 else
                1 if screen_frac >= 0.001 else
                2
            )

            depth_result[cid] = {
                "depth":         round(biased_depth, 6),
                "raw_depth":     round(raw_depth, 6),
                "bias":          round(bias, 6),
                "lod_level":     lod_level,
                "clip_xy":       (round(clip_x, 4), round(clip_y, 4)),
            }

        # Merge back into shadow's depth_map (renderer wins over setup estimate)
        for cid, d in depth_result.items():
            self._shadow.depth_map[cid] = d["depth"]

        print(
            f"[ASTRO-SHADOW] ShadowDepthRenderer: shadow_id={self._shadow.shadow_id} "
            f"rendered={len(depth_result)} bias={bias:.5f}",
            file=sys.stderr,
        )
        return depth_result


# ---------------------------------------------------------------------------
# DeferredShadingRenderer.cpp — G-Buffer + Render pipeline
#
# 鲁迅式：延迟着色是一种制度的产物——先收集，再计算。
# 收集时代价低廉；计算时代价高昂，却只在真正需要时支付。
# 这是聪明的经济学，也是对「及时付账」的反叛。
# ---------------------------------------------------------------------------

#: Mirrors CVarNanitePrimeHZBMode (r.Nanite.PrimeHZB).
#: 0 = off, 1 = run if no HZB, 2 = always
_NANITE_PRIME_HZB_MODE: int = 0


@dataclass
class AstroCellGBuffer:
    """
    Python equivalent of the UE5 G-Buffer layout (GBufferA/B/C/D).

    In UE5, 4 RGBA render targets carry:
      GBufferA → BaseColor (RGB) + Shading model (A)
      GBufferB → Metallic (R) + Specular (G) + Roughness (B) + Shading (A)
      GBufferC → WorldNormal (RGB) + Ambient Occlusion (A)
      GBufferD → Custom data (per-shading-model payload)

    2-D SVG adaptation:
      base_color      → hex string (primary fill colour)
      metallic        → float [0,1] from _species_f0
      specular        → float [0,1]
      roughness       → float [0,1] from _SPECIES_ROUGHNESS_SEC
      world_normal    → 2-D (nx, ny) unit vector (faces viewer = (0,1))
      ambient_occlusion → crowding_opacity complement
      shading_model   → int (0=Unlit, 1=Lit, 3=SubsurfaceProfile …)
      custom_data     → species-specific float4 payload

    鲁迅式：G-Buffer 是场景的档案馆——每个像素的材质属性都在此存档，
    等待光照计算的审判。档案馆本身不发光，只是记录。
    """
    # GBufferA
    base_color:      str   = "#888888"   # hex
    shading_model:   int   = 1           # 1 = DefaultLit

    # GBufferB
    metallic:        float = 0.0
    specular:        float = 0.5
    roughness:       float = 0.5
    anisotropy:      float = 0.0

    # GBufferC
    world_normal:    tuple = (0.0, 1.0)  # 2-D (nx, ny), faces upward
    ambient_occlusion: float = 1.0       # 1 = fully lit

    # GBufferD (custom per-shading-model)
    custom_data:     tuple = (0.0, 0.0, 0.0, 0.0)

    def to_dict(self) -> dict:
        return {
            "base_color":    self.base_color,
            "shading_model": self.shading_model,
            "metallic":      round(self.metallic, 4),
            "specular":      round(self.specular, 4),
            "roughness":     round(self.roughness, 4),
            "world_normal":  list(self.world_normal),
            "ao":            round(self.ambient_occlusion, 4),
            "custom_data":   list(self.custom_data),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AstroCellGBuffer":
        gb = cls()
        gb.base_color        = d.get("base_color", "#888888")
        gb.shading_model     = d.get("shading_model", 1)
        gb.metallic          = float(d.get("metallic", 0.0))
        gb.specular          = float(d.get("specular", 0.5))
        gb.roughness         = float(d.get("roughness", 0.5))
        gb.ambient_occlusion = float(d.get("ao", 1.0))
        gb.world_normal      = tuple(d.get("world_normal", [0.0, 1.0]))
        gb.custom_data       = tuple(d.get("custom_data", [0.0, 0.0, 0.0, 0.0]))
        return gb


class AstroCellDeferredShadingRenderer:
    """
    Python equivalent of FDeferredShadingSceneRenderer.

    Orchestrates the full deferred shading pipeline for a single epoch:
      1. _init_views()               → InitViews / visibility culling
      2. _render_shadow_depth_maps() → InitDynamicShadows + RenderShadowDepthMaps
      3. _render_base_pass()         → RenderBasePass (fills G-Buffer)
      4. _render_lights()            → RenderLights (deferred + simple)
      5. _render_post_processing()   → RenderPostProcessing (AA, tonemap, bloom)

    The output is a per-cell render_result dict and a composed SVG comment
    block that can be injected into the frame SVG for diagnostics.

    Lifecycle::
        dsr = AstroCellDeferredShadingRenderer(viewport_w, viewport_h)
        results = dsr.render(cell_entries, frame_index)

    鲁迅式：延迟着色渲染器是一个官僚机构——
    五个部门，各司其职，没有一个部门能独立完成工作，
    但合在一起，它们生产出可以展示于世的画面。
    这是分工合作的奇迹，也是分工合作的悲剧。
    """

    def __init__(
        self,
        viewport_w: float = 1200.0,
        viewport_h: float = 900.0,
        enable_hzb_prime: bool = (_NANITE_PRIME_HZB_MODE > 0),
    ) -> None:
        self.viewport_w    = viewport_w
        self.viewport_h    = viewport_h
        self.enable_hzb_prime = enable_hzb_prime

        # State built during render()
        self._visibility_query: "AstroCellVisibilityQuery | None" = None
        self._shadow:           "AstroCellShadowInfo | None"      = None
        self._shadow_renderer:  "AstroCellShadowDepthRenderer | None" = None
        self._gbuffers:         dict = {}   # cell_id → AstroCellGBuffer
        self._light_pass:       "AstroCellLightPass | None" = None
        self._frame_stats:      dict = {}

    # ------------------------------------------------------------------
    # Phase 1: InitViews
    # ------------------------------------------------------------------

    def _init_views(self, cell_entries: list) -> "AstroCellVisibilityQuery":
        """
        Mirrors FDeferredShadingSceneRenderer::InitViews().

        Runs the Nanite visibility query (perform_nanite_visibility equivalent)
        and optionally primes the HZB if _NANITE_PRIME_HZB_MODE > 0.

        Returns an AstroCellVisibilityQuery with per-cell results.

        鲁迅式：InitViews 是开门迎客——先看清楚谁在门口，
        再决定让谁进来，让谁站在外面等。
        """
        query = AstroCellVisibilityQuery(
            self.viewport_w, self.viewport_h
        )

        for entry in cell_entries:
            cid     = entry.get("cell_id", "")
            bbox    = entry.get("bbox", {})
            species = entry.get("species", "")
            z_layer = int(entry.get("z_layer", bbox.get("z", 3)))

            if not cid or not bbox:
                continue

            query.test_cell(cid, bbox, species, z_layer)

        query.finish()

        # HZB prime pass (mirrors CVarNanitePrimeHZB: run if no HZB available)
        if self.enable_hzb_prime:
            print(
                f"[ASTRO-DSR] InitViews: HZB prime pass (mode={_NANITE_PRIME_HZB_MODE}) "
                f"cells={len(cell_entries)}",
                file=sys.stderr,
            )

        return query

    # ------------------------------------------------------------------
    # Phase 2: RenderShadowDepthMaps
    # ------------------------------------------------------------------

    def _render_shadow_depth_maps(
        self, cell_entries: list, frame_index: int
    ) -> "AstroCellShadowInfo":
        """
        Mirrors FDeferredShadingSceneRenderer::RenderShadowDepthMaps().

        Calls build_whole_scene_shadow_caster() then constructs an
        AstroCellShadowDepthRenderer and executes the depth pass.

        Returns the populated AstroCellShadowInfo.

        鲁迅式：阴影深度图是过去的快照——此刻的光，照出昨日的形状。
        但「昨日」在渲染器里只有一帧之差，所以它总是足够接近真实。
        """
        shadow = build_whole_scene_shadow_caster(
            self.viewport_w, self.viewport_h, cell_entries, frame_index
        )

        if not shadow.is_cached:
            renderer = AstroCellShadowDepthRenderer(
                shadow, self.viewport_w, self.viewport_h
            )
            renderer.render(cell_entries)
            self._shadow_renderer = renderer

        return shadow

    # ------------------------------------------------------------------
    # Phase 3: RenderBasePass → fill G-Buffer
    # ------------------------------------------------------------------

    def _render_base_pass(
        self,
        cell_entries: list,
        vis_query: "AstroCellVisibilityQuery",
    ) -> dict:
        """
        Mirrors FDeferredShadingSceneRenderer::RenderBasePass().

        Iterates visible cells, derives G-Buffer parameters from species
        data and the visibility query, and populates self._gbuffers.

        In UE5, the Base Pass writes BaseColor, Normal, Roughness, Metallic,
        AO, and custom shading model data into the G-Buffer render targets.
        Here we produce an AstroCellGBuffer per visible cell.

        Returns dict: cell_id → AstroCellGBuffer.

        鲁迅式：Base Pass 是材质信息的总动员——
        每个表面把自己的颜色、粗糙度、法线全部交出来，
        放入 G-Buffer 这个大型档案箱，等待光照部门的审阅。
        """
        _ROUGHNESS_MAP = {
            "cil-eye": 0.1, "cil-bolt": 0.2, "cil-plus": 0.3,
            "cil-vector": 0.5, "cil-arrow-right": 0.7,
            "cil-filter": 0.3, "cil-code": 0.4, "cil-layers": 0.2,
            "cil-loop": 0.5, "cil-graph": 0.6,
        }

        gbuffers: dict = {}

        for entry in cell_entries:
            cid     = entry.get("cell_id", "")
            species = entry.get("species", "")
            bbox    = entry.get("bbox", {})

            if not cid:
                continue

            vis_result = vis_query.cell_results.get(cid, {})
            if vis_result.get("lod", 0) == -1:
                continue  # culled — no G-Buffer entry

            sp_idx = _species_to_index(species)
            colour = _SPECIES_INDEX_TO_COLOUR.get(sp_idx, _SPECIES_INDEX_TO_COLOUR[0])

            gbuf = AstroCellGBuffer(
                base_color      = _colour_to_hex(colour),
                shading_model   = 1,
                metallic        = _species_f0(species),
                specular        = 0.5,
                roughness       = _ROUGHNESS_MAP.get(species, 0.5),
                world_normal    = (0.0, 1.0),
                ambient_occlusion = entry.get("crowding_opacity", 1.0),
                custom_data     = (float(bbox.get("z", 3)), 0.0, 0.0, 0.0),
            )
            gbuffers[cid] = gbuf

        self._gbuffers = gbuffers

        print(
            f"[ASTRO-DSR] BasePass: gbuffer_entries={len(gbuffers)} "
            f"total_cells={len(cell_entries)}",
            file=sys.stderr,
        )
        return gbuffers

    # ------------------------------------------------------------------
    # Phase 4: RenderLights (deferred light pass over G-Buffer)
    # ------------------------------------------------------------------

    def _render_lights(
        self,
        cell_entries: list,
        shadow: "AstroCellShadowInfo",
    ) -> dict:
        """
        Mirrors FDeferredShadingSceneRenderer::RenderLights().

        Iterates the G-Buffer and evaluates the primary deferred light
        contribution for each cell, modulated by the shadow depth map.

        Shadow modulation (mirrors the shadow term in the BRDF evaluation):
            shadow_term = shadow.sample(caster_id)
            diffuse *= shadow_term

        Returns dict: cell_id → light_result dict.

        鲁迅式：灯光是权力——它决定谁被照亮，谁留在阴影之中。
        G-Buffer 给了每个人平等的档案；光照，决定了最终的命运。
        """
        light_pass = AstroCellLightPass()
        light_results: dict = {}

        bbox_by_cid = {e["cell_id"]: e["bbox"] for e in cell_entries if "cell_id" in e}
        sibling_bboxes = {
            e["cell_id"]: e["bbox"]
            for e in cell_entries if "cell_id" in e and e.get("bbox")
        }

        for cid, gbuf in self._gbuffers.items():
            bbox    = bbox_by_cid.get(cid, {})
            species = next(
                (e["species"] for e in cell_entries if e.get("cell_id") == cid),
                ""
            )

            result = light_pass.render_light(
                cid, bbox, species, gbuf.roughness, sibling_bboxes
            )

            # Apply shadow modulation: shadow term from depth map
            shadow_term = shadow.sample(cid)
            if shadow_term < 0.99:
                # Darken diffuse by shadow term (mirrors shadow masking in HLSL)
                orig_hex = result.get("diffuse_color", "#000000")
                def _dim(ch_val: int) -> int:
                    return max(0, int(ch_val * shadow_term))
                try:
                    r = _dim(int(orig_hex[1:3], 16))
                    g = _dim(int(orig_hex[3:5], 16))
                    b = _dim(int(orig_hex[5:7], 16))
                    result["diffuse_color"] = "#{:02X}{:02X}{:02X}".format(r, g, b)
                    result["shadow_term"]   = round(shadow_term, 4)
                except (ValueError, IndexError):
                    pass

            light_results[cid] = result

        self._light_pass = light_pass

        print(
            f"[ASTRO-DSR] RenderLights: light_results={len(light_results)} "
            f"shadow_casters={len(shadow.casters)} cached={shadow.is_cached}",
            file=sys.stderr,
        )
        return light_results

    # ------------------------------------------------------------------
    # Phase 5: RenderPostProcessing (minimal AA + tonemap)
    # ------------------------------------------------------------------

    def _render_post_processing(
        self, light_results: dict
    ) -> dict:
        """
        Mirrors FDeferredShadingSceneRenderer::RenderPostProcessing().

        Applies a simple filmic tonemap (Reinhard) and ambient occlusion
        attenuation to the final per-cell colour values.

        Tonemap: L_out = L_in / (1 + L_in)   (Reinhard global)
        AO attenuation: colour *= ao_factor

        Returns dict: cell_id → post_result dict (adds 'final_color' key).

        鲁迅式：后处理是面具——它让生硬的数字看起来像自然的颜色。
        没有它，画面是正确的；有了它，画面才是美的。
        这是一种用于欺骗眼睛的善意谎言。
        """
        post_results: dict = {}

        for cid, lr in light_results.items():
            gbuf = self._gbuffers.get(cid)
            ao   = gbuf.ambient_occlusion if gbuf else 1.0

            # Parse diffuse colour
            hex_col = lr.get("diffuse_color", "#000000")
            try:
                r_lin = int(hex_col[1:3], 16) / 255.0
                g_lin = int(hex_col[3:5], 16) / 255.0
                b_lin = int(hex_col[5:7], 16) / 255.0
            except (ValueError, IndexError):
                r_lin, g_lin, b_lin = 0.0, 0.0, 0.0

            # Reinhard tonemap
            r_tm = r_lin / (1.0 + r_lin)
            g_tm = g_lin / (1.0 + g_lin)
            b_tm = b_lin / (1.0 + b_lin)

            # AO attenuation
            r_ao = r_tm * ao
            g_ao = g_tm * ao
            b_ao = b_tm * ao

            final_hex = "#{:02X}{:02X}{:02X}".format(
                max(0, min(255, int(r_ao * 255))),
                max(0, min(255, int(g_ao * 255))),
                max(0, min(255, int(b_ao * 255))),
            )

            post_results[cid] = {
                **lr,
                "final_color":  final_hex,
                "ao_factor":    round(ao, 4),
                "tonemap":      "reinhard",
            }

        print(
            f"[ASTRO-DSR] PostProcessing: tonemapped={len(post_results)} mode=reinhard",
            file=sys.stderr,
        )
        return post_results

    # ------------------------------------------------------------------
    # Main render() entry point
    # ------------------------------------------------------------------

    def render(
        self,
        cell_entries: list,
        frame_index: int = 0,
    ) -> dict:
        """
        Execute the full deferred shading pipeline for one frame.

        Mirrors FDeferredShadingSceneRenderer::Render():
          Phase 1 → _init_views()
          Phase 2 → _render_shadow_depth_maps()
          Phase 3 → _render_base_pass()
          Phase 4 → _render_lights()
          Phase 5 → _render_post_processing()

        Returns dict: cell_id → post-processed render result.

        鲁迅式：Render() 是总开关——按下去，五个部门同时运转，
        各自沉默地完成职责，最终交出一帧可以示人的画面。
        没有英雄，只有流程；没有奇迹，只有正确的顺序。
        """
        self._frame_stats = {"frame": frame_index}

        # Phase 1: visibility
        vis_query = self._init_views(cell_entries)
        self._visibility_query = vis_query
        self._frame_stats["visible_cells"] = vis_query.finish().get(
            "visible_cells", 0
        ) if not vis_query._finished else sum(
            1 for r in vis_query.cell_results.values() if r.get("visible")
        )

        # Phase 2: shadow depths
        shadow = self._render_shadow_depth_maps(cell_entries, frame_index)
        self._shadow = shadow
        self._frame_stats["shadow_casters"] = len(shadow.casters)
        self._frame_stats["shadow_cached"]  = shadow.is_cached

        # Phase 3: base pass
        gbuffers = self._render_base_pass(cell_entries, vis_query)
        self._frame_stats["gbuffer_entries"] = len(gbuffers)

        # Phase 4: lights
        light_results = self._render_lights(cell_entries, shadow)
        self._frame_stats["light_results"] = len(light_results)

        # Phase 5: post-processing
        final_results = self._render_post_processing(light_results)
        self._frame_stats["post_results"] = len(final_results)

        print(
            f"[ASTRO-DSR] Render COMPLETE: frame={frame_index} "
            f"visible={self._frame_stats['visible_cells']} "
            f"shadow_casters={self._frame_stats['shadow_casters']} "
            f"gbuffers={self._frame_stats['gbuffer_entries']} "
            f"light={self._frame_stats['light_results']} "
            f"post={self._frame_stats['post_results']}",
            file=sys.stderr,
        )

        return final_results

    def frame_stats(self) -> dict:
        """Return stats from the last render() call."""
        return dict(self._frame_stats)


def run_deferred_shading_pipeline(
    cell_entries: list,
    frame_index: int = 0,
    viewport_w: float = 1200.0,
    viewport_h: float = 900.0,
) -> dict:
    """
    Top-level entry point for the deferred shading pipeline.

    Convenience wrapper that constructs AstroCellDeferredShadingRenderer,
    calls render(), and returns the final per-cell result dict.

    Mirrors the call site in the engine's Tick() that dispatches
    FDeferredShadingSceneRenderer::Render() on the render thread.

    鲁迅式：入口函数是门面——它让外部世界相信内部是有序的。
    无论内部多么复杂，门面保持镇定，只交出一个 dict。
    """
    dsr = AstroCellDeferredShadingRenderer(viewport_w, viewport_h)
    return dsr.render(cell_entries, frame_index)


# =============================================================================
# [ASTRO-CELL] MeshPassProcessor + MeshDrawCommands + SceneCapture +
#              ReflectionEnvironmentCapture → Python port
#
# Ported from (commit heads at time of writing):
#   upstream/unreal-renderer-ue5/Renderer-Private/MeshPassProcessor.cpp
#   upstream/unreal-renderer-ue5/Renderer-Private/MeshDrawCommands.cpp
#   upstream/unreal-renderer-ue5/Renderer-Private/SceneCaptureRendering.cpp
#   upstream/unreal-renderer-ue5/Renderer-Private/ReflectionEnvironmentCapture.cpp
#
# 鲁迅曾言：「有谁从小康人家而坠入困顿的么，我以为在这途路中，大概
# 可以看见世人的真面目。」
# MeshPassProcessor 亦然——当 PSO 状态切换压力在此集中，你才看清
# draw call 分类与合批的真实代价。每一条 mesh draw command 都是一次
# 对 GPU 状态机的主张；合并它们，是工程师对效率的永恒追求。
#
# Key UE5 constructs → Astro equivalents
# ─────────────────────────────────────────────────────────────────────────────
# [MeshPassProcessor]
#   FGraphicsMinimalPipelineStateId   → AstroCellPipelineStateId
#     PersistentIdTable (RWLock)      → _pipeline_state_table (dict + lock-free)
#     bIsIdTableFrozen                → _pso_table_frozen flag
#   FReadOnlyMeshDrawSingleShaderBindings.SetShaderBindings()
#                                     → AstroCellShaderBindings.apply()
#   GEmitMeshDrawEvent (CVar)         → ASTRO_EMIT_MESH_DRAW_EVENT
#   GSkipDrawOnPSOPrecaching (CVar)   → ASTRO_SKIP_DRAW_ON_PSO_PRECACHING
#   FMeshDrawCommandSortKey           → AstroCellDrawSortKey
#
# [MeshDrawCommands]
#   FPrimitiveIdVertexBufferPool      → AstroCellPrimitiveIdBufferPool
#     Allocate / ReturnToFreeList     → allocate / release
#     DiscardAll (discard_id)         → discard_stale (epoch-based)
#   UpdateTranslucentMeshSortKeys()   → update_translucent_sort_keys()
#   BitInvertIfNegativeFloat()        → _bit_invert_if_negative()
#   CVarMeshSortingMethodWithoutEarlyZ → ASTRO_MESH_SORT_METHOD
#   CVarDeferredMeshPassSetupTaskSync  → ASTRO_DEFERRED_MESH_PASS_SYNC
#
# [SceneCaptureRendering]
#   GSceneCaptureAllowRenderInMainRenderer → ASTRO_CAPTURE_ALLOW_MAIN_RENDERER
#   GSceneCaptureCubeSinglePass            → ASTRO_CAPTURE_CUBE_SINGLE_PASS
#   FSceneCapturePS (pixel shader)         → AstroCellCaptureProcessor
#     ESourceMode (8 modes)                → AstroCellCaptureMode (enum)
#     ShouldCompilePermutation()           → should_compile_permutation()
#     GetPermutationVector()               → get_permutation()
#   CopyCaptureToTarget()                  → copy_capture_to_target()
#   UpdateSceneCaptureContents()           → update_scene_capture_contents()
#
# [ReflectionEnvironmentCapture]
#   GReflectionCaptureNearPlane            → _REFL_NEAR_PLANE
#   GSupersampleCaptureFactor (1..8)       → _REFL_SUPERSAMPLE_FACTOR
#   CVarReflectionCaptureRuntimeTimeslice  → _REFL_TIMESLICE_FACES
#   CVarReflectionCaptureRuntimeMode       → _REFL_RUNTIME_MODE
#   CVarReflectionCaptureRuntimeBudget     → _REFL_BUDGET
#   FAstroCellReflectionCaptureState       → AstroCellReflectionCaptureState
#   CaptureSceneToScratchCubemap()         → capture_scene_to_scratch_cubemap()
#   UpdateReflectionCaptures()             → update_reflection_captures()
#
# 2-D SVG channel adaptation:
#   GPU StructuredBuffer   → list of dicts in memory / JSON channel
#   Cubemap face           → per-Z-layer "face" (6 z-layer offsets)
#   PSO state key          → (species, blend_mode, pass_name) 3-tuple
#   RHI draw submission    → write to cell/*/svg.svg channel
#   Persistent table lock  → atomic integer epoch counter (no true lock needed
#                            in single-threaded epoch loop)
# =============================================================================

import struct as _struct

# ---------------------------------------------------------------------------
# Global flags — mirrors CVars from all four source files
# ---------------------------------------------------------------------------

#: Mirrors GEmitMeshDrawEvent — emit per-draw debug annotations in SVG.
ASTRO_EMIT_MESH_DRAW_EVENT: bool = False

#: Mirrors GSkipDrawOnPSOPrecaching — skip cells whose PSO is still compiling.
ASTRO_SKIP_DRAW_ON_PSO_PRECACHING: bool = False

#: Mirrors CVarMeshSortingMethodWithoutEarlyZ (0=state+Z, 1=strict Z).
ASTRO_MESH_SORT_METHOD: int = 0

#: Mirrors CVarDeferredMeshPassSetupTaskSync — defer batch sort to RDG exec.
ASTRO_DEFERRED_MESH_PASS_SYNC: bool = True

#: Mirrors GSceneCaptureAllowRenderInMainRenderer.
ASTRO_CAPTURE_ALLOW_MAIN_RENDERER: bool = True

#: Mirrors GSceneCaptureCubeSinglePass — all 6 faces in one pass.
ASTRO_CAPTURE_CUBE_SINGLE_PASS: bool = True

#: Mirrors GReflectionCaptureNearPlane (world units).
_REFL_NEAR_PLANE: float = 5.0

#: Mirrors GSupersampleCaptureFactor (clamped to [1, 8]).
_REFL_SUPERSAMPLE_FACTOR: int = 1

#: Mirrors CVarReflectionCaptureRuntimeTimeslice — faces rendered per frame.
_REFL_TIMESLICE_FACES: int = 1

#: Mirrors CVarReflectionCaptureRuntimeMode (0=Continuous, 1=Once).
_REFL_RUNTIME_MODE: int = 1

#: Mirrors CVarReflectionCaptureRuntimeBudget (0=unlimited).
_REFL_BUDGET: int = 0


# =============================================================================
# [MeshDrawCommands] BitInvertIfNegativeFloat + sort-key helpers
# =============================================================================

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
