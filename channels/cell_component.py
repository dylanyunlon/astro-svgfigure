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


# Species → generator mapping
SPECIES_GENERATORS = {
    "cil-eye": generate_svg_cil_eye,
    "cil-vector": generate_svg_cil_vector,
    "cil-bolt": generate_svg_cil_bolt,
    "cil-plus": generate_svg_cil_plus,
    "cil-arrow-right": generate_svg_cil_arrow_right,
}


def proc(cell_id: str):
    """
    Apollo Component::Proc() equivalent.
    Reads channels → generates SVG with species algorithm → publishes.
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

    # ── Proc: species algorithm generates SVG ──
    generator = SPECIES_GENERATORS.get(species, generate_svg_cil_arrow_right)
    svg_content, actual_bbox = generator(cell_id, label, bbox, gene_traits)

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
        f'<!-- [ASTRO-CELL] PostProcessAO crowding_opacity={crowding_opacity:.4f} '
        f'(FAstroConstraintAO::ComputeConstraintWeight port) -->\n'
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
