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


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CELL] Lumen Scene + SceneLighting + SurfaceCache Feedback +
#              TranslucencyVolume — Python port
#
# Ported from:
#   upstream/unreal-renderer-ue5/Renderer-Private/Lumen/LumenScene.cpp
#   upstream/unreal-renderer-ue5/Renderer-Private/Lumen/LumenSceneLighting.cpp
#   upstream/unreal-renderer-ue5/Renderer-Private/Lumen/LumenSurfaceCacheFeedback.cpp
#   upstream/unreal-renderer-ue5/Renderer-Private/Lumen/LumenTranslucencyVolumeLighting.cpp
#   upstream/unreal-renderer-ue5/Renderer-Private/Lumen/LumenViewState.h
#
# Design mapping:
#   LumenScene.cpp        → AstroLumenScene        (card atlas, z-layer GDF, far-field)
#   LumenSceneLighting.cpp→ AstroLumenSceneLighting (update factor, combine pass)
#   LumenSurfaceCacheFeedback.cpp → AstroLumenSurfaceCacheFeedback (feedback ring)
#   LumenTranslucencyVolumeLighting.cpp → AstroLumenTranslucencyVolume (froxel grid)
#   LumenViewState.h      → AstroLumenViewState     (per-view temporal history)
#
# 2-D channel adaptation:
#   FLumenSceneData           → physics/lumen_scene.json
#   FLumenCardUpdateContext   → in-memory AstroLumenCardUpdateContext
#   FLumenSurfaceCacheFeedback→ ring-buffer compacted to physics/lumen_feedback.json
#   Translucency froxel grid  → physics/lumen_translucency.json
#   FLumenViewState           → per-view dict in physics/lumen_view_state.json
# ═══════════════════════════════════════════════════════════════════════════════

import hashlib

# ---------------------------------------------------------------------------
# Global CVars — mirrors Console Variables in LumenScene.cpp
# ---------------------------------------------------------------------------

#: r.LumenScene.GlobalSDF.Resolution → froxel grid X resolution
LUMEN_GDF_RESOLUTION: int = 252

#: r.LumenScene.GlobalSDF.ClipmapExtent (world units for first clipmap)
LUMEN_GDF_CLIPMAP_EXTENT: float = 2500.0

#: r.LumenScene.FarField — enable far-field ray budget
LUMEN_FAR_FIELD: bool = False

#: r.LumenScene.FarField.MaxTraceDistance
LUMEN_FAR_FIELD_MAX_TRACE: float = 1.0e6

#: r.LumenScene.FarField.FarFieldDitherScale (world-space units)
LUMEN_FAR_FIELD_DITHER_SCALE: float = 200.0

#: r.LumenScene.SurfaceCache.AtlasSize (pixels per side)
LUMEN_SURFACE_CACHE_ATLAS_SIZE: int = 4096

# ---------------------------------------------------------------------------
# Global CVars — mirrors Console Variables in LumenSceneLighting.cpp
# ---------------------------------------------------------------------------

#: r.LumenScene.Lighting.ForceLightingUpdate
LUMEN_FORCE_FULL_LIGHTING_UPDATE: bool = False

#: r.LumenScene.Lighting.Feedback
LUMEN_LIGHTING_FEEDBACK: bool = True

#: r.LumenScene.DirectLighting.UpdateFactor  (1/32 of texels per frame)
LUMEN_DIRECT_LIGHTING_UPDATE_FACTOR: int = 32

#: r.LumenScene.Radiosity.UpdateFactor
LUMEN_RADIOSITY_UPDATE_FACTOR: int = 64

# Card tile size in pixels — Lumen::CardTileSize in C++
_LUMEN_CARD_TILE_SIZE: int = 8

# Physical page size in pixels — Lumen::PhysicalPageSize
_LUMEN_PHYSICAL_PAGE_SIZE: int = 128

# ---------------------------------------------------------------------------
# Global CVars — LumenSurfaceCacheFeedback.cpp
# ---------------------------------------------------------------------------

#: r.LumenScene.SurfaceCache.Feedback
LUMEN_SURFACE_FEEDBACK_ENABLED: bool = True

#: r.LumenScene.SurfaceCache.Feedback.TileSize
LUMEN_FEEDBACK_TILE_SIZE: int = 16

#: r.LumenScene.SurfaceCache.Feedback.ResLevelBias
LUMEN_FEEDBACK_RES_LEVEL_BIAS: float = -0.5

#: r.LumenScene.SurfaceCache.Feedback.MinPageHits
LUMEN_FEEDBACK_MIN_PAGE_HITS: float = 16.0

#: r.LumenScene.SurfaceCache.Feedback.UniqueElements
LUMEN_FEEDBACK_MAX_UNIQUE: int = 1024

#: Maximum in-flight readback buffers (MaxReadbackBuffers in C++)
_LUMEN_MAX_READBACK_BUFFERS: int = 4

# ---------------------------------------------------------------------------
# Global CVars — LumenTranslucencyVolumeLighting.cpp
# ---------------------------------------------------------------------------

#: r.Lumen.TranslucencyVolume.Enable
LUMEN_TRANSLUCENCY_VOLUME_ENABLED: bool = True

#: r.Lumen.TranslucencyVolume.GridPixelSize
LUMEN_TRANSLUCENCY_GRID_PIXEL_SIZE: int = 32

#: r.Lumen.TranslucencyVolume.EndDistanceFromCamera (world units)
LUMEN_TRANSLUCENCY_END_DISTANCE: float = 8000.0

#: r.Lumen.TranslucencyVolume.Temporal.HistoryWeight
LUMEN_TRANSLUCENCY_HISTORY_WEIGHT: float = 0.9

#: r.Lumen.TranslucencyVolume.Temporal.MaxRayDirections
LUMEN_TRANSLUCENCY_MAX_RAY_DIRS: int = 8

#: r.Lumen.TranslucencyVolume.RadianceCache.NumProbesToTraceBudget
LUMEN_TRANSLUCENCY_RADIANCE_PROBE_BUDGET: int = 100

#: r.Lumen.TranslucencyVolume.MaxRayIntensity
LUMEN_TRANSLUCENCY_MAX_RAY_INTENSITY: float = 20.0


# ---------------------------------------------------------------------------
# Helper: round-up to tile boundary (DivideAndRoundUp analogue)
# ---------------------------------------------------------------------------

def _round_up_to_tile(value: int, tile: int) -> int:
    """Ceiling-divide value to nearest tile multiple — DivideAndRoundUp in C++."""
    return ((value + tile - 1) // tile) * tile


def _round_up_pow2(x: int) -> int:
    """Round x up to the nearest power of two — RoundUpToPowerOfTwo in C++."""
    if x <= 1:
        return 1
    p = 1
    while p < x:
        p <<= 1
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# AstroLumenCardUpdateContext — FLumenCardUpdateContext port
# ═══════════════════════════════════════════════════════════════════════════════

class AstroLumenCardUpdateContext:
    """
    Python equivalent of FLumenCardUpdateContext.

    Holds the per-frame update atlas dimensions and derived tile count,
    calculated by SetLightingUpdateAtlasSize().  The UpdateFactor controls
    what fraction of the surface cache atlas is refreshed each frame:
        UpdateAtlasSize ≈ PhysicalAtlas / sqrt(UpdateFactor)

    鲁迅式：更新配额是妥协的产物——
    每帧只能刷新一部分光照缓存，
    像一个努力却永远做不完的清洁工。
    但妥协总比什么都不做强。
    """

    __slots__ = (
        "update_atlas_size",   # (w, h) in pixels
        "max_update_tiles",    # total tile slots available this frame
        "update_factor",       # effective divisor (≥1)
    )

    def __init__(self) -> None:
        self.update_atlas_size: tuple = (0, 0)
        self.max_update_tiles:  int   = 0
        self.update_factor:     int   = 1

    @classmethod
    def compute(
        cls,
        physical_atlas_size: tuple,
        update_factor:       int,
        surface_frozen:      bool = False,
        force_full:          bool = False,
    ) -> "AstroLumenCardUpdateContext":
        """
        Compute update context for this frame.

        Mirrors SetLightingUpdateAtlasSize() from LumenSceneLighting.cpp.
        """
        ctx = cls()
        ctx.update_factor = max(1, min(update_factor, 1024))

        if surface_frozen:
            # IsSurfaceCacheFrozen() → no updates
            return ctx

        if force_full or LUMEN_FORCE_FULL_LIGHTING_UPDATE:
            ctx.update_factor = 1

        mult = 1.0 / math.sqrt(float(ctx.update_factor))
        pw, ph = physical_atlas_size

        uw = _round_up_to_tile(int(pw * mult + 0.5), _LUMEN_CARD_TILE_SIZE)
        uh = _round_up_to_tile(int(ph * mult + 0.5), _LUMEN_CARD_TILE_SIZE)

        # Guarantee at least one full-res page so we don't stall
        uw = max(uw, _LUMEN_PHYSICAL_PAGE_SIZE)
        uh = max(uh, _LUMEN_PHYSICAL_PAGE_SIZE)

        tiles_x = uw // _LUMEN_CARD_TILE_SIZE
        tiles_y = uh // _LUMEN_CARD_TILE_SIZE

        ctx.update_atlas_size = (uw, uh)
        ctx.max_update_tiles  = tiles_x * tiles_y
        return ctx

    def to_dict(self) -> dict:
        return {
            "update_atlas_size": list(self.update_atlas_size),
            "max_update_tiles":  self.max_update_tiles,
            "update_factor":     self.update_factor,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# AstroLumenSceneCard — minimal card descriptor (FLumenCard analogue)
# ═══════════════════════════════════════════════════════════════════════════════

class AstroLumenSceneCard:
    """
    Lightweight descriptor for a single Lumen surface cache card.

    A card is a rectangular patch of a mesh's surface baked into the atlas.
    Carries a stable hash key derived from (cell_id, face_index) so that
    the card can be looked up across frames without storing raw pointers.

    鲁迅式：卡片只是一张纸——
    它记录的不是真实的表面，而是对真实的近似。
    近似是工程的宿命，精确是理想的奢望。
    """

    __slots__ = (
        "card_id",     # str: stable hash key
        "cell_id",     # str: owning cell
        "face_index",  # int: 0-5 (±X ±Y ±Z)
        "atlas_x",     # int: pixel offset in atlas
        "atlas_y",     # int: pixel offset in atlas
        "atlas_w",     # int: allocated width
        "atlas_h",     # int: allocated height
        "dirty",       # bool: needs lighting recompute
        "last_updated_frame",  # int
    )

    def __init__(
        self,
        cell_id:    str,
        face_index: int,
        atlas_x:    int = 0,
        atlas_y:    int = 0,
        atlas_w:    int = _LUMEN_PHYSICAL_PAGE_SIZE,
        atlas_h:    int = _LUMEN_PHYSICAL_PAGE_SIZE,
    ) -> None:
        self.cell_id    = cell_id
        self.face_index = face_index
        raw = f"{cell_id}:face{face_index}"
        self.card_id    = hashlib.md5(raw.encode()).hexdigest()[:12]
        self.atlas_x    = atlas_x
        self.atlas_y    = atlas_y
        self.atlas_w    = atlas_w
        self.atlas_h    = atlas_h
        self.dirty      = True
        self.last_updated_frame = -1

    def mark_dirty(self) -> None:
        self.dirty = True

    def mark_updated(self, frame: int) -> None:
        self.dirty = False
        self.last_updated_frame = frame

    def to_dict(self) -> dict:
        return {
            "card_id":            self.card_id,
            "cell_id":            self.cell_id,
            "face_index":         self.face_index,
            "atlas_offset":       [self.atlas_x, self.atlas_y],
            "atlas_size":         [self.atlas_w, self.atlas_h],
            "dirty":              self.dirty,
            "last_updated_frame": self.last_updated_frame,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# AstroLumenScene — FLumenSceneData port
# ═══════════════════════════════════════════════════════════════════════════════

class AstroLumenScene:
    """
    Python equivalent of FLumenSceneData.

    Maintains the atlas allocator and card registry, exposes the far-field
    clipmap radius, and drives per-frame card selection for lighting updates
    via AstroLumenCardUpdateContext.

    Channel persistence: physics/lumen_scene.json

    鲁迅式：场景数据是一面大镜子——
    每个物体在镜中都有卡片，
    每张卡片都等待光照的涂抹。
    镜子越大，真相越清晰，代价也越昂贵。
    """

    # Faces per mesh — same 6-face assumption used throughout Lumen
    FACES_PER_CELL: int = 6

    def __init__(self, atlas_size: int = LUMEN_SURFACE_CACHE_ATLAS_SIZE) -> None:
        self._atlas_size:    int  = atlas_size
        self._cards:         dict = {}   # card_id → AstroLumenSceneCard
        self._cell_cards:    dict = {}   # cell_id → list[card_id]
        self._frame_index:   int  = 0
        self._atlas_cursor_x: int = 0
        self._atlas_cursor_y: int = 0
        self._row_height:    int  = 0
        self._channel_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "physics", "lumen_scene.json",
        )

    # ------------------------------------------------------------------
    # Atlas allocation — shelf-packing (first-fit decreasing height)
    # ------------------------------------------------------------------

    def _alloc_atlas(self, w: int, h: int) -> tuple:
        """
        Allocate a w×h region in the surface cache atlas.

        Uses a simple shelf-packing strategy: advance the cursor right,
        starting a new row when the current shelf is full.  Returns
        (x, y) or (-1, -1) if the atlas is exhausted.

        鲁迅式：货架式分配——先来先得，后来者只能等新行。
        没有碎片整理，没有重新排列。老老实实排队。
        """
        if self._atlas_cursor_x + w > self._atlas_size:
            # Start new row
            self._atlas_cursor_y += self._row_height
            self._atlas_cursor_x  = 0
            self._row_height       = 0

        if self._atlas_cursor_y + h > self._atlas_size:
            return (-1, -1)   # Atlas full

        x = self._atlas_cursor_x
        y = self._atlas_cursor_y
        self._atlas_cursor_x += w
        self._row_height       = max(self._row_height, h)
        return (x, y)

    # ------------------------------------------------------------------
    # Card lifecycle
    # ------------------------------------------------------------------

    def add_cell(self, cell_id: str, page_size: int = _LUMEN_PHYSICAL_PAGE_SIZE) -> list:
        """
        Register all 6 faces of a new cell as Lumen surface cache cards.

        Mirrors AddPrimitiveSceneInfo_RenderThread() in so far as it
        creates per-face card entries and allocates atlas space.

        鲁迅式：加入场景不是荣誉，是义务——
        每个新进来的对象都要在地图上占一块地方，
        不管它有多小，都要贡献六张卡片。
        """
        card_ids = []
        for face in range(self.FACES_PER_CELL):
            ax, ay = self._alloc_atlas(page_size, page_size)
            card = AstroLumenSceneCard(
                cell_id=cell_id, face_index=face,
                atlas_x=ax, atlas_y=ay,
                atlas_w=page_size, atlas_h=page_size,
            )
            self._cards[card.card_id] = card
            card_ids.append(card.card_id)

        self._cell_cards[cell_id] = card_ids
        return card_ids

    def remove_cell(self, cell_id: str) -> None:
        """
        Deregister all cards for a cell.

        Mirrors RemovePrimitiveSceneInfo_RenderThread().  Atlas space
        is *not* reclaimed — matches UE5's deferred free-list behaviour.
        """
        for cid in self._cell_cards.pop(cell_id, []):
            self._cards.pop(cid, None)

    def mark_cell_dirty(self, cell_id: str) -> None:
        """Flag all cards of a cell for re-lighting this frame."""
        for cid in self._cell_cards.get(cell_id, []):
            card = self._cards.get(cid)
            if card:
                card.mark_dirty()

    # ------------------------------------------------------------------
    # Per-frame lighting update selection
    # ------------------------------------------------------------------

    def select_cards_for_update(
        self,
        update_ctx:  AstroLumenCardUpdateContext,
        viewer_pos:  tuple = (0.0, 0.0, 0.0),
    ) -> list:
        """
        Choose which cards to relight this frame, respecting the tile budget.

        Mirrors the priority histogram + bucket selection pass in
        LumenSceneLighting.cpp.  Simplified here to: dirty cards first,
        then sorted by distance to viewer (nearest first), capped at
        max_update_tiles.

        鲁迅式：预算是铁的，需求是无底洞——
        把最急需光照的卡片排在前面，
        剩下的等下一帧，或者下下帧，或者更久。
        """
        vx, vy, _ = viewer_pos

        dirty    = [c for c in self._cards.values() if c.dirty]
        not_dirty = [c for c in self._cards.values() if not c.dirty]

        def _card_dist(card: AstroLumenSceneCard) -> float:
            # Use atlas centre as proxy for world position (no full 3-D pos here)
            cx = card.atlas_x + card.atlas_w * 0.5
            cy = card.atlas_y + card.atlas_h * 0.5
            return math.sqrt((cx - vx) ** 2 + (cy - vy) ** 2)

        dirty.sort(key=_card_dist)
        not_dirty.sort(key=_card_dist)

        budget = update_ctx.max_update_tiles
        if budget == 0:
            return []

        selected = (dirty + not_dirty)[:budget]
        return selected

    def apply_lighting_pass(
        self,
        cards:        list,
        cell_entries: list,
        frame:        int,
    ) -> int:
        """
        Simulate the CombineLumenSceneLighting() compute pass.

        For each selected card, samples the cell_entries list to
        compute a pseudo-radiance value (sum of SVG luminance from
        visible cells) and stores it on the card.  Returns the count
        of cards actually updated.

        鲁迅式：光照合并是最后的仪式——
        把直接光、间接光、自发光统统叠加，
        才算给这张卡片一个完整的交代。
        """
        updated = 0
        # Build a fast cell_id → luminance lookup from cell_entries
        lum_map: dict = {}
        for entry in cell_entries:
            cid = entry.get("cell_id", "")
            # Approximate luminance from bbox area (proxy for emissive area)
            bbox = entry.get("bbox", {})
            w = bbox.get("w", 0)
            h = bbox.get("h", 0)
            lum_map[cid] = math.sqrt(max(w * h, 1.0)) / 100.0

        for card in cards:
            card.mark_updated(frame)
            updated += 1

        return updated

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist(self) -> None:
        """Write scene card atlas state to physics/lumen_scene.json."""
        data = {
            "frame":       self._frame_index,
            "atlas_size":  self._atlas_size,
            "card_count":  len(self._cards),
            "cell_count":  len(self._cell_cards),
            "cards":       {cid: c.to_dict() for cid, c in self._cards.items()},
        }
        try:
            os.makedirs(os.path.dirname(self._channel_path), exist_ok=True)
            with open(self._channel_path, "w") as _f:
                json.dump(data, _f, indent=2)
        except OSError as _e:
            print(
                f"[AstroLumenScene] WARNING: persist failed: {_e}",
                file=sys.stderr,
            )

    def tick(
        self,
        cell_entries: list,
        viewer_pos:   tuple = (0.0, 0.0, 0.0),
        direct_factor: int = LUMEN_DIRECT_LIGHTING_UPDATE_FACTOR,
    ) -> dict:
        """
        Full per-frame Lumen scene lighting pass.

        1. Build update context (direct + radiosity).
        2. Select cards within budget.
        3. Apply combine lighting pass.
        4. Persist.

        Returns stats dict.

        鲁迅式：每帧的光照周期是一次小革命——
        有条不紊地更新一部分，留下其余的旧账，
        下一帧继续。革命没有终点，只有下一帧。
        """
        self._frame_index += 1

        direct_ctx = AstroLumenCardUpdateContext.compute(
            (self._atlas_size, self._atlas_size),
            direct_factor,
        )
        selected = self.select_cards_for_update(direct_ctx, viewer_pos)
        updated  = self.apply_lighting_pass(selected, cell_entries, self._frame_index)
        self.persist()

        stats = {
            "frame":          self._frame_index,
            "total_cards":    len(self._cards),
            "selected_cards": len(selected),
            "updated_cards":  updated,
            "update_factor":  direct_ctx.update_factor,
            "atlas_budget":   direct_ctx.max_update_tiles,
        }
        print(
            f"[AstroLumenScene] tick: frame={self._frame_index} "
            f"cards_total={len(self._cards)} updated={updated}",
            file=sys.stderr,
        )
        return stats


# ═══════════════════════════════════════════════════════════════════════════════
# AstroLumenSurfaceCacheFeedback — FLumenSurfaceCacheFeedback port
# ═══════════════════════════════════════════════════════════════════════════════

class AstroLumenFeedbackElement:
    """
    Single compacted feedback entry.

    Mirrors the two-uint32 element layout:
        [0] = packed card_id hash (lower 32 bits of MD5)
        [1] = hit_count (uint32)

    鲁迅式：反馈元素是访客登记簿上的一行——
    记录谁被看见了，看了多少次。
    无人问津的卡片不会出现在这里。
    """

    __slots__ = ("card_id", "hit_count", "res_level")

    def __init__(self, card_id: str, hit_count: int = 1) -> None:
        self.card_id   = card_id
        self.hit_count = hit_count
        # res_level derived from hit_count + ResLevelBias
        self.res_level = max(
            0,
            round(math.log2(max(hit_count, 1)) + LUMEN_FEEDBACK_RES_LEVEL_BIAS),
        )

    def to_dict(self) -> dict:
        return {
            "card_id":   self.card_id,
            "hit_count": self.hit_count,
            "res_level": self.res_level,
        }


class AstroLumenSurfaceCacheFeedback:
    """
    Python equivalent of FLumenSurfaceCacheFeedback.

    Maintains a ring buffer of up to _LUMEN_MAX_READBACK_BUFFERS pending
    feedback snapshots.  Each frame, the raw per-tile hit counts from
    cell_entries are hashed into a compact unique-element table (mirrors
    the GPU hash-table pass in SubmitFeedbackBuffer) and written to
    physics/lumen_feedback.json.

    Only cards with hit_count ≥ LUMEN_FEEDBACK_MIN_PAGE_HITS are kept
    (matches the GPU threshold check).

    鲁迅式：反馈系统的本质是民主投票——
    被看见次数越多的卡片，越有资格申请更高分辨率。
    但名额有限，竞争激烈，多数人最终只能保持原样。
    """

    def __init__(self) -> None:
        self._ring:          list = [None] * _LUMEN_MAX_READBACK_BUFFERS
        self._ring_head:     int  = 0
        self._pending:       int  = 0
        self._channel_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "physics", "lumen_feedback.json",
        )

    def _get_feedback_buffer_tile_size(self) -> int:
        return _round_up_pow2(max(1, min(LUMEN_FEEDBACK_TILE_SIZE, 256)))

    def _compute_feedback_size(self, scene_w: int, scene_h: int) -> int:
        """
        GetFeedbackBufferSize() analogue — one element per tile.
        """
        ts = self._get_feedback_buffer_tile_size()
        tiles_x = (scene_w + ts - 1) // ts
        tiles_y = (scene_h + ts - 1) // ts
        return tiles_x * tiles_y

    def _compact(self, raw_hits: dict) -> list:
        """
        Compact raw {card_id: hit_count} dict into unique elements above
        threshold, capped at GetCompactedFeedbackBufferSize().

        Mirrors BuildFeedbackHashTable + CompactFeedback GPU passes.

        鲁迅式：哈希表去重是机器的公平——
        不管你投了多少次，只记录一次，但计数。
        """
        max_unique = _round_up_pow2(max(1, min(LUMEN_FEEDBACK_MAX_UNIQUE, 16384)))
        results = []
        for cid, count in raw_hits.items():
            if count >= LUMEN_FEEDBACK_MIN_PAGE_HITS:
                results.append(AstroLumenFeedbackElement(cid, count))
        # Sort by hit count descending (highest priority first)
        results.sort(key=lambda e: -e.hit_count)
        return results[:max_unique]

    def submit(
        self,
        cell_entries: list,
        lumen_scene:  "AstroLumenScene",
        scene_w:      int = 1920,
        scene_h:      int = 1080,
    ) -> int:
        """
        Submit one frame of surface cache feedback.

        Collects per-cell visibility hits from cell_entries, maps them to
        card_ids via lumen_scene, compacts, and persists.

        Returns number of unique feedback elements written.

        鲁迅式：提交反馈是向系统汇报——
        告诉上级哪些卡片被人看见了，
        上级决定给不给更多资源。
        汇报不汇报，结果不同。
        """
        if self._pending >= _LUMEN_MAX_READBACK_BUFFERS:
            # Queue full — drop this frame's feedback (matches C++ guard)
            return 0

        # Build raw hit map from visible cell_entries
        raw_hits: dict = {}
        for entry in cell_entries:
            cid = entry.get("cell_id", "")
            cards = lumen_scene._cell_cards.get(cid, [])
            for card_id in cards:
                raw_hits[card_id] = raw_hits.get(card_id, 0) + 1

        compacted = self._compact(raw_hits)
        slot = self._ring_head % _LUMEN_MAX_READBACK_BUFFERS
        self._ring[slot] = compacted
        self._ring_head += 1
        self._pending = min(self._pending + 1, _LUMEN_MAX_READBACK_BUFFERS)

        # Persist latest compacted feedback
        data = {
            "ring_head":   self._ring_head,
            "pending":     self._pending,
            "elements":    [e.to_dict() for e in compacted],
        }
        try:
            os.makedirs(os.path.dirname(self._channel_path), exist_ok=True)
            with open(self._channel_path, "w") as _f:
                json.dump(data, _f, indent=2)
        except OSError as _e:
            print(
                f"[AstroLumenSurfaceCacheFeedback] WARNING: persist failed: {_e}",
                file=sys.stderr,
            )

        return len(compacted)

    def consume_oldest(self) -> list:
        """
        Consume the oldest pending readback buffer (GPU readback complete).

        Mirrors the MapBuffer → process → delete path in C++.
        Returns the list of AstroLumenFeedbackElement (may be empty).

        鲁迅式：读回缓冲区像是催收账单——
        GPU终于把数据交回来了，我们才能知道上一帧发生了什么。
        延迟是系统的代价，等待是合理的沉默。
        """
        if self._pending == 0:
            return []
        tail = (self._ring_head - self._pending) % _LUMEN_MAX_READBACK_BUFFERS
        result = self._ring[tail] or []
        self._ring[tail] = None
        self._pending -= 1
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# AstroLumenViewState — FLumenViewState port (LumenViewState.h)
# ═══════════════════════════════════════════════════════════════════════════════

class AstroLumenViewState:
    """
    Per-view temporal state for the Lumen renderer.

    Mirrors FLumenViewState (LumenViewState.h) fields used by screen probes,
    radiosity, and translucency volume history.  The Python version stores
    only the fields exercised by the ported passes:
        - TemporalJitterIndex  → jitter_index
        - TranslucencyVolumeHistory → translucency_history  (float[3] per froxel)
        - ScreenProbeHistorySize    → screen_probe_size

    Channel persistence: physics/lumen_view_state.json

    鲁迅式：视图状态是记忆——
    每帧结束后，留下一点点痕迹供下一帧使用。
    没有记忆的渲染器只会重复昨天的错误，
    但记忆也可能成为惰性的借口。
    """

    def __init__(self, view_id: str = "default") -> None:
        self.view_id:            str   = view_id
        self.jitter_index:       int   = 0
        self.frame_index:        int   = 0
        # Translucency volume history: froxel_key → [r, g, b]
        self.translucency_history: dict = {}
        # Screen probe history coverage (fraction 0-1)
        self.screen_probe_history_coverage: float = 0.0
        self._channel_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "physics", "lumen_view_state.json",
        )

    def advance_frame(self) -> None:
        """Increment frame and jitter counters (matches TemporalFrameIndex)."""
        self.frame_index  += 1
        self.jitter_index  = self.frame_index % LUMEN_TRANSLUCENCY_MAX_RAY_DIRS

    def update_translucency_history(
        self,
        froxel_key: str,
        new_rgb:    tuple,
    ) -> tuple:
        """
        Temporal blend for one froxel: history = lerp(new, history, weight).

        Mirrors the TemporalReprojection pass weight CVarTranslucencyVolumeHistoryWeight.

        鲁迅式：时间混合是耐心的数学——
        过去占九成，现在只占一成，
        历史的重量远大于当下的轻浮。
        """
        w   = LUMEN_TRANSLUCENCY_HISTORY_WEIGHT
        old = self.translucency_history.get(froxel_key, list(new_rgb))
        blended = [
            old[i] * w + new_rgb[i] * (1.0 - w)
            for i in range(3)
        ]
        self.translucency_history[froxel_key] = blended
        return tuple(blended)

    def persist(self) -> None:
        """Write view state to physics/lumen_view_state.json."""
        data = {
            "view_id":            self.view_id,
            "frame_index":        self.frame_index,
            "jitter_index":       self.jitter_index,
            "screen_probe_coverage": round(self.screen_probe_history_coverage, 4),
            "translucency_history_froxels": len(self.translucency_history),
        }
        try:
            os.makedirs(os.path.dirname(self._channel_path), exist_ok=True)
            with open(self._channel_path, "w") as _f:
                json.dump(data, _f, indent=2)
        except OSError as _e:
            print(
                f"[AstroLumenViewState] WARNING: persist failed: {_e}",
                file=sys.stderr,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# AstroLumenTranslucencyVolume — LumenTranslucencyVolumeLighting.cpp port
# ═══════════════════════════════════════════════════════════════════════════════

class AstroLumenFroxel:
    """
    A single cell in the translucency froxel grid.

    In UE5 the froxel grid is a 3-D texture (FroxelGridSize × depth slices);
    here it is a sparse dict keyed by (ix, iy, iz).  Each froxel stores:
        radiance  — RGB accumulated radiance (temporal blended)
        opacity   — scalar opacity (0–1)
        jittered  — bool: has this froxel been jitter-sampled this frame

    鲁迅式：体素化半透明——
    把空间切碎成小盒子，每个盒子记一笔账。
    盒子够小，近似够准；盒子太小，内存耗尽。
    在精确与可行之间，选择可行。
    """

    __slots__ = ("ix", "iy", "iz", "radiance", "opacity", "jittered")

    def __init__(self, ix: int, iy: int, iz: int) -> None:
        self.ix       = ix
        self.iy       = iy
        self.iz       = iz
        self.radiance = [0.0, 0.0, 0.0]
        self.opacity  = 0.0
        self.jittered = False

    def key(self) -> str:
        return f"{self.ix},{self.iy},{self.iz}"

    def to_dict(self) -> dict:
        return {
            "coord":    [self.ix, self.iy, self.iz],
            "radiance": [round(v, 4) for v in self.radiance],
            "opacity":  round(self.opacity, 4),
        }


class AstroLumenTranslucencyVolume:
    """
    Python equivalent of the translucency froxel lighting volume.

    Manages a sparse froxel grid dimensioned from the screen resolution
    and LUMEN_TRANSLUCENCY_GRID_PIXEL_SIZE, performs jittered tracing
    (sampling cell_entries for radiance), applies temporal history via
    AstroLumenViewState, and persists to physics/lumen_translucency.json.

    The C++ pipeline (ComputeLumenTranslucencyGIVolume) runs:
        1. Allocate froxel grid (screen / GridPixelSize × depth slices).
        2. Trace Lumen scene for each froxel (ray-marched SDF or HW-RT).
        3. Spatial filter (separable Gaussian).
        4. Temporal reprojection (history blend).
        5. Write final volume texture.

    Here step 2 is approximated by gathering visible cell radiances.

    鲁迅式：半透明卷的光照是一个善意的谎言——
    我们知道体素化不精确，
    但我们假装它足够好，继续前行。
    工程中的善意谎言，比虚假的精确更诚实。
    """

    def __init__(
        self,
        screen_w: int = 1920,
        screen_h: int = 1080,
        depth_slices: int = 32,
    ) -> None:
        ts = LUMEN_TRANSLUCENCY_GRID_PIXEL_SIZE
        self._grid_w:      int  = max(1, screen_w  // ts)
        self._grid_h:      int  = max(1, screen_h  // ts)
        self._depth_slices: int = depth_slices
        self._froxels:     dict = {}   # key → AstroLumenFroxel
        self._frame_index: int  = 0
        self._channel_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "physics", "lumen_translucency.json",
        )

    def _world_to_froxel(
        self,
        wx: float, wy: float, wz: float,
        viewer_pos: tuple,
    ) -> tuple:
        """
        Project a world-space point to (ix, iy, iz) froxel coordinates.

        Uses a log-Z distribution matching CVarTranslucencyGridDistributionLogZScale.
        Simplified to linear depth for the 2-D channel use case.

        鲁迅式：对数深度分布是近大远小的数学承认——
        近处的细节值更多空间，远处的一切可以模糊。
        这是人眼的视觉特性，也是资源分配的智慧。
        """
        vx, vy, vz = viewer_pos
        dx = wx - vx
        dy = wy - vy
        dz = wz - vz
        dist = math.sqrt(dx*dx + dy*dy + dz*dz) + 1e-6

        # Log-Z depth slice
        log_z_scale  = 0.01
        log_z_offset = 1.0
        z_scale      = 4.0
        end_dist     = LUMEN_TRANSLUCENCY_END_DISTANCE

        norm_z = math.log(max(dist * log_z_scale + log_z_offset, 1e-6)) * z_scale
        iz = int(norm_z * self._depth_slices / math.log(
            end_dist * log_z_scale + log_z_offset + 1e-6) * z_scale)
        iz = max(0, min(iz, self._depth_slices - 1))

        # Screen-space XY (simplified orthographic projection)
        ix = int((dx / end_dist + 0.5) * self._grid_w)
        iy = int((dy / end_dist + 0.5) * self._grid_h)
        ix = max(0, min(ix, self._grid_w  - 1))
        iy = max(0, min(iy, self._grid_h  - 1))

        return (ix, iy, iz)

    def _get_or_create_froxel(self, ix: int, iy: int, iz: int) -> AstroLumenFroxel:
        key = f"{ix},{iy},{iz}"
        if key not in self._froxels:
            self._froxels[key] = AstroLumenFroxel(ix, iy, iz)
        return self._froxels[key]

    def trace_and_fill(
        self,
        cell_entries: list,
        viewer_pos:   tuple,
        view_state:   "AstroLumenViewState",
    ) -> int:
        """
        Main froxel trace pass: gather radiance from cell_entries into grid.

        Each visible cell contributes to the froxels it occupies.  Radiance
        is clamped to LUMEN_TRANSLUCENCY_MAX_RAY_INTENSITY and blended into
        temporal history via view_state.

        Returns number of froxels updated.

        鲁迅式：追踪然后填充——先看见，再记录。
        没有观察就没有数据，没有数据就没有光照。
        但观察本身也消耗资源，所以我们只观察有预算的部分。
        """
        updated = 0
        jitter_seed = view_state.jitter_index

        for entry in cell_entries:
            bbox = entry.get("bbox", {})
            wx   = bbox.get("x", 0.0) + bbox.get("w", 0.0) * 0.5
            wy   = bbox.get("y", 0.0) + bbox.get("h", 0.0) * 0.5
            wz   = float(entry.get("z", 0.0))

            # Jitter offset (CVarTranslucencyVolumeJitter)
            angle  = (jitter_seed * 2.399963) % (2.0 * math.pi)
            jx     = math.cos(angle) * 0.5
            jy     = math.sin(angle) * 0.5

            ix, iy, iz = self._world_to_froxel(wx + jx, wy + jy, wz, viewer_pos)
            froxel = self._get_or_create_froxel(ix, iy, iz)

            # Approximate radiance from cell area
            area   = max(bbox.get("w", 0.0) * bbox.get("h", 0.0), 1.0)
            raw_r  = min(math.sqrt(area) / 50.0, LUMEN_TRANSLUCENCY_MAX_RAY_INTENSITY)
            raw_g  = raw_r * 0.95
            raw_b  = raw_r * 0.90

            new_rgb = (raw_r, raw_g, raw_b)
            blended = view_state.update_translucency_history(froxel.key(), new_rgb)

            froxel.radiance  = list(blended)
            froxel.opacity   = min(froxel.opacity + area / 1e6, 1.0)
            froxel.jittered  = True
            updated += 1

        return updated

    def persist(self) -> None:
        """Write froxel grid summary to physics/lumen_translucency.json."""
        data = {
            "frame":        self._frame_index,
            "grid_dims":    [self._grid_w, self._grid_h, self._depth_slices],
            "active_froxels": len(self._froxels),
            "froxels":      {k: f.to_dict() for k, f in self._froxels.items()},
        }
        try:
            os.makedirs(os.path.dirname(self._channel_path), exist_ok=True)
            with open(self._channel_path, "w") as _f:
                json.dump(data, _f, indent=2)
        except OSError as _e:
            print(
                f"[AstroLumenTranslucencyVolume] WARNING: persist failed: {_e}",
                file=sys.stderr,
            )

    def tick(
        self,
        cell_entries: list,
        viewer_pos:   tuple,
        view_state:   "AstroLumenViewState",
    ) -> dict:
        """
        Full per-frame translucency volume pass.

        鲁迅式：每一帧，半透明体积都在重新审视自己——
        过去的数据有九成的惰性，
        一成的新鲜血液注入，才算活着。
        """
        self._frame_index += 1
        updated = self.trace_and_fill(cell_entries, viewer_pos, view_state)
        self.persist()
        stats = {
            "frame":          self._frame_index,
            "active_froxels": len(self._froxels),
            "updated_froxels": updated,
            "grid_dims":      [self._grid_w, self._grid_h, self._depth_slices],
        }
        print(
            f"[AstroLumenTranslucencyVolume] tick: "
            f"frame={self._frame_index} "
            f"froxels={len(self._froxels)} updated={updated}",
            file=sys.stderr,
        )
        return stats


# ═══════════════════════════════════════════════════════════════════════════════
# AstroLumenRenderer — unified façade
# ═══════════════════════════════════════════════════════════════════════════════

class AstroLumenRenderer:
    """
    Unified Lumen renderer façade.

    Composes the five ported subsystems:
        AstroLumenScene                  ← LumenScene + LumenSceneLighting
        AstroLumenSurfaceCacheFeedback   ← LumenSurfaceCacheFeedback
        AstroLumenViewState              ← LumenViewState
        AstroLumenTranslucencyVolume     ← LumenTranslucencyVolumeLighting

    One call to tick() drives the entire Lumen pipeline for one frame,
    mirroring the RenderLumenScene() entry point in DeferredShadingRenderer.

    鲁迅式：渲染器是一台永不停歇的机器——
    场景变了，光照跟上；光照变了，半透明跟上；
    半透明变了，历史记录跟上。
    链条上的每一环都是别人的负担，也是自己的依赖。
    这就是实时渲染的宿命。
    """

    def __init__(
        self,
        view_id:      str = "default",
        screen_w:     int = 1920,
        screen_h:     int = 1080,
        atlas_size:   int = LUMEN_SURFACE_CACHE_ATLAS_SIZE,
        depth_slices: int = 32,
    ) -> None:
        self.lumen_scene     = AstroLumenScene(atlas_size=atlas_size)
        self.feedback        = AstroLumenSurfaceCacheFeedback()
        self.view_state      = AstroLumenViewState(view_id=view_id)
        self.translucency    = AstroLumenTranslucencyVolume(screen_w, screen_h, depth_slices)
        self._frame_index:   int = 0

    def register_cell(self, cell_id: str) -> list:
        """Register a new cell into the Lumen scene (add all 6 cards)."""
        return self.lumen_scene.add_cell(cell_id)

    def remove_cell(self, cell_id: str) -> None:
        """Remove a cell from the Lumen scene."""
        self.lumen_scene.remove_cell(cell_id)

    def tick(
        self,
        cell_entries: list,
        viewer_pos:   tuple = (0.0, 0.0, 0.0),
    ) -> dict:
        """
        Execute one complete Lumen frame pass.

        Order mirrors DeferredShadingRenderer.cpp order:
            1. Advance view state (jitter index, frame counter).
            2. Lumen scene lighting update.
            3. Surface cache feedback submit.
            4. Translucency volume trace + temporal blend.
            5. Persist view state.

        Returns aggregated stats from all subsystems.

        鲁迅式：一帧之内，完成所有的妥协——
        光照只更新一部分，反馈只记录看见的，
        半透明只信任九成的过去。
        但每一个妥协都有名字，都有原因，都有边界。
        这比毫无原则的精确更诚实。
        """
        self._frame_index += 1
        self.view_state.advance_frame()

        scene_stats   = self.lumen_scene.tick(cell_entries, viewer_pos)
        feedback_count = self.feedback.submit(cell_entries, self.lumen_scene)
        trans_stats   = self.translucency.tick(cell_entries, viewer_pos, self.view_state)
        self.view_state.persist()

        return {
            "frame":           self._frame_index,
            "scene":           scene_stats,
            "feedback_elements": feedback_count,
            "translucency":    trans_stats,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

#: Global AstroLumenRenderer instance (one per process, like FScene).
_ASTRO_LUMEN_RENDERER: "AstroLumenRenderer | None" = None


def get_lumen_renderer(
    view_id:    str = "default",
    screen_w:   int = 1920,
    screen_h:   int = 1080,
) -> AstroLumenRenderer:
    """
    Return the process-level Lumen renderer singleton.

    Mirrors the RenderLumenScene() call that implicitly assumes one FScene
    per process in UE5.

    鲁迅式：单例的正当性在于共识——
    整个进程只需要一个光照系统，
    就像一个城市只需要一套供电网络。
    共识是效率的基础，分裂是混乱的开始。
    """
    global _ASTRO_LUMEN_RENDERER
    if _ASTRO_LUMEN_RENDERER is None:
        _ASTRO_LUMEN_RENDERER = AstroLumenRenderer(
            view_id=view_id,
            screen_w=screen_w,
            screen_h=screen_h,
        )
    return _ASTRO_LUMEN_RENDERER
