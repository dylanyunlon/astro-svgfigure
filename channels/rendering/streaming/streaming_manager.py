
# ─────────────────────────────────────────────────────────────────────────────
# M824: Streaming LOD Manager
# ─────────────────────────────────────────────────────────────────────────────

import sys
import time
import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import IntEnum


class LODLevel(IntEnum):
    """Level-of-Detail tiers. Lower = more detail."""
    LOD0_FULL = 0       # Full detail: SDF icon + MSDF label + all filters
    LOD1_SIMPLIFIED = 1  # Simplified: flat icon + bitmap label + reduced filters
    LOD2_ICON_ONLY = 2   # Icon only: colored rectangle + species badge
    LOD3_DOT = 3         # Dot: single colored circle at cell center


@dataclass(order=True)
class StreamingRequest:
    """Priority queue entry for LOD streaming."""
    priority: float
    cell_id: str = field(compare=False)
    target_lod: LODLevel = field(compare=False)
    distance: float = field(compare=False)
    timestamp: float = field(compare=False, default_factory=time.time)


class StreamingManager:
    """
    Distance-based LOD streaming manager.

    Assigns LOD levels to cells based on distance from camera/viewport center.
    Manages a priority queue for LOD transitions — closer cells upgrade first.

    LOD thresholds (in world units, configurable):
      LOD0: distance < 200   (full detail)
      LOD1: distance < 500   (simplified)
      LOD2: distance < 1000  (icon only)
      LOD3: distance >= 1000 (dot)

    References:
      - upstream/potree/ octree LOD strategy
      - channels/rendering/nanite/ for UE5-style LOD concepts

    [ASTRO-STREAMING] debug tag family.
    """

    def __init__(self,
                 lod0_dist: float = 200.0,
                 lod1_dist: float = 500.0,
                 lod2_dist: float = 1000.0,
                 max_upgrades_per_frame: int = 4):
        self.thresholds = {
            LODLevel.LOD0_FULL: lod0_dist,
            LODLevel.LOD1_SIMPLIFIED: lod1_dist,
            LODLevel.LOD2_ICON_ONLY: lod2_dist,
        }
        self.max_upgrades_per_frame = max_upgrades_per_frame
        self._current_lods: Dict[str, LODLevel] = {}
        self._queue: List[StreamingRequest] = []
        self._frame_upgrades = 0

    def classify(self, cell_id: str, distance: float) -> LODLevel:
        """Classify a cell's LOD level based on distance."""
        if distance < self.thresholds[LODLevel.LOD0_FULL]:
            lod = LODLevel.LOD0_FULL
        elif distance < self.thresholds[LODLevel.LOD1_SIMPLIFIED]:
            lod = LODLevel.LOD1_SIMPLIFIED
        elif distance < self.thresholds[LODLevel.LOD2_ICON_ONLY]:
            lod = LODLevel.LOD2_ICON_ONLY
        else:
            lod = LODLevel.LOD3_DOT

        old_lod = self._current_lods.get(cell_id, LODLevel.LOD3_DOT)
        if lod != old_lod:
            # LOD upgrade (higher detail) needs streaming
            priority = distance  # closer = higher priority (lower number)
            if lod < old_lod:
                # Upgrading — queue it
                heapq.heappush(self._queue, StreamingRequest(
                    priority=priority, cell_id=cell_id,
                    target_lod=lod, distance=distance))
                print(f"[ASTRO-STREAMING] queue upgrade: {cell_id} "
                      f"LOD{old_lod}→LOD{lod} dist={distance:.1f}",
                      file=sys.stderr)
            else:
                # Downgrading — immediate (free resources)
                self._current_lods[cell_id] = lod
                print(f"[ASTRO-STREAMING] downgrade: {cell_id} "
                      f"LOD{old_lod}→LOD{lod} dist={distance:.1f}",
                      file=sys.stderr)
        return self._current_lods.get(cell_id, lod)

    def process_frame(self) -> List[Tuple[str, LODLevel]]:
        """
        Process queued LOD upgrades for this frame.
        Returns list of (cell_id, new_lod) that were upgraded.
        """
        upgraded = []
        self._frame_upgrades = 0
        while self._queue and self._frame_upgrades < self.max_upgrades_per_frame:
            req = heapq.heappop(self._queue)
            old = self._current_lods.get(req.cell_id, LODLevel.LOD3_DOT)
            if req.target_lod < old:
                self._current_lods[req.cell_id] = req.target_lod
                upgraded.append((req.cell_id, req.target_lod))
                self._frame_upgrades += 1
                print(f"[ASTRO-STREAMING] upgraded: {req.cell_id} → LOD{req.target_lod}",
                      file=sys.stderr)
        return upgraded

    def get_lod(self, cell_id: str) -> LODLevel:
        return self._current_lods.get(cell_id, LODLevel.LOD3_DOT)

    @property
    def stats(self) -> dict:
        from collections import Counter
        lod_counts = Counter(self._current_lods.values())
        return {
            "cells_tracked": len(self._current_lods),
            "pending_upgrades": len(self._queue),
            "lod_distribution": {f"LOD{k}": v for k, v in sorted(lod_counts.items())},
        }


class MipMapScheduler:
    """
    Schedules MSDF texture mipmap loading by cell distance and visibility.

    [ASTRO-STREAMING] Mipmap scheduling prevents bandwidth spikes by
    spreading texture loads across multiple frames.
    """

    def __init__(self, max_loads_per_frame: int = 2):
        self.max_loads = max_loads_per_frame
        self._pending: List[Tuple[float, str, int]] = []  # (priority, cell_id, mip_level)
        self._loaded: Dict[str, int] = {}  # cell_id → highest loaded mip

    def request_mip(self, cell_id: str, mip_level: int, distance: float):
        current = self._loaded.get(cell_id, 99)
        if mip_level < current:
            heapq.heappush(self._pending, (distance, cell_id, mip_level))

    def process_frame(self) -> List[Tuple[str, int]]:
        loaded = []
        count = 0
        while self._pending and count < self.max_loads:
            _, cell_id, mip = heapq.heappop(self._pending)
            current = self._loaded.get(cell_id, 99)
            if mip < current:
                self._loaded[cell_id] = mip
                loaded.append((cell_id, mip))
                count += 1
                print(f"[ASTRO-STREAMING] mipmap loaded: {cell_id} mip={mip}",
                      file=sys.stderr)
        return loaded
