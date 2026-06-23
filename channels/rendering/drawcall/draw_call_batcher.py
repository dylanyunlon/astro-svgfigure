"""
draw_call_batcher.py — M823: DrawCall batching + instanced cell rendering.

Merges cells of the same species into instanced draw calls to reduce
CPU→GPU overhead.  Maps to UE4 MeshDrawCommands.cpp batch-and-sort logic.

[ASTRO-DRAWCALL] debug tag family.
"""

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# DrawCall primitive
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DrawCall:
    """A single GPU draw call — one species, one z-layer, multiple cells."""
    species: str
    z_layer: int
    cell_ids: List[str] = field(default_factory=list)
    instance_count: int = 0
    blend_mode: str = "normal"        # normal | additive | multiply | screen
    shader_program: str = ""          # e.g. "cil-eye.frag"
    # Per-instance data (packed for GPU upload)
    instance_positions: List[Tuple[float, float]] = field(default_factory=list)
    instance_sizes: List[Tuple[float, float]] = field(default_factory=list)
    instance_opacities: List[float] = field(default_factory=list)
    instance_colors: List[str] = field(default_factory=list)

    def add_cell(self, cell_id: str, x: float, y: float,
                 w: float, h: float, opacity: float, color: str):
        """Add a cell instance to this draw call."""
        self.cell_ids.append(cell_id)
        self.instance_positions.append((x, y))
        self.instance_sizes.append((w, h))
        self.instance_opacities.append(opacity)
        self.instance_colors.append(color)
        self.instance_count = len(self.cell_ids)


# ─────────────────────────────────────────────────────────────────────────────
# DrawCallBatcher — merges same-species cells into instanced calls
# ─────────────────────────────────────────────────────────────────────────────

class DrawCallBatcher:
    """
    Collects cell render requests and batches them by (species, z_layer).

    Instanced rendering means we draw all cil-eye cells in one GPU call,
    all cil-bolt cells in another, etc — instead of one call per cell.

    [ASTRO-DRAWCALL] Batch key = (species, z_layer, blend_mode).
    """

    def __init__(self):
        self._batches: Dict[Tuple[str, int, str], DrawCall] = {}
        self._sorted_calls: List[DrawCall] = []
        self._dirty = True

    def submit(self, cell_id: str, species: str, z_layer: int,
               x: float, y: float, w: float, h: float,
               opacity: float = 1.0, color: str = "#FFFFFF",
               blend_mode: str = "normal"):
        """Submit a cell for batched rendering."""
        key = (species, z_layer, blend_mode)
        if key not in self._batches:
            self._batches[key] = DrawCall(
                species=species, z_layer=z_layer, blend_mode=blend_mode,
                shader_program=f"{species}.frag")
        self._batches[key].add_cell(cell_id, x, y, w, h, opacity, color)
        self._dirty = True
        print(f"[ASTRO-DRAWCALL] submit cell={cell_id} species={species} "
              f"z={z_layer} batch_size={self._batches[key].instance_count}",
              file=sys.stderr)

    def flush(self) -> List[DrawCall]:
        """
        Sort and return all batched draw calls.

        Sort order (back-to-front, then by species for coherence):
          1. z_layer ascending (back first)
          2. blend_mode: normal → additive → multiply → screen
          3. species alphabetical (minimise shader switches)
        """
        if not self._dirty and self._sorted_calls:
            return self._sorted_calls

        _blend_order = {"normal": 0, "additive": 1, "multiply": 2, "screen": 3}
        self._sorted_calls = sorted(
            self._batches.values(),
            key=lambda dc: (dc.z_layer, _blend_order.get(dc.blend_mode, 9), dc.species))

        total_instances = sum(dc.instance_count for dc in self._sorted_calls)
        print(f"[ASTRO-DRAWCALL] flush: {len(self._sorted_calls)} draw calls, "
              f"{total_instances} total instances", file=sys.stderr)
        self._dirty = False
        return self._sorted_calls

    def clear(self):
        """Reset for next frame."""
        self._batches.clear()
        self._sorted_calls.clear()
        self._dirty = True

    @property
    def stats(self) -> dict:
        calls = self.flush()
        return {
            "draw_call_count": len(calls),
            "total_instances": sum(dc.instance_count for dc in calls),
            "species_groups": len(set(dc.species for dc in calls)),
            "z_layers_used": sorted(set(dc.z_layer for dc in calls)),
        }


# ─────────────────────────────────────────────────────────────────────────────
# DrawCallSorter — separate sorter for custom orderings
# ─────────────────────────────────────────────────────────────────────────────

class DrawCallSorter:
    """
    Provides alternative sort strategies for draw calls.

    [ASTRO-DRAWCALL] Sort strategies:
      - back_to_front: standard painter's algorithm (default)
      - front_to_back: early-z optimisation for opaque passes
      - species_coherent: minimise shader/texture switches
    """

    @staticmethod
    def back_to_front(calls: List[DrawCall]) -> List[DrawCall]:
        return sorted(calls, key=lambda dc: dc.z_layer)

    @staticmethod
    def front_to_back(calls: List[DrawCall]) -> List[DrawCall]:
        return sorted(calls, key=lambda dc: -dc.z_layer)

    @staticmethod
    def species_coherent(calls: List[DrawCall]) -> List[DrawCall]:
        """Group by species first (fewer shader switches), then z."""
        return sorted(calls, key=lambda dc: (dc.species, dc.z_layer))


# ─────────────────────────────────────────────────────────────────────────────
# IndirectDrawBuffer — GPU indirect draw args buffer
# ─────────────────────────────────────────────────────────────────────────────

class IndirectDrawBuffer:
    """
    Prepares indirect draw arguments for GPU-driven rendering.

    Each entry = (vertex_count, instance_count, first_vertex, first_instance)
    matching WebGL2 drawArraysInstanced / drawElementsInstanced layout.

    [ASTRO-DRAWCALL] Indirect buffer reduces CPU→GPU draw call overhead.
    """

    def __init__(self):
        self._entries: List[Tuple[int, int, int, int]] = []

    def build_from_calls(self, calls: List[DrawCall],
                         vertices_per_cell: int = 6):
        """
        Build indirect draw buffer from sorted draw calls.

        Args:
            calls: sorted draw calls from DrawCallBatcher.flush()
            vertices_per_cell: vertices per cell quad (6 for two triangles)
        """
        self._entries.clear()
        first_instance = 0
        for dc in calls:
            self._entries.append((
                vertices_per_cell,      # vertex_count
                dc.instance_count,      # instance_count
                0,                      # first_vertex
                first_instance,         # first_instance
            ))
            first_instance += dc.instance_count
            print(f"[ASTRO-DRAWCALL] indirect: species={dc.species} "
                  f"instances={dc.instance_count} offset={first_instance - dc.instance_count}",
                  file=sys.stderr)

    @property
    def buffer(self) -> List[Tuple[int, int, int, int]]:
        return self._entries

    @property
    def total_draw_calls(self) -> int:
        return len(self._entries)

    @property
    def total_instances(self) -> int:
        return sum(e[1] for e in self._entries)
