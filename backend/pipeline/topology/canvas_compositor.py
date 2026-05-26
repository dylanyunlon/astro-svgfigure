"""canvas_compositor.py — The DoubleBuffer + dispatch().

CCCL f984c90's dispatch() function is 95 lines of pure orchestration:

    // Pass 0: dedicated histogram-only kernel
    { launcher.doit(histogram_kernel, d_keys_in, ...); }

    // Passes 1..N: fused filter+histogram with DoubleBuffer
    DoubleBuffer<key_in_t> key_bufs(alloc[3], alloc[2]);
    DoubleBuffer<OffsetT>  idx_bufs(alloc[5], alloc[4]);
    int pass = 1;
    for (; pass < num_passes; pass++) {
        launcher.doit(topk_kernel,
            key_bufs.Current(),  key_bufs.Alternate(),
            idx_bufs.Current(),  idx_bufs.Alternate(), ...);
        key_bufs.selector ^= 1;   // zero-copy swap
        idx_bufs.selector ^= 1;
    }

    // Final: invoke_last_filter on key_bufs.Current()

Three insights make this elegant:
  1. The kernels don't know about buffer management — they just read
     from their input and write to their output.  The dispatch function
     handles the plumbing.
  2. `selector ^= 1` is the entire swap operation.  No memcpy, no
     allocation, no deallocation.  Just flip a bit.
  3. After the loop, `Current()` always points to the latest result.
     The final filter reads from it without caring which physical
     buffer it is.

We apply the same pattern to our canvas composition:
  - Each region's subgraph is a "buffer" — a Dict[str, Any] (ELK JSON)
  - draft → validate → refine → final is our multi-pass loop
  - After each pass, the region's "current" buffer is the latest output
  - The compositor reads from Current() to build the unified canvas
  - Cross-region edges are the "global histogram merge" — resolved
    after all regions have been generated

This module also handles:
  - Absolute coordinate transform (region-local → canvas-global)
  - Collision resolution (overlapping regions)
  - Layer management (each region is a named layer)
  - Export to ELK, MastergoLayout, and per-region layer data
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from backend.pipeline.topology.region_planner import PlannedRegion
from backend.pipeline.topology.user_intent_parser import UserIntent
from backend.pipeline.topology.finalize_pass import PassContext

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  §1  DoubleBuffer — the zero-copy swap primitive
# ═══════════════════════════════════════════════════════════════════════════

class DoubleBuffer:
    """Zero-copy buffer swap for iterative region refinement.

    Direct port of CCCL's DoubleBuffer<T>:

        template <typename T>
        struct DoubleBuffer {
            T* d_buffers[2];
            int selector;
            T* Current()   { return d_buffers[selector]; }
            T* Alternate() { return d_buffers[selector ^ 1]; }
        };

    We store ELK subgraph dicts instead of GPU memory pointers.
    `selector ^= 1` swaps Current/Alternate with zero copy.
    """
    __slots__ = ("_buffers", "selector")

    def __init__(
        self,
        initial: Dict[str, Any],
        alternate: Optional[Dict[str, Any]] = None,
    ):
        self._buffers: list = [initial, alternate or {}]
        self.selector: int = 0

    def current(self) -> Dict[str, Any]:
        """Read buffer — the latest version of this region's subgraph."""
        return self._buffers[self.selector]

    def alternate(self) -> Dict[str, Any]:
        """Write buffer — where the next refinement pass writes to."""
        return self._buffers[self.selector ^ 1]

    def swap(self) -> None:
        """Flip selector.  O(1), no data copy.

        After swap(), what was Alternate becomes Current.
        Like CCCL's `key_bufs.selector ^= 1`.
        """
        self.selector ^= 1

    def set_alternate(self, data: Dict[str, Any]) -> None:
        """Write a new version to the alternate buffer, then swap."""
        self._buffers[self.selector ^ 1] = data
        self.swap()

    @property
    def pass_count(self) -> int:
        """How many swaps have occurred (= how many passes completed)."""
        return self.selector  # 0 after init, alternates with each swap


# ═══════════════════════════════════════════════════════════════════════════
#  §2  RegionBuffer — per-region state with DoubleBuffer
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RegionBuffer:
    """One region's mutable state during composition.

    Like CCCL's per-pass state: the region has a plan (from histogram
    kernel), a subgraph (from filter+histogram kernel), and a buffer
    for iterative refinement.
    """
    plan: PlannedRegion
    buffer: DoubleBuffer = field(init=False)
    is_finalized: bool = False
    cross_region_edges: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.buffer = DoubleBuffer(initial={})

    @property
    def subgraph(self) -> Dict[str, Any]:
        return self.buffer.current()

    @subgraph.setter
    def subgraph(self, value: Dict[str, Any]):
        self.buffer.set_alternate(value)


# ═══════════════════════════════════════════════════════════════════════════
#  §3  Layer — MasterGo/Figma-style layer abstraction
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Layer:
    """A named layer in the canvas, corresponding to one region.

    MasterGo's layer model:
      - Each layer has a name, visibility, lock state, z-index
      - Layers can be reordered, hidden, locked
      - Each layer contains the elements from one region
    """
    id: str
    name: str
    region_id: str
    z_index: int = 0
    visible: bool = True
    locked: bool = False
    opacity: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "region_id": self.region_id,
            "z_index": self.z_index,
            "visible": self.visible,
            "locked": self.locked,
            "opacity": self.opacity,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  §4  ComposedCanvas — the unified output
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ComposedCanvas:
    """The final output of composition: all regions merged into one canvas.

    Like CCCL's output after all passes: d_keys_out contains the top-k
    items in sorted order.  Our canvas contains all regions with absolute
    coordinates and resolved cross-region edges.
    """
    width: int
    height: int
    elk_graph: Dict[str, Any] = field(default_factory=dict)
    layers: List[Layer] = field(default_factory=list)
    cross_region_edges: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canvas": {"width": self.width, "height": self.height},
            "elk": self.elk_graph,
            "layers": [l.to_dict() for l in self.layers],
            "cross_region_edges": self.cross_region_edges,
            "diagnostics": self.diagnostics,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  §5  Composition — the dispatch() function
# ═══════════════════════════════════════════════════════════════════════════

def compose(
    regions: List[PlannedRegion],
    subgraphs: List[Dict[str, Any]],
    canvas_width: int,
    canvas_height: int,
    context: Optional[PassContext] = None,
) -> ComposedCanvas:
    """Compose N region subgraphs into a unified canvas.

    This is the dispatch() function — the orchestration layer that wires
    together the outputs of our kernels (region_planner + per_region_gen).

    Like CCCL's dispatch():
      1. Initialize buffers (RegionBuffer for each region)
      2. Place each subgraph at its region's bbox (coordinate transform)
      3. Resolve cross-region edges
      4. Build the unified ELK graph
      5. Generate layer metadata

    CCCL dispatch pattern:
        counter_t* counter = allocations[0];
        OffsetT* histogram = allocations[1];
        { histogram_kernel(...); }                    // pass 0
        DoubleBuffer key_bufs(alloc[3], alloc[2]);
        for (pass = 1; pass < num_passes; pass++) {   // passes 1..N
            topk_kernel(key_bufs.Current(), key_bufs.Alternate(), ...);
            key_bufs.selector ^= 1;
        }
        invoke_last_filter(key_bufs.Current(), ...);  // final

    Our pattern:
        region_buffers = [RegionBuffer(plan) for plan in regions]
        for i, buf in enumerate(region_buffers):      // "place" pass
            buf.subgraph = transform(subgraphs[i], buf.plan.bbox)
        cross_edges = resolve_cross_edges(region_buffers)  // final
        elk = merge_to_elk(region_buffers, cross_edges)

    Args:
        regions: Planned regions from M1 (histogram kernel output)
        subgraphs: Generated ELK subgraphs from M3 (one per region)
        canvas_width: Total canvas width in pixels
        canvas_height: Total canvas height in pixels
        context: Pipeline counter state

    Returns:
        ComposedCanvas with unified ELK graph and layer metadata
    """
    if context is None:
        context = PassContext()
    context.advance("composition", is_last=True)

    diag: Dict[str, Any] = {"pass": "composition"}

    # ── Initialize region buffers (like allocating DoubleBuffers) ──
    buffers: List[RegionBuffer] = []
    for i, region in enumerate(regions):
        rb = RegionBuffer(plan=region)
        if i < len(subgraphs):
            rb.buffer = DoubleBuffer(initial=subgraphs[i])
        buffers.append(rb)

    # ── Pass: coordinate transform (local → absolute) ──
    for rb in buffers:
        transformed = _transform_to_absolute(rb.subgraph, rb.plan.bbox)
        rb.buffer.set_alternate(transformed)
        # After set_alternate + implicit swap, Current() = transformed

    diag["regions_placed"] = len(buffers)

    # ── Resolve collisions between regions ──
    collision_count = _resolve_collisions(buffers)
    diag["collisions_resolved"] = collision_count

    # ── Extract and resolve cross-region edges ──
    cross_edges = _resolve_cross_region_edges(buffers)
    diag["cross_region_edges"] = len(cross_edges)

    # ── Build unified ELK graph ──
    elk = _merge_to_elk(buffers, cross_edges, canvas_width, canvas_height)
    diag["total_nodes"] = _count_all_nodes(elk)
    diag["total_edges"] = len(elk.get("edges", []))

    # ── Generate layers ──
    layers = _generate_layers(buffers)

    # ── Update context counters ──
    context.total_entities = diag["total_nodes"]
    context.total_edges = diag["total_edges"]

    return ComposedCanvas(
        width=canvas_width,
        height=canvas_height,
        elk_graph=elk,
        layers=layers,
        cross_region_edges=cross_edges,
        diagnostics=diag,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  §6  Coordinate transform — region-local → canvas-absolute
# ═══════════════════════════════════════════════════════════════════════════

def _transform_to_absolute(
    subgraph: Dict[str, Any],
    bbox: Dict[str, int],
) -> Dict[str, Any]:
    """Transform region-local coordinates to canvas-absolute.

    Like CCCL's index management: the histogram kernel works with
    absolute indices (0..num_items), but the filter kernel writes to
    a buffer-relative position via atomicAdd.  When reading from
    the original input, indices are absolute; when reading from a
    filtered buffer, they're relative.  The dispatch function handles
    the translation.

    We do the same: each region's subgraph has nodes at (0,0)-relative
    positions.  The compositor offsets them by the region's bbox origin.

    Node positioning within the region:
      - If nodes already have x, y: offset by bbox origin
      - If nodes lack x, y: distribute evenly within bbox
    """
    if not subgraph or not subgraph.get("children"):
        return subgraph

    result = {**subgraph}
    ox, oy = bbox["x"], bbox["y"]
    bw, bh = bbox["width"], bbox["height"]

    # Internal padding
    pad = 20
    inner_w = max(bw - 2 * pad, 100)
    inner_h = max(bh - 2 * pad, 80)

    children = result.get("children", [])
    needs_layout = any(
        not isinstance(c, dict) or ("x" not in c and "y" not in c)
        for c in children
    )

    if needs_layout:
        # Distribute nodes in a simple grid within the bbox
        n = len([c for c in children if isinstance(c, dict)])
        if n > 0:
            import math
            cols = max(1, math.ceil(math.sqrt(n * (inner_w / max(inner_h, 1)))))
            rows = max(1, math.ceil(n / cols))
            cell_w = inner_w / cols
            cell_h = inner_h / rows

            idx = 0
            for child in children:
                if not isinstance(child, dict):
                    continue
                col = idx % cols
                row = idx // cols
                w = child.get("width", 150)
                h = child.get("height", 50)
                # Center node within its cell
                child["x"] = ox + pad + col * cell_w + (cell_w - w) / 2
                child["y"] = oy + pad + row * cell_h + (cell_h - h) / 2
                idx += 1
    else:
        # Offset existing coordinates by bbox origin
        for child in children:
            if isinstance(child, dict):
                child["x"] = child.get("x", 0) + ox + pad
                child["y"] = child.get("y", 0) + oy + pad
                # Recurse into compound nodes
                if child.get("children"):
                    for sub in child["children"]:
                        if isinstance(sub, dict):
                            sub["x"] = sub.get("x", 0) + child["x"]
                            sub["y"] = sub.get("y", 0) + child["y"]

    result["children"] = children
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  §7  Collision resolution — no overlapping regions
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_collisions(buffers: List[RegionBuffer]) -> int:
    """Detect and resolve overlapping region bounding boxes.

    Like CCCL's candidate_class: items that are "selected" go to output,
    items that are "candidates" go to the next pass's buffer.  Overlapping
    regions are "candidates" that need repositioning.

    Algorithm: sweep-line on x-axis, then resolve y-overlaps.
    Simpler than full constraint solving — regions don't overlap often
    because the planner (M1) already assigns non-overlapping bboxes.
    """
    collisions = 0
    n = len(buffers)

    for i in range(n):
        for j in range(i + 1, n):
            bi = buffers[i].plan.bbox
            bj = buffers[j].plan.bbox

            if _rects_overlap(bi, bj):
                collisions += 1
                # Push the later region to the right or down
                # (whichever requires less displacement)
                dx = _overlap_x(bi, bj)
                dy = _overlap_y(bi, bj)

                if abs(dx) <= abs(dy):
                    bj["x"] += dx + 16  # 16px gap
                else:
                    bj["y"] += dy + 16

    return collisions


def _rects_overlap(a: Dict[str, int], b: Dict[str, int]) -> bool:
    """Check if two rects overlap (with 8px tolerance)."""
    gap = 8
    return not (
        a["x"] + a["width"] + gap <= b["x"] or
        b["x"] + b["width"] + gap <= a["x"] or
        a["y"] + a["height"] + gap <= b["y"] or
        b["y"] + b["height"] + gap <= a["y"]
    )


def _overlap_x(a: Dict[str, int], b: Dict[str, int]) -> int:
    """Horizontal overlap amount (positive = b needs to move right)."""
    return (a["x"] + a["width"]) - b["x"]


def _overlap_y(a: Dict[str, int], b: Dict[str, int]) -> int:
    """Vertical overlap amount (positive = b needs to move down)."""
    return (a["y"] + a["height"]) - b["y"]


# ═══════════════════════════════════════════════════════════════════════════
#  §8  Cross-region edge resolution — the global histogram merge
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_cross_region_edges(
    buffers: List[RegionBuffer],
) -> List[Dict[str, Any]]:
    """Resolve edges that cross region boundaries.

    In CCCL, each thread block builds a local histogram, then the last
    block merges all local histograms into the global histogram via
    atomicAdd.  Cross-region edges are our "global histogram" — they
    emerge from individual region subgraphs but connect the whole canvas.

    Detection: edges whose targets are prefixed with another region's ID
    (e.g., "other_region.node_id" in per_region_generator.py M3).

    Resolution: create global edges in the unified ELK graph with
    sources/targets referencing the absolute node IDs.
    """
    cross_edges: List[Dict[str, Any]] = []

    # Build a global node ID registry
    all_nodes: Dict[str, str] = {}  # node_id → region_id
    for rb in buffers:
        region_id = rb.plan.id
        for node_id in _collect_node_ids(rb.subgraph):
            all_nodes[node_id] = region_id

    edge_id_counter = 0

    for rb in buffers:
        region_id = rb.plan.id
        subgraph = rb.subgraph

        for edge in subgraph.get("edges", []):
            sources = edge.get("sources", [])
            targets = edge.get("targets", [])

            for target in targets:
                # Check if target is a cross-region reference
                # Convention: "other_region.node_id" or "other_region_node_id"
                if "." in target:
                    parts = target.split(".", 1)
                    target_region = parts[0]
                    target_node = parts[1]

                    # Find the actual node
                    resolved = _find_node_in_region(
                        buffers, target_region, target_node,
                    )
                    if resolved:
                        edge_id_counter += 1
                        cross_edges.append({
                            "id": f"cross_e{edge_id_counter}",
                            "sources": sources,
                            "targets": [resolved],
                            "source_region": region_id,
                            "target_region": target_region,
                        })
                elif target not in all_nodes or all_nodes[target] != region_id:
                    # Target exists in a different region
                    if target in all_nodes:
                        edge_id_counter += 1
                        cross_edges.append({
                            "id": f"cross_e{edge_id_counter}",
                            "sources": sources,
                            "targets": [target],
                            "source_region": region_id,
                            "target_region": all_nodes[target],
                        })

    return cross_edges


def _find_node_in_region(
    buffers: List[RegionBuffer],
    region_id: str,
    node_id: str,
) -> Optional[str]:
    """Find a node by region + local ID, return its absolute ID."""
    for rb in buffers:
        if rb.plan.id == region_id:
            all_ids = _collect_node_ids(rb.subgraph)
            # Try exact match first
            if node_id in all_ids:
                return node_id
            # Try prefixed match (region_id_node_id)
            prefixed = f"{region_id}_{node_id}"
            if prefixed in all_ids:
                return prefixed
            # Try fuzzy match (partial)
            for aid in all_ids:
                if node_id in aid or aid in node_id:
                    return aid
    return None


def _collect_node_ids(node: Dict[str, Any]) -> Set[str]:
    """Recursively collect all node IDs from an ELK graph."""
    ids: Set[str] = set()
    nid = node.get("id")
    if nid and nid != "root":
        ids.add(nid)
    for child in node.get("children", []):
        if isinstance(child, dict):
            ids.update(_collect_node_ids(child))
    return ids


# ═══════════════════════════════════════════════════════════════════════════
#  §9  Merge — build the unified ELK graph
# ═══════════════════════════════════════════════════════════════════════════

def _merge_to_elk(
    buffers: List[RegionBuffer],
    cross_edges: List[Dict[str, Any]],
    canvas_width: int,
    canvas_height: int,
) -> Dict[str, Any]:
    """Merge all region subgraphs into one ELK graph.

    Like CCCL's final output: after all passes, d_keys_out contains
    the complete top-k result.  After our composition, the ELK graph
    contains all nodes from all regions with absolute coordinates.

    Structure:
        {
          "id": "root",
          "children": [
            // Region containers (compound nodes with children)
            { "id": "region_1", "group": true, "children": [...], ... },
            { "id": "region_2", "group": true, "children": [...], ... },
          ],
          "edges": [
            // Cross-region edges
          ],
          "layoutOptions": { ... }
        }
    """
    root_children: List[Dict[str, Any]] = []

    for rb in buffers:
        region_node = _build_region_container(rb)
        root_children.append(region_node)

    # Cross-region edges go at the root level
    root_edges: List[Dict[str, Any]] = []
    for ce in cross_edges:
        root_edges.append({
            "id": ce["id"],
            "sources": ce["sources"],
            "targets": ce["targets"],
        })

    elk = {
        "id": "root",
        "children": root_children,
        "edges": root_edges,
        "layoutOptions": {
            "elk.algorithm": "layered",
            "elk.direction": "DOWN",
            "elk.spacing.nodeNode": "16",
            "elk.padding": "[top=20,left=20,bottom=20,right=20]",
        },
    }

    return elk


def _build_region_container(rb: RegionBuffer) -> Dict[str, Any]:
    """Build an ELK compound node for one region.

    The region container:
      - Has the region's bbox as its position and size
      - Contains all nodes from the subgraph as children
      - Contains all intra-region edges
      - Has visual style (group border, label)
    """
    plan = rb.plan
    bbox = plan.bbox
    subgraph = rb.subgraph

    # Collect internal children and edges
    children = subgraph.get("children", [])
    internal_edges = []
    for edge in subgraph.get("edges", []):
        sources = edge.get("sources", [])
        targets = edge.get("targets", [])
        # Only keep edges where both ends are in this region
        if not any("." in t for t in targets):
            internal_edges.append(edge)

    region_node: Dict[str, Any] = {
        "id": plan.id,
        "x": bbox["x"],
        "y": bbox["y"],
        "width": bbox["width"],
        "height": bbox["height"],
        "labels": [{"text": plan.name}],
        "group": True,
        "borderless": False,
        "children": children,
        "edges": internal_edges,
        "layoutOptions": {
            "elk.padding": "[top=35,left=12,bottom=12,right=12]",
        },
    }

    # Apply style from region plan
    if plan.style:
        color = plan.style.get("color_family")
        if color:
            region_node["style"] = {"color_family": color}

    return region_node


# ═══════════════════════════════════════════════════════════════════════════
#  §10  Layer generation — MasterGo/Figma layer model
# ═══════════════════════════════════════════════════════════════════════════

def _generate_layers(buffers: List[RegionBuffer]) -> List[Layer]:
    """Generate layer metadata for each region.

    Each region becomes a named layer in the canvas, ordered by
    priority (z-index).  The user can reorder, hide, or lock layers
    in the frontend (M10).
    """
    layers: List[Layer] = []
    for i, rb in enumerate(buffers):
        layer = Layer(
            id=f"layer_{rb.plan.id}",
            name=rb.plan.name,
            region_id=rb.plan.id,
            z_index=i,
            visible=True,
            locked=False,
        )
        layers.append(layer)

    return layers


# ═══════════════════════════════════════════════════════════════════════════
#  §11  Utility — count all nodes
# ═══════════════════════════════════════════════════════════════════════════

def _count_all_nodes(elk: Dict[str, Any]) -> int:
    """Count all leaf nodes (non-group) in the ELK graph."""
    count = 0
    for child in elk.get("children", []):
        if isinstance(child, dict):
            if child.get("children"):
                count += _count_all_nodes(child)
            else:
                count += 1
    return count


# ═══════════════════════════════════════════════════════════════════════════
#  §12  Re-compose — update a single region without re-running everything
# ═══════════════════════════════════════════════════════════════════════════

def recompose_region(
    canvas: ComposedCanvas,
    region_id: str,
    new_subgraph: Dict[str, Any],
    regions: List[PlannedRegion],
) -> ComposedCanvas:
    """Replace one region's subgraph and recompose.

    Like CCCL's approach to re-running a single pass: you don't
    restart the entire pipeline.  You swap in the new buffer for
    that region and re-merge.

    This enables the "regenerate this region" feature in the frontend:
    the user clicks "regenerate" on one region, we re-run M3 for just
    that region, then call this function to update the canvas.
    """
    # Find the region and update its subgraph
    region_plan = None
    for r in regions:
        if r.id == region_id:
            region_plan = r
            break

    if region_plan is None:
        logger.warning(f"Region {region_id} not found, returning unchanged canvas")
        return canvas

    # Rebuild the buffers list from the current canvas
    buffers: List[RegionBuffer] = []
    for r in regions:
        rb = RegionBuffer(plan=r)
        if r.id == region_id:
            # Swap in the new subgraph
            transformed = _transform_to_absolute(new_subgraph, r.bbox)
            rb.buffer = DoubleBuffer(initial=transformed)
        else:
            # Use existing children from the elk graph
            for child in canvas.elk_graph.get("children", []):
                if isinstance(child, dict) and child.get("id") == r.id:
                    rb.buffer = DoubleBuffer(initial=child)
                    break
        buffers.append(rb)

    # Re-resolve cross-region edges
    cross_edges = _resolve_cross_region_edges(buffers)

    # Re-merge
    elk = _merge_to_elk(buffers, cross_edges, canvas.width, canvas.height)
    layers = _generate_layers(buffers)

    return ComposedCanvas(
        width=canvas.width,
        height=canvas.height,
        elk_graph=elk,
        layers=layers,
        cross_region_edges=cross_edges,
        diagnostics={
            "recomposed_region": region_id,
            "total_nodes": _count_all_nodes(elk),
            "total_edges": len(elk.get("edges", [])),
        },
    )
