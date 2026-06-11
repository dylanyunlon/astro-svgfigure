#!/usr/bin/env python3
"""
Loop Orchestrator — runs one full epoch of the cell pub/sub loop.

In production: each cell is a separate sub-Claude via claude_hk_chat.sh
Here: we simulate the loop locally to verify convergence.

Flow:
  1. Each cell reads its skeleton signal + force_field (subscribe)
  2. Each cell generates SVG with its species algorithm (proc)
  3. Each cell publishes bbox + svg (publish)
  4. Physics engine reads all bboxes → detects 3D collisions → updates force_field
  5. Convergence judge checks if all cells are stable
  6. If not converged → next epoch
  7. If converged → assemble final SVG by z-layer order
"""
import json
import math
import os
import glob

CHANNELS = os.path.dirname(os.path.abspath(__file__))

def read_channel(path):
    with open(os.path.join(CHANNELS, path)) as f:
        return json.load(f)

def write_channel(path, data):
    full = os.path.join(CHANNELS, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    if isinstance(data, str):
        with open(full, "w") as f:
            f.write(data)
    else:
        with open(full, "w") as f:
            json.dump(data, f, indent=2)


def run_all_cells():
    """Execute all cell components (simulate parallel sub-Claude dispatch)."""
    from cell_component import proc
    cells = [f.replace(".json", "") for f in os.listdir(os.path.join(CHANNELS, "skeleton", "cell"))]
    for cell_id in sorted(cells):
        proc(cell_id)
    return cells


def _sweep_line_overlaps(rects_by_z):
    """
    FAstroCellSweepLine — ported from upstream/unreal-renderer/SceneSoftwareOcclusion.cpp
    commit 0b4b199.

    Algorithm: O((N+K) log N) sweep-line 2D overlap detection.
      1. For every cell bbox emit an OPEN event at min_x and a CLOSE event at max_x.
      2. Sort all events by x; ties: OPEN before CLOSE so touching edges are caught.
      3. Scan events left→right, maintaining an active set of currently open rects.
      4. On OPEN: insert into active set; test new rect against every active rect for
         Y-interval overlap → emit collision pair.
      5. On CLOSE: remove from active set.
      6. Cells on different z-layers never collide (same logic as the original O(N²)).

    Returns a list of (cell_id_a, cell_id_b, overlap_x, overlap_y, z) tuples.
    """
    # Event type constants  (mirrors FSweepEvent.bOpen)
    OPEN  = 1   # process opens BEFORE closes at same x (sort key trick from C++)
    CLOSE = 0

    pairs = []

    for z_layer, rects in rects_by_z.items():
        # rects: list of (cell_id, min_x, min_y, max_x, max_y)
        if len(rects) < 2:
            continue

        # Build events — two per rect, mirroring AddRect()
        # Sort key: (x, -is_open) so OPEN (1) sorts before CLOSE (0) at equal x
        events = []
        for idx, (cell_id, min_x, min_y, max_x, max_y) in enumerate(rects):
            if min_x > max_x or min_y > max_y:   # degenerate guard (same as C++)
                continue
            events.append((min_x, -OPEN,  idx))  # OPEN  at min_x
            events.append((max_x, -CLOSE, idx))  # CLOSE at max_x

        events.sort()   # ascending x; at equal x: -OPEN (-1) < -CLOSE (0) → open first

        # Active set: indices of rects whose x-interval contains current sweep position
        active = []   # list of rect indices (mirrors TArray<int32> ActiveSet in C++)

        for ev_x, neg_kind, idx in events:
            is_open = (-neg_kind == OPEN)
            cell_id, min_x, min_y, max_x, max_y = rects[idx]

            if is_open:
                # Test new rect against every currently active rect for Y overlap.
                # In the C++ version this happens implicitly when an occludee leaves
                # the sweep; here we emit pairs eagerly on OPEN so both directions are
                # symmetric and we don't need occluder/occludee distinction.
                for aidx in active:
                    _, _, a_min_y, _, a_max_y = rects[aidx]
                    # Y-interval overlap test (mirrors OR.MaxY >= OccludeeMinY && OR.MinY <= OccludeeMaxY)
                    if a_min_y <= max_y and a_max_y >= min_y:
                        ov_y = min(max_y, a_max_y) - max(min_y, a_min_y)
                        # x-overlap is guaranteed because we're inside the active sweep window
                        a_cell_id, a_min_x, _, a_max_x, _ = rects[aidx]
                        ov_x = min(max_x, a_max_x) - max(min_x, a_min_x)
                        if ov_x > 0 and ov_y > 0:
                            pairs.append((cell_id, a_cell_id, ov_x, ov_y, z_layer))
                active.append(idx)
            else:
                # CLOSE: remove from active set (mirrors RemoveSingleSwap)
                try:
                    active.remove(idx)
                except ValueError:
                    pass   # already removed (shouldn't happen, but be defensive)

    return pairs


def physics_engine():
    """
    Physics Engine organ — reads all cell/*/bbox.json, detects 3D collisions,
    computes force field. Publishes to physics/force_field.json.

    Collision detection replaced with FAstroCellSweepLine sweep-line algorithm
    ported from commit 0b4b199 of upstream/unreal-renderer/SceneSoftwareOcclusion.cpp.
    Complexity: O((N+K) log N) vs the previous O(N²) brute-force double loop.

    Only same-z cells can collide. Different z = no collision (preserved).
    """
    bbox_files = glob.glob(os.path.join(CHANNELS, "cell", "*", "bbox.json"))
    bboxes = {}
    for bf in bbox_files:
        cell_id = os.path.basename(os.path.dirname(bf))
        with open(bf) as f:
            bboxes[cell_id] = json.load(f)

    force_field = {cid: {"dx": 0, "dy": 0, "dz": 0} for cid in bboxes}

    # --- 1. Bucket cells by z-layer (same-z-only rule preserved from original) ---
    rects_by_z = {}
    for cell_id, b in bboxes.items():
        z = b.get("z", 3)
        min_x = int(b["x"])
        min_y = int(b["y"])
        max_x = int(b["x"] + b["w"])
        max_y = int(b["y"] + b["h"])
        rects_by_z.setdefault(z, []).append((cell_id, min_x, min_y, max_x, max_y))

    # --- 2. Run sweep-line O((N+K) log N) overlap detection ---------------------
    overlap_pairs = _sweep_line_overlaps(rects_by_z)

    # --- 3. Convert overlapping pairs → collisions + repulsion forces -----------
    collisions = []
    for cell_a, cell_b, overlap_x, overlap_y, z in overlap_pairs:
        collisions.append({
            "a": cell_a, "b": cell_b,
            "overlap": overlap_x * overlap_y,
            "z": z
        })
        a = bboxes[cell_a]
        b = bboxes[cell_b]
        # Repulsion force: push apart along axis of least overlap (logic unchanged)
        if overlap_x < overlap_y:
            push = overlap_x / 2 + 5
            if a["x"] < b["x"]:
                force_field[cell_a]["dx"] -= push
                force_field[cell_b]["dx"] += push
            else:
                force_field[cell_a]["dx"] += push
                force_field[cell_b]["dx"] -= push
        else:
            push = overlap_y / 2 + 5
            if a["y"] < b["y"]:
                force_field[cell_a]["dy"] -= push
                force_field[cell_b]["dy"] += push
            else:
                force_field[cell_a]["dy"] += push
                force_field[cell_b]["dy"] -= push

    write_channel("physics/force_field.json", force_field)
    write_channel("physics/collision.json", {"collisions": collisions, "count": len(collisions)})

    print(f"[Physics] {len(collisions)} collisions detected (sweep-line O((N+K)logN))")
    return collisions


def convergence_judge():
    """Check if all cells have converged (no collisions)."""
    collision_data = read_channel("physics/collision.json")
    count = collision_data.get("count", 999)
    epoch = read_channel("skeleton/epoch.json")

    converged = count == 0
    write_channel("physics/converged.json", {
        "converged": converged,
        "epoch": epoch["current"],
        "conflicts": count
    })
    print(f"[Judge] epoch={epoch['current']} conflicts={count} converged={converged}")
    return converged


# ---------------------------------------------------------------------------
# FAstroCellRegistry + FConstraintCollector
# Ported from upstream/unreal-renderer/SceneRendering.cpp commit 7eae73c.
#
# Original: C++ structs FAstroCellSlot / AstroConstraintCollector operating
# on FViewInfo arrays with GPU-upload uint32 packing.
# Python port: dataclass registry + collector class over cell bbox/topology
# data; packing repurposed as constraint_score for z-layer sort priority.
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class AstroCellSlot:
    """
    FAstroCellSlot — one registered cell in the topology graph.

    Fields mirror the C++ struct:
      cell_id        ↔  CellId   (str here instead of int32)
      z_layer        ↔  ZLayer
      constraint_mask↔  ConstraintMask  (packed outgoing-edge bits)
      active         ↔  bActive
      bbox           — bbox dict (x,y,w,h); not in C++ slot, added for Python
      species        — species string derived from bbox.json
      constraints    — {"incoming": [...], "outgoing": [...], "neighbors": [...]}
    """
    cell_id: str
    z_layer: int
    constraint_mask: int = 0          # low 8 bits: outgoing-edge count (clamped)
    active: bool = True
    bbox: Dict = field(default_factory=dict)
    species: str = ""
    constraints: Dict = field(default_factory=lambda: {
        "incoming": [], "outgoing": [], "neighbors": []
    })


class AstroConstraintCollector:
    """
    AstroConstraintCollector — walks all registered slots and aggregates
    constraints into a packed buffer, mirroring AstroConstraintCollector::Collect()
    from commit 7eae73c.

    Pack layout (uint32 equivalent → Python int):
      bits 31-16 : cell_id hash (low 16 bits of hash(cell_id))
      bits 15- 8 : z_layer  & 0xFF
      bits  7- 0 : constraint_mask & 0xFF   (= outgoing-edge count, clamped 0-255)

    Divergence from C++: we also expose per-cell outgoing_edge_count() for the
    z-layer sort step — the C++ version implicitly carries this in ConstraintMask
    which we use the same way.
    """

    def __init__(self, registry: List[AstroCellSlot]) -> None:
        self._registry = registry
        self._constraint_buffer: List[int] = []

    def collect(self) -> None:
        """Iterate active slots; pack constraints → _constraint_buffer."""
        self._constraint_buffer.clear()
        for slot in self._registry:
            if not slot.active:
                continue
            # Pack: cell_id hash (16b) | z_layer (8b) | constraint_mask (8b)
            cell_id_bits = hash(slot.cell_id) & 0xFFFF
            packed = (
                (cell_id_bits          << 16) |
                ((slot.z_layer & 0xFF) <<  8) |
                (slot.constraint_mask  &  0xFF)
            )
            self._constraint_buffer.append(packed)

        print(
            f"[ASTRO-CONSTRAINT] Collected {len(self._constraint_buffer)} constraints "
            f"from {len(self._registry)} registered cells"
        )

    @property
    def constraint_buffer(self) -> List[int]:
        return self._constraint_buffer

    def active_cell_count(self) -> int:
        return len(self._constraint_buffer)

    def outgoing_edge_count(self, cell_id: str) -> int:
        """Return the outgoing-edge priority score for a given cell_id."""
        for slot in self._registry:
            if slot.cell_id == cell_id and slot.active:
                return len(slot.constraints["outgoing"])
        return 0


def _build_cell_registry(
    bbox_map: Dict[str, Dict],
    topology_edges: List[Dict],
    species_map: Dict[str, str],
) -> List[AstroCellSlot]:
    """
    [ASTRO-CELL-REGISTRY] Populate registry from topology, mirroring the
    InitViews block in SceneRendering.cpp (commit 7eae73c).

    Each cell_id → AstroCellSlot; ZLayer from bbox["z"]; constraint_mask
    encodes outgoing-edge count (clamped to 0xFF), analogous to how the C++
    code encodes StereoPass into ConstraintMask as a lightweight hint without
    touching the public API.

    Neighbor relations: cells sharing a z-layer are mutual neighbors,
    mirroring how the C++ registry groups view-ports by GPU z-layer slot.
    """
    # --- Phase 1: build incoming / outgoing maps from topology edges ----------
    outgoing: Dict[str, List[str]] = {cid: [] for cid in bbox_map}
    incoming: Dict[str, List[str]] = {cid: [] for cid in bbox_map}
    for edge in topology_edges:
        for src in edge.get("sources", []):
            for tgt in edge.get("targets", []):
                if src in outgoing:
                    outgoing[src].append(tgt)
                if tgt in incoming:
                    incoming[tgt].append(src)

    # --- Phase 2: bucket cells by z-layer for neighbor detection --------------
    z_buckets: Dict[int, List[str]] = {}
    for cid, b in bbox_map.items():
        z = b.get("z", 3)
        z_buckets.setdefault(z, []).append(cid)

    # --- Phase 3: build slots (monotonically assigned, epoch-coherent) --------
    registry: List[AstroCellSlot] = []
    for cell_id in sorted(bbox_map.keys()):           # stable sort = epoch-coherent
        b = bbox_map[cell_id]
        z_layer = b.get("z", 3)

        neighbors = [c for c in z_buckets.get(z_layer, []) if c != cell_id]

        out_edges = outgoing[cell_id]
        in_edges  = incoming[cell_id]

        # constraint_mask: low 8 bits = outgoing-edge count (clamped), same
        # semantic as ConstraintMask in C++ — a cheap per-slot priority hint.
        constraint_mask = min(len(out_edges), 0xFF)

        slot = AstroCellSlot(
            cell_id=cell_id,
            z_layer=z_layer,
            constraint_mask=constraint_mask,
            active=True,
            bbox=b,
            species=species_map.get(cell_id, ""),
            constraints={
                "incoming":  in_edges,
                "outgoing":  out_edges,
                "neighbors": neighbors,
            },
        )
        registry.append(slot)
        print(
            f"[ASTRO-CELL-REGISTRY] Registered cell id={cell_id} "
            f"z-layer={z_layer} constraint_mask=0x{constraint_mask:02X} "
            f"out={len(out_edges)} in={len(in_edges)}"
        )

    return registry


# Species that carry partial transparency / overlay semantics —
# analogous to EMeshPass::TranslucencyAll in the C++ sort.
# Cells in this set are "translucent passes" and render after opaque ones
# within the same z-layer (改 20%: driven by species family, not pass index).
_TRANSLUCENT_SPECIES = {"cil-eye", "cil-plus"}


def assemble_final_svg():
    """
    Assemble all cell SVGs into final.svg.

    Algorithm (ported from SceneRendering.cpp commit 7eae73c):
      1. FAstroCellRegistry  — build slot registry from bbox + topology + species.
      2. FConstraintCollector— collect & pack constraints; derive outgoing_edge_count.
      3. Z-layer sort        — opaque cells precede translucent cells within a layer;
                               among same-opacity cells, higher outgoing-edge count
                               renders first (constraint priority, mirrors the C++
                               SortedPassIndices.StableSort weighted by ViewZLayer).
    """
    svg_files  = glob.glob(os.path.join(CHANNELS, "cell", "*", "svg.svg"))
    bbox_files = glob.glob(os.path.join(CHANNELS, "cell", "*", "bbox.json"))

    # ── Load bbox map ──────────────────────────────────────────────────────────
    bbox_map: Dict[str, Dict] = {}
    for bf in bbox_files:
        cell_id = os.path.basename(os.path.dirname(bf))
        with open(bf) as f:
            bbox_map[cell_id] = json.load(f)

    # ── Load topology edges ────────────────────────────────────────────────────
    topology_path = os.path.join(CHANNELS, "skeleton", "topology.json")
    topology_edges: List[Dict] = []
    if os.path.exists(topology_path):
        with open(topology_path) as f:
            topology_edges = json.load(f).get("edges", [])

    # ── Load species map ───────────────────────────────────────────────────────
    species_path = os.path.join(CHANNELS, "physics", "species_assignment.json")
    species_map: Dict[str, str] = {}
    if os.path.exists(species_path):
        with open(species_path) as f:
            raw = json.load(f)
            species_map = {cid: v.get("species", "") for cid, v in raw.items()}
    # Fallback: read species directly from bbox.json
    for cid, b in bbox_map.items():
        if cid not in species_map and "species" in b:
            species_map[cid] = b["species"]

    # ── [ASTRO-CELL-REGISTRY] Populate registry ────────────────────────────────
    registry = _build_cell_registry(bbox_map, topology_edges, species_map)

    # ── [ASTRO-CONSTRAINT] Collect constraints ─────────────────────────────────
    collector = AstroConstraintCollector(registry)
    collector.collect()
    print(
        f"[ASTRO-CELL-REGISTRY] Registry init complete: {len(registry)} cells, "
        f"{collector.active_cell_count()} active constraints"
    )

    # ── Load SVG fragments ─────────────────────────────────────────────────────
    svg_map: Dict[str, str] = {}
    for sf in svg_files:
        cell_id = os.path.basename(os.path.dirname(sf))
        with open(sf) as f:
            svg_map[cell_id] = f.read()

    # ── [ASTRO-ZLAYER-SORT] Build sort key and order fragments ─────────────────
    #
    # Mirrors SetupMeshPass SortedPassIndices.StableSort (commit 7eae73c):
    #   • "Opaque" cells (non-translucent species) sort before translucent ones.
    #   • Within the same opacity class, cells with more outgoing edges (higher
    #     constraint priority) render first — they establish the visual substrate
    #     that later cells layer on top of.
    #   • Final tie-break: ascending z_layer, then cell_id (stable/deterministic).
    #
    # key tuple: (z_layer, is_translucent, -outgoing_edge_count, cell_id)
    def _sort_key(slot: AstroCellSlot) -> Tuple:
        is_translucent = int(slot.species in _TRANSLUCENT_SPECIES)
        out_priority   = -collector.outgoing_edge_count(slot.cell_id)  # negate → descending
        return (slot.z_layer, is_translucent, out_priority, slot.cell_id)

    sorted_slots = sorted(
        [s for s in registry if s.active and s.cell_id in svg_map],
        key=_sort_key,
    )

    print(
        f"[ASTRO-ZLAYER-SORT] Dispatching {len(sorted_slots)} cells in "
        f"z-layer + constraint-priority order"
    )
    for s in sorted_slots:
        translucent_tag = "translucent" if s.species in _TRANSLUCENT_SPECIES else "opaque"
        print(
            f"  cell={s.cell_id:15s} z={s.z_layer} {translucent_tag:11s} "
            f"out_edges={collector.outgoing_edge_count(s.cell_id)}"
        )

    # ── Compute canvas bounds ──────────────────────────────────────────────────
    max_x, max_y = 0, 0
    for b in bbox_map.values():
        max_x = max(max_x, b["x"] + b["w"])
        max_y = max(max_y, b["y"] + b["h"])
    width  = max_x + 60
    height = max_y + 60

    # ── Group sorted slots by z-layer (preserve intra-layer order) ────────────
    z_groups: Dict[int, List[AstroCellSlot]] = {}
    for slot in sorted_slots:
        z_groups.setdefault(slot.z_layer, []).append(slot)

    # ── Defs ───────────────────────────────────────────────────────────────────
    defs = '''  <defs>
    <marker id="arrow-green" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#2E7D32"/>
    </marker>
  </defs>'''

    # ── Build SVG ──────────────────────────────────────────────────────────────
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" style="max-width:100%;height:auto;">',
        f'  <rect width="{width}" height="{height}" fill="#FAFAFA" rx="4"/>',
        defs,
    ]

    for z in sorted(z_groups.keys()):
        lines.append(f'  <g id="z{z}-layer">')
        for slot in z_groups[z]:
            out_cnt = collector.outgoing_edge_count(slot.cell_id)
            lines.append(
                f'    <!-- cell: {slot.cell_id} species={slot.species} '
                f'constraint_mask=0x{slot.constraint_mask:02X} out_edges={out_cnt} -->'
            )
            lines.append(f'    {svg_map[slot.cell_id]}')
        lines.append(f'  </g>')

    lines.append('</svg>')

    final_svg  = "\n".join(lines)
    output_path = os.path.join(CHANNELS, "..", "output_cell_loop.svg")
    with open(output_path, "w") as f:
        f.write(final_svg)

    print(
        f"[Assemble] final SVG: {len(sorted_slots)} cells, "
        f"{len(z_groups)} z-layers, {width}x{height}"
    )
    return output_path


def run_loop(max_epochs=5):
    """Main pub/sub loop."""
    print("=" * 60)
    print("astro-svgfigure Cell Pub/Sub Loop")
    print("=" * 60)

    for epoch in range(max_epochs):
        print(f"\n--- Epoch {epoch} ---")
        write_channel("skeleton/epoch.json", {
            "current": epoch, "max": max_epochs, "status": "running"
        })

        # 1. All cells develop (in production: parallel sub-Claude dispatch)
        cells = run_all_cells()

        # 2. Physics engine computes forces
        collisions = physics_engine()

        # 3. Convergence check
        if convergence_judge():
            print(f"\n✓ Converged at epoch {epoch}!")
            break
    else:
        print(f"\n⚠ Max epochs ({max_epochs}) reached without full convergence")

    # 4. Assemble final SVG
    write_channel("skeleton/epoch.json", {
        "current": epoch, "max": max_epochs, "status": "converged"
    })
    output = assemble_final_svg()
    print(f"\n{'=' * 60}")
    print(f"Output: {output}")
    return output


if __name__ == "__main__":
    os.chdir(CHANNELS)
    run_loop()
