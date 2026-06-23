import os, sys, json, math
from typing import Any, Optional

def _dbg(tag, msg):
    if os.environ.get(f"ASTRO_{tag.replace('-','_')}_VERBOSE", "0") == "1":
        print(f"[{tag}] {msg}", file=sys.stderr)






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









# [ASTRO-VISIBILITY] ─────────────────────────────────────────────────────────
# HierarchicalZBuffer + PortalVisibility + PVS
# Mirrors UE5 HZB occlusion culling, portal-graph visibility, and
# Quake3-era PVS precomputation — adapted for 2-D cell-based scene graph.
# ─────────────────────────────────────────────────────────────────────────────

import bisect
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NANITE_VIS_GUARD_BAND: float = 50.0          # px — frustum expansion
_NANITE_CULL_THRESHOLD: float = 0.0005        # < 0.05 % viewport → culled
_NANITE_LOD2_THRESHOLD: float = 0.002         # < 0.2 % viewport → rect-only

_HZB_MIP_LEVELS: int = 8                      # log2(256) — max pyramid depth
_HZB_TILE_SIZE: int = 4                        # px per HZB leaf tile
_PVS_CLUSTER_MAX: int = 64                     # max cells per PVS cluster
_PORTAL_MAX_DEPTH: int = 16                    # max portal-chain recursion


# ---------------------------------------------------------------------------
# § 1  HierarchicalZBuffer — layered occlusion reject
# ---------------------------------------------------------------------------
# Mirrors FHZBOcclusionTester / BuildHZB in UE5 SceneRendering.
# For the 2-D cell scene we store *minimum z-layer* per tile rather than
# max GPU depth, because cells with a *higher* z-layer are rendered on top
# and should occlude those below them.
# ---------------------------------------------------------------------------

class HierarchicalZBuffer:
    """
    [ASTRO-VISIBILITY] Hierarchical Z-Buffer for fast occlusion rejection.

    Construction:
        Build the mip pyramid from a flat list of (cell_id, bbox, z_layer).
        Leaf tiles are _HZB_TILE_SIZE × _HZB_TILE_SIZE pixel buckets; each
        stores the *maximum* z-layer of any opaque cell that covers that tile
        (higher z = drawn on top = occluder).

    Test:
        To test whether a candidate cell is occluded, select the mip level
        whose tile(s) minimally cover the cell's bbox, then compare the cell's
        own z_layer against the stored maximum.  If z_layer < stored max the
        cell is fully behind an occluder → cull.

    Mirrors::
        UE5: FScene::BuildHZB → RHI UAV compute pass
        Here: pure-Python tile-grid updated via update() + queried via test()
    """

    def __init__(self, viewport_w: int, viewport_h: int):
        self.viewport_w = viewport_w
        self.viewport_h = viewport_h

        # Pyramid of tile grids.  mip[0] = finest (1 tile per _HZB_TILE_SIZE px).
        # Each grid is a 2-D list of floats (max z_layer seen so far).
        self.mips: List[List[List[float]]] = []
        self._build_empty_pyramid()

        _dbg("ASTRO-VISIBILITY",
             f"HZB init vp=({viewport_w},{viewport_h}) "
             f"mips={len(self.mips)} tile={_HZB_TILE_SIZE}px")

    # ── internal ─────────────────────────────────────────────────────────────

    def _build_empty_pyramid(self) -> None:
        """Allocate mip chain; all tiles initialised to -∞ (no occluder)."""
        tile_w = math.ceil(self.viewport_w / _HZB_TILE_SIZE)
        tile_h = math.ceil(self.viewport_h / _HZB_TILE_SIZE)
        for level in range(_HZB_MIP_LEVELS):
            w = max(1, tile_w >> level)
            h = max(1, tile_h >> level)
            self.mips.append([[float("-inf")] * w for _ in range(h)])

    def _tile_coords(self, x: float, y: float, mip: int) -> Tuple[int, int]:
        scale = _HZB_TILE_SIZE * (1 << mip)
        return int(x / scale), int(y / scale)

    def _bbox_mip_level(self, w: float, h: float) -> int:
        """
        Select mip level such that the bbox spans ≈ 2×2 tiles.
        Mirrors: ComputeOcclusionMeshLOD heuristic.
        """
        min_dim = max(min(w, h), 1.0)
        level = int(math.log2(min_dim / (_HZB_TILE_SIZE * 2.0)))
        return max(0, min(level, _HZB_MIP_LEVELS - 1))

    # ── public API ───────────────────────────────────────────────────────────

    def update(self, cell_id: str, bbox: dict, z_layer: float) -> None:
        """
        [ASTRO-VISIBILITY] Mark occluder tiles in mip 0, propagate upward.

        Call this for every opaque cell after sort-by-z (back-to-front) so
        that the maximum z_layer per tile is correct when test() is called.
        """
        x0 = float(bbox.get("x", 0))
        y0 = float(bbox.get("y", 0))
        w  = float(bbox.get("w", 1))
        h  = float(bbox.get("h", 1))
        x1, y1 = x0 + w, y0 + h

        # Write into mip 0
        grid = self.mips[0]
        tw = len(grid[0]) if grid else 1
        th = len(grid)
        tx0 = max(0, int(x0 / _HZB_TILE_SIZE))
        ty0 = max(0, int(y0 / _HZB_TILE_SIZE))
        tx1 = min(tw - 1, int(x1 / _HZB_TILE_SIZE))
        ty1 = min(th - 1, int(y1 / _HZB_TILE_SIZE))

        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                if grid[ty][tx] < z_layer:
                    grid[ty][tx] = z_layer

        # Propagate: each parent tile = max of its 2×2 children
        for level in range(1, _HZB_MIP_LEVELS):
            prev = self.mips[level - 1]
            curr = self.mips[level]
            ptx0, pty0 = tx0 >> 1, ty0 >> 1
            ptx1, pty1 = tx1 >> 1, ty1 >> 1
            pw = len(curr[0]) if curr else 1
            ph = len(curr)
            for ry in range(max(0, pty0), min(ph - 1, pty1) + 1):
                for rx in range(max(0, ptx0), min(pw - 1, ptx1) + 1):
                    cy0_ = ry * 2
                    cx0_ = rx * 2
                    best = float("-inf")
                    for dy in range(2):
                        row = cy0_ + dy
                        if row < len(prev):
                            for dx in range(2):
                                col = cx0_ + dx
                                if col < len(prev[0]):
                                    best = max(best, prev[row][col])
                    curr[ry][rx] = best
            tx0, ty0, tx1, ty1 = ptx0, pty0, ptx1, pty1

    def test(self, bbox: dict, z_layer: float) -> bool:
        """
        [ASTRO-VISIBILITY] Return True if the cell passes HZB (not occluded).

        A cell is occluded when ALL tiles at the chosen mip level have a
        stored z_layer *greater* than the candidate's own z_layer.
        """
        x0 = float(bbox.get("x", 0))
        y0 = float(bbox.get("y", 0))
        w  = float(bbox.get("w", 1))
        h  = float(bbox.get("h", 1))
        x1, y1 = x0 + w, y0 + h

        mip = self._bbox_mip_level(w, h)
        grid = self.mips[mip]
        if not grid or not grid[0]:
            return True  # no HZB data → conservatively visible

        scale = _HZB_TILE_SIZE * (1 << mip)
        tw = len(grid[0])
        th = len(grid)
        tx0 = max(0, int(x0 / scale))
        ty0 = max(0, int(y0 / scale))
        tx1 = min(tw - 1, int(x1 / scale))
        ty1 = min(th - 1, int(y1 / scale))

        # If any tile has max_z <= candidate z → candidate might be visible
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                if grid[ty][tx] <= z_layer:
                    return True  # not fully occluded

        _dbg("ASTRO-VISIBILITY",
             f"HZB cull z={z_layer} bbox=({x0:.0f},{y0:.0f},{w:.0f},{h:.0f}) "
             f"mip={mip}")
        return False  # every tile shows a closer occluder

    def reset(self) -> None:
        """Clear pyramid for next frame."""
        for level, grid in enumerate(self.mips):
            for row in grid:
                for i in range(len(row)):
                    row[i] = float("-inf")


# ---------------------------------------------------------------------------
# § 2  PortalVisibility — sector/portal graph traversal
# ---------------------------------------------------------------------------
# Mirrors UE4 FPrecomputedVisibilityHandler::GetPrecomputedVisibility and
# the portal-based PVS used in classic BSP renderers (Quake, Half-Life).
#
# The scene is partitioned into *sectors* (named groups of cells).  Sectors
# are connected by *portals* (rectangular openings in shared walls/borders).
# Starting from the sector containing the camera, we recursively trace
# portal-chains to collect all potentially visible sectors.
# ---------------------------------------------------------------------------

@dataclass
class Portal:
    """
    [ASTRO-VISIBILITY] Directed opening between two sectors.

    Fields mirror FPortal in static lighting / visibility computations:
        sector_a, sector_b  — connected sector names
        rect                — bounding rect of the opening {x,y,w,h}
        two_way             — if False, only visible a→b (one-sided glass)
    """
    portal_id: str
    sector_a: str
    sector_b: str
    rect: dict                    # {x, y, w, h}
    two_way: bool = True


@dataclass
class Sector:
    """
    [ASTRO-VISIBILITY] Named group of cells sharing a convex sub-space.
    """
    sector_id: str
    cell_ids: List[str] = field(default_factory=list)
    bounds: dict = field(default_factory=lambda: {"x": 0, "y": 0, "w": 0, "h": 0})


class PortalVisibility:
    """
    [ASTRO-VISIBILITY] Portal-graph visibility determination.

    Usage::

        pv = PortalVisibility()
        pv.add_sector(Sector("hall", ["cell_0", "cell_1"], bounds=...))
        pv.add_sector(Sector("room_a", ["cell_2"], bounds=...))
        pv.add_portal(Portal("p0", "hall", "room_a", rect=...))
        visible_sectors = pv.query_visible_sectors("hall")
        visible_cells   = pv.visible_cells("hall")

    Algorithm:
        BFS from camera sector through portals, bounded by _PORTAL_MAX_DEPTH.
        Each portal is clipped against the current *view frustum clip rect*
        (initially the full viewport) to avoid light leaking around corners.
        Mirrors: EvalPortals() in BSP renderers / UE4 portal precompute.
    """

    def __init__(self):
        self._sectors: Dict[str, Sector] = {}
        self._portals: List[Portal] = []
        # adjacency: sector_id → list of (portal, neighbour_sector_id)
        self._adj: Dict[str, List[Tuple[Portal, str]]] = {}

    # ── construction ─────────────────────────────────────────────────────────

    def add_sector(self, sector: Sector) -> None:
        """[ASTRO-VISIBILITY] Register a sector into the portal graph."""
        self._sectors[sector.sector_id] = sector
        self._adj.setdefault(sector.sector_id, [])

    def add_portal(self, portal: Portal) -> None:
        """[ASTRO-VISIBILITY] Register a portal (edge in the graph)."""
        self._portals.append(portal)
        self._adj.setdefault(portal.sector_a, []).append((portal, portal.sector_b))
        if portal.two_way:
            self._adj.setdefault(portal.sector_b, []).append((portal, portal.sector_a))

    # ── internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _rect_intersect(a: dict, b: dict) -> Optional[dict]:
        """Clip rect *a* against rect *b*; return intersection or None."""
        ax0, ay0 = a["x"], a["y"]
        ax1, ay1 = ax0 + a["w"], ay0 + a["h"]
        bx0, by0 = b["x"], b["y"]
        bx1, by1 = bx0 + b["w"], by0 + b["h"]
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        if ix0 >= ix1 or iy0 >= iy1:
            return None
        return {"x": ix0, "y": iy0, "w": ix1 - ix0, "h": iy1 - iy0}

    # ── public API ───────────────────────────────────────────────────────────

    def query_visible_sectors(
        self,
        camera_sector: str,
        clip_rect: Optional[dict] = None,
    ) -> Set[str]:
        """
        [ASTRO-VISIBILITY] BFS portal traversal; returns set of visible sector IDs.

        *clip_rect* is the view frustum window (defaults to ±∞).
        """
        if camera_sector not in self._sectors:
            return set()

        visible: Set[str] = {camera_sector}
        # queue entries: (sector_id, frustum_clip_rect, depth)
        default_clip = clip_rect or {"x": -1e9, "y": -1e9, "w": 2e9, "h": 2e9}
        queue = [(camera_sector, default_clip, 0)]
        idx = 0

        while idx < len(queue):
            cur_sector, cur_clip, depth = queue[idx]
            idx += 1

            if depth >= _PORTAL_MAX_DEPTH:
                continue

            for portal, neighbour in self._adj.get(cur_sector, []):
                if neighbour in visible:
                    continue
                # Clip portal rect against current view frustum
                portal_clipped = self._rect_intersect(portal.rect, cur_clip)
                if portal_clipped is None:
                    continue   # portal not visible through current frustum
                visible.add(neighbour)
                queue.append((neighbour, portal_clipped, depth + 1))

        _dbg("ASTRO-VISIBILITY",
             f"Portal BFS from '{camera_sector}' → {len(visible)} sectors visible")
        return visible

    def visible_cells(
        self,
        camera_sector: str,
        clip_rect: Optional[dict] = None,
    ) -> List[str]:
        """
        [ASTRO-VISIBILITY] Convenience wrapper: returns flat list of visible cell IDs.
        """
        visible_sectors = self.query_visible_sectors(camera_sector, clip_rect)
        cells: List[str] = []
        for sid in visible_sectors:
            cells.extend(self._sectors[sid].cell_ids)
        return cells


# ---------------------------------------------------------------------------
# § 3  PVS — Potentially Visible Set (precomputed)
# ---------------------------------------------------------------------------
# Mirrors Quake/BSP PVS + UE4 FPrecomputedVisibilityHandler.
# Cells are grouped into *clusters* of ≤ _PVS_CLUSTER_MAX members.
# For each cluster we precompute the set of clusters visible from any point
# within it (offline), then at runtime we just look up the camera cluster.
#
# Precomputation strategy (simplified from exact portal visibility):
#   For each cluster C:
#       Start with C itself in PVS.
#       For each other cluster D ≠ C:
#           If any cell in C has unobstructed line-of-sight to any cell in D
#           (tested via axis-aligned AABB gap check), add D to PVS[C].
# ---------------------------------------------------------------------------

@dataclass
class PVSCluster:
    """[ASTRO-VISIBILITY] A spatial cluster of cells for PVS precomputation."""
    cluster_id: int
    cell_ids: List[str] = field(default_factory=list)
    # Bounding box of the whole cluster
    bounds: dict = field(default_factory=lambda: {"x": 0, "y": 0, "w": 0, "h": 0})


class PrecomputedVisibilitySet:
    """
    [ASTRO-VISIBILITY] Precomputed Visibility Set manager.

    Offline phase (build):
        Cluster cells spatially, then test pairwise AABB line-of-sight to
        build a visibility bitset for each cluster.

    Runtime phase (query):
        Given a camera position, find its cluster and look up the PVS.
        Returns the list of visible cell IDs in O(clusters) time.

    Mirrors FPrecomputedVisibilityHandler and Quake vis compiler.
    """

    def __init__(self):
        self._clusters: List[PVSCluster] = []
        # pvs[i] = set of cluster indices visible from cluster i
        self._pvs: Dict[int, Set[int]] = {}
        self._built = False

    # ── construction ─────────────────────────────────────────────────────────

    def add_cluster(self, cluster: PVSCluster) -> None:
        """[ASTRO-VISIBILITY] Register a cluster before calling build()."""
        self._clusters.append(cluster)

    def build(self, occluder_bboxes: Optional[List[dict]] = None) -> None:
        """
        [ASTRO-VISIBILITY] Precompute PVS for all registered clusters.

        For each pair (A, B) we test whether the gap between their bounding
        boxes is unobstructed by any occluder_bbox.  The occluder list is
        optional; without it visibility degrades to pure frustum overlap test.

        Time complexity: O(C² × O) where C = #clusters, O = #occluders.
        Acceptable offline; not suitable for per-frame updates.
        """
        n = len(self._clusters)
        for i in range(n):
            visible: Set[int] = {i}  # cluster always sees itself
            ci = self._clusters[i]
            for j in range(n):
                if i == j:
                    continue
                cj = self._clusters[j]
                if self._clusters_have_los(ci.bounds, cj.bounds, occluder_bboxes or []):
                    visible.add(j)
            self._pvs[i] = visible

        self._built = True
        _dbg("ASTRO-VISIBILITY",
             f"PVS built: {n} clusters, avg_vis="
             f"{sum(len(v) for v in self._pvs.values()) / max(n, 1):.1f}")

    @staticmethod
    def _clusters_have_los(
        a: dict, b: dict, occluders: List[dict]
    ) -> bool:
        """
        [ASTRO-VISIBILITY] Simplified LOS: check AABB-to-AABB visibility gap.

        We test whether the *gap rect* between A and B is free of occluders.
        This is a coarse conservative test — it can give false positives (mark
        invisible clusters as visible) but never false negatives.
        """
        # Gap rect = bounding box enclosing both clusters
        ax0, ay0 = a["x"], a["y"]
        ax1, ay1 = ax0 + a["w"], ay0 + a["h"]
        bx0, by0 = b["x"], b["y"]
        bx1, by1 = bx0 + b["w"], by0 + b["h"]

        gap = {
            "x": min(ax0, bx0),
            "y": min(ay0, by0),
            "w": max(ax1, bx1) - min(ax0, bx0),
            "h": max(ay1, by1) - min(ay0, by0),
        }

        for occ in occluders:
            ox0, oy0 = occ.get("x", 0), occ.get("y", 0)
            ox1 = ox0 + occ.get("w", 0)
            oy1 = oy0 + occ.get("h", 0)
            gx0, gy0 = gap["x"], gap["y"]
            gx1 = gx0 + gap["w"]
            gy1 = gy0 + gap["h"]
            # If occluder fully covers the gap → no LOS
            if ox0 <= gx0 and ox1 >= gx1 and oy0 <= gy0 and oy1 >= gy1:
                return False
        return True

    # ── runtime ──────────────────────────────────────────────────────────────

    def find_cluster(self, px: float, py: float) -> Optional[int]:
        """
        [ASTRO-VISIBILITY] Return index of the cluster containing point (px, py).
        Returns None if no cluster covers the point.
        """
        for i, c in enumerate(self._clusters):
            x0, y0 = c.bounds["x"], c.bounds["y"]
            x1 = x0 + c.bounds["w"]
            y1 = y0 + c.bounds["h"]
            if x0 <= px <= x1 and y0 <= py <= y1:
                return i
        return None

    def query(self, camera_x: float, camera_y: float) -> List[str]:
        """
        [ASTRO-VISIBILITY] Runtime PVS lookup; returns list of visible cell IDs.

        O(V × M) where V = visible clusters, M = cells per cluster.
        """
        if not self._built:
            raise RuntimeError("[ASTRO-VISIBILITY] PVS not built; call build() first")

        cam_cluster = self.find_cluster(camera_x, camera_y)
        if cam_cluster is None:
            # Camera outside all clusters → return everything (safe fallback)
            return [cid for c in self._clusters for cid in c.cell_ids]

        visible_clusters = self._pvs.get(cam_cluster, {cam_cluster})
        cells: List[str] = []
        for ci in visible_clusters:
            cells.extend(self._clusters[ci].cell_ids)

        _dbg("ASTRO-VISIBILITY",
             f"PVS query cam=({camera_x:.0f},{camera_y:.0f}) "
             f"cluster={cam_cluster} → {len(visible_clusters)} clusters "
             f"{len(cells)} cells")
        return cells


# ---------------------------------------------------------------------------
# § 4  Integrated visibility pass
# ---------------------------------------------------------------------------

def perform_full_visibility(
    cells: List[dict],
    viewport_w: int,
    viewport_h: int,
    scroll_x: float = 0.0,
    scroll_y: float = 0.0,
    hzb: Optional["HierarchicalZBuffer"] = None,
    portal_vis: Optional["PortalVisibility"] = None,
    pvs: Optional["PrecomputedVisibilitySet"] = None,
    camera_sector: Optional[str] = None,
) -> List[dict]:
    """
    [ASTRO-VISIBILITY] Unified visibility pass combining HZB + Portal + PVS.

    Each *cell* dict must have at minimum::
        cell_id: str
        bbox: {x, y, w, h, z}    (z is z-layer integer)
        species: str
        sector: str               (optional, for portal vis)

    Returns the subset of cells that pass ALL active visibility tests,
    annotated with an ``astro_vis`` sub-dict.

    Pipeline (mirrors UE5 visibility chain):
        1. Frustum cull  (AstroCellVisibilityQuery._intersect_box)
        2. PVS lookup    (skip cells outside camera's PVS clusters)
        3. Portal filter (skip cells in sectors not reachable via portals)
        4. HZB test      (skip cells fully behind closer occluders)
    """
    query = AstroCellVisibilityQuery(
        viewport_w, viewport_h, scroll_x=scroll_x, scroll_y=scroll_y
    )

    # PVS: collect allowed cell_ids
    pvs_allowed: Optional[Set[str]] = None
    if pvs is not None:
        cam_cx = scroll_x + viewport_w / 2.0
        cam_cy = scroll_y + viewport_h / 2.0
        pvs_allowed = set(pvs.query(cam_cx, cam_cy))

    # Portal: collect allowed cell_ids
    portal_allowed: Optional[Set[str]] = None
    if portal_vis is not None and camera_sector is not None:
        portal_allowed = set(portal_vis.visible_cells(camera_sector))

    # Sort back-to-front so HZB update order is correct
    sorted_cells = sorted(cells, key=lambda c: c.get("bbox", {}).get("z", 0))

    visible_cells: List[dict] = []
    for cell in sorted_cells:
        cid = cell["cell_id"]
        bbox = cell.get("bbox", {})
        z_layer = int(bbox.get("z", 0))
        species = cell.get("species", "default")

        # 1. Frustum
        result = query.test_cell(cid, bbox, species, z_layer)
        if not result["visible"]:
            continue

        # 2. PVS filter
        if pvs_allowed is not None and cid not in pvs_allowed:
            _dbg("ASTRO-VISIBILITY", f"PVS reject cell={cid}")
            continue

        # 3. Portal filter
        if portal_allowed is not None and cid not in portal_allowed:
            _dbg("ASTRO-VISIBILITY", f"Portal reject cell={cid}")
            continue

        # 4. HZB test
        if hzb is not None and not hzb.test(bbox, z_layer):
            _dbg("ASTRO-VISIBILITY", f"HZB reject cell={cid}")
            continue

        # Cell passes — update HZB with this occluder (front-to-back order
        # means we already sorted; now build pyramid as cells become visible)
        if hzb is not None:
            hzb.update(cid, bbox, z_layer)

        cell["astro_vis"] = result
        visible_cells.append(cell)

    summary = query.finish()
    _dbg("ASTRO-VISIBILITY",
         f"Full vis pass: {len(visible_cells)}/{len(cells)} cells passed")
    return visible_cells
