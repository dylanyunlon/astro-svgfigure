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








