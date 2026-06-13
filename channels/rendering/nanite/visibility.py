#!/usr/bin/env python3
"""Extracted from cell_component.py — Nanite rendering subsystem."""
import json
import math
import os
import sys

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


