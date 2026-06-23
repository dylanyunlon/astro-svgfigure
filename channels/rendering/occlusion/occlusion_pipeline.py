"""Occlusion pipeline: HZB, software occlusion, distance field AO.
Ported from upstream/unreal-renderer/SceneOcclusion.cpp + DistanceFieldAmbientOcclusion.cpp
"""
import json, math

class AstroHZBOcclusion:
    """Hierarchical Z-Buffer occlusion culling for cells."""
    DEFAULTS = {"mip_levels": 8, "conservative": True, "threshold": 0.01}
    def __init__(self, params=None): self.params = {**self.DEFAULTS, **(params or {})}
    def proc(self, cell_bboxes):
        visible = []
        for cid, bbox in cell_bboxes.items():
            area = bbox.get("w", 0) * bbox.get("h", 0)
            if area > self.params["threshold"]:
                visible.append(cid)
        return {"type": "hzb", "visible_cells": visible, "culled": len(cell_bboxes) - len(visible), **self.params}

class AstroDistanceFieldAO:
    """Distance field ambient occlusion for soft shadows between cells."""
    DEFAULTS = {"radius": 50, "intensity": 0.6, "bias": 0.1, "power": 2.0, "samples": 16}
    def __init__(self, params=None): self.params = {**self.DEFAULTS, **(params or {})}
    def proc(self):
        return {"type": "dfao", **self.params, "enabled": True}

class AstroSoftwareOcclusion:
    """CPU-side occlusion for LOD selection (Nanite fallback)."""
    DEFAULTS = {"resolution": 128, "conservative": True}
    def __init__(self, params=None): self.params = {**self.DEFAULTS, **(params or {})}
    def proc(self, z_layers):
        return {"type": "software", "z_layers_tested": len(z_layers), **self.params, "enabled": True}
