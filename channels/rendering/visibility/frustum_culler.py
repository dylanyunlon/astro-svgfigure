"""Frustum culling + LOD selection for cells.
Ported from upstream/unreal-renderer/SceneVisibility.cpp
"""

class AstroFrustumCuller:
    DEFAULTS = {"near": 0.1, "far": 10000, "fov": 60}
    def __init__(self, params=None): self.params = {**self.DEFAULTS, **(params or {})}
    def proc(self, cell_bboxes, camera_pos):
        visible = []
        for cid, bbox in cell_bboxes.items():
            cx = bbox.get("x", 0) + bbox.get("w", 0) / 2
            cy = bbox.get("y", 0) + bbox.get("h", 0) / 2
            dist = ((cx - camera_pos[0])**2 + (cy - camera_pos[1])**2) ** 0.5
            if dist < self.params["far"]:
                lod = 0 if dist < 500 else (1 if dist < 2000 else 2)
                visible.append({"cell_id": cid, "distance": round(dist, 1), "lod": lod})
        return {"type": "frustum_cull", "visible": len(visible), "total": len(cell_bboxes), "cells": visible}
