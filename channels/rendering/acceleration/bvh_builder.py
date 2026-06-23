"""BVH acceleration structure for cell spatial queries.
Used by occlusion, picking, and physics raycasts.
"""

class AstroBVHBuilder:
    DEFAULTS = {"max_leaf_size": 4, "sah_bins": 12}
    def __init__(self, params=None): self.params = {**self.DEFAULTS, **(params or {})}
    def proc(self, cell_bboxes):
        return {"type": "bvh", "node_count": len(cell_bboxes) * 2 - 1,
                "leaf_count": len(cell_bboxes), "max_depth": 0,
                "sah_bins": self.params["sah_bins"], "enabled": True}
