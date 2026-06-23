"""Translucency sorting for overlapping cells and skip edges.
Ported from upstream/unreal-renderer/TranslucentRendering.cpp
"""

class AstroTranslucencySort:
    DEFAULTS = {"sort_mode": "per_pixel", "max_layers": 8, "oit_method": "weighted_blended"}
    def __init__(self, params=None): self.params = {**self.DEFAULTS, **(params or {})}
    def proc(self, translucent_items):
        sorted_items = sorted(translucent_items, key=lambda x: x.get("z", 0), reverse=True)
        return {"type": "translucency_sort", "method": self.params["oit_method"],
                "item_count": len(sorted_items), "max_layers": self.params["max_layers"], "enabled": True}
