"""PostProcess bloom + tonemap pass.
Ported from upstream/unreal-renderer/PostProcess/ — ACES tonemap + Kawase bloom.
"""
import json, math

class AstroBloomTonemapPass:
    """Multi-pass bloom + ACES tonemap for cell rendering pipeline."""
    
    DEFAULTS = {
        "bloom_threshold": 0.8,
        "bloom_intensity": 0.6,
        "bloom_radius": 4,
        "bloom_passes": 5,
        "tonemap_method": "aces",
        "exposure": 1.0,
        "gamma": 2.2,
        "saturation": 1.05,
        "contrast": 1.1,
    }

    def __init__(self, params=None):
        self.params = {**self.DEFAULTS, **(params or {})}

    def proc(self, cells_composite):
        """Generate bloom + tonemap JSON params for the composite."""
        p = self.params
        bloom_kernels = []
        for i in range(p["bloom_passes"]):
            radius = p["bloom_radius"] * (2 ** i)
            weight = p["bloom_intensity"] / (1 + i * 0.5)
            bloom_kernels.append({"pass": i, "radius": radius, "weight": round(weight, 4)})

        return {
            "bloom": {
                "threshold": p["bloom_threshold"],
                "kernels": bloom_kernels,
                "mix_strength": p["bloom_intensity"],
            },
            "tonemap": {
                "method": p["tonemap_method"],
                "exposure": p["exposure"],
                "gamma": p["gamma"],
                "saturation": p["saturation"],
                "contrast": p["contrast"],
            },
            "enabled": True,
        }


class AstroDOFPass:
    """Depth of field — circle of confusion + bokeh simulation."""
    
    DEFAULTS = {
        "focus_distance": 500,
        "aperture": 2.8,
        "focal_length": 50,
        "bokeh_shape": "hexagonal",
        "max_blur": 8,
    }

    def __init__(self, params=None):
        self.params = {**self.DEFAULTS, **(params or {})}

    def proc(self, z_layers):
        p = self.params
        coc_scale = p["focal_length"] / (p["aperture"] * p["focus_distance"])
        return {
            "focus_distance": p["focus_distance"],
            "coc_scale": round(coc_scale, 6),
            "bokeh_shape": p["bokeh_shape"],
            "max_blur": p["max_blur"],
            "near_start": max(0, p["focus_distance"] - 100),
            "far_start": p["focus_distance"] + 100,
            "enabled": True,
        }
