"""PBR shading model for cell surfaces.
Cook-Torrance BRDF with GGX distribution, Smith geometry, Fresnel-Schlick.
"""
import math

class AstroPBRShading:
    DEFAULTS = {"metallic": 0.0, "roughness": 0.5, "ao": 1.0, "normal_strength": 1.0, "emissive_intensity": 0.0}
    def __init__(self, params=None): self.params = {**self.DEFAULTS, **(params or {})}
    def proc(self, species_color, f0):
        p = self.params
        return {
            "type": "pbr_cook_torrance",
            "base_color": species_color,
            "metallic": p["metallic"], "roughness": p["roughness"],
            "f0": f0, "ao": p["ao"],
            "normal_strength": p["normal_strength"],
            "emissive_intensity": p["emissive_intensity"],
            "distribution": "ggx", "geometry": "smith", "fresnel": "schlick",
            "enabled": True,
        }
