"""Effects pipeline: godray, caustics, heat distortion, vignette.
References: AT pixijs-filters, lygia generative/snoise, Finding-Love-Shaders.
"""
import json, math

class AstroGodrayEffect:
    """Screen-space god rays from dominant light source."""
    DEFAULTS = {"density": 0.5, "weight": 0.6, "decay": 0.93, "exposure": 0.3, "samples": 64}
    def __init__(self, params=None): self.params = {**self.DEFAULTS, **(params or {})}
    def proc(self, light_pos):
        return {"type": "godray", "light_screen_pos": light_pos, **self.params, "enabled": True}

class AstroCausticsEffect:
    """Water caustics via Voronoi distance field animation."""
    DEFAULTS = {"scale": 3.0, "speed": 0.4, "intensity": 0.3, "color": [0.4, 0.7, 1.0]}
    def __init__(self, params=None): self.params = {**self.DEFAULTS, **(params or {})}
    def proc(self):
        return {"type": "caustics", "shader": "lygia/generative/voronoise", **self.params, "enabled": True}

class AstroHeatDistortion:
    """Heat distortion via animated UV offset (simplex noise)."""
    DEFAULTS = {"strength": 0.01, "speed": 1.5, "frequency": 8.0, "falloff": 0.5}
    def __init__(self, params=None): self.params = {**self.DEFAULTS, **(params or {})}
    def proc(self):
        return {"type": "heat_distortion", "shader": "lygia/generative/snoise", **self.params, "enabled": True}

class AstroVignetteEffect:
    """Vignette darkening at screen edges."""
    DEFAULTS = {"intensity": 0.4, "smoothness": 0.5, "roundness": 1.0, "color": [0, 0, 0]}
    def __init__(self, params=None): self.params = {**self.DEFAULTS, **(params or {})}
    def proc(self):
        return {"type": "vignette", **self.params, "enabled": True}
