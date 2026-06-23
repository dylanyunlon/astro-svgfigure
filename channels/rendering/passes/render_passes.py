"""Render pass definitions: base pass, lighting pass, translucency pass.
Ported from upstream/unreal-renderer/BasePassRendering.cpp + LightRendering.cpp
"""

class AstroBasePass:
    """GBuffer fill pass — writes albedo, normal, roughness, metallic."""
    def proc(self, cells):
        return {"type": "base_pass", "cell_count": len(cells), "output": ["albedo", "normal", "roughness", "metallic", "depth"], "enabled": True}

class AstroLightingPass:
    """Deferred lighting — evaluates all lights against GBuffer."""
    DEFAULTS = {"max_lights": 4, "shadow_quality": "medium", "ambient_color": [0.1, 0.1, 0.15]}
    def __init__(self, params=None): self.params = {**self.DEFAULTS, **(params or {})}
    def proc(self):
        return {"type": "lighting_pass", **self.params, "output": ["lit_color", "specular"], "enabled": True}

class AstroTranslucencyPass:
    """Forward translucency for skip edges and cell overlays."""
    DEFAULTS = {"sort_mode": "back_to_front", "max_layers": 4}
    def __init__(self, params=None): self.params = {**self.DEFAULTS, **(params or {})}
    def proc(self):
        return {"type": "translucency_pass", **self.params, "enabled": True}
