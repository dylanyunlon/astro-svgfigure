"""Reflection pipeline: SSR probe, planar reflection, cubemap capture.
Ported from upstream/unreal-renderer/ReflectionEnvironment*.cpp
"""
import json, math

class AstroSSRProbe:
    """Screen-space reflections via ray march in depth buffer."""
    DEFAULTS = {"max_steps": 32, "thickness": 0.5, "stride": 2, "jitter": 0.5, "fade_end": 0.95, "roughness_cutoff": 0.6}
    def __init__(self, params=None): self.params = {**self.DEFAULTS, **(params or {})}
    def proc(self):
        return {"type": "ssr", **self.params, "enabled": True}

class AstroPlanarReflection:
    """Planar reflection for flat surfaces (cell backgrounds)."""
    DEFAULTS = {"clip_plane_offset": 0.01, "distortion": 0.02, "opacity": 0.3, "blur": 2.0}
    def __init__(self, params=None): self.params = {**self.DEFAULTS, **(params or {})}
    def proc(self, surface_normal=[0, 1, 0]):
        return {"type": "planar", "surface_normal": surface_normal, **self.params, "enabled": True}

class AstroCubemapCapture:
    """Environment cubemap capture for IBL reflections."""
    DEFAULTS = {"resolution": 256, "mip_levels": 7, "update_interval": 10, "blend": 0.5}
    def __init__(self, params=None): self.params = {**self.DEFAULTS, **(params or {})}
    def proc(self, capture_pos=[0, 0, 0]):
        return {"type": "cubemap", "capture_position": capture_pos, **self.params, "enabled": True}
