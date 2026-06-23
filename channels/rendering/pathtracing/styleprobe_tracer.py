"""Style probe path tracer: sample neighbor cells for color diffusion.
Conceptual mapping of UE5 path tracing to species color bleeding.
"""

class AstroStyleProbeTracer:
    DEFAULTS = {"max_bounces": 3, "samples_per_cell": 8, "diffusion_radius": 100, "decay": 0.7}
    def __init__(self, params=None): self.params = {**self.DEFAULTS, **(params or {})}
    def proc(self, cell_colors):
        return {"type": "styleprobe_trace", "cell_count": len(cell_colors),
                "max_bounces": self.params["max_bounces"],
                "samples": self.params["samples_per_cell"],
                "radius": self.params["diffusion_radius"],
                "decay": self.params["decay"], "enabled": True}
