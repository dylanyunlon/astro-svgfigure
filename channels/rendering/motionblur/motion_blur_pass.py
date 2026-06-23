"""Motion blur pass for epoch transitions.
Per-cell velocity buffer → directional blur.
"""

class AstroMotionBlurPass:
    DEFAULTS = {"intensity": 0.5, "samples": 8, "max_velocity": 20, "center_weight": 0.5}
    def __init__(self, params=None): self.params = {**self.DEFAULTS, **(params or {})}
    def proc(self, cell_velocities):
        return {
            "type": "motion_blur",
            "cell_velocities": {cid: round(v, 3) for cid, v in cell_velocities.items()},
            "max_velocity": self.params["max_velocity"],
            "samples": self.params["samples"],
            "intensity": self.params["intensity"],
            "enabled": True,
        }
