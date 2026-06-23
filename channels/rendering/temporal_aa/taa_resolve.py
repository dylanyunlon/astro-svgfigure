"""Temporal anti-aliasing resolve pass.
Ported from upstream/unreal-renderer-ue5/Renderer-Private/ TSR concepts.
"""

class AstroTAAResolve:
    DEFAULTS = {"jitter_sequence": "halton", "feedback_weight": 0.9, "velocity_rejection": 0.1, "sharpness": 0.5}
    def __init__(self, params=None): self.params = {**self.DEFAULTS, **(params or {})}
    def proc(self, frame_index):
        import math
        # Halton(2,3) jitter
        def halton(i, base):
            f, r = 1.0, 0.0
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r
        jx = halton(frame_index % 16, 2) - 0.5
        jy = halton(frame_index % 16, 3) - 0.5
        return {
            "type": "taa_resolve",
            "jitter": [round(jx, 4), round(jy, 4)],
            "feedback_weight": self.params["feedback_weight"],
            "velocity_rejection": self.params["velocity_rejection"],
            "sharpness": self.params["sharpness"],
            "enabled": True,
        }
