import os, sys, json, math
from typing import Any, Optional

def _dbg(tag, msg):
    if os.environ.get(f"ASTRO_{tag.replace('-','_')}_VERBOSE", "0") == "1":
        print(f"[{tag}] {msg}", file=sys.stderr)









def _colour_to_hex(rgb: tuple) -> str:
    """Convert (r, g, b) int tuple to #RRGGBB hex string."""
    return "#{:02X}{:02X}{:02X}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

















def _lerp_colour(c_own: tuple, c_target: tuple, t: float) -> tuple:
    """
    Linear interpolation between two (r,g,b) tuples.

    t=0 → c_own unchanged; t=1 → fully c_target.
    Mirrors FMath::Lerp(CubemapSample, PaletteAvg, t) from BlendWithCubemap.
    """
    t = max(0.0, min(1.0, t))
    return (
        c_own[0] + (c_target[0] - c_own[0]) * t,
        c_own[1] + (c_target[1] - c_own[1]) * t,
        c_own[2] + (c_target[2] - c_own[2]) * t,
    )








