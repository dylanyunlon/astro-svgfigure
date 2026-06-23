
# ─────────────────────────────────────────────────────────────────────────────
# M825: Color Grading — Tone mapping + Species harmony + LUT pipeline
# ─────────────────────────────────────────────────────────────────────────────

import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Academic journal color presets
# ─────────────────────────────────────────────────────────────────────────────

JOURNAL_LUTS = {
    "nature": {
        "description": "Nature — cool blues, muted warm accents",
        "shadows": (0.05, 0.08, 0.15),   # deep blue shadows
        "midtones": (0.45, 0.50, 0.55),  # neutral cool midtones
        "highlights": (0.95, 0.93, 0.90), # warm cream highlights
        "saturation": 0.75,
        "contrast": 1.1,
    },
    "science": {
        "description": "Science — high contrast, vivid primary colors",
        "shadows": (0.03, 0.03, 0.05),
        "midtones": (0.50, 0.50, 0.50),
        "highlights": (1.0, 1.0, 1.0),
        "saturation": 0.90,
        "contrast": 1.25,
    },
    "ieee": {
        "description": "IEEE — clean, bright, technical diagram style",
        "shadows": (0.10, 0.10, 0.12),
        "midtones": (0.55, 0.55, 0.58),
        "highlights": (0.98, 0.98, 1.0),
        "saturation": 0.65,
        "contrast": 1.0,
    },
    "arxiv_dark": {
        "description": "arXiv dark — dark background, neon accents",
        "shadows": (0.02, 0.02, 0.04),
        "midtones": (0.25, 0.27, 0.35),
        "highlights": (0.80, 0.85, 0.95),
        "saturation": 1.1,
        "contrast": 1.3,
    },
}


def hex_to_rgb(h: str) -> Tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def rgb_to_hsl(r: float, g: float, b: float) -> Tuple[float, float, float]:
    mx, mn = max(r, g, b), min(r, g, b)
    l = (mx + mn) / 2.0
    if mx == mn:
        h = s = 0.0
    else:
        d = mx - mn
        s = d / (2.0 - mx - mn) if l > 0.5 else d / (mx + mn)
        if mx == r:
            h = (g - b) / d + (6 if g < b else 0)
        elif mx == g:
            h = (b - r) / d + 2
        else:
            h = (r - g) / d + 4
        h /= 6.0
    return h, s, l


class ColorGrading:
    """
    Global tone mapping and color grading for the academic figure world.

    [ASTRO-COLOR] Applies journal-style LUT to the entire rendered output.
    """

    def __init__(self, preset: str = "nature"):
        self.set_preset(preset)

    def set_preset(self, name: str):
        if name not in JOURNAL_LUTS:
            name = "nature"
        self.preset_name = name
        self.lut = JOURNAL_LUTS[name]
        print(f"[ASTRO-COLOR] grading preset: {name} — {self.lut['description']}",
              file=sys.stderr)

    def grade_color(self, r: float, g: float, b: float) -> Tuple[float, float, float]:
        """Apply LUT color grading to a single RGB color (0-1 range)."""
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b

        # Lift-gamma-gain style grading
        shadows = self.lut["shadows"]
        mids = self.lut["midtones"]
        highs = self.lut["highlights"]
        sat = self.lut["saturation"]
        con = self.lut["contrast"]

        # Shadow/midtone/highlight blend based on luminance
        if lum < 0.33:
            w = lum / 0.33
            base = tuple(shadows[i] * (1 - w) + mids[i] * w for i in range(3))
        else:
            w = (lum - 0.33) / 0.67
            base = tuple(mids[i] * (1 - w) + highs[i] * w for i in range(3))

        # Blend original with LUT base
        out = tuple(0.6 * c + 0.4 * base[i] for i, c in enumerate((r, g, b)))

        # Saturation adjustment
        gray = sum(out) / 3.0
        out = tuple(gray + (c - gray) * sat for c in out)

        # Contrast around midpoint
        out = tuple(0.5 + (c - 0.5) * con for c in out)

        return tuple(max(0.0, min(1.0, c)) for c in out)


class SpeciesColorHarmony:
    """
    Ensures adjacent species have harmonious colors.

    [ASTRO-COLOR] Uses color wheel relationships to adjust species palettes.
    Strategies: complementary, triadic, analogous, split-complementary.
    """

    @staticmethod
    def check_harmony(colors: List[str]) -> Dict[str, float]:
        """Analyze color harmony score for a set of hex colors."""
        hues = []
        for c in colors:
            r, g, b = hex_to_rgb(c)
            h, s, l = rgb_to_hsl(r, g, b)
            hues.append(h * 360)

        if len(hues) < 2:
            return {"score": 1.0, "type": "single"}

        # Check angular distances between adjacent hues
        hues_sorted = sorted(hues)
        gaps = []
        for i in range(len(hues_sorted)):
            next_h = hues_sorted[(i + 1) % len(hues_sorted)]
            gap = (next_h - hues_sorted[i]) % 360
            gaps.append(gap)

        avg_gap = 360.0 / len(hues)
        variance = sum((g - avg_gap) ** 2 for g in gaps) / len(gaps)
        evenness = 1.0 / (1.0 + variance / 1000.0)

        # Detect harmony type
        if len(hues) == 2:
            diff = abs(hues[0] - hues[1]) % 360
            if 150 < diff < 210:
                htype = "complementary"
            elif diff < 60 or diff > 300:
                htype = "analogous"
            else:
                htype = "split"
        else:
            htype = "multi"

        return {"score": evenness, "type": htype, "hue_gaps": gaps}
