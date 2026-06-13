import os, sys, json, math
from typing import Any, Optional

def _dbg(tag, msg):
    if os.environ.get(f"ASTRO_{tag.replace('-','_')}_VERBOSE", "0") == "1":
        print(f"[{tag}] {msg}", file=sys.stderr)





def build_astro_cell_style_probes(channels_dir: str) -> dict:
    """
    Build one AstroCellStyleProbe per active cell in the registry snapshot.

    Python equivalent of BuildAstroCellStyleProbes() — iterates all published
    cell/*/bbox.json entries, constructs probes, samples surrounding cells,
    returns a dict keyed by cell_id.

    Called from proc() before SVG parameter finalisation so that the probe
    data is ready when adjust_svg_params_for_style_consistency() is called.
    """
    probes: dict = {}

    cell_base = os.path.join(channels_dir, "cell")
    if not os.path.isdir(cell_base):
        return probes

    for cid in os.listdir(cell_base):
        bbox_path = os.path.join(cell_base, cid, "bbox.json")
        if not os.path.isfile(bbox_path):
            continue
        try:
            with open(bbox_path) as _f:
                bbox = json.load(_f)
        except (json.JSONDecodeError, OSError):
            continue

        probe = AstroCellStyleProbe(cid, bbox)
        probe.sample_surrounding_cells(channels_dir)
        probes[cid] = probe

    return probes

# ═══════════════════════════════════════════════
# Channel I/O — Apollo Reader/Writer equivalent
# ═══════════════════════════════════════════════

# ═══════════════════════════════════════════════

