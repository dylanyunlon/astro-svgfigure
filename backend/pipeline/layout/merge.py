from __future__ import annotations
import base64, hashlib, io, json, logging, time
from typing import Any, Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)
try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

# ═══════════════════════════════════════════════════════════════════════
#  §11  Multi-State Merge (Hungarian + Union-Find)
# ═══════════════════════════════════════════════════════════════════════

def merge_multi_state_layouts(
    layouts: List[List[Dict[str, Any]]],
    max_distance: float = 30.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Merge N frames' layouts via optimal matching + transitive grouping."""
    if not layouts:
        return [], {"frames": 0}
    if len(layouts) == 1:
        return layouts[0], {"frames": 1, "shared": 0, "unique": len(layouts[0])}

    from backend.pipeline.layout_algorithms import group_elements_across_frames

    groups, flat = group_elements_across_frames(layouts, max_distance)
    merged, shared = [], 0
    for group in groups:
        frames_in = set()
        best = None
        for idx in group:
            fi, elem = flat[idx]
            frames_in.add(fi)
            if best is None or "_refined" in elem:
                best = elem
        r = {"id": best["id"], "name": best["name"], "bbox": best["bbox"],
             "_shared": len(frames_in) > 1}
        if len(frames_in) > 1:
            r["_appears_in_frames"] = sorted(frames_in)
            shared += 1
        else:
            r["_state_index"] = min(frames_in)
        for k in ("_elk", "_refined"):
            if k in best: r[k] = best[k]
        merged.append(r)

    total = sum(len(l) for l in layouts)
    return merged, {"frames": len(layouts), "total_input": total,
                     "shared": shared, "unique": len(merged)-shared,
                     "merged": len(merged), "dedup": round(1-len(merged)/max(total,1), 2)}


# ═══════════════════════════════════════════════════════════════════════
#  §12  Hidden Element Inference (from ELK edges)
# ═══════════════════════════════════════════════════════════════════════

def infer_hidden_elements(
    layout: List[Dict[str, Any]],
    elk_edges: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Infer elements referenced by edges but not detected in any frame."""
    if not elk_edges:
        return layout, []
    known = {e["id"] for e in layout}
    inferred = []
    for edge in elk_edges:
        for nid in (edge.get("source"), edge.get("target")):
            if nid and nid not in known:
                inferred.append({"id": nid, "name": f"inferred_{nid}",
                                 "bbox": {"x":0,"y":0,"width":0,"height":0},
                                 "_inferred": True, "_from_edge": edge.get("id","")})
                known.add(nid)
    return layout + inferred, inferred


# ═══════════════════════════════════════════════════════════════════════
#  §13  Grid Snap (histogram inference + binary search)
# ═══════════════════════════════════════════════════════════════════════

def grid_snap_layout(
    layout: List[Dict[str, Any]],
    grid: int = 0,
    snap_sizes: bool = True,
    size_tolerance: int = 3,
) -> List[Dict[str, Any]]:
    """Snap layout to auto-inferred grid + standard sizes via binary search."""
    from backend.pipeline.layout_algorithms import infer_grid, snap_to_grid, snap_dimension

    if grid == 0:
        grid = infer_grid(layout)

    snapped = []
    for elem in layout:
        b = elem.get("bbox", {})
        x, y, w, h = int(b.get("x",0)), int(b.get("y",0)), int(b.get("width",0)), int(b.get("height",0))
        if w <= 0 or h <= 0:
            snapped.append(elem); continue
        sx, sy = snap_to_grid(x, grid), snap_to_grid(y, grid)
        sw = snap_dimension(w, size_tolerance) if snap_sizes else snap_to_grid(w, grid)
        sh = snap_dimension(h, size_tolerance) if snap_sizes else snap_to_grid(h, grid)
        r = dict(elem)
        r["bbox"] = {"x": sx, "y": sy, "width": sw, "height": sh}
        if sx != x or sy != y or sw != w or sh != h:
            r["_snapped"] = {"original": {"x":x,"y":y,"width":w,"height":h},
                             "delta": {"dx":sx-x,"dy":sy-y,"dw":sw-w,"dh":sh-h}}
        snapped.append(r)
    return snapped