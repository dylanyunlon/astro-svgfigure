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
#  §9  ELK Structured Layout → Mastergo Format
# ═══════════════════════════════════════════════════════════════════════

def elk_to_mastergo(
    elk_graph: Dict[str, Any],
    parent_x: float = 0,
    parent_y: float = 0,
) -> List[Dict[str, Any]]:
    """Convert ELK structured layout data to mastergo format.

    ELK format (after elkjs layout, has coordinates):
        {
          "id": "source_files", "x": 32, "y": 35,
          "width": 160, "height": 50,
          "labels": [{"text": "Source Code Files"}],
          "children": [...], "edges": [...]
        }

    Mastergo format:
        {"id": "source_files", "name": "Source Code Files",
         "bbox": {"x": 32, "y": 35, "width": 160, "height": 50}}

    From NVIDIA CCCL's cub::DeviceSegmentedReduce flatten pattern start.
    Each ELK group has children with coordinates RELATIVE to the group.
    We recursively walk children, accumulating the parent offset to get
    ABSOLUTE coordinates — same as mastergo's global canvas coords.

    Then, follow that pattern to implement a recursive flattener that
    collects all leaf nodes (and optionally groups) into a flat array
    with absolute coordinates. Next, edges are extracted separately
    with their label text and source/target IDs for downstream use.
    """
    objects = []

    node_id = elk_graph.get("id", "root")
    labels = elk_graph.get("labels", [])
    name = labels[0].get("text", "") if labels else elk_graph.get("name", "")

    # This node's absolute position
    node_x = parent_x + float(elk_graph.get("x", 0))
    node_y = parent_y + float(elk_graph.get("y", 0))
    node_w = float(elk_graph.get("width", 0))
    node_h = float(elk_graph.get("height", 0))

    is_group = bool(elk_graph.get("children")) or elk_graph.get("group", False)

    # Add this node (skip root container)
    if node_id != "root" and node_w > 0 and node_h > 0:
        obj = {
            "id": node_id,
            "name": name or node_id,
            "bbox": {
                "x": round(node_x),
                "y": round(node_y),
                "width": round(node_w),
                "height": round(node_h),
            },
        }
        # Preserve ELK-specific metadata (nothing thrown away)
        elk_meta = {}
        if elk_graph.get("iconHint"):
            elk_meta["iconHint"] = elk_graph["iconHint"]
        if is_group:
            elk_meta["group"] = True
        if elk_graph.get("borderless"):
            elk_meta["borderless"] = True
        if elk_graph.get("layoutOptions"):
            elk_meta["layoutOptions"] = elk_graph["layoutOptions"]
        if elk_meta:
            obj["_elk"] = elk_meta
        objects.append(obj)

    # Recurse into children (coordinates are relative to this node)
    for child in elk_graph.get("children", []):
        child_objects = elk_to_mastergo(child, node_x, node_y)
        objects.extend(child_objects)

    return objects


def elk_extract_edges(elk_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract edges from ELK graph with FULL geometry preserved.

    Returns:
        [{
          "id": "s1_e1",
          "source": "source_files", "target": "semantic_dep_graph",
          "label": "parse & extract",
          "type": "data_flow",
          "sections": [{"startPoint":{x,y}, "endPoint":{x,y}, "bendPoints":[...]}],
          "style": {"strokeColor": "#90A4AE", "lineStyle": "dashed", "strokeWidth": 2}
        }]
    """
    edges = []

    for edge in elk_graph.get("edges", []):
        sources = edge.get("sources", [])
        targets = edge.get("targets", [])
        adv = edge.get("advanced", {})
        labels = adv.get("edgeLabels", [])
        label_text = labels[0].get("text", "") if labels else ""

        # Preserve full routing geometry
        sections = edge.get("sections", [])

        # Preserve visual style
        style = {}
        if adv.get("strokeColor"):
            style["strokeColor"] = adv["strokeColor"]
        if adv.get("lineStyle"):
            style["lineStyle"] = adv["lineStyle"]
        if adv.get("strokeWidth"):
            style["strokeWidth"] = adv["strokeWidth"]
        if adv.get("routing"):
            style["routing"] = adv["routing"]

        for src in sources:
            for tgt in targets:
                edges.append({
                    "id": edge.get("id", f"{src}_to_{tgt}"),
                    "source": src,
                    "target": tgt,
                    "label": label_text,
                    "type": adv.get("semanticType", "data_flow"),
                    "sections": sections,
                    "style": style if style else None,
                })

    # Recurse into children groups
    for child in elk_graph.get("children", []):
        edges.extend(elk_extract_edges(child))

    return edges
