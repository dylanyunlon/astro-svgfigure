"""
Scaffold Builder — ELK Layouted → NanoBanana JSON Scaffold
============================================================
Converts ELK layouted graph (with computed x, y coordinates) into
the NanoBanana JSON scaffold format (json_example_user1 template).

This is the bridge between Step 2 (ELK layout) and Step 3 (NanoBanana SVG).

The scaffold provides:
  - Precise pixel positions for every element
  - Connection routing points from ELK edge sections
  - Style hints for NanoBanana to produce beautiful SVG
  - Canvas dimensions computed from the layout bounds

GitHub references:
  - kieler/elkjs (layout output format)
  - gemini-cli-extensions/nanobanana (scaffold input format)
  - EmilStenstrom/elkjs-svg (ELK → SVG conversion reference)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..schemas import (
    ElkGraph,
    NanoBananaScaffold,
    ScaffoldConnection,
    ScaffoldElement,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Color Palette (academic figure style)
# ============================================================================

# Soft pastel palette for academic figures
NODE_COLORS = [
    "#E3F2FD",  # Light Blue
    "#E8F5E9",  # Light Green
    "#FFF3E0",  # Light Orange
    "#F3E5F5",  # Light Purple
    "#E0F7FA",  # Light Cyan
    "#FBE9E7",  # Light Deep Orange
    "#F1F8E9",  # Light Lime
    "#EDE7F6",  # Light Deep Purple
    "#FCE4EC",  # Light Pink
    "#E0F2F1",  # Light Teal
]

BORDER_COLORS = [
    "#42A5F5",  # Blue
    "#66BB6A",  # Green
    "#FFA726",  # Orange
    "#AB47BC",  # Purple
    "#26C6DA",  # Cyan
    "#FF7043",  # Deep Orange
    "#9CCC65",  # Lime
    "#7E57C2",  # Deep Purple
    "#EC407A",  # Pink
    "#26A69A",  # Teal
]


# ============================================================================
# Main Function
# ============================================================================

def build_scaffold(
    layouted,
    *,
    padding: float = 40.0,
    style_hints: Optional[Dict[str, Any]] = None,
) -> NanoBananaScaffold:
    """
    Convert ELK layouted graph to NanoBanana scaffold.

    Args:
        layouted: ElkGraph or dict with computed x, y coordinates (from ELK.js)
        padding: Canvas padding around the graph bounds
        style_hints: Optional style customization

    Returns:
        NanoBananaScaffold ready for NanoBanana SVG generation
    """
    # Normalize input: accept both ElkGraph and dict
    if isinstance(layouted, dict):
        children = layouted.get("children", [])
        edges = layouted.get("edges", [])
    else:
        children = layouted.children
        edges = layouted.edges

    elements: List[ScaffoldElement] = []
    connections: List[ScaffoldConnection] = []

    total_nodes = len(children)

    # ── Convert nodes to scaffold elements (recursive for compound nodes) ──
    _build_elements_recursive(children, elements, connections, padding, depth=0, parent_id=None)

    # ── Convert top-level edges to scaffold connections ─────────────────
    for edge in edges:
        conn = _build_connection_from_edge(edge, elements, padding)
        if conn:
            connections.append(conn)

    # ── Compute canvas dimensions ───────────────────────────────────────
    canvas_width, canvas_height = _compute_canvas_size(elements, padding)

    scaffold = NanoBananaScaffold(
        figure_type="academic_architecture",
        canvas={"width": canvas_width, "height": canvas_height},
        elements=elements,
        connections=connections,
        style_hints=style_hints or {
            "palette": "academic_pastel",
            "font": "system-ui, sans-serif",
            "border_radius": 8,
            "edge_style": "smooth_bezier",
            "arrow_style": "filled",
        },
    )

    logger.info(
        f"Scaffold built: {len(elements)} elements, {len(connections)} connections, "
        f"canvas={canvas_width}x{canvas_height}"
    )

    return scaffold


# ============================================================================
# Helper Functions
# ============================================================================

def _classify_node(node_id: str, index: int, total: int) -> str:
    """Classify node type based on ID and position for styling hints."""
    nid = node_id.lower()

    if any(k in nid for k in ("input", "embed", "encode")):
        return "input"
    if any(k in nid for k in ("output", "result", "predict", "softmax")):
        return "output"
    if any(k in nid for k in ("attention", "attn")):
        return "attention"
    if any(k in nid for k in ("norm", "layer_norm", "batch_norm")):
        return "normalization"
    if any(k in nid for k in ("loss", "criterion", "objective")):
        return "loss"
    if any(k in nid for k in ("add", "residual", "skip", "concat")):
        return "operation"

    # Position-based heuristic
    if index == 0:
        return "input"
    if index == total - 1:
        return "output"

    return "processing"


# ── Group background tints per nesting depth ────────────────────────────
GROUP_BACKGROUND_TINTS = [
    "rgba(100,150,255,0.06)",   # depth 0: very subtle blue
    "rgba(100,200,150,0.06)",   # depth 1: very subtle green
    "rgba(200,150,100,0.06)",   # depth 2: very subtle orange
    "rgba(150,100,200,0.06)",   # depth 3: very subtle purple
    "rgba(100,200,200,0.06)",   # depth 4: very subtle teal
]


def _build_elements_recursive(
    children: Any,
    elements: List[ScaffoldElement],
    connections: List[ScaffoldConnection],
    padding: float,
    depth: int = 0,
    parent_id: Optional[str] = None,
) -> None:
    """
    Recursively build scaffold elements from ELK nodes, handling compound nodes.

    For compound (group) nodes:
      - Creates a group element with borderless background tint
      - Recursively processes nested children
      - Processes nested edges within the compound node
    """
    total = len(children) if isinstance(children, list) else 0

    for i, node in enumerate(children if isinstance(children, list) else []):
        if isinstance(node, dict):
            node_id = node.get("id", f"node_{depth}_{i}")
            x = node.get("x", 0) or 0
            y = node.get("y", 0) or 0
            w = node.get("width", 150)
            h = node.get("height", 50)
            labels = node.get("labels", [])
            label = labels[0].get("text", node_id) if labels else node_id
            is_group = node.get("group", False)
            is_borderless = node.get("borderless", False)
            icon_hint = node.get("iconHint", None)
            nested_children = node.get("children", [])
            nested_edges = node.get("edges", [])
        else:
            node_id = node.id
            x = node.x if node.x is not None else 0
            y = node.y if node.y is not None else 0
            w = node.width
            h = node.height
            label = node.labels[0].text if node.labels else node.id
            is_group = getattr(node, "group", False)
            is_borderless = getattr(node, "borderless", False)
            icon_hint = getattr(node, "iconHint", None)
            nested_children = getattr(node, "children", [])
            nested_edges = getattr(node, "edges", [])

        node_type = _classify_node(node_id, i, total)

        # Determine fill color
        if is_group and is_borderless:
            # Borderless group: use subtle tint based on nesting depth
            fill = GROUP_BACKGROUND_TINTS[depth % len(GROUP_BACKGROUND_TINTS)]
            style = "borderless_group"
        elif is_group:
            fill = NODE_COLORS[i % len(NODE_COLORS)]
            style = "group_rect"
        else:
            fill = NODE_COLORS[(i + depth * 3) % len(NODE_COLORS)]
            style = "rounded_rect"

        elem = ScaffoldElement(
            id=node_id,
            type=node_type if not is_group else "group",
            label=label,
            x=x + padding,
            y=y + padding,
            width=w,
            height=h,
            style=style,
            fill=fill,
        )
        elements.append(elem)

        # Recursively process nested children (compound nodes)
        if nested_children and isinstance(nested_children, list):
            _build_elements_recursive(
                nested_children, elements, connections,
                padding, depth + 1, parent_id=node_id,
            )

        # Process nested edges inside compound nodes
        if nested_edges and isinstance(nested_edges, list):
            for edge in nested_edges:
                conn = _build_connection_from_edge(edge, elements, padding)
                if conn:
                    connections.append(conn)


def _build_connection_from_edge(
    edge: Any,
    elements: List[ScaffoldElement],
    padding: float,
) -> Optional[ScaffoldConnection]:
    """Build a ScaffoldConnection from an ELK edge, with advanced style detection."""
    if isinstance(edge, dict):
        source = (edge.get("sources") or [""])[0]
        target = (edge.get("targets") or [""])[0]
        sections = edge.get("sections", [])
    else:
        source = edge.sources[0] if edge.sources else ""
        target = edge.targets[0] if edge.targets else ""
        sections = getattr(edge, "sections", [])

    if not source or not target:
        return None

    # Extract routing points
    points = _extract_edge_points_from_sections(sections, padding)

    # If no routing points, compute simple center-to-center connection
    if not points:
        points = _compute_simple_connection(source, target, elements)

    # Determine edge style from advanced properties
    edge_style = "arrow"
    adv = edge.get("advanced") if isinstance(edge, dict) else getattr(edge, "advanced", None)
    if adv and isinstance(adv, dict):
        sem = adv.get("semanticType", "")
        ls = adv.get("lineStyle", "")
        if ls == "dashed" or sem in ("gradient_flow", "optional_path", "inference_only", "feedback"):
            edge_style = "dashed"
        elif ls == "dotted" or sem == "attention":
            edge_style = "dotted"
        elif adv.get("directionality") == "bidirectional":
            edge_style = "bidirectional"

    return ScaffoldConnection(
        **{"from": source, "to": target},
        style=edge_style,
        points=points,
    )


def _extract_edge_points_from_sections(
    sections: Any,
    padding: float,
) -> List[Dict[str, float]]:
    """
    Extract routing points from ELK edge sections.
    Accepts both list of dicts and list of objects.
    """
    points: List[Dict[str, float]] = []

    if not sections or not isinstance(sections, list):
        return points

    for section in sections:
        if isinstance(section, dict):
            start = section.get("startPoint")
            if start:
                points.append({
                    "x": start.get("x", 0) + padding,
                    "y": start.get("y", 0) + padding,
                })

            for bp in section.get("bendPoints", []):
                points.append({
                    "x": bp.get("x", 0) + padding,
                    "y": bp.get("y", 0) + padding,
                })

            end = section.get("endPoint")
            if end:
                points.append({
                    "x": end.get("x", 0) + padding,
                    "y": end.get("y", 0) + padding,
                })

    return points


def _compute_simple_connection(
    source_id: str,
    target_id: str,
    elements: List[ScaffoldElement],
) -> List[Dict[str, float]]:
    """Compute simple center-bottom to center-top connection."""
    source = next((e for e in elements if e.id == source_id), None)
    target = next((e for e in elements if e.id == target_id), None)

    if not source or not target:
        return []

    return [
        {"x": source.x + source.width / 2, "y": source.y + source.height},
        {"x": target.x + target.width / 2, "y": target.y},
    ]


def _compute_canvas_size(
    elements: List[ScaffoldElement],
    padding: float,
) -> tuple[float, float]:
    """Compute canvas dimensions to fit all elements."""
    if not elements:
        return (800, 600)

    max_x = max(e.x + e.width for e in elements) + padding
    max_y = max(e.y + e.height for e in elements) + padding

    # Ensure minimum size
    return (max(max_x, 400), max(max_y, 300))