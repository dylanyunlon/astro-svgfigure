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
    layouted: ElkGraph,
    *,
    padding: float = 40.0,
    style_hints: Optional[Dict[str, Any]] = None,
) -> NanoBananaScaffold:
    """
    Convert ELK layouted graph to NanoBanana scaffold.

    Args:
        layouted: ElkGraph with computed x, y coordinates (from ELK.js)
        padding: Canvas padding around the graph bounds
        style_hints: Optional style customization

    Returns:
        NanoBananaScaffold ready for NanoBanana SVG generation
    """
    elements: List[ScaffoldElement] = []
    connections: List[ScaffoldConnection] = []

    # Build node ID → index mapping for coloring
    node_ids = [child.id for child in layouted.children]

    # ── Convert nodes to scaffold elements ──────────────────────────────
    for i, node in enumerate(layouted.children):
        x = node.x if node.x is not None else 0
        y = node.y if node.y is not None else 0
        w = node.width
        h = node.height
        label = node.labels[0].text if node.labels else node.id

        # Determine node type from position (heuristic)
        node_type = _classify_node(node.id, i, len(layouted.children))

        elements.append(ScaffoldElement(
            id=node.id,
            type=node_type,
            label=label,
            x=x + padding,
            y=y + padding,
            width=w,
            height=h,
            style="rounded_rect",
            fill=NODE_COLORS[i % len(NODE_COLORS)],
        ))

    # ── Convert edges to scaffold connections ───────────────────────────
    for edge in layouted.edges:
        source = edge.sources[0] if edge.sources else ""
        target = edge.targets[0] if edge.targets else ""

        # Extract routing points from ELK edge sections
        points = _extract_edge_points(edge, layouted, padding)

        # If no routing points, compute simple center-to-center connection
        if not points:
            points = _compute_simple_connection(source, target, elements)

        connections.append(ScaffoldConnection(
            **{"from": source, "to": target},
            style="arrow",
            points=points,
        ))

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


def _extract_edge_points(
    edge: Any,
    graph: ElkGraph,
    padding: float,
) -> List[Dict[str, float]]:
    """
    Extract routing points from ELK edge sections.

    ELK stores edge routing in:
      edge.sections[i].startPoint, endPoint, bendPoints[]
    """
    points: List[Dict[str, float]] = []

    # ELK edge sections (if available after layout)
    sections = getattr(edge, "sections", None)
    if sections and isinstance(sections, list):
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
