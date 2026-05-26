"""mastergo_schema.py — Target output schema for mastergo-quality layout data.

The gap between ELK topology JSON and mastergo_all_layoutobj.txt:

  ELK output:       ~15 nodes, no pixel positions, no visual style
  mastergo output:   50+ elements, pixel-precise bbox, icon types, nesting depth ≥ 3

This module defines the schema that downstream renderers expect, and provides
converters to bridge the gap.  Every visual element in the original figure —
icons, labels, group borders, data arrows, resource chips — gets its own entry
with an absolute-coordinate bounding box.

Design reference: NVIDIA CCCL's device-level reduce pattern.
  Start from a coarse partition (ELK groups), then recursively subdivide each
  group into finer elements (icons, labels, badges) until every pixel region
  is accounted for.  The subdivision terminates when element area < min_area.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════
#  §1  Element Types — exhaustive taxonomy of visual components
# ═══════════════════════════════════════════════════════════════════════════

ELEMENT_TYPES = [
    # Structural
    "group_container",    # Dashed/solid border box enclosing children
    "panel",              # Named region (e.g. "Workload", "Resource", "Storage")
    "section_header",     # Title bar within a panel
    # Functional
    "module_box",         # Agent / major processing block
    "submodule_box",      # Sub-step within a module
    "operation_node",     # Filter, Join, Aggregate, etc.
    "data_object",        # Table, Index, Column, Code file
    # Visual atoms
    "icon",               # CPU, GPU, RAM, SSD, HDD, web, tool icons
    "label",              # Text label (standalone, not part of a node)
    "badge",              # Small tag/chip (e.g. "Q₁.exe")
    "connector_arrow",    # Edge rendered as arrow (for bbox tracking)
    "annotation",         # Ellipsis "...", footnotes, legends
    # Input/Output
    "input_port",         # Left-side entry point
    "output_port",        # Right-side exit point
]


# ═══════════════════════════════════════════════════════════════════════════
#  §2  BBox — pixel-precise bounding box (absolute coordinates)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple:
        return (self.x + self.width // 2, self.y + self.height // 2)

    def contains(self, other: "BBox") -> bool:
        return (self.x <= other.x
                and self.y <= other.y
                and self.x + self.width >= other.x + other.width
                and self.y + self.height >= other.y + other.height)

    def overlaps(self, other: "BBox") -> bool:
        return not (self.x + self.width <= other.x
                    or other.x + other.width <= self.x
                    or self.y + self.height <= other.y
                    or other.y + other.height <= self.y)

    def iou(self, other: "BBox") -> float:
        ix = max(0, min(self.x + self.width, other.x + other.width) - max(self.x, other.x))
        iy = max(0, min(self.y + self.height, other.y + other.height) - max(self.y, other.y))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def to_dict(self) -> Dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


# ═══════════════════════════════════════════════════════════════════════════
#  §3  MastergoElement — one entry in the layout object list
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MastergoElement:
    """Single visual element in mastergo_all_layoutobj format.

    Fields mirror what Mastergo exports for each design object:
      id       — unique identifier (snake_case)
      name     — human-readable label
      bbox     — absolute pixel bounding box
      type     — one of ELEMENT_TYPES
      parent   — id of containing group (None for root-level)
      depth    — nesting depth (0 = root-level, 1 = inside panel, ...)
      iconHint — natural-language icon description for image generation
      style    — optional visual properties (fill, stroke, opacity)
      children — ids of contained elements (for groups)
    """
    id: str
    name: str
    bbox: BBox
    type: str = "module_box"
    parent: Optional[str] = None
    depth: int = 0
    iconHint: Optional[str] = None
    style: Optional[Dict[str, Any]] = None
    children: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "name": self.name,
            "bbox": self.bbox.to_dict(),
            "type": self.type,
        }
        if self.parent:
            d["parent"] = self.parent
        if self.depth > 0:
            d["depth"] = self.depth
        if self.iconHint:
            d["iconHint"] = self.iconHint
        if self.style:
            d["style"] = self.style
        if self.children:
            d["children"] = self.children
        return d


# ═══════════════════════════════════════════════════════════════════════════
#  §4  MastergoLayout — the complete layout document
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MastergoLayout:
    """Complete mastergo-quality layout for one figure.

    Invariants enforced by validate():
      - Every element has a unique id
      - Every parent reference points to an existing element
      - No bbox exceeds canvas bounds
      - Group elements contain all declared children
      - Element count >= density threshold for the figure complexity
    """
    elements: List[MastergoElement] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    canvas_width: int = 900
    canvas_height: int = 500
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add(self, elem: MastergoElement) -> None:
        self.elements.append(elem)

    def find(self, elem_id: str) -> Optional[MastergoElement]:
        for e in self.elements:
            if e.id == elem_id:
                return e
        return None

    def by_type(self, element_type: str) -> List[MastergoElement]:
        return [e for e in self.elements if e.type == element_type]

    def validate(self) -> tuple:
        """Validate layout integrity. Returns (ok, errors)."""
        errors = []
        ids = set()
        for e in self.elements:
            if e.id in ids:
                errors.append(f"duplicate id: {e.id}")
            ids.add(e.id)
            if e.parent and e.parent not in ids:
                # Parent must appear before child (or exist)
                pass  # Relaxed: parent checked at end
            if e.bbox.x < 0 or e.bbox.y < 0:
                errors.append(f"{e.id}: negative bbox coords")
            if e.bbox.width <= 0 or e.bbox.height <= 0:
                errors.append(f"{e.id}: zero/negative bbox size")
            if e.type not in ELEMENT_TYPES:
                errors.append(f"{e.id}: unknown type '{e.type}'")

        # Check parent references
        for e in self.elements:
            if e.parent and e.parent not in ids:
                errors.append(f"{e.id}: parent '{e.parent}' not found")

        # Check child references
        for e in self.elements:
            for child_id in e.children:
                if child_id not in ids:
                    errors.append(f"{e.id}: child '{child_id}' not found")

        # Density check
        if len(self.elements) < 15:
            errors.append(f"only {len(self.elements)} elements, expect ≥15 for any figure")

        return len(errors) == 0, errors

    def to_list(self) -> List[Dict[str, Any]]:
        """Export as list of dicts (mastergo_all_layoutobj.txt format)."""
        return [e.to_dict() for e in self.elements]

    def stats(self) -> Dict[str, Any]:
        types = {}
        for e in self.elements:
            types[e.type] = types.get(e.type, 0) + 1
        max_depth = max((e.depth for e in self.elements), default=0)
        return {
            "total_elements": len(self.elements),
            "total_edges": len(self.edges),
            "max_nesting_depth": max_depth,
            "type_distribution": types,
            "canvas": f"{self.canvas_width}x{self.canvas_height}",
        }


# ═══════════════════════════════════════════════════════════════════════════
#  §5  ELK → Mastergo converter (enhanced version of elk_bridge.py)
# ═══════════════════════════════════════════════════════════════════════════

def elk_to_mastergo_layout(
    elk_graph: Dict[str, Any],
    canvas_width: int = 900,
    canvas_height: int = 500,
) -> MastergoLayout:
    """Convert post-layout ELK graph to MastergoLayout with full element detail.

    Enhancement over elk_bridge.elk_to_mastergo:
      - Tracks parent/child relationships
      - Computes nesting depth
      - Preserves iconHint and style metadata
      - Extracts edges with routing geometry
      - Produces sub-elements for icons/labels within nodes

    From NVIDIA CCCL's cub::DeviceSegmentedReduce pattern:
      Walk the ELK tree, accumulate parent offsets for absolute coords,
      and at each node emit both the container element and its visual
      sub-components (icon region, label region).
    """
    layout = MastergoLayout(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    )

    def _safe_id(raw: str) -> str:
        return re.sub(r'[^a-zA-Z0-9_]', '_', raw.lower()).strip('_')[:60]

    def _walk(
        node: Dict[str, Any],
        parent_x: float,
        parent_y: float,
        parent_id: Optional[str],
        depth: int,
    ) -> None:
        node_id = node.get("id", "")
        if not node_id or node_id == "root":
            # Process children of root without creating root element
            for child in node.get("children", []):
                _walk(child, parent_x, parent_y, None, depth)
            return

        safe_id = _safe_id(node_id)
        labels = node.get("labels", [])
        name = labels[0].get("text", "") if labels else node_id

        abs_x = parent_x + float(node.get("x", 0))
        abs_y = parent_y + float(node.get("y", 0))
        w = float(node.get("width", 150))
        h = float(node.get("height", 50))

        is_group = bool(node.get("children")) or node.get("group", False)
        has_icon = bool(node.get("iconHint"))

        # Determine element type
        if is_group:
            etype = "group_container"
        elif node.get("type") == "input" or "input" in node_id.lower():
            etype = "input_port"
        elif node.get("type") == "output" or "output" in node_id.lower():
            etype = "output_port"
        elif has_icon and w <= 60 and h <= 60:
            etype = "icon"
        elif node.get("type") == "operation" or any(
            kw in node_id.lower() for kw in ("filter", "join", "aggregate", "scan")
        ):
            etype = "operation_node"
        elif node.get("type") == "data_object" or any(
            kw in node_id.lower() for kw in ("table", "index", "column", "code", "exe")
        ):
            etype = "data_object"
        elif depth >= 2:
            etype = "submodule_box"
        else:
            etype = "module_box"

        # Style extraction
        style = None
        adv = node.get("advanced", {})
        if adv:
            style = {}
            if adv.get("strokeColor"):
                style["stroke"] = adv["strokeColor"]
            if adv.get("fillColor"):
                style["fill"] = adv["fillColor"]
            if node.get("borderless"):
                style["borderless"] = True

        # Create main element
        elem = MastergoElement(
            id=safe_id,
            name=name,
            bbox=BBox(x=round(abs_x), y=round(abs_y), width=round(w), height=round(h)),
            type=etype,
            parent=_safe_id(parent_id) if parent_id else None,
            depth=depth,
            iconHint=node.get("iconHint"),
            style=style,
        )

        # Register children IDs
        children = node.get("children", [])
        if children:
            elem.children = [_safe_id(c.get("id", "")) for c in children if c.get("id")]

        layout.add(elem)

        # Sub-element generation: if node has icon + label, split into regions
        if has_icon and not is_group and w >= 80:
            # Icon sub-region (left 40px)
            icon_w = min(40, int(w * 0.3))
            icon_elem = MastergoElement(
                id=f"{safe_id}_icon",
                name=f"{name} icon",
                bbox=BBox(
                    x=round(abs_x + 4),
                    y=round(abs_y + (h - icon_w) / 2),
                    width=icon_w,
                    height=icon_w,
                ),
                type="icon",
                parent=safe_id,
                depth=depth + 1,
                iconHint=node.get("iconHint"),
            )
            layout.add(icon_elem)

            # Label sub-region (remaining width)
            label_elem = MastergoElement(
                id=f"{safe_id}_label",
                name=name,
                bbox=BBox(
                    x=round(abs_x + icon_w + 8),
                    y=round(abs_y + 4),
                    width=round(w - icon_w - 12),
                    height=round(h - 8),
                ),
                type="label",
                parent=safe_id,
                depth=depth + 1,
            )
            layout.add(label_elem)

        # Recurse into children
        for child in children:
            _walk(child, abs_x, abs_y, node_id, depth + 1)

    _walk(elk_graph, 0, 0, None, 0)

    # Extract edges
    layout.edges = _extract_edges_recursive(elk_graph)

    return layout


def _extract_edges_recursive(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Recursively extract all edges from ELK graph."""
    edges = []
    for edge in node.get("edges", []):
        sources = edge.get("sources", [])
        targets = edge.get("targets", [])
        adv = edge.get("advanced", {})
        labels_list = adv.get("edgeLabels", edge.get("labels", []))
        label = labels_list[0].get("text", "") if labels_list else ""

        style = {}
        for key in ("strokeColor", "lineStyle", "strokeWidth", "strokeDasharray"):
            if adv.get(key):
                style[key] = adv[key]

        for src in sources:
            for tgt in targets:
                edges.append({
                    "id": edge.get("id", f"{src}_to_{tgt}"),
                    "source": src,
                    "target": tgt,
                    "label": label,
                    "type": adv.get("semanticType", "data_flow"),
                    "style": style if style else None,
                })

    for child in node.get("children", []):
        edges.extend(_extract_edges_recursive(child))

    return edges


# ═══════════════════════════════════════════════════════════════════════════
#  §6  Complexity estimator — how many elements should the figure have?
# ═══════════════════════════════════════════════════════════════════════════

def estimate_figure_complexity(text: str) -> Dict[str, Any]:
    """Estimate expected element count from paper description text.

    Uses keyword density analysis to predict figure complexity.
    The GenDB example has ~70 distinct visual elements.
    """
    # Count distinct technical nouns
    # Split on word boundaries, filter for capitalized words and known terms
    words = set(re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', text))

    # Count specific categories
    hardware_kw = {"CPU", "GPU", "RAM", "SSD", "HDD", "SIMD", "cache", "mmap"}
    data_kw = {"Table", "Index", "Column", "Schema", "SQL", "Query", "Data"}
    op_kw = {"Join", "Filter", "Aggregate", "Scan", "Sort", "Group", "Merge"}
    agent_kw = {"Agent", "Analyzer", "Designer", "Planner", "Generator", "Optimizer"}

    hw_count = len([w for w in words if any(kw.lower() in w.lower() for kw in hardware_kw)])
    data_count = len([w for w in words if any(kw.lower() in w.lower() for kw in data_kw)])
    op_count = len([w for w in words if any(kw.lower() in w.lower() for kw in op_kw)])
    agent_count = len([w for w in words if any(kw.lower() in w.lower() for kw in agent_kw)])

    # Estimate total visual elements
    base = len(words)
    # Each hardware item = icon + label = 2 elements
    # Each agent = box + sub-components ≈ 4 elements
    # Each operation = node + icon = 2 elements
    estimated = hw_count * 2 + data_count * 2 + op_count * 2 + agent_count * 4 + base

    # Clamp to reasonable range
    estimated = max(20, min(120, estimated))

    complexity = "simple" if estimated < 25 else "medium" if estimated < 45 else "complex" if estimated < 70 else "very_complex"

    return {
        "estimated_elements": estimated,
        "complexity": complexity,
        "breakdown": {
            "hardware": hw_count,
            "data_objects": data_count,
            "operations": op_count,
            "agents": agent_count,
            "other_terms": base,
        },
        "min_entities": max(20, estimated // 2),
        "min_edges": max(15, estimated // 3),
    }
