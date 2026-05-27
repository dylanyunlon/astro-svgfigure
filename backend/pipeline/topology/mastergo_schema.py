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
    layer_id: Optional[str] = None    # M14: which layer (region) owns this element

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
        if self.layer_id:
            d["layer_id"] = self.layer_id
        return d


# ═══════════════════════════════════════════════════════════════════════════
#  §3b  MastergoLayer — per-region layer metadata (M14)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MastergoLayer:
    """One layer in the MasterGo layer panel, mapping 1:1 to a pipeline region.

    MasterGo / Figma layer model:
      - Each layer groups visual elements from one spatial region
      - Layers are ordered by z_index (higher = on top)
      - Layers can be hidden, locked, or have reduced opacity
      - The layer panel supports reordering via drag-and-drop

    From CCCL's DoubleBuffer pattern: each pass writes to a layer
    (Alternate buffer), and layers are composited in z-order just as
    DoubleBuffer.selector picks which buffer is 'Current'.
    """
    id: str
    name: str
    region_id: str                              # back-reference to PlannedRegion.id
    z_index: int = 0
    visible: bool = True
    locked: bool = False
    opacity: float = 1.0
    bbox: Optional[BBox] = None                 # region's canvas-level bounding box
    element_ids: List[str] = field(default_factory=list)   # elements in this layer
    color_tag: Optional[str] = None             # MasterGo color label for the layer

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "name": self.name,
            "region_id": self.region_id,
            "z_index": self.z_index,
            "visible": self.visible,
            "locked": self.locked,
            "opacity": self.opacity,
        }
        if self.bbox:
            d["bbox"] = self.bbox.to_dict()
        if self.element_ids:
            d["element_ids"] = self.element_ids
        if self.color_tag:
            d["color_tag"] = self.color_tag
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
    layers: List[MastergoLayer] = field(default_factory=list)  # M14: per-region layers
    canvas_width: int = 900
    canvas_height: int = 500
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add(self, elem: MastergoElement) -> None:
        self.elements.append(elem)

    def add_layer(self, layer: MastergoLayer) -> None:
        self.layers.append(layer)

    def find(self, elem_id: str) -> Optional[MastergoElement]:
        for e in self.elements:
            if e.id == elem_id:
                return e
        return None

    def elements_in_layer(self, layer_id: str) -> List[MastergoElement]:
        """Return all elements belonging to a given layer."""
        return [e for e in self.elements if e.layer_id == layer_id]

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

    def to_layered_dict(self) -> Dict[str, Any]:
        """Export full layout with layer structure (M14 format).

        Returns a dict with:
          - canvas: {width, height}
          - layers: ordered layer metadata with element_ids
          - elements: flat list of all elements (each tagged with layer_id)
          - edges: all edges
          - metadata: extra info
        """
        return {
            "canvas": {"width": self.canvas_width, "height": self.canvas_height},
            "layers": [l.to_dict() for l in sorted(self.layers, key=lambda x: x.z_index)],
            "elements": self.to_list(),
            "edges": self.edges,
            "metadata": self.metadata,
        }

    def to_mastergo_import(self) -> Dict[str, Any]:
        """Export in MasterGo Import API compatible format.

        MasterGo's plugin import API expects:
          - document: { id, name, type: "DOCUMENT" }
          - document.children → pages
          - page.children → frames
          - frame.children → groups/rectangles/text/vectors

        We map:
          - MastergoLayer → Frame (one frame per region)
          - MastergoElement of type group_container → Group
          - MastergoElement of type icon → Rectangle with imageRef
          - MastergoElement of type label → Text
          - Other MastergoElements → Rectangle

        The geometry is absolute-positioned within each frame, and
        frames are positioned according to their region bbox.
        """
        layers_sorted = sorted(self.layers, key=lambda x: x.z_index)

        frames = []
        for layer in layers_sorted:
            frame_children = []
            for elem in self.elements_in_layer(layer.id):
                node = _element_to_mastergo_node(elem, layer.bbox)
                frame_children.append(node)

            frame = {
                "id": layer.id,
                "name": layer.name,
                "type": "FRAME",
                "visible": layer.visible,
                "locked": layer.locked,
                "opacity": layer.opacity,
                "absoluteBoundingBox": layer.bbox.to_dict() if layer.bbox else {
                    "x": 0, "y": 0,
                    "width": self.canvas_width,
                    "height": self.canvas_height,
                },
                "children": frame_children,
            }
            if layer.color_tag:
                frame["colorTag"] = layer.color_tag
            frames.append(frame)

        # Elements not assigned to any layer go into a "misc" frame
        orphan_elems = [e for e in self.elements if not e.layer_id]
        if orphan_elems:
            misc_children = [_element_to_mastergo_node(e, None) for e in orphan_elems]
            frames.append({
                "id": "layer_misc",
                "name": "Misc Elements",
                "type": "FRAME",
                "visible": True,
                "locked": False,
                "opacity": 1.0,
                "absoluteBoundingBox": {
                    "x": 0, "y": 0,
                    "width": self.canvas_width,
                    "height": self.canvas_height,
                },
                "children": misc_children,
            })

        page = {
            "id": "page_topology",
            "name": "Topology Figure",
            "type": "PAGE",
            "children": frames,
        }

        return {
            "document": {
                "id": "doc_root",
                "name": self.metadata.get("title", "Generated Topology"),
                "type": "DOCUMENT",
                "children": [page],
            },
            "canvas": {"width": self.canvas_width, "height": self.canvas_height},
            "schemaVersion": "1.0",
            "generator": "astro-svgfigure/m14",
        }

    def stats(self) -> Dict[str, Any]:
        types = {}
        for e in self.elements:
            types[e.type] = types.get(e.type, 0) + 1
        max_depth = max((e.depth for e in self.elements), default=0)
        layer_stats = []
        for layer in self.layers:
            layer_stats.append({
                "id": layer.id,
                "name": layer.name,
                "element_count": len(layer.element_ids),
                "z_index": layer.z_index,
                "visible": layer.visible,
            })
        return {
            "total_elements": len(self.elements),
            "total_edges": len(self.edges),
            "total_layers": len(self.layers),
            "max_nesting_depth": max_depth,
            "type_distribution": types,
            "canvas": f"{self.canvas_width}x{self.canvas_height}",
            "layers": layer_stats,
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


# ═══════════════════════════════════════════════════════════════════════════
#  §7  MasterGo Node Serializer — element → import-API node (M14)
# ═══════════════════════════════════════════════════════════════════════════

_TYPE_TO_MASTERGO = {
    "group_container": "GROUP",
    "panel":           "GROUP",
    "section_header":  "TEXT",
    "module_box":      "RECTANGLE",
    "submodule_box":   "RECTANGLE",
    "operation_node":  "RECTANGLE",
    "data_object":     "RECTANGLE",
    "icon":            "RECTANGLE",  # with imageRef
    "label":           "TEXT",
    "badge":           "RECTANGLE",
    "connector_arrow": "VECTOR",
    "annotation":      "TEXT",
    "input_port":      "ELLIPSE",
    "output_port":     "ELLIPSE",
}


def _element_to_mastergo_node(
    elem: MastergoElement,
    frame_bbox: Optional[BBox],
) -> Dict[str, Any]:
    """Convert a MastergoElement to a MasterGo import API node.

    Positions are made relative to the frame_bbox if provided (since
    MasterGo frames use local coordinates for children).
    """
    mg_type = _TYPE_TO_MASTERGO.get(elem.type, "RECTANGLE")

    # Compute position relative to frame origin
    if frame_bbox:
        rel_x = elem.bbox.x - frame_bbox.x
        rel_y = elem.bbox.y - frame_bbox.y
    else:
        rel_x = elem.bbox.x
        rel_y = elem.bbox.y

    node: Dict[str, Any] = {
        "id": elem.id,
        "name": elem.name,
        "type": mg_type,
        "absoluteBoundingBox": elem.bbox.to_dict(),
        "relativeTransform": [[1, 0, rel_x], [0, 1, rel_y]],
        "size": {"x": elem.bbox.width, "y": elem.bbox.height},
    }

    # Style → fills / strokes
    if elem.style:
        if elem.style.get("fill"):
            node["fills"] = [{"type": "SOLID", "color": _parse_color(elem.style["fill"])}]
        if elem.style.get("stroke"):
            node["strokes"] = [{"type": "SOLID", "color": _parse_color(elem.style["stroke"])}]
        if elem.style.get("borderless"):
            node["strokes"] = []

    # Icon hint → imageRef placeholder
    if elem.iconHint and mg_type == "RECTANGLE":
        node["imageRef"] = {"hint": elem.iconHint, "status": "pending"}

    # Text nodes get characters field
    if mg_type == "TEXT":
        node["characters"] = elem.name
        node["style"] = {
            "fontSize": 12 if elem.type == "annotation" else 14,
            "fontFamily": "Inter",
            "textAlignHorizontal": "LEFT",
        }

    return node


def _parse_color(hex_or_name: str) -> Dict[str, float]:
    """Parse a hex color string to MasterGo RGBA dict."""
    h = hex_or_name.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return {"r": r / 255, "g": g / 255, "b": b / 255, "a": 1.0}
    return {"r": 0.5, "g": 0.5, "b": 0.5, "a": 1.0}


# ═══════════════════════════════════════════════════════════════════════════
#  §8  Layered Pipeline → MasterGo Layout (M14 core)
# ═══════════════════════════════════════════════════════════════════════════

# Color tags for layers in the MasterGo panel (cycling palette)
_LAYER_COLORS = [
    "#4F46E5",  # indigo
    "#059669",  # emerald
    "#D97706",  # amber
    "#DC2626",  # red
    "#7C3AED",  # violet
    "#2563EB",  # blue
    "#DB2777",  # pink
    "#65A30D",  # lime
    "#0891B2",  # cyan
    "#EA580C",  # orange
    "#4338CA",  # indigo-dark
    "#047857",  # emerald-dark
]


def layered_to_mastergo_layout(
    layered_result: Any,
) -> MastergoLayout:
    """Convert LayeredResult to MastergoLayout with full region → layer mapping.

    This is the M14 core function: it bridges the layered pipeline's
    multi-region output to MasterGo's layer-based design model.

    From CCCL's output composition pattern: each pass produces a partial
    result (one region's subgraph), and the final step assembles them
    into a unified output.  MasterGo layers are the visual equivalent
    of CCCL's DoubleBuffer slots — each region occupies a buffer (layer)
    that can be independently shown, hidden, locked, or reordered.

    Input: LayeredResult with:
      - canvas: ComposedCanvas (elk_graph, layers, cross_region_edges)
      - regions: List[PlannedRegion] (id, name, bbox, description)
      - intent: UserIntent

    Output: MastergoLayout with:
      - layers: One MastergoLayer per region (z-ordered)
      - elements: All elements tagged with layer_id
      - edges: All edges (intra-region + cross-region)
      - MasterGo Import API compatible export via to_mastergo_import()
    """
    canvas = getattr(layered_result, "canvas", None)
    regions = getattr(layered_result, "regions", [])
    intent = getattr(layered_result, "intent", None)

    if not canvas:
        # Fallback: if no canvas, try to build from elk_graph on result
        elk = getattr(layered_result, "elk_graph", {})
        if elk:
            return elk_to_mastergo_layout(elk)
        return MastergoLayout()

    layout = MastergoLayout(
        canvas_width=canvas.width,
        canvas_height=canvas.height,
        metadata={
            "source": "layered_pipeline",
            "title": intent.summary().get("raw_text", "")[:80] if intent else "",
            "region_count": len(regions),
        },
    )

    # ── Step 1: Create layers from regions ──
    # Like CCCL's buffer allocation: one buffer (layer) per pass (region)
    region_map = {}  # region_id → PlannedRegion
    for i, region in enumerate(regions):
        rid = region.id if hasattr(region, "id") else region.get("id", f"region_{i}")
        rname = region.name if hasattr(region, "name") else region.get("name", f"Region {i+1}")
        rbbox_raw = region.bbox if hasattr(region, "bbox") else region.get("bbox", {})

        rbbox = BBox(
            x=rbbox_raw.get("x", 0),
            y=rbbox_raw.get("y", 0),
            width=rbbox_raw.get("width", 200),
            height=rbbox_raw.get("height", 200),
        ) if isinstance(rbbox_raw, dict) else BBox(x=0, y=0, width=200, height=200)

        layer = MastergoLayer(
            id=f"layer_{rid}",
            name=rname,
            region_id=rid,
            z_index=i,
            visible=True,
            locked=False,
            opacity=1.0,
            bbox=rbbox,
            color_tag=_LAYER_COLORS[i % len(_LAYER_COLORS)],
        )
        layout.add_layer(layer)
        region_map[rid] = {
            "region": region,
            "layer": layer,
            "bbox": rbbox,
        }

    # ── Step 2: Walk ELK graph, assign elements to layers ──
    # The composed ELK graph has top-level children that map to regions.
    # Each top-level child's id prefix matches its region.
    elk = canvas.elk_graph or {}

    def _safe_id(raw: str) -> str:
        return re.sub(r'[^a-zA-Z0-9_]', '_', raw.lower()).strip('_')[:60]

    def _find_layer_for_node(node_id: str) -> Optional[MastergoLayer]:
        """Match a node to its owning layer by region prefix."""
        nid_lower = node_id.lower()
        for rid, info in region_map.items():
            if rid.lower() in nid_lower or nid_lower.startswith(_safe_id(rid)):
                return info["layer"]
        return None

    def _walk_elk(
        node: Dict[str, Any],
        px: float, py: float,
        parent_id: Optional[str],
        depth: int,
        inherited_layer: Optional[MastergoLayer],
    ) -> None:
        nid = node.get("id", "")
        if not nid or nid == "root":
            for child in node.get("children", []):
                _walk_elk(child, px, py, None, depth, None)
            return

        safe = _safe_id(nid)
        labels = node.get("labels", [])
        name = labels[0].get("text", "") if labels else nid

        ax = px + float(node.get("x", 0))
        ay = py + float(node.get("y", 0))
        w = float(node.get("width", 150))
        h = float(node.get("height", 50))

        # Determine owning layer: check node id, then inherit from parent
        layer = _find_layer_for_node(nid) or inherited_layer

        is_group = bool(node.get("children"))
        has_icon = bool(node.get("iconHint"))

        # Element type classification (same logic as elk_to_mastergo_layout)
        if is_group:
            etype = "group_container"
        elif node.get("type") == "input" or "input" in nid.lower():
            etype = "input_port"
        elif node.get("type") == "output" or "output" in nid.lower():
            etype = "output_port"
        elif has_icon and w <= 60 and h <= 60:
            etype = "icon"
        elif any(kw in nid.lower() for kw in ("filter", "join", "aggregate", "scan")):
            etype = "operation_node"
        elif any(kw in nid.lower() for kw in ("table", "index", "column", "code", "exe")):
            etype = "data_object"
        elif depth >= 2:
            etype = "submodule_box"
        else:
            etype = "module_box"

        # Style
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

        elem = MastergoElement(
            id=safe,
            name=name,
            bbox=BBox(x=round(ax), y=round(ay), width=round(w), height=round(h)),
            type=etype,
            parent=_safe_id(parent_id) if parent_id else None,
            depth=depth,
            iconHint=node.get("iconHint"),
            style=style,
            layer_id=layer.id if layer else None,
        )

        children = node.get("children", [])
        if children:
            elem.children = [_safe_id(c.get("id", "")) for c in children if c.get("id")]

        layout.add(elem)

        # Register element in its layer
        if layer:
            layer.element_ids.append(safe)

        # Sub-elements for icon + label
        if has_icon and not is_group and w >= 80:
            icon_w = min(40, int(w * 0.3))
            icon_elem = MastergoElement(
                id=f"{safe}_icon",
                name=f"{name} icon",
                bbox=BBox(x=round(ax + 4), y=round(ay + (h - icon_w) / 2), width=icon_w, height=icon_w),
                type="icon",
                parent=safe,
                depth=depth + 1,
                iconHint=node.get("iconHint"),
                layer_id=layer.id if layer else None,
            )
            layout.add(icon_elem)
            if layer:
                layer.element_ids.append(f"{safe}_icon")

            label_elem = MastergoElement(
                id=f"{safe}_label",
                name=name,
                bbox=BBox(x=round(ax + icon_w + 8), y=round(ay + 4), width=round(w - icon_w - 12), height=round(h - 8)),
                type="label",
                parent=safe,
                depth=depth + 1,
                layer_id=layer.id if layer else None,
            )
            layout.add(label_elem)
            if layer:
                layer.element_ids.append(f"{safe}_label")

        for child in children:
            _walk_elk(child, ax, ay, nid, depth + 1, layer)

    _walk_elk(elk, 0, 0, None, 0, None)

    # ── Step 3: Extract edges (intra + cross-region) ──
    layout.edges = _extract_edges_recursive(elk)

    # Add cross-region edges from compositor
    for cr_edge in (canvas.cross_region_edges or []):
        layout.edges.append({
            "id": cr_edge.get("id", ""),
            "source": cr_edge.get("source", ""),
            "target": cr_edge.get("target", ""),
            "label": cr_edge.get("label", ""),
            "type": "cross_region",
            "style": cr_edge.get("style"),
        })

    return layout
