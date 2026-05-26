"""solver.py — Deterministic constraint-based layout solver.

Takes canonical elements, edges, and groups from the canonicalizer,
produces absolute pixel positions for every element.

Layer architecture (like osdk Store.ts):
    Layer 0: Type defaults from registry
    Layer 1: Text-derived size overrides (already computed by canonicalizer)
    Layer 2: Group constraint propagation (parent sizes from children)
    Layer 3: Position assignment (topological order + group layout)
    Layer 4: Collision resolution (no overlapping elements)

The output is a SolvedLayout: every element has (x, y, width, height)
ready for rendering. No LLM involvement in position/size.

Analogy to osdk-ts:
    Store.ts     → ConstraintSolver (state management + computation)
    Layers.ts    → Solve layers (base → override → resolved)
    Changes.ts   → Layout diff tracking
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.pipeline.topology.constraint.registry import (
    ConstraintRegistry, GroupConstraint,
)
from backend.pipeline.topology.constraint.canonicalizer import (
    CanonicalElement, CanonicalEdge, CanonicalGroup,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  §1  SolvedElement — position-resolved element
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SolvedElement:
    """Element with absolute pixel position (the final output)."""
    id: str
    name: str
    type: str
    x: int
    y: int
    width: int
    height: int
    depth: int = 0
    parent_group: Optional[str] = None
    icon_hint: str = ""
    # For ELK compatibility
    labels: List[Dict[str, str]] = field(default_factory=list)
    children: List["SolvedElement"] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    is_group: bool = False
    borderless: bool = True

    def to_elk_node(self) -> Dict[str, Any]:
        """Convert to ELK JSON node format."""
        node: Dict[str, Any] = {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "labels": self.labels or [{"text": self.name}],
        }
        if self.icon_hint:
            node["iconHint"] = self.icon_hint
        if self.is_group:
            node["group"] = True
            node["borderless"] = self.borderless
            node["layoutOptions"] = {
                "elk.padding": f"[top=35,left=12,bottom=12,right=12]"
            }
            if self.children:
                node["children"] = [c.to_elk_node() for c in self.children]
            if self.edges:
                node["edges"] = self.edges
        return node

    def to_mastergo(self) -> Dict[str, Any]:
        """Convert to mastergo_all_layoutobj.txt format."""
        return {
            "id": self.id,
            "name": self.name,
            "bbox": {
                "x": self.x,
                "y": self.y,
                "width": self.width,
                "height": self.height,
            },
            "type": self.type,
            "depth": self.depth,
            "parent": self.parent_group,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  §2  SolvedLayout — the complete layout solution
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SolvedLayout:
    """Complete layout with all elements positioned."""
    elements: List[SolvedElement] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    canvas_width: int = 900
    canvas_height: int = 500

    def to_elk_graph(self) -> Dict[str, Any]:
        """Convert to complete ELK JSON graph."""
        # Separate root-level elements from group children
        root_elements = [e for e in self.elements if e.parent_group is None]

        return {
            "id": "root",
            "layoutOptions": {
                "elk.algorithm": "layered",
                "elk.direction": "DOWN",
                "elk.layered.spacing.nodeNodeBetweenLayers": "60",
                "elk.spacing.nodeNode": "40",
            },
            "children": [e.to_elk_node() for e in root_elements],
            "edges": self.edges,
        }

    def to_mastergo_list(self) -> List[Dict[str, Any]]:
        """Convert to flat mastergo format."""
        result = []
        for e in self.elements:
            result.append(e.to_mastergo())
        return result

    def stats(self) -> Dict[str, Any]:
        types: Dict[str, int] = {}
        for e in self.elements:
            types[e.type] = types.get(e.type, 0) + 1
        max_depth = max((e.depth for e in self.elements), default=0)
        return {
            "total_elements": len(self.elements),
            "total_edges": len(self.edges),
            "max_depth": max_depth,
            "types": types,
            "canvas": f"{self.canvas_width}x{self.canvas_height}",
        }


# ═══════════════════════════════════════════════════════════════════════════
#  §3  ConstraintSolver — the deterministic layout engine
# ═══════════════════════════════════════════════════════════════════════════

class ConstraintSolver:
    """Deterministic position computation from canonical elements.

    Usage:
        solver = ConstraintSolver(registry, canvas_width=900)
        layout = solver.solve(elements, edges, groups)
    """

    def __init__(
        self,
        registry: Optional[ConstraintRegistry] = None,
        canvas_width: int = 900,
        canvas_height: int = 500,
        margin: int = 40,
    ):
        self._registry = registry or ConstraintRegistry()
        self._canvas_w = canvas_width
        self._canvas_h = canvas_height
        self._margin = margin

    def solve(
        self,
        elements: List[CanonicalElement],
        edges: List[CanonicalEdge],
        groups: List[CanonicalGroup],
    ) -> SolvedLayout:
        """Constraint solver with Sugiyama layered layout.

        Phase 0: Sizes computed by canonicalizer (type-aware)
        Phase 1: Group sizes from children (bottom-up propagation)
        Phase 2: Sugiyama layout for root-level items:
                 - Cycle removal (DFS back-edge reversal)
                 - Longest-path layer assignment
                 - Barycenter crossing minimization (O(|V|·|E|) per sweep)
                 - Median-aligned coordinate assignment (Brandes-Köpf inspired)
        Phase 3: Intra-group child layout (direction-aware)
        Phase 4: Collision resolution
        """
        from backend.pipeline.topology.constraint.sugiyama import sugiyama_layout

        elem_map = {e.id: e for e in elements}
        group_map = {g.id: g for g in groups}

        # ── Build parent-child index ──
        children_of: Dict[str, List[str]] = defaultdict(list)
        for g in groups:
            for child_id in g.children_ids:
                children_of[g.id].append(child_id)

        # ── Phase 1: Group sizes (bottom-up) ──
        group_sizes = self._compute_group_sizes(groups, elem_map)

        # ── Phase 2: Sugiyama layout for root-level items ──
        # Identify root-level items (not inside any group)
        grouped_elements = set()
        for g in groups:
            grouped_elements.update(g.children_ids)
        group_set = {g.id for g in groups}

        root_ids = []
        root_widths: Dict[str, int] = {}
        root_heights: Dict[str, int] = {}

        for g in groups:
            if g.parent_group is None:
                root_ids.append(g.id)
                gw, gh = group_sizes.get(g.id, (200, 100))
                root_widths[g.id] = gw
                root_heights[g.id] = gh
        for e in elements:
            if e.id not in grouped_elements and e.id not in group_set:
                root_ids.append(e.id)
                root_widths[e.id] = e.width
                root_heights[e.id] = e.height

        # Build root-level edge list (map child→parent group for cross-group edges)
        elem_to_root: Dict[str, str] = {}
        for g in groups:
            for child_id in g.children_ids:
                elem_to_root[child_id] = g.id

        root_edges: List[tuple] = []
        seen_root_edges: set = set()
        for edge in edges:
            src_root = elem_to_root.get(edge.source, edge.source)
            tgt_root = elem_to_root.get(edge.target, edge.target)
            if src_root != tgt_root and src_root in set(root_ids) and tgt_root in set(root_ids):
                pair = (src_root, tgt_root)
                if pair not in seen_root_edges:
                    seen_root_edges.add(pair)
                    root_edges.append(pair)

        # Run Sugiyama
        positions, sugiyama_diag = sugiyama_layout(
            root_ids, root_edges, root_widths, root_heights,
            canvas_width=self._canvas_w,
            min_node_gap=40,
            layer_gap=60,
            margin=self._margin,
        )

        # ── Phase 3: Build solved elements with Sugiyama positions ──
        solved: Dict[str, SolvedElement] = {}

        for item_id in root_ids:
            x, y = positions.get(item_id, (self._margin, self._margin))

            if item_id in group_map:
                group = group_map[item_id]
                gw, gh = group_sizes.get(item_id, (200, 100))
                group_elem = SolvedElement(
                    id=item_id, name=group.label, type="group_container",
                    x=x, y=y, width=gw, height=gh,
                    is_group=True, depth=group.depth,
                )
                # Layout children inside group
                group_children = self._layout_children_in_group(
                    group, elem_map, x, y, gw, gh,
                )
                group_elem.children = group_children
                # Build intra-group edges
                child_ids = {c.id for c in group_children}
                group_elem.edges = [
                    self._make_elk_edge(e)
                    for e in edges
                    if e.source in child_ids and e.target in child_ids
                ]
                solved[item_id] = group_elem
                for c in group_children:
                    solved[c.id] = c
            elif item_id in elem_map:
                e = elem_map[item_id]
                solved[item_id] = SolvedElement(
                    id=e.id, name=e.name, type=e.type,
                    x=x, y=y, width=e.width, height=e.height,
                    icon_hint=e.icon_hint, depth=e.depth,
                )

        solved_elements = list(solved.values())

        # ── Phase 4: Collision resolution ──
        solved_elements = self._resolve_collisions(solved_elements)

        # ── Build edge list for ELK ──
        elk_edges = self._build_edges(edges, solved_elements, groups)

        # ── Compute canvas size ──
        max_x = max((e.x + e.width for e in solved_elements), default=self._canvas_w)
        max_y = max((e.y + e.height for e in solved_elements), default=self._canvas_h)

        layout = SolvedLayout(
            elements=solved_elements,
            edges=elk_edges,
            canvas_width=max(self._canvas_w, max_x + self._margin),
            canvas_height=max(self._canvas_h, max_y + self._margin),
        )

        return layout

    # ── Layer 1: Bottom-up group sizing ──

    def _compute_group_sizes(
        self,
        groups: List[CanonicalGroup],
        elem_map: Dict[str, CanonicalElement],
    ) -> Dict[str, Tuple[int, int]]:
        """Compute group sizes from children (bottom-up).

        For each group, compute:
            width  = f(children_widths, direction, padding)
            height = f(children_heights, direction, padding)

        This is fully deterministic — sizes are computed from registry
        constraints and children dimensions, never from LLM.
        """
        group_sizes: Dict[str, Tuple[int, int]] = {}

        for group in groups:
            constraint = self._registry.get_element("group_container")
            group_constraint = self._registry.get_group(group.layout_type)

            # Collect children sizes
            children_widths = []
            children_heights = []
            for child_id in group.children_ids:
                if child_id in elem_map:
                    children_widths.append(elem_map[child_id].width)
                    children_heights.append(elem_map[child_id].height)
                elif child_id in group_sizes:
                    w, h = group_sizes[child_id]
                    children_widths.append(w)
                    children_heights.append(h)

            if not children_widths:
                group_sizes[group.id] = (constraint.min_w, constraint.min_h)
                continue

            w, h = constraint.compute_group_size(
                children_widths, children_heights,
                direction=group_constraint.direction,
            )
            group_sizes[group.id] = (w, h)

        return group_sizes

    def _layout_children_in_group(
        self,
        group: CanonicalGroup,
        elem_map: Dict[str, CanonicalElement],
        group_x: int,
        group_y: int,
        group_w: int,
        group_h: int,
    ) -> List[SolvedElement]:
        """Lay out children inside a group container.

        Direction is deterministic from GroupConstraint:
            DOWN  → stack vertically, centered
            RIGHT → arrange horizontally
            GRID  → grid layout with max_children_per_row
        """
        gc = self._registry.get_group(group.layout_type)
        ec = self._registry.get_element("group_container")

        children = [elem_map[cid] for cid in group.children_ids if cid in elem_map]
        if not children:
            return []

        pad_top = ec.child_padding_top
        pad_side = ec.child_padding_sides
        gap = gc.min_gap

        solved = []

        if gc.direction == "RIGHT":
            # Horizontal layout
            total_w = sum(c.width for c in children) + (len(children) - 1) * gap
            start_x = group_x + pad_side
            if gc.align == "center":
                available = group_w - 2 * pad_side
                start_x = group_x + pad_side + max(0, (available - total_w) // 2)

            cx = start_x
            for c in children:
                cy = group_y + pad_top
                solved.append(SolvedElement(
                    id=c.id, name=c.name, type=c.type,
                    x=cx, y=cy, width=c.width, height=c.height,
                    parent_group=group.id, depth=group.depth + 1,
                    icon_hint=c.icon_hint,
                ))
                cx += c.width + gap

        elif gc.direction == "GRID":
            # Grid layout
            cols = gc.max_children_per_row
            cx = group_x + pad_side
            cy = group_y + pad_top
            col_idx = 0
            row_max_h = 0

            for c in children:
                if col_idx >= cols:
                    col_idx = 0
                    cy += row_max_h + gap
                    cx = group_x + pad_side
                    row_max_h = 0

                solved.append(SolvedElement(
                    id=c.id, name=c.name, type=c.type,
                    x=cx, y=cy, width=c.width, height=c.height,
                    parent_group=group.id, depth=group.depth + 1,
                    icon_hint=c.icon_hint,
                ))
                cx += c.width + gap
                row_max_h = max(row_max_h, c.height)
                col_idx += 1

        else:
            # DOWN: vertical stack (default)
            total_h = sum(c.height for c in children) + (len(children) - 1) * gap
            cy = group_y + pad_top
            for c in children:
                cx = group_x + pad_side
                if gc.align == "center":
                    available = group_w - 2 * pad_side
                    cx = group_x + pad_side + max(0, (available - c.width) // 2)

                solved.append(SolvedElement(
                    id=c.id, name=c.name, type=c.type,
                    x=cx, y=cy, width=c.width, height=c.height,
                    parent_group=group.id, depth=group.depth + 1,
                    icon_hint=c.icon_hint,
                ))
                cy += c.height + gap

        return solved

    # ── Collision resolution ──

    def _resolve_collisions(
        self,
        elements: List[SolvedElement],
    ) -> List[SolvedElement]:
        """Ensure no two root-level elements overlap.

        Simple sweep: for each pair at the same level,
        if bbox overlaps, push the right one further right.
        """
        # Only check root-level elements (children are inside groups)
        root = [e for e in elements if e.parent_group is None]
        others = [e for e in elements if e.parent_group is not None]

        # Sort by (y, x) for sweep
        root.sort(key=lambda e: (e.y, e.x))

        for i in range(len(root)):
            for j in range(i + 1, len(root)):
                a, b = root[i], root[j]
                # Check overlap
                if (a.x < b.x + b.width and a.x + a.width > b.x
                        and a.y < b.y + b.height and a.y + a.height > b.y):
                    # Push b to the right of a
                    b.x = a.x + a.width + 40

        return root + others

    # ── Edge building ──

    def _build_edges(
        self,
        canonical_edges: List[CanonicalEdge],
        solved: List[SolvedElement],
        groups: List[CanonicalGroup],
    ) -> List[Dict[str, Any]]:
        """Build ELK-format edges from canonical edges.

        Cross-group edges go in root edges list.
        Intra-group edges are handled in group.edges.
        """
        solved_map = {e.id: e for e in solved}
        group_children: Set[str] = set()
        for g in groups:
            group_children.update(g.children_ids)

        root_edges = []
        for edge in canonical_edges:
            src_in_group = edge.source in group_children
            tgt_in_group = edge.target in group_children

            # Skip intra-group edges (handled by group.edges)
            if src_in_group and tgt_in_group:
                # Find if they're in the SAME group
                src_group = None
                tgt_group = None
                for g in groups:
                    if edge.source in g.children_ids:
                        src_group = g.id
                    if edge.target in g.children_ids:
                        tgt_group = g.id
                if src_group == tgt_group:
                    continue  # intra-group, already handled

            root_edges.append(self._make_elk_edge(edge))

        return root_edges

    def _make_elk_edge(self, edge: CanonicalEdge) -> Dict[str, Any]:
        """Convert canonical edge to ELK format."""
        elk_edge: Dict[str, Any] = {
            "id": edge.id,
            "sources": [edge.source],
            "targets": [edge.target],
        }
        adv: Dict[str, Any] = {
            "semanticType": edge.edge_type,
        }
        if edge.label:
            adv["edgeLabels"] = [{"text": edge.label}]
        if edge.line_style != "solid":
            adv["lineStyle"] = edge.line_style
        if edge.stroke_color:
            adv["strokeColor"] = edge.stroke_color
        adv["strokeWidth"] = edge.stroke_width
        elk_edge["advanced"] = adv
        return elk_edge


# ═══════════════════════════════════════════════════════════════════════════
#  §4  Convenience: one-shot solve from raw LLM output
# ═══════════════════════════════════════════════════════════════════════════

def solve_from_raw(
    entities: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    groups: List[Dict[str, Any]],
    canvas_width: int = 900,
    canvas_height: int = 500,
) -> SolvedLayout:
    """One-shot: raw LLM output → deterministic layout.

    This is the function that replaces assemble_elk().
    The LLM decides WHAT. This function decides WHERE and HOW BIG.
    """
    from backend.pipeline.topology.constraint.canonicalizer import LayoutCanonicalizer

    registry = ConstraintRegistry()
    canon = LayoutCanonicalizer(registry)

    # Canonicalize
    c_elements = canon.canonicalize_entities(entities)
    c_element_ids = {e.id for e in c_elements}
    c_edges = canon.canonicalize_edges(edges, c_element_ids)
    c_groups = canon.canonicalize_groups(groups, c_elements)

    # Solve
    solver = ConstraintSolver(registry, canvas_width, canvas_height)
    return solver.solve(c_elements, c_edges, c_groups)
