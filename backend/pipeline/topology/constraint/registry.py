"""registry.py — Element type registry with deterministic size constraints.

This is the core of the white-box approach. Instead of:
    width = len(name) * 10 + 40   # ← black box, no semantic awareness

We use:
    constraint = REGISTRY.get(element.type)
    width = constraint.compute_width(element)  # ← typed rule

Every element type has:
    - Fixed min/max dimensions
    - Padding rules (how much space around text/icon)
    - Icon slot dimensions (if applicable)
    - Label metrics (char width at the target font size)
    - Nesting rules (can this contain children? max depth?)

Analogy to osdk-ts:
    ObjectCacheKey   → ElementConstraint   (per-type identity + rules)
    CacheKeyRegistry → ConstraintRegistry  (type → constraint lookup)
    KnownCacheKey    → ELEMENT_DEFAULTS    (built-in type table)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
#  §1  ElementConstraint — rules for a single element type
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ElementConstraint:
    """Deterministic sizing rules for one element type.

    All dimensions in pixels at the target canvas scale (900×500 default).

    The width/height computation is:
        width  = clamp(label_width + icon_slot + 2*h_pad, min_w, max_w)
        height = clamp(line_count * line_h + 2*v_pad, min_h, max_h)

    Where label_width = len(label) * char_width_px.
    """
    type_name: str

    # ── Size bounds ──
    min_w: int = 120
    max_w: int = 300
    min_h: int = 40
    max_h: int = 80

    # ── Padding ──
    h_pad: int = 16          # horizontal padding (left + right of text)
    v_pad: int = 10          # vertical padding (top + bottom of text)

    # ── Icon slot ──
    has_icon: bool = False
    icon_w: int = 32          # icon width (0 if no icon)
    icon_h: int = 32          # icon height
    icon_gap: int = 8         # gap between icon and label

    # ── Text metrics ──
    char_width_px: float = 7.5   # average character width at 13px font
    line_height_px: int = 18     # line height for multi-line labels
    max_label_chars: int = 30    # truncate labels longer than this

    # ── Nesting ──
    can_contain: bool = False    # can this type have children?
    max_nesting_depth: int = 0   # how deep can children nest?
    child_padding_top: int = 35  # padding above first child (for group title)
    child_padding_sides: int = 12
    child_gap: int = 16          # gap between children

    # ── Visual ──
    border_radius: int = 8
    default_fill: str = ""       # empty = use theme default
    default_stroke: str = ""

    def compute_width(self, label: str) -> int:
        """Deterministic width from label text."""
        text = label[:self.max_label_chars]
        label_w = len(text) * self.char_width_px
        icon_slot = (self.icon_w + self.icon_gap) if self.has_icon else 0
        raw = label_w + icon_slot + 2 * self.h_pad
        return max(self.min_w, min(self.max_w, int(raw)))

    def compute_height(self, line_count: int = 1) -> int:
        """Deterministic height from line count."""
        text_h = line_count * self.line_height_px
        raw = text_h + 2 * self.v_pad
        return max(self.min_h, min(self.max_h, int(raw)))

    def compute_group_size(
        self,
        children_widths: List[int],
        children_heights: List[int],
        direction: str = "DOWN",
    ) -> Tuple[int, int]:
        """Compute group container size from children.

        direction: "DOWN" = stack vertically, "RIGHT" = stack horizontally
        """
        if not children_widths:
            return (self.min_w, self.min_h)

        if direction == "RIGHT":
            total_w = sum(children_widths) + (len(children_widths) - 1) * self.child_gap
            max_h = max(children_heights)
            w = total_w + 2 * self.child_padding_sides
            h = max_h + self.child_padding_top + self.child_padding_sides
        else:  # DOWN
            max_w = max(children_widths)
            total_h = sum(children_heights) + (len(children_heights) - 1) * self.child_gap
            w = max_w + 2 * self.child_padding_sides
            h = total_h + self.child_padding_top + self.child_padding_sides

        return (max(self.min_w, w), max(self.min_h, h))


# ═══════════════════════════════════════════════════════════════════════════
#  §2  GroupConstraint — layout rules for group containers
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class GroupConstraint:
    """Layout direction and child arrangement rules for a group type.

    Analogy to osdk-ts ListQueryOptions — defines how children are
    arranged within a container.
    """
    type_name: str
    direction: str = "DOWN"      # "DOWN" | "RIGHT" | "GRID"
    max_children_per_row: int = 5  # for GRID layout
    align: str = "center"         # "start" | "center" | "end"
    distribute: bool = False      # equal-space children?
    min_gap: int = 16
    # If True, children are arranged in the direction of the parent flow
    follow_parent_flow: bool = False


# ═══════════════════════════════════════════════════════════════════════════
#  §3  ELEMENT_DEFAULTS — the built-in type table
# ═══════════════════════════════════════════════════════════════════════════

# Every element type the system can produce, with its deterministic rules.
# This is the single source of truth — like osdk's KnownCacheKey enum.

ELEMENT_DEFAULTS: Dict[str, ElementConstraint] = {
    # ── Agents / Major modules ──
    "module": ElementConstraint(
        type_name="module",
        min_w=140, max_w=280, min_h=50, max_h=70,
        has_icon=True, icon_w=28, icon_h=28,
        char_width_px=7.5,
    ),
    # ── Sub-modules (inside a group) ──
    "submodule": ElementConstraint(
        type_name="submodule",
        min_w=120, max_w=220, min_h=40, max_h=55,
        has_icon=True, icon_w=24, icon_h=24,
        char_width_px=7.0,
    ),
    # ── Operations (Join, Filter, Aggregate) ──
    "operation": ElementConstraint(
        type_name="operation",
        min_w=100, max_w=200, min_h=40, max_h=55,
        has_icon=True, icon_w=24, icon_h=24,
        char_width_px=7.0,
        default_fill="#FFF3E0",
    ),
    # ── Data objects (Table, Index, Column, Code) ──
    "data_object": ElementConstraint(
        type_name="data_object",
        min_w=90, max_w=180, min_h=36, max_h=50,
        has_icon=True, icon_w=20, icon_h=20,
        char_width_px=7.0,
    ),
    "data_store": ElementConstraint(
        type_name="data_store",
        min_w=90, max_w=180, min_h=36, max_h=50,
        has_icon=True, icon_w=20, icon_h=20,
        char_width_px=7.0,
    ),
    # ── Resources (CPU, GPU, RAM, SSD, HDD) ──
    "resource": ElementConstraint(
        type_name="resource",
        min_w=70, max_w=120, min_h=50, max_h=65,
        has_icon=True, icon_w=32, icon_h=32, icon_gap=4,
        h_pad=8, v_pad=6,
        char_width_px=7.0,
    ),
    # ── I/O ports ──
    "input": ElementConstraint(
        type_name="input",
        min_w=80, max_w=180, min_h=36, max_h=50,
        has_icon=True, icon_w=20, icon_h=20,
        char_width_px=7.0,
    ),
    "output": ElementConstraint(
        type_name="output",
        min_w=80, max_w=180, min_h=36, max_h=50,
        has_icon=True, icon_w=20, icon_h=20,
        char_width_px=7.0,
    ),
    # ── Standalone icons (small visual elements) ──
    "icon": ElementConstraint(
        type_name="icon",
        min_w=40, max_w=60, min_h=40, max_h=60,
        has_icon=True, icon_w=32, icon_h=32,
        h_pad=4, v_pad=4,
        char_width_px=0,  # icons have no text
    ),
    # ── Labels (text-only annotations) ──
    "label": ElementConstraint(
        type_name="label",
        min_w=40, max_w=200, min_h=20, max_h=30,
        has_icon=False,
        h_pad=4, v_pad=2,
        char_width_px=6.5,
    ),
    "annotation": ElementConstraint(
        type_name="annotation",
        min_w=30, max_w=80, min_h=20, max_h=30,
        has_icon=False,
        h_pad=4, v_pad=2,
        char_width_px=6.5,
    ),
    # ── Group containers ──
    "group_container": ElementConstraint(
        type_name="group_container",
        min_w=200, max_w=800, min_h=100, max_h=600,
        can_contain=True, max_nesting_depth=4,
        child_padding_top=35, child_padding_sides=12, child_gap=16,
        border_radius=12,
    ),
    "panel": ElementConstraint(
        type_name="panel",
        min_w=150, max_w=600, min_h=80, max_h=500,
        can_contain=True, max_nesting_depth=3,
        child_padding_top=30, child_padding_sides=10, child_gap=12,
        border_radius=8,
    ),
}


GROUP_DEFAULTS: Dict[str, GroupConstraint] = {
    # GenDB-style: agent tool set = horizontal row
    "agent_row": GroupConstraint(
        type_name="agent_row", direction="RIGHT",
        max_children_per_row=6, align="center", min_gap=20,
    ),
    # Input panel: vertical stack of input items
    "input_panel": GroupConstraint(
        type_name="input_panel", direction="DOWN",
        align="start", min_gap=8,
    ),
    # Output panel: vertical stack of output items
    "output_panel": GroupConstraint(
        type_name="output_panel", direction="DOWN",
        align="start", min_gap=8,
    ),
    # Resource panel: grid of hardware icons
    "resource_grid": GroupConstraint(
        type_name="resource_grid", direction="GRID",
        max_children_per_row=3, align="center", min_gap=10,
    ),
    # Default: vertical stack
    "default": GroupConstraint(
        type_name="default", direction="DOWN",
        align="center", min_gap=16,
    ),
    # Horizontal flow (for pipeline stages)
    "pipeline": GroupConstraint(
        type_name="pipeline", direction="RIGHT",
        align="center", min_gap=24,
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
#  §4  ConstraintRegistry — the lookup table
# ═══════════════════════════════════════════════════════════════════════════

class ConstraintRegistry:
    """Type → Constraint lookup with fallback defaults.

    Analogy to osdk-ts ObjectCacheKeyRegistry:
        registry.get("module") → ElementConstraint with fixed rules
        registry.get("unknown_type") → fallback to "module" defaults
    """

    def __init__(
        self,
        elements: Optional[Dict[str, ElementConstraint]] = None,
        groups: Optional[Dict[str, GroupConstraint]] = None,
    ):
        self._elements = dict(elements or ELEMENT_DEFAULTS)
        self._groups = dict(groups or GROUP_DEFAULTS)

    def get_element(self, type_name: str) -> ElementConstraint:
        """Get constraint for element type, with fallback."""
        return self._elements.get(type_name, self._elements["module"])

    def get_group(self, type_name: str) -> GroupConstraint:
        """Get constraint for group type, with fallback."""
        return self._groups.get(type_name, self._groups["default"])

    def register_element(self, constraint: ElementConstraint) -> None:
        """Add or override an element type constraint."""
        self._elements[constraint.type_name] = constraint

    def register_group(self, constraint: GroupConstraint) -> None:
        """Add or override a group type constraint."""
        self._groups[constraint.type_name] = constraint

    def classify_group_type(self, group_label: str, children_types: List[str]) -> str:
        """Infer group layout type from label and children.

        Rules (deterministic, no LLM):
            - All children are "resource" → "resource_grid"
            - All children are "input" → "input_panel"
            - All children are "output" → "output_panel"
            - Label contains "agent"/"tool" → "agent_row"
            - Label contains "pipeline"/"flow" → "pipeline"
            - Default → "default" (vertical stack)
        """
        label_lower = group_label.lower()

        # Count type frequencies
        type_counts: Dict[str, int] = {}
        for t in children_types:
            type_counts[t] = type_counts.get(t, 0) + 1

        dominant_type = max(type_counts, key=type_counts.get) if type_counts else ""

        if dominant_type == "resource":
            return "resource_grid"
        if dominant_type == "input":
            return "input_panel"
        if dominant_type == "output":
            return "output_panel"
        if any(kw in label_lower for kw in ("agent", "tool set", "toolset")):
            return "agent_row"
        if any(kw in label_lower for kw in ("pipeline", "flow", "chain", "sequence")):
            return "pipeline"
        return "default"

    @property
    def element_types(self) -> List[str]:
        return list(self._elements.keys())

    @property
    def group_types(self) -> List[str]:
        return list(self._groups.keys())
