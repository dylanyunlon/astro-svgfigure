"""canonicalizer.py — Normalize LLM output into canonical typed form.

The LLM outputs free-form JSON with inconsistent naming, missing types,
and no size information. The canonicalizer:

  1. Normalizes names → deterministic snake_case IDs
  2. Resolves types → canonical element types from the registry
  3. Applies constraints → computes width/height from rules, not LLM
  4. Deduplicates → merges near-identical entities

Analogy to osdk-ts:
    Canonicalizer.ts         → LayoutCanonicalizer
    GenericCanonicalizer.ts  → type-resolution logic
    IntersectCanonicalizer   → dedup/merge logic
    SelectCanonicalizer      → field selection/normalization

The LLM decides WHAT. The canonicalizer decides the CANONICAL FORM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.pipeline.topology.constraint.registry import (
    ConstraintRegistry, ElementConstraint, GroupConstraint,
    ELEMENT_DEFAULTS,
)


# ═══════════════════════════════════════════════════════════════════════════
#  §1  Canonical data types (the clean output)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CanonicalElement:
    """One element after canonicalization.

    Every field is deterministic — computed from rules, not LLM guesses.
    """
    id: str                    # deterministic snake_case
    name: str                  # display label (from LLM, but truncated)
    type: str                  # canonical type from registry
    width: int                 # computed by ElementConstraint.compute_width
    height: int                # computed by ElementConstraint.compute_height
    parent_group: Optional[str] = None  # id of containing group
    depth: int = 0             # nesting depth (0 = root level)
    icon_hint: str = ""        # natural language icon description
    visual_hint: str = ""      # from dense extractor
    # Source metadata (for debugging, not used in layout)
    _raw_type: str = ""        # original type string from LLM
    _raw_name: str = ""        # original name before canonicalization


@dataclass
class CanonicalEdge:
    """One edge after canonicalization."""
    id: str
    source: str                # canonical element id
    target: str                # canonical element id
    label: str = ""
    edge_type: str = "data_flow"   # data_flow|feedback|contains|uses
    # Visual style (deterministic from edge_type)
    line_style: str = "solid"
    stroke_color: str = ""
    stroke_width: float = 1.5


@dataclass
class CanonicalGroup:
    """One group container after canonicalization."""
    id: str
    label: str
    children_ids: List[str] = field(default_factory=list)
    layout_type: str = "default"   # from GroupConstraint classification
    direction: str = "DOWN"
    depth: int = 0
    parent_group: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════
#  §2  LayoutCanonicalizer — the main transformer
# ═══════════════════════════════════════════════════════════════════════════

class LayoutCanonicalizer:
    """Transforms raw LLM output into canonical form with deterministic sizes.

    Usage:
        canon = LayoutCanonicalizer(registry)
        elements = canon.canonicalize_entities(raw_entities)
        edges = canon.canonicalize_edges(raw_edges, element_ids)
        groups = canon.canonicalize_groups(raw_groups, element_ids)
    """

    def __init__(self, registry: Optional[ConstraintRegistry] = None):
        self._registry = registry or ConstraintRegistry()
        self._id_counter: Dict[str, int] = {}
        self._seen_ids: Set[str] = set()

    # ── Public API ──

    def canonicalize_entities(
        self,
        raw_entities: List[Dict[str, Any]],
    ) -> List[CanonicalElement]:
        """Transform raw LLM entities into canonical form with deterministic sizes."""
        elements = []
        for raw in raw_entities:
            elem = self._canonicalize_one_entity(raw)
            if elem:
                elements.append(elem)
        return elements

    def canonicalize_edges(
        self,
        raw_edges: List[Dict[str, Any]],
        valid_ids: Set[str],
    ) -> List[CanonicalEdge]:
        """Transform raw edges, filtering to valid element IDs."""
        edges = []
        seen_pairs: Set[Tuple[str, str]] = set()

        for raw in raw_edges:
            src = self._make_id(raw.get("source", ""))
            tgt = self._make_id(raw.get("target", ""))

            if src not in valid_ids or tgt not in valid_ids:
                continue
            if src == tgt:
                continue
            if (src, tgt) in seen_pairs:
                continue
            seen_pairs.add((src, tgt))

            edge_type = raw.get("type", "data_flow")
            line_style, stroke_color = _edge_style(edge_type)

            edges.append(CanonicalEdge(
                id=f"e_{src}_{tgt}",
                source=src,
                target=tgt,
                label=str(raw.get("label", ""))[:40],
                edge_type=edge_type,
                line_style=line_style,
                stroke_color=stroke_color,
            ))

        return edges

    def canonicalize_groups(
        self,
        raw_groups: List[Dict[str, Any]],
        elements: List[CanonicalElement],
    ) -> List[CanonicalGroup]:
        """Transform raw groups with deterministic layout classification."""
        element_map = {e.id: e for e in elements}
        groups = []

        for raw in raw_groups:
            group_id = self._make_id(raw.get("name", "group"))
            label = str(raw.get("label", raw.get("name", "")))[:40]

            # Map children names to canonical IDs
            children_raw = raw.get("children", [])
            children_ids = []
            children_types = []
            for child_name in children_raw:
                child_id = self._make_id(child_name)
                if child_id in element_map:
                    children_ids.append(child_id)
                    children_types.append(element_map[child_id].type)

            # Deterministic layout classification
            layout_type = self._registry.classify_group_type(label, children_types)
            group_constraint = self._registry.get_group(layout_type)

            groups.append(CanonicalGroup(
                id=group_id,
                label=label,
                children_ids=children_ids,
                layout_type=layout_type,
                direction=group_constraint.direction,
            ))

            # Update element parent references
            for child_id in children_ids:
                if child_id in element_map:
                    element_map[child_id].parent_group = group_id

        return groups

    # ── Internal methods ──

    def _canonicalize_one_entity(self, raw: Dict[str, Any]) -> Optional[CanonicalElement]:
        """Transform one raw entity dict into CanonicalElement."""
        raw_name = str(raw.get("name", "")).strip()
        if not raw_name:
            return None

        canonical_id = self._make_unique_id(raw_name)
        raw_type = str(raw.get("type", "module")).lower()

        # Resolve to canonical type
        canonical_type = _resolve_type(raw_type, raw_name)

        # Get constraint for this type
        constraint = self._registry.get_element(canonical_type)

        # Compute deterministic size
        display_name = raw_name[:constraint.max_label_chars]
        width = constraint.compute_width(display_name)
        height = constraint.compute_height(line_count=1)

        # Icon hint: prefer explicit, then from raw, then from type
        icon_hint = str(raw.get("visual_hint", "") or raw.get("icon_hint", ""))

        return CanonicalElement(
            id=canonical_id,
            name=display_name,
            type=canonical_type,
            width=width,
            height=height,
            icon_hint=icon_hint,
            visual_hint=str(raw.get("visual_hint", "")),
            _raw_type=raw_type,
            _raw_name=raw_name,
        )

    def _make_id(self, name: str) -> str:
        """Deterministic name → ID mapping."""
        clean = re.sub(r'[^a-zA-Z0-9_]', '_', name.lower()).strip('_')
        return clean[:50] if clean else "unnamed"

    def _make_unique_id(self, name: str) -> str:
        """Deterministic unique ID — appends counter if collision."""
        base = self._make_id(name)
        if base not in self._seen_ids:
            self._seen_ids.add(base)
            return base
        # Collision: append counter
        counter = self._id_counter.get(base, 1)
        while f"{base}_{counter}" in self._seen_ids:
            counter += 1
        self._id_counter[base] = counter + 1
        unique = f"{base}_{counter}"
        self._seen_ids.add(unique)
        return unique


# ═══════════════════════════════════════════════════════════════════════════
#  §3  Type resolution — deterministic mapping from LLM types to canonical
# ═══════════════════════════════════════════════════════════════════════════

# LLM outputs messy types like "component", "step", "block".
# This table maps them to canonical types deterministically.

_TYPE_ALIASES: Dict[str, str] = {
    # Direct matches
    "module": "module",
    "submodule": "submodule",
    "operation": "operation",
    "data_object": "data_object",
    "data_store": "data_store",
    "resource": "resource",
    "input": "input",
    "output": "output",
    "icon": "icon",
    "label": "label",
    "annotation": "annotation",
    # LLM-generated aliases → canonical
    "component": "module",
    "step": "module",
    "stage": "module",
    "block": "module",
    "agent": "module",
    "processor": "module",
    "analyzer": "module",
    "designer": "module",
    "planner": "module",
    "generator": "module",
    "optimizer": "module",
    "service": "module",
    "engine": "module",
    "system": "module",
    "layer": "submodule",
    "sub_component": "submodule",
    "tool": "submodule",
    "utility": "submodule",
    "function": "operation",
    "transform": "operation",
    "process": "operation",
    "filter": "operation",
    "join": "operation",
    "aggregate": "operation",
    "table": "data_object",
    "index": "data_object",
    "column": "data_object",
    "file": "data_object",
    "document": "data_object",
    "code": "data_object",
    "executable": "data_object",
    "database": "data_store",
    "storage": "data_store",
    "cache": "data_store",
    "queue": "data_store",
    "cpu": "resource",
    "gpu": "resource",
    "ram": "resource",
    "memory": "resource",
    "ssd": "resource",
    "hdd": "resource",
    "hardware": "resource",
    "disk": "resource",
    "config": "input",
    "configuration": "input",
    "parameter": "input",
    "schema": "input",
    "sql": "input",
    "query": "input",
    "result": "output",
    "response": "output",
    "product": "output",
    "artifact": "output",
}


def _resolve_type(raw_type: str, name: str) -> str:
    """Resolve LLM type string to canonical type.

    Strategy (ordered by priority):
        1. Direct alias match
        2. Name-based keyword match (if type is generic like "component")
        3. Fallback to "module"
    """
    # Strategy 1: direct alias
    if raw_type in _TYPE_ALIASES:
        return _TYPE_ALIASES[raw_type]

    # Strategy 2: name-based keywords
    name_lower = name.lower()
    name_keywords = {
        "resource": ("cpu", "gpu", "ram", "ssd", "hdd", "memory", "simd"),
        "data_object": ("table", "index", "column", "code", "executable", ".exe"),
        "operation": ("filter", "join", "aggregate", "scan", "sort", "merge"),
        "input": ("schema", "sql", "config", "data", "workload", "query"),
        "output": ("result", "output", "storage", "index"),
    }
    for canonical, keywords in name_keywords.items():
        if any(kw in name_lower for kw in keywords):
            return canonical

    # Strategy 3: fallback
    return "module"


# ═══════════════════════════════════════════════════════════════════════════
#  §4  Edge style resolution — deterministic from edge type
# ═══════════════════════════════════════════════════════════════════════════

_EDGE_STYLES: Dict[str, Tuple[str, str]] = {
    "data_flow":     ("solid", ""),
    "feedback":      ("dashed", "#9C27B0"),
    "contains":      ("dotted", "#607D8B"),
    "uses":          ("solid", "#009688"),
    "cross_boundary": ("solid", "#009688"),
    "dependency":    ("dashed", "#FF9800"),
}


def _edge_style(edge_type: str) -> Tuple[str, str]:
    """Deterministic edge style from type. Returns (line_style, stroke_color)."""
    return _EDGE_STYLES.get(edge_type, ("solid", ""))
