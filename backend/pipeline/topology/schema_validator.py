"""schema_validator.py - Validation gates for each topology step."""
from __future__ import annotations
from typing import Any, Dict, List, Tuple


def validate_entities(entities: List[Dict], min_count: int = 20) -> Tuple[bool, List[str]]:
    errors = []
    if not isinstance(entities, list):
        return False, ["entities must be a list"]
    if len(entities) < min_count:
        errors.append(f"only {len(entities)} entities, need >= {min_count}")
    names = set()
    for i, e in enumerate(entities):
        if not isinstance(e, dict):
            errors.append(f"entity[{i}] not a dict"); continue
        if not e.get("name"):
            errors.append(f"entity[{i}] missing name")
        if not e.get("type"):
            errors.append(f"entity[{i}] missing type")
        if e.get("name") in names:
            errors.append(f"duplicate: {e.get('name')}")
        names.add(e.get("name"))
    return len(errors) == 0, errors


def validate_edges(edges: List[Dict], entity_names: set, min_count: int = 15) -> Tuple[bool, List[str]]:
    errors = []
    if len(edges) < min_count:
        errors.append(f"only {len(edges)} edges, need >= {min_count}")
    for i, e in enumerate(edges):
        if e.get("source") not in entity_names:
            errors.append(f"edge[{i}] source '{e.get('source')}' unknown")
        if e.get("target") not in entity_names:
            errors.append(f"edge[{i}] target '{e.get('target')}' unknown")
    return len(errors) == 0, errors


def validate_hierarchy(groups: List[Dict], entity_names: set) -> Tuple[bool, List[str]]:
    errors = []
    assigned = set()
    for g in groups:
        for child in g.get("children", []):
            if child in assigned:
                errors.append(f"'{child}' in multiple groups")
            assigned.add(child)
    group_names = {g.get("name") for g in groups}
    unassigned = entity_names - assigned - group_names
    if unassigned:
        errors.append(f"unassigned: {unassigned}")
    return len(errors) == 0, errors


def validate_icons(icons: Dict[str, str], entity_names: set) -> Tuple[bool, List[str]]:
    errors = []
    missing = entity_names - set(icons.keys())
    if missing:
        errors.append(f"missing icons: {missing}")
    return len(errors) == 0, errors


def validate_elk(elk: Dict) -> Tuple[bool, List[str]]:
    errors = []
    if not isinstance(elk, dict):
        return False, ["ELK must be a dict"]
    def count_nodes(node):
        n = 1 if node.get("id") and node["id"] != "root" else 0
        if n and not node.get("labels"):
            errors.append(f"node '{node.get('id')}' missing labels")
        for child in node.get("children", []):
            n += count_nodes(child)
        return n
    def count_edges(node):
        n = len(node.get("edges", []))
        for child in node.get("children", []):
            n += count_edges(child)
        return n
    total_n = count_nodes(elk)
    total_e = count_edges(elk)
    if total_n < 15:
        errors.append(f"only {total_n} nodes, need >= 15")
    if total_e < 10:
        errors.append(f"only {total_e} edges, need >= 10")
    return len(errors) == 0, errors
