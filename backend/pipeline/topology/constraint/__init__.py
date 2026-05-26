"""constraint/ — Deterministic layout constraint system.

Replaces black-box LLM sizing/positioning with typed, validated rules.
Follows the osdk-ts Canonicalizer → Validator → Helper → Store pattern.

The LLM decides WHAT to draw (entities, relationships).
The constraint system decides WHERE and HOW BIG — deterministically.
"""
from backend.pipeline.topology.constraint.registry import (
    ConstraintRegistry, ElementConstraint, GroupConstraint,
    ELEMENT_DEFAULTS, GROUP_DEFAULTS,
)
from backend.pipeline.topology.constraint.canonicalizer import (
    LayoutCanonicalizer, CanonicalElement, CanonicalEdge, CanonicalGroup,
)
from backend.pipeline.topology.constraint.solver import (
    ConstraintSolver, SolvedLayout, SolvedElement,
)
