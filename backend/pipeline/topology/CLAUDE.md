# Topology Generation Pipeline

## Basic Pipeline (generate_rich_topology)
Single LLM call produces ~8 nodes. Actual figure needs ~50+.
Fix: 5-step pipeline with schema validation gates.

Step 1: entity_extractor    -> 30+ entities (reject if <20)
Step 2: relationship_mapper -> edges (reject if <15)
Step 3: hierarchy_builder   -> groups (every entity assigned)
Step 4: icon_classifier     -> iconHints (rule-based, no API)
Step 5: elk_assembler       -> complete ELK JSON

Each step retries up to 3x. Never silently drops content.

## Mastergo Pipeline (generate_mastergo_topology) — NEW
Targets mastergo_all_layoutobj.txt quality: 50+ elements with pixel-precise bboxes.

Step 0: complexity_estimator -> adaptive min_entities/min_edges thresholds
Step 1: dense_extractor      -> two-pass: modules (Pass1) + sub-elements (Pass2)
Step 2: implicit_inject      -> add hardware icons, data objects LLM missed
Step 3: relationship_mapper  -> edges with adaptive thresholds
Step 4: hierarchy_builder    -> 3+ nesting levels (not flat)
Step 5: icon_classifier      -> visual_hint from dense extractor preferred
Step 6: elk_assembler        -> compound nodes with sub-element children
Step 7: vision_constraint    -> align ELK positions to screenshot CCL bboxes
Step 8: mastergo_convert     -> MastergoLayout with full element taxonomy

New modules:
  - dense_extractor.py     — MapReduce-style two-pass entity extraction
  - mastergo_schema.py     — Target output schema + ELK converter
  - vision_constraint.py   — Screenshot CCL + vision LLM alignment

## Constrained Pipeline (generate_constrained_topology) — WHITE BOX
The LLM decides WHAT. Rules decide WHERE and HOW BIG. No LLM in sizing.

Steps 1-3 use LLM (entity extraction, relationships, hierarchy).
Steps 4-5 are PURE DETERMINISTIC RULES — no LLM calls:

Step 4: LayoutCanonicalizer  -> normalize types + compute sizes from registry
Step 5: ConstraintSolver     -> compute positions from rules + collision resolve

Constraint system (constraint/ directory, osdk-ts pattern):
  registry.py       — ElementConstraint per type (min/max w/h, padding, icon slot)
                       GroupConstraint per layout (direction, gap, align)
                       ConstraintRegistry: type → constraint lookup
  canonicalizer.py  — Raw LLM output → CanonicalElement/Edge/Group
                       Deterministic ID, type resolution, size computation
  solver.py         — CanonicalElements → SolvedLayout with absolute positions
                       Layer 1: group sizes from children (bottom-up)
                       Layer 2: position assignment (topological layers)
                       Layer 3: collision resolution

Key insight: assemble_elk() uses `len(name)*10+40` for width — black box.
             ConstraintSolver uses typed rules: resource=70-120px, module=140-280px.
             Every size is traceable to a rule in registry.py.
