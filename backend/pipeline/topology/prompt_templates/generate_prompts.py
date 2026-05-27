"""generate_prompts.py — Per-region generation prompts.

Like CCCL's kernel-specific tuning: the same DeviceTopKKernel runs
with different extract_bin_op and identify_candidates_op depending
on the pass number.  Our generation prompts are the same kernel code
with different "operators" (type supplements) plugged in.
"""
from backend.pipeline.topology.user_intent_parser import DiagramType


# ═══════════════════════════════════════════════════════════════════════════
#  Base system prompt — shared across all diagram types
# ═══════════════════════════════════════════════════════════════════════════

GEN_SYSTEM_BASE = """\
You are generating ONE REGION of a technical architecture diagram.
You will receive:
  1. The complete system description (for context)
  2. This region's name, bbox, and expected entities
  3. What other regions exist (for cross-region edge awareness)

Output ONLY an ELK JSON subgraph for THIS REGION.

=== OUTPUT FORMAT ===
{{
  "id": "region_id",
  "children": [
    {{"id": "node_id", "width": <int>, "height": <int>,
     "labels": [{{"text": "Label"}}],
     "iconHint": "short description of what icon to show"}}
  ],
  "edges": [
    {{"id": "e1", "sources": ["node_a"], "targets": ["node_b"]}}
  ]
}}

=== ICON HINTS ===
Use iconHint to describe the icon you want in plain language.
The system will search 275k+ open-source icons to find the best match.
Good examples: "database", "CPU chip", "gear settings", "brain neural",
"filter funnel", "code brackets", "cloud server", "lock security".
Keep it 1-3 words. Describe the visual, not the concept.

=== RULES ===
1. EVERY entity in entity_hints MUST appear as a node.
2. Node sizes must fit within the region bbox:
   - Total node area ≤ 70% of region area
   - No node wider than region_width - 40px
   - No node taller than region_height / 2
3. Use compound nodes (children inside children) for sub-structure.
4. Edge sources/targets MUST reference existing node IDs.
5. Cross-region connections: target = "other_region.node_id".
6. Node IDs: snake_case, descriptive, unique within this region.
7. Do NOT include x, y coordinates — the compositor assigns positions.
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Type supplements — plugged into the base system prompt
# ═══════════════════════════════════════════════════════════════════════════

GEN_TYPE_SUPPLEMENTS = {
    DiagramType.ARCHITECTURE: """
=== ARCHITECTURE-SPECIFIC ===
- Use compound nodes: parent modules contain child sub-modules.
- Add "group": true to container nodes.
- Nesting depth ≥ 2 for complex modules.
- Add layoutOptions: {"elk.padding": "[top=35,left=12,bottom=12,right=12]"}
  to compound nodes.
- iconHint examples: "gear" for modules, "chip" for hardware,
  "database" for storage, "code brackets" for code components.
""",

    DiagramType.ENGINEERING_FLOW: """
=== ENGINEERING FLOW-SPECIFIC ===
- Nodes represent tools/actions: "git clone", "tree inspect", "build".
- Edges are sequential: action1 → action2 → action3.
- Include resource nodes: repositories, configs, outputs.
- iconHint examples: "git branch" for git ops, "terminal" for CLI,
  "hammer build" for compilation, "arrow right" for data flow.
""",

    DiagramType.FLOWCHART: """
=== FLOWCHART-SPECIFIC ===
- Linear chain of processing steps.
- Each step has clear input → processing → output.
- Decision points: add "shape": "diamond" to branch nodes.
- Parallel paths: multiple edges from a branch node.
- iconHint examples: "filter funnel", "transform arrows", "merge".
""",

    DiagramType.RECURSIVE_CHAIN: """
=== RECURSIVE CHAIN-SPECIFIC ===
- Pattern: "from C start, implement D, let E do F".
- Each step is a compound node containing:
  - The new component being implemented
  - The capability it enables
- Show the dependency chain clearly.
- iconHint examples: "gear" for implementations, "arrow" for deps.
""",

    DiagramType.COMPARISON: """
=== COMPARISON-SPECIFIC ===
- Side-by-side layout within this region.
- Matching nodes at the same height for compared items.
- Shared elements appear once with edges to both sides.
- Use similar iconHints for parallel items to show correspondence.
""",

    DiagramType.DATA_FLOW: """
=== DATA FLOW-SPECIFIC ===
- Source → transform → sink pipeline pattern.
- Include buffer/queue nodes between stages.
- Show data format changes (raw → parsed → enriched → stored).
- iconHint examples: "database" for stores, "transform" for processing,
  "queue" for buffers.
""",
}


# ═══════════════════════════════════════════════════════════════════════════
#  User template — shared across all types
# ═══════════════════════════════════════════════════════════════════════════

GEN_USER = """\
=== REGION TO GENERATE ===
Region ID: {region_id}
Region name: {region_name}
Bounding box: {bbox_w}×{bbox_h} pixels
Expected entities: {entity_hints}

=== OTHER REGIONS (for cross-region edge awareness) ===
{other_regions}

=== FULL SYSTEM DESCRIPTION ===
{text}

=== PREVIOUSLY EXTRACTED ENTITIES IN THIS REGION ===
{region_entities}

Generate the ELK subgraph for THIS REGION ONLY:"""
