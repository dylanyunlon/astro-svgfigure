"""plan_prompts.py — Region planning prompts per diagram type.

Like CCCL's per-architecture policies: same histogram kernel,
different occupancy and block size tuning.

Each diagram type gets a system prompt optimized for its spatial
structure.  The user template is shared across all types.
"""
from backend.pipeline.topology.user_intent_parser import DiagramType


# ═══════════════════════════════════════════════════════════════════════════
#  Base rules shared by all diagram types
# ═══════════════════════════════════════════════════════════════════════════

_PLAN_BASE = """\
You are a spatial layout planner for technical diagrams.
Given a system description and canvas dimensions, divide the canvas
into non-overlapping rectangular regions.

Each region is a logical section rendered independently (like layers
in MasterGo/Figma).

Output ONLY a JSON array.  Each element:
{{
  "id": "snake_case_id",
  "name": "Human Label",
  "bbox": {{"x": <int>, "y": <int>, "width": <int>, "height": <int>}},
  "description": "What this region contains",
  "entity_hints": ["entity1", "entity2"],
  "style": {{"color_family": "blue|green|purple|amber|teal|coral"}}
}}

UNIVERSAL RULES:
1. Regions must NOT overlap.  Leave ≥16px gap between them.
2. Every entity in the text must appear in exactly one entity_hints.
3. Larger regions for sections with more entities.
4. Minimum region size: 200×150 pixels.
5. Maximum regions: 8 (merge small sections if needed).
"""

# ═══════════════════════════════════════════════════════════════════════════
#  Per-type specializations
# ═══════════════════════════════════════════════════════════════════════════

_PLAN_ARCHITECTURE = _PLAN_BASE + """
ARCHITECTURE-SPECIFIC:
- Each major module/component/subsystem is a separate region.
- If a module contains sub-modules, the region should be large enough
  to hold nested compound nodes inside.
- Reading order: top-to-bottom for hierarchy, or left-to-right for
  pipeline-style architectures.
- Input/output ports (external data sources, user interfaces) get
  narrow regions on the left/right edges.
- Use color_family to group related modules (same subsystem = same color).
"""

_PLAN_FLOWCHART = _PLAN_BASE + """
FLOWCHART-SPECIFIC:
- Each pipeline stage or processing phase is a region.
- Regions should be arranged sequentially: left-to-right or top-to-bottom.
- Decision branches: place parallel paths in side-by-side regions.
- Input region on the left/top, output region on the right/bottom.
- Keep regions roughly equal width for visual balance.
"""

_PLAN_ENGINEERING = _PLAN_BASE + """
ENGINEERING FLOW-SPECIFIC:
- Divide by workflow phases: setup → build → test → deploy.
- Each phase region contains the tools, commands, and artifacts
  involved in that phase.
- Timeline flows left-to-right (horizontal layout preferred).
- Resource/environment region at the top or bottom (configs, repos).
- Output/artifact region on the right edge.
"""

_PLAN_COMPARISON = _PLAN_BASE + """
COMPARISON-SPECIFIC:
- One region per compared item, arranged side-by-side.
- Equal-width regions for fair visual comparison.
- Shared elements (if any) get a separate region above or below.
- Reading order: left-to-right matching the comparison order.
"""

_PLAN_RECURSIVE = _PLAN_BASE + """
RECURSIVE CHAIN-SPECIFIC:
- The "seed" example gets one region at the top or left.
- Each derived implementation step gets its own region.
- Regions arranged to show the dependency chain clearly:
  seed → step 1 → step 2 → ... (cascade layout).
- Each region should show what was built and what it enables.
"""

_PLAN_DATA_FLOW = _PLAN_BASE + """
DATA FLOW-SPECIFIC:
- Sources (data ingestion) on the left, sinks (output) on the right.
- Processing stages flow left-to-right.
- Storage/buffer regions between processing stages.
- Keep the pipeline linear; branches get side-by-side sub-regions.
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Lookup table — the PolicySelector dispatch
# ═══════════════════════════════════════════════════════════════════════════

PLAN_SYSTEMS = {
    DiagramType.ARCHITECTURE:    _PLAN_ARCHITECTURE,
    DiagramType.FLOWCHART:       _PLAN_FLOWCHART,
    DiagramType.ENGINEERING_FLOW: _PLAN_ENGINEERING,
    DiagramType.COMPARISON:      _PLAN_COMPARISON,
    DiagramType.RECURSIVE_CHAIN: _PLAN_RECURSIVE,
    DiagramType.DATA_FLOW:       _PLAN_DATA_FLOW,
    DiagramType.UNKNOWN:         _PLAN_BASE,
}

PLAN_USER = """\
Canvas size: {width} × {height} pixels
Diagram type: {diagram_type}
Estimated entities: {entity_count}

System description (FULL — read every word):
---
{text}
---

{intent_summary}

Output ONLY the JSON array of regions:"""
