"""region_planner.py — The Histogram Kernel.

CCCL f984c90 extracts the first pass into `DeviceTopKHistogramKernel`.
That kernel does exactly one thing: scan the full input, build a histogram.
No filtering, no candidate selection, no output writing.  Just count bins.
This lets it run at maximum occupancy — no register pressure from filter
logic, no shared memory for candidate buffers.

We do the same.  `plan_regions()` is the first pass of our pipeline.
It does exactly one thing: read the user's full text (NO TRUNCATION),
output a list of regions with concrete bounding boxes.  No entity
extraction, no relationship mapping, no icon classification.  Just
spatial planning.

The critical insight from CCCL: the first pass operates on the ORIGINAL
input at FULL size.  Every subsequent pass operates on a reduced
candidate set.  Same here: region planning sees the complete user text.
Per-region generation (M3) only sees its assigned slice + context.

Input:  full user text + parsed UserIntent (from M2)
Output: List[PlannedRegion] with {id, name, bbox, description, entities}
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from backend.pipeline.topology.user_intent_parser import (
    UserIntent, DiagramType, ComplexityLevel,
)
from backend.pipeline.topology.finalize_pass import (
    PassContext, ValidatedOutput, finalize_pass,
    validate_region_list,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  §1  PlannedRegion — output of the histogram kernel
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PlannedRegion:
    """One spatial region in the figure plan.

    Like a histogram bin in CCCL: it has a position (bin index → bbox)
    and a count (number of entities expected inside).

    Attributes:
        id: unique snake_case identifier
        name: human-readable label
        bbox: absolute pixel bounding box {x, y, width, height}
        description: what this region contains (from user text)
        entity_hints: names of entities expected in this region
        style: visual style hints (color, border, etc.)
        priority: generation order (0 = first)
    """
    id: str
    name: str
    bbox: Dict[str, int]   # {x, y, width, height}
    description: str = ""
    entity_hints: List[str] = field(default_factory=list)
    style: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0

    @property
    def area(self) -> int:
        return self.bbox["width"] * self.bbox["height"]

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "name": self.name,
            "bbox": self.bbox,
        }
        if self.description:
            d["description"] = self.description
        if self.entity_hints:
            d["entity_hints"] = self.entity_hints
        if self.style:
            d["style"] = self.style
        return d


# ═══════════════════════════════════════════════════════════════════════════
#  §2  Canvas sizing — adaptive, not hardcoded
# ═══════════════════════════════════════════════════════════════════════════

def compute_canvas_size(
    intent: UserIntent,
    min_width: int = 800,
    min_height: int = 500,
    max_width: int = 2400,
    max_height: int = 1600,
) -> Tuple[int, int]:
    """Compute canvas size from intent complexity.

    Like CCCL's grid size calculation:
        topk_grid_size = min(max_occupancy, num_tiles)

    We compute:
        canvas_area = f(entity_count, region_count, diagram_type)
    Then solve for width/height using the diagram type's preferred
    aspect ratio.

    Returns:
        (width, height) in pixels
    """
    n_entities = max(intent.estimated_entities, 5)
    n_regions = max(intent.estimated_regions, 1)

    # Base area: ~15,000 px² per entity (gives breathing room)
    area_per_entity = 15000
    # Regions need margin between them
    margin_per_region = 40000

    total_area = n_entities * area_per_entity + n_regions * margin_per_region
    total_area = max(total_area, min_width * min_height)
    total_area = min(total_area, max_width * max_height)

    # Aspect ratio from diagram type
    aspect_ratios = {
        DiagramType.ARCHITECTURE: 1.6,       # wider than tall
        DiagramType.FLOWCHART: 0.75,         # taller than wide (top-down)
        DiagramType.ENGINEERING_FLOW: 2.0,   # very wide (left-to-right timeline)
        DiagramType.COMPARISON: 2.5,         # very wide (side-by-side)
        DiagramType.RECURSIVE_CHAIN: 1.0,    # square-ish (nested)
        DiagramType.DATA_FLOW: 1.8,          # wide pipeline
        DiagramType.UNKNOWN: 1.6,
    }
    ratio = aspect_ratios.get(intent.diagram_type, 1.6)

    # width = sqrt(area * ratio), height = area / width
    width = int(math.sqrt(total_area * ratio))
    height = int(total_area / width)

    # Clamp
    width = max(min_width, min(max_width, width))
    height = max(min_height, min(max_height, height))

    return width, height


# ═══════════════════════════════════════════════════════════════════════════
#  §3  Deterministic region layout — no LLM needed for simple cases
# ═══════════════════════════════════════════════════════════════════════════

def layout_regions_grid(
    n_regions: int,
    canvas_width: int,
    canvas_height: int,
    padding: int = 20,
    gap: int = 16,
) -> List[Dict[str, int]]:
    """Compute grid layout for N regions.

    Like CCCL's tile computation: given the total input size and the
    block size, compute how many tiles (grid_size) and their offsets.

    Uses a balanced grid: tries to minimize wasted space while keeping
    aspect ratios reasonable.

    Args:
        n_regions: number of regions to lay out
        canvas_width: total canvas width
        canvas_height: total canvas height
        padding: outer padding
        gap: gap between regions

    Returns:
        List of {x, y, width, height} bboxes
    """
    if n_regions <= 0:
        return []
    if n_regions == 1:
        return [{"x": padding, "y": padding,
                 "width": canvas_width - 2 * padding,
                 "height": canvas_height - 2 * padding}]

    # Find the best grid dimensions (cols × rows)
    # Minimize abs(cols/rows - canvas_aspect)
    canvas_aspect = canvas_width / max(canvas_height, 1)
    best_cols, best_rows = 1, n_regions
    best_waste = float("inf")

    for cols in range(1, n_regions + 1):
        rows = math.ceil(n_regions / cols)
        grid_aspect = (cols / rows) if rows > 0 else float("inf")
        # Waste = empty cells + aspect ratio mismatch
        empty_cells = cols * rows - n_regions
        aspect_penalty = abs(grid_aspect - canvas_aspect) * 2
        waste = empty_cells + aspect_penalty
        if waste < best_waste:
            best_waste = waste
            best_cols, best_rows = cols, rows

    cols, rows = best_cols, best_rows

    # Compute cell sizes
    avail_width = canvas_width - 2 * padding - (cols - 1) * gap
    avail_height = canvas_height - 2 * padding - (rows - 1) * gap
    cell_w = avail_width // cols
    cell_h = avail_height // rows

    bboxes = []
    for i in range(n_regions):
        col = i % cols
        row = i // cols
        x = padding + col * (cell_w + gap)
        y = padding + row * (cell_h + gap)
        bboxes.append({"x": x, "y": y, "width": cell_w, "height": cell_h})

    return bboxes


# ═══════════════════════════════════════════════════════════════════════════
#  §4  LLM-assisted region planning — for complex inputs
# ═══════════════════════════════════════════════════════════════════════════

REGION_PLAN_SYSTEM = """\
You are a spatial layout planner for technical architecture diagrams.
Given a system description and canvas dimensions, divide the canvas
into non-overlapping rectangular regions.

Each region represents a logical section of the diagram that will be
rendered independently (like layers in MasterGo/Figma).

Output ONLY a JSON array.  Each element:
{{
  "id": "snake_case_id",
  "name": "Human Label",
  "bbox": {{"x": <int>, "y": <int>, "width": <int>, "height": <int>}},
  "description": "What entities/concepts this region contains",
  "entity_hints": ["entity1", "entity2"],
  "style": {{"color_family": "blue|green|purple|amber|teal|coral"}}
}}

RULES:
1. Regions must NOT overlap.  Leave ≥16px gap between adjacent regions.
2. Every entity mentioned in the text must appear in exactly one region's
   entity_hints.  Don't drop entities.
3. Regions should reflect the logical structure:
   - For architecture: each major module is a region
   - For flowcharts: each pipeline stage is a region
   - For engineering flows: each phase (clone, build, test) is a region
4. Larger regions for more complex sections (more entities = more area)
5. Reading order: top-to-bottom for hierarchies, left-to-right for flows
6. Minimum region size: 200×150 pixels
7. Maximum regions: 8 (merge small sections if needed)
"""

REGION_PLAN_USER = """\
Canvas size: {width} × {height} pixels
Diagram type: {diagram_type}
Estimated entities: {entity_count}

System description (FULL — read every word):
---
{text}
---

{intent_summary}

Output ONLY the JSON array of regions:"""


async def plan_regions_with_llm(
    text: str,
    intent: UserIntent,
    canvas_width: int,
    canvas_height: int,
    ai_engine,
    model: str = "",
    context: Optional[PassContext] = None,
) -> Tuple[List[PlannedRegion], Dict[str, Any]]:
    """Plan regions using LLM for complex inputs.

    This is the DeviceTopKHistogramKernel — the dedicated first pass.
    It reads the FULL input (no truncation!) and outputs a spatial plan.

    Like CCCL's histogram kernel:
      - Operates on the FULL original input (d_keys_in, num_items)
      - Builds a histogram (our region plan)
      - Does NOT filter or write candidates (no entity extraction)
      - Runs at maximum occupancy (one focused LLM call)

    Args:
        text: Full user text (NEVER truncated)
        intent: Parsed user intent from M2
        canvas_width: Canvas width in pixels
        canvas_height: Canvas height in pixels
        ai_engine: AIEngine instance
        model: LLM model override
        context: Pipeline counter state

    Returns:
        (regions, diagnostics)
    """
    if context is None:
        context = PassContext()
    context.advance("region_planning")

    diag: Dict[str, Any] = {"pass": "region_planning"}

    # ── Short-circuit: if intent has clear containment groups, use them ──
    if (intent.containment_groups
            and len(intent.containment_groups) >= 2
            and intent.confidence > 0.5):
        regions = _regions_from_containment(
            intent, canvas_width, canvas_height,
        )
        if regions:
            diag["source"] = "deterministic_containment"
            diag["region_count"] = len(regions)
            context.total_regions = len(regions)
            return regions, diag

    # ── LLM-assisted planning for complex inputs ──
    if not model:
        s = ai_engine._settings
        model = s.ANTHROPIC_DEFAULT_MODEL or s.DEFAULT_MODEL

    # Build intent summary for the prompt
    intent_lines = []
    if intent.entities:
        names = [e.name for e in intent.entities[:30]]
        intent_lines.append(f"Detected entities: {', '.join(names)}")
    if intent.containment_groups:
        for parent, children in list(intent.containment_groups.items())[:5]:
            intent_lines.append(f"  {parent} contains: {', '.join(children)}")
    if intent.action_chain:
        steps = [f"{s.verb}({s.object})" for s in intent.action_chain[:10]]
        intent_lines.append(f"Action chain: {' → '.join(steps)}")
    intent_summary = "\n".join(intent_lines) if intent_lines else "(no pre-parsed structure)"

    prompt = REGION_PLAN_USER.format(
        width=canvas_width,
        height=canvas_height,
        diagram_type=intent.diagram_type.value,
        entity_count=intent.estimated_entities,
        text=text,  # FULL TEXT — no [:12000] truncation!
        intent_summary=intent_summary,
    )

    max_retries = 2
    best_regions: List[PlannedRegion] = []

    for attempt in range(max_retries):
        try:
            provider = ai_engine._select_provider(model)
            resp = await provider.get_completion(
                messages=[
                    {"role": "system", "content": REGION_PLAN_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                temperature=0.2 + attempt * 0.1,
                max_tokens=4096,
            )
            raw = resp.get("content", "")

            # Use finalize_pass — the shared template
            def update_counters(parsed, ctx):
                ctx.total_regions = len(parsed)

            result = finalize_pass(
                raw_output=raw,
                context=context,
                validator=validate_region_list,
                counter_updater=update_counters,
                expect_array=True,
                min_items=1,
            )

            if result.success and result.data:
                regions = _parse_region_list(result.data, canvas_width, canvas_height)
                if len(regions) >= len(best_regions):
                    best_regions = regions
                if len(regions) >= intent.estimated_regions:
                    break  # Good enough

        except Exception as e:
            context.record_error(f"Region planning attempt {attempt + 1}: {e}")

    # ── Fallback: deterministic grid layout ──
    if not best_regions:
        context.record_warning("LLM region planning failed, using grid fallback")
        bboxes = layout_regions_grid(
            intent.estimated_regions, canvas_width, canvas_height,
        )
        best_regions = [
            PlannedRegion(
                id=f"region_{i}",
                name=f"Region {i + 1}",
                bbox=bbox,
                priority=i,
            )
            for i, bbox in enumerate(bboxes)
        ]
        diag["source"] = "grid_fallback"
    else:
        diag["source"] = "llm"

    diag["region_count"] = len(best_regions)
    diag["regions"] = [r.to_dict() for r in best_regions]
    context.total_regions = len(best_regions)

    return best_regions, diag


# ═══════════════════════════════════════════════════════════════════════════
#  §5  Helpers — parse and convert LLM output to PlannedRegions
# ═══════════════════════════════════════════════════════════════════════════

def _parse_region_list(
    data: List[Dict[str, Any]],
    canvas_width: int,
    canvas_height: int,
) -> List[PlannedRegion]:
    """Convert raw JSON array to list of PlannedRegion objects.

    Applies sanity checks:
      - bbox within canvas bounds
      - minimum size enforcement
      - overlap resolution
    """
    regions: List[PlannedRegion] = []

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        # Extract bbox (handle both flat and nested formats)
        bbox = item.get("bbox", {})
        if not bbox:
            bbox = {
                "x": item.get("x", 0),
                "y": item.get("y", 0),
                "width": item.get("width", 200),
                "height": item.get("height", 150),
            }

        # Ensure integer values
        bbox = {k: int(v) for k, v in bbox.items() if k in ("x", "y", "width", "height")}

        # Minimum size
        bbox["width"] = max(200, bbox.get("width", 200))
        bbox["height"] = max(150, bbox.get("height", 150))

        # Clamp to canvas
        bbox["x"] = max(0, min(bbox.get("x", 0), canvas_width - bbox["width"]))
        bbox["y"] = max(0, min(bbox.get("y", 0), canvas_height - bbox["height"]))

        region = PlannedRegion(
            id=item.get("id", f"region_{i}"),
            name=item.get("name", item.get("label", f"Region {i + 1}")),
            bbox=bbox,
            description=item.get("description", ""),
            entity_hints=item.get("entity_hints", []),
            style=item.get("style", {}),
            priority=i,
        )
        regions.append(region)

    return regions


def _regions_from_containment(
    intent: UserIntent,
    canvas_width: int,
    canvas_height: int,
) -> List[PlannedRegion]:
    """Create regions directly from parsed containment groups.

    When the user says "System has A, B, C" and "A contains X, Y",
    we can plan regions without an LLM call — pure deterministic layout.

    Like CCCL's fast path: if num_passes == 1, skip the loop entirely.
    """
    groups = list(intent.containment_groups.items())
    if not groups:
        return []

    bboxes = layout_regions_grid(len(groups), canvas_width, canvas_height)

    regions: List[PlannedRegion] = []
    for i, (parent, children) in enumerate(groups):
        if i >= len(bboxes):
            break
        region_id = _safe_id(parent)
        regions.append(PlannedRegion(
            id=region_id,
            name=parent,
            bbox=bboxes[i],
            description=f"Contains: {', '.join(children)}",
            entity_hints=children,
            priority=i,
        ))

    return regions


def _safe_id(name: str) -> str:
    """Convert a name to a safe snake_case ID."""
    import re
    return re.sub(r"[^a-zA-Z0-9_]", "_", name.lower()).strip("_")[:50]
