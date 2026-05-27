"""per_region_generator.py — The Fused Filter+Histogram Kernel (passes 1..N).

CCCL f984c90's main loop after pass 0:

    int pass = 1;
    for (; pass < num_passes; pass++) {
        extract_bin_op extract_op(pass, total_bits, decomposer);
        identify_candidates_op identify_op(&counter->kth_key_bits, ...);
        launcher.doit(topk_kernel,
            d_keys_in, d_keys_out, d_values_in, d_values_out,
            key_bufs.Current(), idx_bufs.Current(),    // input (read)
            key_bufs.Alternate(), idx_bufs.Alternate(), // output (write)
            counter, histogram, ...);
        key_bufs.selector ^= 1;   // swap buffers
    }

Each pass reads from Current(), writes to Alternate(), then swaps.
The candidate set shrinks each pass: num_items → filter_cnt → smaller filter_cnt.

We do the same.  For each planned region:
  1. Read: full user text + region plan + global entity registry
  2. Write: ELK subgraph with nodes, edges, sizes within the region's bbox
  3. Swap: the generated subgraph becomes the "current" for that region
     (available for refinement in a second pass if needed)

Each region is independent — like CCCL's thread blocks, they don't
communicate during the pass.  Cross-region edges are resolved in M5
(canvas_compositor), just as CCCL's global histogram merge happens
in finalize_pass.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from backend.pipeline.topology.region_planner import PlannedRegion
from backend.pipeline.topology.user_intent_parser import (
    UserIntent, DiagramType,
)
from backend.pipeline.topology.finalize_pass import (
    PassContext, ValidatedOutput, finalize_pass,
    validate_elk_subgraph,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  §1  Prompt templates — per-diagram-type (the PolicySelector)
# ═══════════════════════════════════════════════════════════════════════════

REGION_GEN_SYSTEM = """\
You are generating ONE REGION of a technical architecture diagram.
You will receive:
  1. The complete system description (for context)
  2. This region's name, bbox, and expected entities
  3. What other regions exist (for cross-region edge awareness)

Output ONLY an ELK JSON subgraph for THIS REGION.  The subgraph will be
placed at the region's bbox coordinates by the compositor.

=== OUTPUT FORMAT ===
{{
  "id": "region_id",
  "children": [
    {{"id": "node_id", "width": <int>, "height": <int>,
     "labels": [{{"text": "Label"}}], "iconHint": "description"}}
  ],
  "edges": [
    {{"id": "e1", "sources": ["node_a"], "targets": ["node_b"]}}
  ]
}}

=== RULES ===
1. EVERY entity in the region's entity_hints MUST appear as a node.
2. Node sizes must fit within the region bbox:
   - Total node area ≤ 70% of region area (leave room for edges)
   - No single node wider than region_width - 40px
   - No single node taller than region_height / 2
3. Use compound nodes (children inside children) for sub-structure.
4. Edge sources/targets MUST reference existing node IDs.
5. For cross-region connections: add edges with target IDs prefixed by
   the other region's id (e.g., "other_region.node_id").  The compositor
   will resolve these.
6. Use iconHint for visual elements (natural language icon descriptions).
7. Do NOT include x, y coordinates — the compositor assigns positions.
8. Node IDs: snake_case, descriptive, unique within this region.
"""

# Diagram-type-specific additions to the system prompt
_TYPE_SUPPLEMENTS = {
    DiagramType.ARCHITECTURE: """
=== ARCHITECTURE-SPECIFIC ===
- Use compound nodes extensively: parent modules contain child sub-modules
- Add "group": true, "borderless": true to parent nodes
- Nesting depth ≥ 2 for complex modules
- Add layoutOptions with elk.padding for compound nodes
""",
    DiagramType.ENGINEERING_FLOW: """
=== ENGINEERING FLOW-SPECIFIC ===
- Nodes represent tools/actions: "git clone", "tree inspect", "build"
- Edges are sequential: action1 → action2 → action3
- Include resource nodes: repositories, configs, outputs
- Action nodes should have iconHint matching the tool (e.g., "git logo",
  "terminal window", "file tree")
""",
    DiagramType.FLOWCHART: """
=== FLOWCHART-SPECIFIC ===
- Linear chain of processing steps
- Each step has clear input → processing → output
- Decision points use diamond-shaped nodes (add "shape": "diamond")
- Parallel paths: use multiple edges from a branch node
""",
    DiagramType.RECURSIVE_CHAIN: """
=== RECURSIVE CHAIN-SPECIFIC ===
- Pattern: "from C start, implement D, let E do F"
- Each implementation step is a compound node containing:
  - The new component being implemented
  - The capability it enables
  - The pattern it follows from the example
- Show the chain of dependencies clearly
""",
    DiagramType.COMPARISON: """
=== COMPARISON-SPECIFIC ===
- Side-by-side layout within this region
- Matching nodes between compared items at the same height
- Highlight differences with distinct colors
- Shared elements appear once with edges to both sides
""",
}

REGION_GEN_USER = """\
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


# ═══════════════════════════════════════════════════════════════════════════
#  §2  Per-region generation — the kernel launch
# ═══════════════════════════════════════════════════════════════════════════

async def generate_region(
    region: PlannedRegion,
    text: str,
    intent: UserIntent,
    all_regions: List[PlannedRegion],
    ai_engine,
    model: str = "",
    context: Optional[PassContext] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate ELK subgraph for one region.

    This is one invocation of the fused filter+histogram kernel.
    Like CCCL's kernel launch:
        launcher.doit(topk_kernel,
            d_keys_in,              // full user text (context)
            key_bufs.Current(),     // this region's entity hints
            key_bufs.Alternate(),   // output: ELK subgraph
            counter, histogram,     // pipeline state
            extract_op,             // region-specific extraction
            identify_op,            // entity identification
            pass, is_last_pass)

    Args:
        region: The planned region to generate
        text: Full user text (for context — NEVER truncated)
        intent: Parsed user intent
        all_regions: All planned regions (for cross-region awareness)
        ai_engine: AIEngine instance
        model: LLM model override
        context: Pipeline counter state

    Returns:
        (elk_subgraph, diagnostics)
    """
    if context is None:
        context = PassContext()
    context.advance(f"region_gen:{region.id}")

    diag: Dict[str, Any] = {"region_id": region.id, "region_name": region.name}

    if not model:
        s = ai_engine._settings
        model = s.ANTHROPIC_DEFAULT_MODEL or s.DEFAULT_MODEL

    # Build the system prompt with diagram-type supplement
    system = REGION_GEN_SYSTEM
    supplement = _TYPE_SUPPLEMENTS.get(intent.diagram_type, "")
    if supplement:
        system += supplement

    # Format other regions for context
    other_lines = []
    for r in all_regions:
        if r.id != region.id:
            other_lines.append(
                f"  - {r.id} ({r.name}): {r.bbox['width']}×{r.bbox['height']}px, "
                f"entities: {', '.join(r.entity_hints[:5])}"
            )
    other_regions_str = "\n".join(other_lines) if other_lines else "(this is the only region)"

    # Format region-specific entities from intent
    region_entities_str = ""
    if region.entity_hints:
        # Find matching entities from intent for richer context
        entity_details = []
        for hint in region.entity_hints:
            hint_lower = hint.lower()
            for e in intent.entities:
                if e.name.lower() == hint_lower or hint_lower in e.name.lower():
                    entity_details.append(
                        f"  - {e.name} (type: {e.entity_type})"
                    )
                    break
            else:
                entity_details.append(f"  - {hint}")
        region_entities_str = "\n".join(entity_details)

    prompt = REGION_GEN_USER.format(
        region_id=region.id,
        region_name=region.name,
        bbox_w=region.bbox["width"],
        bbox_h=region.bbox["height"],
        entity_hints=", ".join(region.entity_hints) if region.entity_hints else "(discover from text)",
        other_regions=other_regions_str,
        text=text,  # FULL TEXT — the kernel reads the full input
        region_entities=region_entities_str or "(discover from text)",
    )

    try:
        provider = ai_engine._select_provider(model)
        resp = await provider.get_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=0.3,
            max_tokens=8192,
        )
        raw = resp.get("content", "")

        # Finalize with the shared template
        def update_counters(parsed, ctx):
            children = parsed.get("children", [])
            edges = parsed.get("edges", [])
            ctx.total_entities += _count_nodes_recursive(parsed)
            ctx.total_edges += len(edges)

        result = finalize_pass(
            raw_output=raw,
            context=context,
            validator=validate_elk_subgraph,
            counter_updater=update_counters,
        )

        if result.success and result.data:
            subgraph = result.data
            # Ensure the region ID is set
            subgraph["id"] = region.id
            # Constrain node sizes to bbox
            _constrain_to_bbox(subgraph, region.bbox)
            diag["success"] = True
            diag["node_count"] = _count_nodes_recursive(subgraph)
            diag["edge_count"] = len(subgraph.get("edges", []))
            return subgraph, diag
        else:
            diag["success"] = False
            diag["errors"] = result.errors
            # Return a minimal fallback subgraph
            return _fallback_subgraph(region), diag

    except Exception as e:
        context.record_error(f"Region gen {region.id}: {e}")
        diag["success"] = False
        diag["error"] = str(e)
        return _fallback_subgraph(region), diag


# ═══════════════════════════════════════════════════════════════════════════
#  §3  Batch generation — the dispatch loop
# ═══════════════════════════════════════════════════════════════════════════

async def generate_all_regions(
    regions: List[PlannedRegion],
    text: str,
    intent: UserIntent,
    ai_engine,
    model: str = "",
    context: Optional[PassContext] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate ELK subgraphs for all regions.

    This is the dispatch loop — the `for (; pass < num_passes; pass++)`
    in CCCL's dispatch function.

    Currently sequential (each region depends on knowing what others
    generated for cross-region edges).  Future optimization: parallel
    generation for independent regions, sequential for dependent ones.

    Like CCCL:
        int pass = 1;
        for (; pass < num_passes; pass++) {
            launcher.doit(topk_kernel, ...);
            key_bufs.selector ^= 1;  // swap
        }

    Args:
        regions: Planned regions from M1
        text: Full user text
        intent: Parsed user intent from M2
        ai_engine: AIEngine instance
        model: LLM model override
        context: Pipeline counter state

    Returns:
        (list_of_subgraphs, diagnostics)
    """
    if context is None:
        context = PassContext()

    diag: Dict[str, Any] = {
        "pass": "per_region_generation",
        "region_count": len(regions),
        "regions": {},
    }

    subgraphs: List[Dict[str, Any]] = []

    # Sort by priority (region_planner assigns priority)
    sorted_regions = sorted(regions, key=lambda r: r.priority)

    for region in sorted_regions:
        subgraph, region_diag = await generate_region(
            region=region,
            text=text,
            intent=intent,
            all_regions=regions,
            ai_engine=ai_engine,
            model=model,
            context=context,
        )
        subgraphs.append(subgraph)
        diag["regions"][region.id] = region_diag

    diag["total_nodes"] = context.total_entities
    diag["total_edges"] = context.total_edges

    return subgraphs, diag


# ═══════════════════════════════════════════════════════════════════════════
#  §4  Helpers — constraint enforcement and fallbacks
# ═══════════════════════════════════════════════════════════════════════════

def _count_nodes_recursive(node: Dict[str, Any]) -> int:
    """Count all nodes in an ELK graph recursively."""
    count = 0
    for child in node.get("children", []):
        if isinstance(child, dict):
            count += 1
            count += _count_nodes_recursive(child)
    return count


def _constrain_to_bbox(
    subgraph: Dict[str, Any],
    bbox: Dict[str, int],
) -> None:
    """Ensure all nodes in the subgraph fit within the region's bbox.

    Like CCCL's buffer_length check: if candidates exceed the buffer,
    skip writing to out_buf.  We clamp node sizes to the bbox instead.
    """
    max_w = bbox["width"] - 40  # 20px margin on each side
    max_h = bbox["height"] - 40
    max_node_w = max(120, max_w - 20)
    max_node_h = max(40, max_h // 2)

    for child in subgraph.get("children", []):
        if not isinstance(child, dict):
            continue
        w = child.get("width", 150)
        h = child.get("height", 50)
        child["width"] = min(w, max_node_w)
        child["height"] = min(h, max_node_h)
        # Recurse into compound nodes
        if child.get("children"):
            inner_bbox = {
                "x": 0, "y": 0,
                "width": child["width"] - 24,
                "height": child["height"] - 40,
            }
            _constrain_to_bbox(child, inner_bbox)


def _fallback_subgraph(region: PlannedRegion) -> Dict[str, Any]:
    """Create a minimal fallback subgraph when generation fails.

    Like CCCL's early_stop path: when we know the answer, skip
    further processing and write directly to output.
    """
    children = []
    for i, hint in enumerate(region.entity_hints[:8]):
        safe_id = re.sub(r"[^a-zA-Z0-9_]", "_", hint.lower()).strip("_")[:50]
        children.append({
            "id": f"{region.id}_{safe_id}" if safe_id else f"{region.id}_node_{i}",
            "width": min(180, region.bbox["width"] - 40),
            "height": 50,
            "labels": [{"text": hint}],
        })

    # If no hints, create a single placeholder node
    if not children:
        children.append({
            "id": f"{region.id}_placeholder",
            "width": min(200, region.bbox["width"] - 40),
            "height": 60,
            "labels": [{"text": region.name}],
        })

    # Create sequential edges between entities
    edges = []
    for i in range(len(children) - 1):
        edges.append({
            "id": f"e_{children[i]['id']}_{children[i + 1]['id']}",
            "sources": [children[i]["id"]],
            "targets": [children[i + 1]["id"]],
        })

    return {
        "id": region.id,
        "children": children,
        "edges": edges,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  §5  Refinement pass — the second radix pass (DoubleBuffer swap)
# ═══════════════════════════════════════════════════════════════════════════

REFINE_SYSTEM = """\
You are refining ONE REGION of a technical architecture diagram.
You will receive the current subgraph JSON and its neighboring regions.

Your task: improve the subgraph WITHOUT changing the node set.
  - Adjust node widths/heights for visual balance with neighbors
  - Fix edge routing hints if nodes are reordered
  - Add iconHint if missing (1-3 word natural language description)
  - Ensure compound node nesting is correct

RULES:
1. Do NOT add or remove nodes. Only adjust sizes, order, and metadata.
2. Do NOT change node IDs.
3. Keep all existing edges. You may adjust edge routing hints.
4. Output the complete refined subgraph JSON.
"""

REFINE_USER = """\
=== CURRENT SUBGRAPH (from Pass 1) ===
{current_json}

=== REGION INFO ===
Region: {region_name} ({bbox_w}×{bbox_h} pixels)

=== NEIGHBORING REGIONS (for size reference) ===
{neighbor_info}

Refine the subgraph. Output ONLY the JSON:"""


async def refine_region(
    region: PlannedRegion,
    current_subgraph: Dict[str, Any],
    neighbor_subgraphs: List[Tuple[PlannedRegion, Dict[str, Any]]],
    ai_engine,
    model: str = "",
    context: Optional[PassContext] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Refine a region's subgraph using neighbor context.

    Like CCCL's pass 2+ with higher radix bits: reads from
    Current() (the pass 1 result), writes to Alternate()
    (the refined subgraph).  After this, the caller does
    selector ^= 1.

    The refinement adjusts node sizes, adds missing iconHints,
    and ensures visual balance across regions — things that
    the initial generation (pass 1) can't do because it doesn't
    see the other regions' results.

    Returns (refined_subgraph, diagnostics).
    """
    import json

    if context:
        context.advance("refine_region")

    diag: Dict[str, Any] = {"region": region.id}

    # Build neighbor summary
    neighbor_lines = []
    for nr, ns in neighbor_subgraphs:
        n_nodes = len(ns.get("children", []))
        avg_w = 0
        if ns.get("children"):
            avg_w = sum(c.get("width", 150) for c in ns["children"]) // n_nodes
        neighbor_lines.append(
            f"  {nr.name}: {nr.bbox['width']}×{nr.bbox['height']}, "
            f"{n_nodes} nodes, avg_width={avg_w}"
        )
    neighbor_info = "\n".join(neighbor_lines) if neighbor_lines else "  (none)"

    current_json = json.dumps(current_subgraph, indent=2, ensure_ascii=False)

    user_prompt = REFINE_USER.format(
        current_json=current_json[:4000],  # truncate if very large
        region_name=region.name,
        bbox_w=region.bbox["width"],
        bbox_h=region.bbox["height"],
        neighbor_info=neighbor_info,
    )

    try:
        messages = [
            {"role": "system", "content": REFINE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        raw_response = await ai_engine.chat(
            messages=messages, model=model,
        )

        from backend.pipeline.topology.finalize_pass import (
            finalize_pass, validate_elk_subgraph,
        )

        validated = finalize_pass(
            raw_output=raw_response,
            context=context,
            validator=validate_elk_subgraph,
            counter_updater=lambda ctx, data: None,
        )

        if validated and validated.get("children"):
            # Preserve original node IDs — don't let LLM rename them
            original_ids = {
                c["id"] for c in current_subgraph.get("children", [])
                if isinstance(c, dict) and c.get("id")
            }
            refined_ids = {
                c["id"] for c in validated.get("children", [])
                if isinstance(c, dict) and c.get("id")
            }

            if original_ids and refined_ids == original_ids:
                _constrain_to_bbox(validated, region.bbox)
                diag["status"] = "refined"
                diag["node_count"] = len(validated.get("children", []))
                return validated, diag
            else:
                diag["status"] = "id_mismatch"
                diag["original_ids"] = len(original_ids)
                diag["refined_ids"] = len(refined_ids)
                logger.warning(
                    "Refinement changed node IDs for %s, keeping original",
                    region.id,
                )
                return current_subgraph, diag

    except Exception as e:
        diag["status"] = "error"
        diag["error"] = str(e)
        logger.warning("Refinement failed for %s: %s", region.id, e)

    return current_subgraph, diag


async def refine_all_regions(
    regions: List[PlannedRegion],
    subgraphs: List[Dict[str, Any]],
    ai_engine,
    model: str = "",
    context: Optional[PassContext] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Refine all regions with neighbor context.

    Like CCCL's multi-pass loop:
        for pass in range(1, num_passes):
            topk_kernel(key_bufs.Current(), key_bufs.Alternate(), ...);
            key_bufs.selector ^= 1;

    Each region reads the pass-1 results of its neighbors.
    """
    from backend.pipeline.topology.canvas_compositor import DoubleBuffer

    diag: Dict[str, Any] = {"regions": {}}

    # Initialize DoubleBuffers for each region
    buffers = [DoubleBuffer(initial=sg) for sg in subgraphs]

    for i, region in enumerate(regions):
        # Collect neighbors: adjacent regions by bbox proximity
        neighbors = []
        for j, other in enumerate(regions):
            if i == j:
                continue
            neighbors.append((other, buffers[j].current()))

        refined, region_diag = await refine_region(
            region=region,
            current_subgraph=buffers[i].current(),
            neighbor_subgraphs=neighbors[:4],  # limit context size
            ai_engine=ai_engine,
            model=model,
            context=context,
        )

        # Write to Alternate, then swap — selector ^= 1
        buffers[i].set_alternate(refined)
        diag["regions"][region.id] = region_diag

    # Read final results from Current() of each buffer
    results = [buf.current() for buf in buffers]
    return results, diag
