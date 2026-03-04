"""
Topology Generator — Step 1 of the Forward Pipeline
=====================================================
Takes paper method description text, calls LLM (Gemini/Claude/GPT) via AIEngine,
outputs an ELK-compatible topology JSON with zero coordinates.

The LLM only needs to output the *topological relationships* (nodes + edges).
ELK.js then computes precise pixel positions (Step 2).

GitHub references:
  - ResearAI/AutoFigure — LLM-driven academic figure generation
  - kieler/elkjs — ELK JSON format specification
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ..ai_engine import AIEngine
from .edge_routing_prompts import get_topology_prompt_with_edge_routing
from ..schemas import (
    ElkAlgorithm,
    ElkDirection,
    ElkGraph,
    TopologyResponse,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Prompt Templates
# ============================================================================

TOPOLOGY_SYSTEM_PROMPT = """\
You are an expert at analyzing academic paper method descriptions and extracting
the computational/data flow as a structured graph topology.

Your output must be valid ELK JSON (Eclipse Layout Kernel format) with:
- id: "root"
- children: array of nodes, each with {id, width, height, labels: [{text}]}
- edges: array of connections, each with {id, sources: [nodeId], targets: [nodeId]}
- layoutOptions: {"elk.algorithm": "layered", "elk.direction": "DOWN"}

=== ARCHITECTURE vs FLOWCHART DETECTION ===
Analyze the input text to determine the diagram type:

(A) ARCHITECTURE DIAGRAM (hierarchical, modular, nested components):
  - Use COMPOUND NODES (nested children) for parent-child relationships
  - A parent node contains its children as nested "children" array
  - Siblings at the same level are children of the same parent
  - Grandchild nodes are nested inside child nodes
  - Add "group" property to parent nodes: {"group": true, "borderless": true}
  - Borderless group nodes represent visual grouping layers — they should
    have NO visible border in the final rendering, only a subtle background tint
  - Deep nesting (≥3 levels) is expected for complex architectures

(B) FLOWCHART / PIPELINE (sequential data flow):
  - Mostly linear chain of nodes with directional edges
  - Minimal nesting, emphasis on edge routing and flow direction
  - Use lane grouping for parallel paths if applicable

=== COMPLEXITY-AWARE NODE GENERATION ===
Count the distinct modules/components/steps mentioned in the text:
  - ≤15 components → generate ~10-15 nodes (simple)
  - 16-30 components → generate ~20-30 nodes (medium)
  - 31-50 components → generate ~35-50 nodes (complex)
  - 51+ components → generate ~50-80 nodes (very complex)
Generate ALL components the paper describes. Do NOT oversimplify.

Rules:
1. Each major component/step/module becomes a node.
2. Data flow or sequential connections become edges.
3. Use descriptive but short labels (max 4 words per label).
4. Node IDs should be snake_case, descriptive (e.g., "self_attention", "feed_forward").
5. Set reasonable default sizes: width=150, height=50 for standard nodes,
   width=200, height=60 for complex nodes, width=250, height=80 for group containers.
6. Do NOT include x, y coordinates — ELK will compute them.
7. For hierarchical structures, use nested children (compound nodes).
   Example: a "Transformer Block" parent containing "Multi-Head Attention",
   "Feed Forward", and "Layer Norm" as children.
8. Every edge's sources and targets MUST reference existing node IDs from children.
9. Edge IDs must be unique strings (e.g., "e1", "e2", "e3").
10. Output ONLY the JSON object, no markdown fences, no explanation.
11. For architecture diagrams: parent nodes with children MUST have
    "layoutOptions": {"elk.padding": "[top=30,left=10,bottom=10,right=10]"}
12. Add iconHint field to nodes that should have an icon/illustration.
    Use natural language descriptions (e.g., "microscope", "DNA helix", "brain"),
    NOT emoji or Unicode. The image generator will create these from text.

Example output with nesting (architecture diagram):
{
  "id": "root",
  "layoutOptions": {"elk.algorithm": "layered", "elk.direction": "DOWN"},
  "children": [
    {"id": "input", "width": 150, "height": 50, "labels": [{"text": "Input"}]},
    {
      "id": "encoder_block", "width": 250, "height": 200,
      "labels": [{"text": "Encoder Block"}],
      "group": true, "borderless": true,
      "layoutOptions": {"elk.padding": "[top=30,left=10,bottom=10,right=10]"},
      "children": [
        {"id": "self_attn", "width": 150, "height": 50, "labels": [{"text": "Self Attention"}], "iconHint": "attention mechanism"},
        {"id": "ffn", "width": 150, "height": 50, "labels": [{"text": "Feed Forward"}]},
        {"id": "layer_norm", "width": 150, "height": 50, "labels": [{"text": "Layer Norm"}]}
      ],
      "edges": [
        {"id": "inner_e1", "sources": ["self_attn"], "targets": ["ffn"]},
        {"id": "inner_e2", "sources": ["ffn"], "targets": ["layer_norm"]}
      ]
    },
    {"id": "output", "width": 150, "height": 50, "labels": [{"text": "Output"}]}
  ],
  "edges": [
    {"id": "e1", "sources": ["input"], "targets": ["encoder_block"]},
    {"id": "e2", "sources": ["encoder_block"], "targets": ["output"]}
  ]
}

Example output (simple flowchart):
{
  "id": "root",
  "layoutOptions": {"elk.algorithm": "layered", "elk.direction": "DOWN"},
  "children": [
    {"id": "input", "width": 150, "height": 50, "labels": [{"text": "Input"}]},
    {"id": "encoder", "width": 160, "height": 50, "labels": [{"text": "Encoder"}]},
    {"id": "output", "width": 150, "height": 50, "labels": [{"text": "Output"}]}
  ],
  "edges": [
    {"id": "e1", "sources": ["input"], "targets": ["encoder"]},
    {"id": "e2", "sources": ["encoder"], "targets": ["output"]}
  ]
}
"""

TOPOLOGY_USER_PROMPT_TEMPLATE = """\
Analyze the following paper method description and generate an ELK topology JSON
that captures the computational/data flow as a directed graph.

Algorithm hint: {algorithm}
Direction hint: {direction}

Paper method description:
---
{text}
---

Output the ELK JSON topology:"""


# ============================================================================
# Main Function
# ============================================================================

async def generate_topology(
    ai_engine: AIEngine,
    text: str,
    model: Optional[str] = None,
    algorithm: ElkAlgorithm = ElkAlgorithm.LAYERED,
    direction: ElkDirection = ElkDirection.DOWN,
) -> TopologyResponse:
    """
    Generate ELK topology from paper method text using LLM.

    Args:
        ai_engine: Initialized AIEngine instance
        text: Paper method description
        model: LLM model to use (auto-detected if None)
        algorithm: ELK layout algorithm hint for the LLM
        direction: ELK layout direction hint

    Returns:
        TopologyResponse with the ELK graph or error
    """
    prompt = TOPOLOGY_USER_PROMPT_TEMPLATE.format(
        text=text,
        algorithm=algorithm.value,
        direction=direction.value,
    )

    try:
        logger.info(f"Generating topology with model={model or 'default'}")

        result = await ai_engine.get_completion(
            messages=[
                {"role": "system", "content": get_topology_prompt_with_edge_routing(TOPOLOGY_SYSTEM_PROMPT)},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=0.3,  # Low temperature for structured output
            max_tokens=8192,
        )

        raw_output = result["content"]
        logger.debug(f"Raw LLM topology output: {raw_output[:500]}...")

        # Parse the JSON response
        topology_dict = _parse_topology_json(raw_output)

        # Validate and auto-fix common LLM output issues
        topology_dict = _validate_and_fix_topology(topology_dict)

        # Inject layout options if not present
        if "layoutOptions" not in topology_dict or not topology_dict["layoutOptions"]:
            topology_dict["layoutOptions"] = {
                "elk.algorithm": algorithm.value,
                "elk.direction": direction.value,
                "elk.layered.spacing.nodeNodeBetweenLayers": "100",
                "elk.spacing.nodeNode": "80",
            }

        # Validate with Pydantic
        topology = ElkGraph(**topology_dict)

        logger.info(
            f"Topology generated: {len(topology.children)} nodes, "
            f"{len(topology.edges)} edges"
        )

        return TopologyResponse(
            success=True,
            topology=topology,
            raw_llm_output=raw_output,
            model_used=result.get("model", model),
        )

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse topology JSON: {e}")
        return TopologyResponse(
            success=False,
            error=f"LLM output is not valid JSON: {str(e)}",
            raw_llm_output=raw_output if "raw_output" in dir() else None,
        )
    except Exception as e:
        logger.error(f"Topology generation failed: {e}")
        return TopologyResponse(
            success=False,
            error=str(e),
        )


# ============================================================================
# Helpers
# ============================================================================

def _parse_topology_json(raw: str) -> Dict[str, Any]:
    """Parse LLM output into a topology dict, handling common issues."""
    text = raw.strip()

    # Remove markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try finding JSON object boundaries
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("Could not extract JSON from LLM output", text, 0)


def _validate_and_fix_topology(topology: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and auto-fix common topology issues from LLM output.

    Fixes:
      - Missing/empty children
      - Nodes without width/height
      - Nodes without labels
      - Duplicate node IDs
      - Edges referencing non-existent nodes
      - Duplicate edge IDs
      - Missing layoutOptions
      - Preserves compound node fields: group, borderless, iconHint
      - Recursively validates nested children and their edges
    """
    # Ensure children exist
    children = topology.get("children", [])
    if not isinstance(children, list) or len(children) == 0:
        logger.warning("Topology has no children — cannot fix")
        return topology

    # Fix nodes (recursive for compound nodes)
    all_valid_ids: set = set()
    fixed_children = _fix_node_list(children, all_valid_ids, depth=0)
    topology["children"] = fixed_children

    # Fix top-level edges
    edges = topology.get("edges", [])
    if not isinstance(edges, list):
        edges = []

    fixed_edges = _fix_edge_list(edges, all_valid_ids, edge_id_prefix="e")
    topology["edges"] = fixed_edges

    original_child_count = len(children)
    original_edge_count = len(edges)

    logger.info(
        f"Topology validated: {len(fixed_children)} top-level nodes "
        f"({len(all_valid_ids)} total including nested), "
        f"{len(fixed_edges)} top-level edges "
        f"(dropped {original_child_count - len(fixed_children)} nodes, "
        f"{original_edge_count - len(fixed_edges)} edges)"
    )

    return topology


def _fix_node_list(
    children: List[Dict[str, Any]],
    all_valid_ids: set,
    depth: int = 0,
) -> List[Dict[str, Any]]:
    """Recursively fix a list of nodes, handling compound nodes with nested children."""
    seen_ids: set = set()
    fixed_children = []

    for i, node in enumerate(children):
        if not isinstance(node, dict):
            continue

        node_id = node.get("id", f"node_d{depth}_{i}")
        # Deduplicate IDs
        if node_id in seen_ids or node_id in all_valid_ids:
            node_id = f"{node_id}_{depth}_{i}"
        seen_ids.add(node_id)
        all_valid_ids.add(node_id)

        fixed_node = {
            "id": node_id,
            "width": node.get("width") or (250 if "children" in node else 150),
            "height": node.get("height") or (200 if "children" in node else 50),
        }

        # Ensure labels
        labels = node.get("labels", [])
        if isinstance(labels, list) and len(labels) > 0:
            fixed_node["labels"] = labels
        else:
            fixed_node["labels"] = [{"text": node_id.replace("_", " ").title()}]

        # Preserve compound node fields
        if node.get("group"):
            fixed_node["group"] = True
        if node.get("borderless"):
            fixed_node["borderless"] = True
        if node.get("iconHint"):
            fixed_node["iconHint"] = node["iconHint"]
        if node.get("layoutOptions"):
            fixed_node["layoutOptions"] = node["layoutOptions"]

        # Recursively fix nested children (compound nodes)
        if "children" in node and isinstance(node["children"], list):
            nested_children = _fix_node_list(node["children"], all_valid_ids, depth + 1)
            fixed_node["children"] = nested_children

            # Fix nested edges inside compound nodes
            nested_edges = node.get("edges", [])
            if isinstance(nested_edges, list) and nested_edges:
                fixed_node["edges"] = _fix_edge_list(
                    nested_edges, all_valid_ids, edge_id_prefix=f"inner_{node_id}_e"
                )

        fixed_children.append(fixed_node)

    return fixed_children


def _fix_edge_list(
    edges: List[Dict[str, Any]],
    valid_ids: set,
    edge_id_prefix: str = "e",
) -> List[Dict[str, Any]]:
    """Fix a list of edges, validating references against valid node IDs."""
    seen_edge_ids: set = set()
    fixed_edges = []

    for i, edge in enumerate(edges):
        if not isinstance(edge, dict):
            continue

        sources = edge.get("sources", [])
        targets = edge.get("targets", [])

        # Handle common LLM mistakes: "source"/"target" instead of "sources"/"targets"
        if not sources and "source" in edge:
            src = edge["source"]
            sources = [src] if isinstance(src, str) else src
        if not targets and "target" in edge:
            tgt = edge["target"]
            targets = [tgt] if isinstance(tgt, str) else tgt

        # Handle "from"/"to" format
        if not sources and "from" in edge:
            src = edge["from"]
            sources = [src] if isinstance(src, str) else src
        if not targets and "to" in edge:
            tgt = edge["to"]
            targets = [tgt] if isinstance(tgt, str) else tgt

        # Ensure sources/targets are lists
        if isinstance(sources, str):
            sources = [sources]
        if isinstance(targets, str):
            targets = [targets]

        if not isinstance(sources, list) or not isinstance(targets, list):
            continue

        # Filter to valid node references
        valid_sources = [s for s in sources if s in valid_ids]
        valid_targets = [t for t in targets if t in valid_ids]

        if not valid_sources or not valid_targets:
            logger.debug(f"Dropping edge {edge.get('id', i)}: invalid sources={sources} or targets={targets}")
            continue

        edge_id = edge.get("id", f"{edge_id_prefix}_{i}")
        if edge_id in seen_edge_ids:
            edge_id = f"{edge_id}_{i}"
        seen_edge_ids.add(edge_id)

        fixed_edge = {
            "id": edge_id,
            "sources": valid_sources,
            "targets": valid_targets,
        }

        # Preserve advanced edge properties
        if edge.get("advanced"):
            fixed_edge["advanced"] = edge["advanced"]
        if edge.get("labels"):
            fixed_edge["labels"] = edge["labels"]

        fixed_edges.append(fixed_edge)

    return fixed_edges


def create_example_topology(
    name: str = "transformer",
    algorithm: ElkAlgorithm = ElkAlgorithm.LAYERED,
    direction: ElkDirection = ElkDirection.DOWN,
) -> ElkGraph:
    """
    Create a built-in example topology for testing / demos.

    Args:
        name: Example name (transformer, diffusion, gan, cnn)
        algorithm: ELK algorithm
        direction: ELK direction

    Returns:
        Pre-built ElkGraph
    """
    examples = {
        "transformer": {
            "id": "root",
            "layoutOptions": {
                "elk.algorithm": algorithm.value,
                "elk.direction": direction.value,
                "elk.layered.spacing.nodeNodeBetweenLayers": "100",
                "elk.spacing.nodeNode": "80",
            },
            "children": [
                {"id": "input_embed", "width": 160, "height": 50, "labels": [{"text": "Input Embedding"}]},
                {"id": "pos_encode", "width": 160, "height": 50, "labels": [{"text": "Positional Encoding"}]},
                {"id": "multi_head_attn", "width": 180, "height": 60, "labels": [{"text": "Multi-Head Attention"}]},
                {"id": "add_norm_1", "width": 150, "height": 50, "labels": [{"text": "Add & Norm"}]},
                {"id": "feed_forward", "width": 160, "height": 50, "labels": [{"text": "Feed Forward"}]},
                {"id": "add_norm_2", "width": 150, "height": 50, "labels": [{"text": "Add & Norm"}]},
                {"id": "output_linear", "width": 150, "height": 50, "labels": [{"text": "Linear"}]},
                {"id": "softmax", "width": 150, "height": 50, "labels": [{"text": "Softmax"}]},
            ],
            "edges": [
                {"id": "e1", "sources": ["input_embed"], "targets": ["pos_encode"]},
                {"id": "e2", "sources": ["pos_encode"], "targets": ["multi_head_attn"]},
                {"id": "e3", "sources": ["multi_head_attn"], "targets": ["add_norm_1"]},
                {"id": "e4", "sources": ["pos_encode"], "targets": ["add_norm_1"]},
                {"id": "e5", "sources": ["add_norm_1"], "targets": ["feed_forward"]},
                {"id": "e6", "sources": ["feed_forward"], "targets": ["add_norm_2"]},
                {"id": "e7", "sources": ["add_norm_1"], "targets": ["add_norm_2"]},
                {"id": "e8", "sources": ["add_norm_2"], "targets": ["output_linear"]},
                {"id": "e9", "sources": ["output_linear"], "targets": ["softmax"]},
            ],
        },
        "diffusion": {
            "id": "root",
            "layoutOptions": {
                "elk.algorithm": algorithm.value,
                "elk.direction": direction.value,
                "elk.layered.spacing.nodeNodeBetweenLayers": "100",
                "elk.spacing.nodeNode": "80",
            },
            "children": [
                {"id": "noise", "width": 150, "height": 50, "labels": [{"text": "Gaussian Noise"}]},
                {"id": "unet", "width": 180, "height": 60, "labels": [{"text": "U-Net Backbone"}]},
                {"id": "time_embed", "width": 160, "height": 50, "labels": [{"text": "Time Embedding"}]},
                {"id": "cross_attn", "width": 170, "height": 50, "labels": [{"text": "Cross Attention"}]},
                {"id": "text_encoder", "width": 160, "height": 50, "labels": [{"text": "Text Encoder"}]},
                {"id": "noise_pred", "width": 160, "height": 50, "labels": [{"text": "Noise Prediction"}]},
                {"id": "scheduler", "width": 150, "height": 50, "labels": [{"text": "Scheduler"}]},
                {"id": "output", "width": 150, "height": 50, "labels": [{"text": "Denoised Image"}]},
            ],
            "edges": [
                {"id": "e1", "sources": ["noise"], "targets": ["unet"]},
                {"id": "e2", "sources": ["time_embed"], "targets": ["unet"]},
                {"id": "e3", "sources": ["text_encoder"], "targets": ["cross_attn"]},
                {"id": "e4", "sources": ["cross_attn"], "targets": ["unet"]},
                {"id": "e5", "sources": ["unet"], "targets": ["noise_pred"]},
                {"id": "e6", "sources": ["noise_pred"], "targets": ["scheduler"]},
                {"id": "e7", "sources": ["scheduler"], "targets": ["output"]},
            ],
        },
    }

    topology_dict = examples.get(name, examples["transformer"])
    return ElkGraph(**topology_dict)