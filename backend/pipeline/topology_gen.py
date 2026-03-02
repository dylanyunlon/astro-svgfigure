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
from typing import Any, Dict, Optional

from ..ai_engine import AIEngine
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
- layoutOptions: {elk.algorithm, elk.direction}

Rules:
1. Each major component/step/module becomes a node.
2. Data flow or sequential connections become edges.
3. Use descriptive but short labels (max 3 words per label).
4. Node IDs should be snake_case, descriptive (e.g., "self_attention", "feed_forward").
5. Set reasonable default sizes: width=150, height=50 for standard nodes,
   width=200, height=60 for complex nodes.
6. Do NOT include x, y coordinates — ELK will compute them.
7. For hierarchical structures, use nested children (compound nodes).
8. Output ONLY the JSON object, no markdown fences, no explanation.
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
                {"role": "system", "content": TOPOLOGY_SYSTEM_PROMPT},
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
