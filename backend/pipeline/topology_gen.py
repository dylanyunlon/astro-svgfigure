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
- layoutOptions: {"elk.algorithm": "layered", "elk.direction": "RIGHT"}

=== CRITICAL RULE: EDGE INTEGRITY ===
EVERY edge's "sources" and "targets" MUST reference node IDs that EXIST in
the "children" array (at the same level or inside a compound node's "children").

**BEFORE outputting**, mentally verify:
  - For each edge, check that the source ID exists as a node.
  - For each edge, check that the target ID exists as a node.
  - If a node should contain sub-components, it MUST have a "children" array
    with those sub-components listed as nodes inside it.

COMMON MISTAKE TO AVOID:
  ❌ WRONG: Creating "input_group" as a big empty box (width=280, height=140)
     and then referencing "source_context" in edges — but "source_context" does
     not exist anywhere in children!
  ✅ RIGHT: Either (A) make "input_group" a compound node WITH children
     containing "source_context", OR (B) list "source_context" as a
     top-level node in the root children array.

=== ARCHITECTURE vs FLOWCHART DETECTION ===
Analyze the input text to determine the diagram type:

(A) ARCHITECTURE DIAGRAM (hierarchical, modular, nested components):
  - Use COMPOUND NODES (nested children) for parent-child relationships
  - A parent node MUST contain its children as a nested "children" array
    — never create empty group boxes without actual children inside
  - Siblings at the same level are children of the same parent
  - Grandchild nodes are nested inside child nodes
  - Add "group" property to parent nodes: {"group": true, "borderless": true}
  - Borderless group nodes represent visual grouping layers — they should
    have NO visible border in the final rendering, only a subtle background tint
  - Deep nesting (≥3 levels) is expected for complex architectures
  - Edges BETWEEN groups go in the root "edges" array, referencing child node IDs
  - Edges WITHIN a group go in that group's "edges" array

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
   TRIPLE-CHECK this — the most common failure mode is edges referencing
   nodes that were never created. If you mention a node in an edge, it MUST
   exist in some children array.
9. Edge IDs must be unique strings (e.g., "e1", "e2", "e3").
10. Output ONLY the JSON object, no markdown fences, no explanation.
11. Follow the COMPLEXITY-AWARE NODE GENERATION guide above to determine
    how many nodes to generate.  Do NOT oversimplify complex descriptions.
    {complexity_override}
12. For architecture diagrams: parent nodes with children MUST have
    "layoutOptions": {"elk.padding": "[top=30,left=10,bottom=10,right=10]"}
12. Add iconHint field to nodes that should have an icon/illustration.
    Use natural language descriptions (e.g., "microscope", "DNA helix", "brain"),
    NOT emoji or Unicode. The image generator will create these from text.
13. Add "labelOnly": true to nodes that are ANNOTATION LABELS — short text
    that should float naked on the diagram WITHOUT any box/rect around them.
    Academic figures use this for descriptive labels like "Join Pattern",
    "Selectivity", "Cardinality", "Code", "Columnar Storage", dimension
    annotations like "180x50 px", or category headers like "Workload",
    "Resource", "User".  These appear near the actual processing nodes
    they describe, connected by dashed/dotted edges or no edge at all.
    Nodes that are actual processing components (Encoder, Decoder, Filter,
    Query Planner) should NOT be labelOnly — they get the normal box.

=== CRITICAL: NO HYPEREDGES ===
Each edge MUST have EXACTLY ONE source and EXACTLY ONE target:
  ✅ CORRECT: {"id": "e1", "sources": ["nodeA"], "targets": ["nodeB"]}
  ❌ WRONG:  {"id": "e1", "sources": ["nodeA", "nodeB"], "targets": ["nodeC"]}
  ❌ WRONG:  {"id": "e1", "sources": ["nodeA"], "targets": ["nodeB", "nodeC"]}
If a node connects to multiple targets, create SEPARATE edges for each connection.
Example: nodeA → nodeB AND nodeA → nodeC requires TWO edges, not one with two targets.

=== CRITICAL: NO DOT-NOTATION IN NODE IDS ===
Node IDs MUST be simple snake_case without dots.
  ✅ CORRECT: "adjacency_matrix_w"
  ❌ WRONG:  "stage_1.adjacency_matrix_w"
Cross-group edges reference child IDs directly (ELK resolves hierarchy automatically).

Example output with nesting (architecture diagram):
{
  "id": "root",
  "layoutOptions": {"elk.algorithm": "layered", "elk.direction": "RIGHT"},
  "children": [
    {"id": "input", "width": 150, "height": 50, "labels": [{"text": "Input"}]},
    {"id": "input_dim", "width": 80, "height": 24, "labels": [{"text": "768-dim"}], "labelOnly": true},
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
  "layoutOptions": {"elk.algorithm": "layered", "elk.direction": "RIGHT"},
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

{intent_supplement}

Paper method description:
---
{text}
---

{entity_hint_section}

Output the ELK JSON topology:"""


# ============================================================================
# Intent-Aware Prompt Supplements
# ============================================================================

_INTENT_SUPPLEMENTS = {
    "engineering": """\
=== ENGINEERING / DEVOPS FLOW DETECTED ===
The user's description includes engineering actions (git clone, tree, build, etc.).
- Generate nodes for EACH action step: "git clone", "tree inspect", "build", etc.
- Include resource nodes: repositories, configs, output artifacts
- Action nodes should have iconHint matching the tool (e.g., "git logo",
  "terminal window", "file tree", "docker container")
- Use sequential edges: clone → inspect → configure → build → test → deploy
- If the user mentions specific repos/tools (CCCL, NCCL, NVLINK, megatron-core),
  each becomes a node or compound node with its sub-components
- Group related actions into compound nodes (e.g., "Setup Phase", "Build Phase")
""",
    "recursive": """\
=== RECURSIVE CHAIN PATTERN DETECTED ===
The user's description follows the pattern:
  "从 C 这个好例子开始 → 实现 D → 让 E 可以 F → G 引入 H..."
  (from C start → implement D → let E do F → G introduces H...)
- Each "implement/introduce/integrate" step is a compound node containing:
  - The component being implemented (child node)
  - The capability it enables (child node with edge)
  - The pattern it follows from the previous example (edge back)
- Show the full chain, do NOT truncate steps
- Each step builds on the previous: edges flow forward AND reference back
- Use descriptive labels that capture BOTH the component and its purpose
""",
    "architecture": """\
=== ARCHITECTURE DIAGRAM DETECTED ===
- Use COMPOUND NODES extensively: modules contain sub-modules
- Nesting depth ≥ 2 for complex modules (e.g., PLTR OSDK containing CCCL, NCCL)
- Every major system/library/framework the user mentions becomes a compound node
  with its internal components as children
- Cross-boundary edges connect components across different systems
""",
    "comparison": """\
=== COMPARISON LAYOUT DETECTED ===
- Side-by-side or parallel columns for compared items
- Matching nodes at the same level for comparable components
- Shared elements appear once with edges to both sides
""",
    "data_flow": """\
=== DATA FLOW / PIPELINE DETECTED ===
- Each processing stage is a node
- Data transformations are labeled on edges
- Branch/merge points are explicit nodes
""",
}

_COMPLEXITY_OVERRIDES = {
    "simple": "Aim for 8-15 nodes and up to 20 edges.",
    "medium": "Generate 20-30 nodes and up to 40 edges.  Do NOT oversimplify.",
    "complex": "Generate 35-50 nodes and up to 60 edges.  Capture ALL components.",
    "dense": "Generate 50-80 nodes and up to 100 edges.  This is a complex system — every component matters.",
}


def _build_intent_supplement(intent) -> str:
    """Build the intent-specific prompt supplement from parsed UserIntent."""
    parts = []
    dtype = getattr(intent, "diagram_type", None)
    if dtype:
        dtype_val = dtype.value if hasattr(dtype, "value") else str(dtype)
        supplement = _INTENT_SUPPLEMENTS.get(dtype_val, "")
        if supplement:
            parts.append(supplement)

    # Inject detected action chain as explicit hints
    action_chain = getattr(intent, "action_chain", [])
    if action_chain:
        steps_str = " → ".join(
            f"{step.verb} {step.object}" + (f" (via {step.tool})" if step.tool else "")
            for step in action_chain
        )
        parts.append(f"=== DETECTED ACTION CHAIN ===\n{steps_str}\nGenerate a node for EACH step above.")

    # Inject containment groups
    containment = getattr(intent, "containment_groups", {})
    if containment:
        lines = []
        for parent, children in containment.items():
            lines.append(f"  {parent} contains: {', '.join(children)}")
        parts.append(f"=== DETECTED CONTAINMENT ===\n" + "\n".join(lines) +
                      "\nModel these as compound nodes with nested children.")

    # Inject style cues
    style_cues = getattr(intent, "style_cues", [])
    for cue in style_cues:
        if cue.cue_type == "density" and cue.value in ("dense", "detailed", "comprehensive"):
            parts.append("The user explicitly requested a DETAILED diagram — maximize node count.")
        elif cue.cue_type == "nesting":
            parts.append(f"The user requested {cue.value} levels of nesting depth.")

    return "\n\n".join(parts)


def _build_entity_hint_section(intent) -> str:
    """Build a section listing pre-extracted entities for the LLM."""
    entities = getattr(intent, "entities", [])
    if not entities:
        return ""

    entity_lines = []
    for e in entities[:80]:  # cap at 80 to avoid prompt overflow
        parent_str = f" (inside {e.parent_hint})" if e.parent_hint else ""
        entity_lines.append(f"  - {e.name} [{e.entity_type}]{parent_str}")

    return (
        "=== PRE-EXTRACTED ENTITIES (from user text — include ALL of these as nodes) ===\n"
        + "\n".join(entity_lines)
        + f"\n\nTotal: {len(entities)} entities detected.  Generate at least this many nodes."
    )


def _get_complexity_override(intent) -> str:
    """Return the complexity-appropriate node limit string."""
    complexity = getattr(intent, "complexity", None)
    if complexity:
        cval = complexity.value if hasattr(complexity, "value") else str(complexity)
        return _COMPLEXITY_OVERRIDES.get(cval, _COMPLEXITY_OVERRIDES["simple"])
    return _COMPLEXITY_OVERRIDES["simple"]


# ============================================================================
# Main Function
# ============================================================================

async def generate_topology(
    ai_engine: AIEngine,
    text: str,
    model: Optional[str] = None,
    algorithm: ElkAlgorithm = ElkAlgorithm.LAYERED,
    direction: ElkDirection = ElkDirection.RIGHT,
    intent=None,
) -> TopologyResponse:
    """
    Generate ELK topology from paper method text using LLM.

    Args:
        ai_engine: Initialized AIEngine instance
        text: Paper method description
        model: LLM model to use (auto-detected if None)
        algorithm: ELK layout algorithm hint for the LLM
        direction: ELK layout direction hint
        intent: Optional pre-parsed UserIntent from user_intent_parser.
                If None, the parser will be called automatically.

    Returns:
        TopologyResponse with the ELK graph or error
    """
    # ── Parse user intent if not provided ──
    if intent is None:
        try:
            from .topology.user_intent_parser import parse_user_intent
            intent = parse_user_intent(text)
            logger.info(
                f"Auto-parsed intent: type={intent.diagram_type.value} "
                f"entities={len(intent.entities)} complexity={intent.complexity.value}"
            )
        except Exception as e:
            logger.warning(f"Intent parsing failed (non-fatal): {e}")
            intent = None

    # ── Build intent-aware prompt components ──
    intent_supplement = _build_intent_supplement(intent) if intent else ""
    entity_hint_section = _build_entity_hint_section(intent) if intent else ""
    complexity_override = _get_complexity_override(intent) if intent else _COMPLEXITY_OVERRIDES["simple"]

    # ── Inject complexity_override into system prompt ──
    system_prompt = TOPOLOGY_SYSTEM_PROMPT.replace("{complexity_override}", complexity_override)

    prompt = TOPOLOGY_USER_PROMPT_TEMPLATE.format(
        text=text,
        algorithm=algorithm.value,
        direction=direction.value,
        intent_supplement=intent_supplement,
        entity_hint_section=entity_hint_section,
    )

    # Track raw_output for error reporting
    raw_output: Optional[str] = None

    try:
        logger.info(f"Generating topology with model={model or 'default'}")

        # ── Pre-flight check: Ensure AI engine has available providers ──
        if not ai_engine.available_providers:
            error_msg = (
                "No AI providers configured. Please set at least one of: "
                "GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY in your .env file."
            )
            logger.error(error_msg)
            return TopologyResponse(
                success=False,
                error=error_msg,
                raw_llm_output=None,
            )

        result = await ai_engine.get_completion(
            messages=[
                {"role": "system", "content": get_topology_prompt_with_edge_routing(system_prompt)},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=0.3,  # Low temperature for structured output
            max_tokens=16384,  # Increased for complex topology JSON
        )

        # ── Robust extraction of raw_output with detailed error handling ──
        raw_output = result.get("content", "")

        # Handle case where content is None or empty
        if raw_output is None:
            raw_output = ""

        if not isinstance(raw_output, str):
            raw_output = str(raw_output) if raw_output else ""

        # ── Validate that we got meaningful content ──
        if not raw_output or not raw_output.strip():
            # Check for common API configuration issues
            provider_hint = _diagnose_empty_response(ai_engine, model)
            error_msg = f"LLM returned empty response. {provider_hint}"
            logger.error(error_msg)
            return TopologyResponse(
                success=False,
                error=error_msg,
                raw_llm_output=raw_output,
                model_used=result.get("model", model),
            )

        logger.debug(f"Raw LLM topology output ({len(raw_output)} chars): {raw_output[:500]}...")

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
        # ── Enhanced JSON error with diagnostic context ──
        error_context = _diagnose_json_error(raw_output, e)
        logger.error(f"Failed to parse topology JSON: {error_context}")
        return TopologyResponse(
            success=False,
            error=error_context,
            raw_llm_output=raw_output,
        )
    except RuntimeError as e:
        # ── Specific handling for provider configuration errors ──
        error_msg = str(e)
        if "No provider available" in error_msg:
            error_msg = (
                f"AI provider not configured: {error_msg}. "
                "Please check your .env file and ensure the required API keys are set."
            )
        logger.error(f"Topology generation failed (RuntimeError): {error_msg}")
        return TopologyResponse(
            success=False,
            error=error_msg,
            raw_llm_output=raw_output,
        )
    except Exception as e:
        logger.error(f"Topology generation failed: {e}", exc_info=True)
        return TopologyResponse(
            success=False,
            error=str(e),
            raw_llm_output=raw_output,
        )


def _diagnose_empty_response(ai_engine: AIEngine, model: Optional[str]) -> str:
    """
    Diagnose why the LLM returned an empty response.
    Returns a helpful hint message.
    """
    providers = ai_engine.available_providers
    if not providers:
        return (
            "No AI providers are configured. "
            "Set GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY in .env"
        )

    if model:
        model_lower = model.lower()
        if model_lower.startswith("gemini") and "gemini" not in providers:
            return f"Model '{model}' requires GEMINI_API_KEY which is not set."
        if model_lower.startswith("claude") and "anthropic" not in providers and "claude_compatible" not in providers:
            return f"Model '{model}' requires ANTHROPIC_API_KEY which is not set."
        if model_lower.startswith(("gpt-", "o1-", "o3-", "grok-")) and "openai" not in providers:
            return f"Model '{model}' requires OPENAI_API_KEY which is not set."

    return (
        f"Available providers: {providers}. "
        "The model may have returned an empty response due to content filtering or rate limits. "
        "Check API logs for details."
    )


def _diagnose_json_error(raw_output: Optional[str], error: json.JSONDecodeError) -> str:
    """
    Provide diagnostic context for JSON parsing errors.
    """
    if raw_output is None or raw_output == "":
        return (
            "LLM returned empty response (no JSON to parse). "
            "This usually indicates an API configuration issue. "
            "Check that your API keys are valid and the model is accessible."
        )

    content_preview = raw_output[:200] if raw_output else "(empty)"
    content_len = len(raw_output) if raw_output else 0

    # Check for common non-JSON patterns
    stripped = raw_output.strip() if raw_output else ""

    if stripped.startswith("I ") or stripped.startswith("Here") or stripped.startswith("The "):
        return (
            f"LLM returned prose text instead of JSON ({content_len} chars). "
            f"Preview: '{content_preview}...'. "
            "The model may have misunderstood the prompt or refused the request."
        )

    if stripped.startswith("<"):
        return (
            f"LLM returned XML/HTML instead of JSON ({content_len} chars). "
            f"Preview: '{content_preview}...'. "
            "Try a different model or simplify the input text."
        )

    if "error" in stripped.lower()[:100]:
        return (
            f"LLM response contains error message ({content_len} chars). "
            f"Preview: '{content_preview}...'"
        )

    return (
        f"Failed to parse JSON from LLM output ({content_len} chars): {str(error)}. "
        f"Preview: '{content_preview}...'"
    )


# ============================================================================
# Helpers
# ============================================================================

def _parse_topology_json(raw: str) -> Dict[str, Any]:
    """
    Parse LLM output into a topology dict, handling common issues.
    
    Handles:
    - Empty/None input
    - Markdown code fences (```json ... ```)
    - Leading/trailing whitespace and text
    - Common LLM JSON syntax errors (missing commas, trailing commas, etc.)
    
    Raises:
        json.JSONDecodeError: If no valid JSON can be extracted
    """
    # ── Guard against None/empty input ──
    if raw is None:
        raise json.JSONDecodeError(
            "LLM returned None (no response content)", "", 0
        )
    
    if not isinstance(raw, str):
        raw = str(raw)
    
    text = raw.strip()
    
    if not text:
        raise json.JSONDecodeError(
            "LLM returned empty response (no content to parse)", "", 0
        )

    # ── Remove markdown fences ──
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove opening fence (```json or ```)
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        
        # After removing fences, check again if empty
        if not text:
            raise json.JSONDecodeError(
                "LLM returned empty content within markdown fences", raw, 0
            )

    # ── Try direct parse ──
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        raise json.JSONDecodeError(
            f"LLM returned valid JSON but not an object (got {type(result).__name__})",
            text, 0
        )
    except json.JSONDecodeError:
        pass

    # ── Try finding JSON object boundaries ──
    start = text.find("{")
    end = text.rfind("}") + 1
    
    if start < 0:
        preview = text[:100] + "..." if len(text) > 100 else text
        raise json.JSONDecodeError(
            f"No JSON object found in LLM output (no '{{' character). Preview: {preview}",
            text, 0
        )
    
    if end <= start:
        preview = text[:100] + "..." if len(text) > 100 else text
        raise json.JSONDecodeError(
            f"Incomplete JSON object in LLM output (no matching '}}' found). Preview: {preview}",
            text, start
        )
    
    # Extract the JSON substring
    json_candidate = text[start:end]
    
    # ── First attempt: direct parse ──
    try:
        result = json.loads(json_candidate)
        if isinstance(result, dict):
            return result
        raise json.JSONDecodeError(
            f"Extracted JSON is not an object (got {type(result).__name__})",
            json_candidate, 0
        )
    except json.JSONDecodeError as first_error:
        # ── Second attempt: fix common LLM JSON errors ──
        logger.warning(f"JSON parse failed at position {first_error.pos}, attempting repair...")
        
        fixed_json = _repair_json(json_candidate, first_error)
        
        if fixed_json:
            try:
                result = json.loads(fixed_json)
                if isinstance(result, dict):
                    logger.info("JSON repair successful")
                    return result
            except json.JSONDecodeError:
                pass
        
        # ── All repair attempts failed ──
        preview = json_candidate[:150] + "..." if len(json_candidate) > 150 else json_candidate
        raise json.JSONDecodeError(
            f"Could not parse extracted JSON: {first_error.msg}. Preview: {preview}",
            json_candidate, first_error.pos
        )


def _repair_json(json_str: str, error: json.JSONDecodeError) -> Optional[str]:
    """
    Attempt to repair common LLM JSON syntax errors.
    
    Common errors:
    - Missing comma between elements
    - Trailing comma before closing bracket
    - Unescaped quotes in strings
    - Truncated JSON (incomplete)
    
    Returns repaired JSON string or None if repair failed.
    """
    import re
    
    text = json_str
    error_pos = error.pos if error.pos else 0
    
    # ── Strategy 1: Fix at error position ──
    # If we know the exact position, try to fix there first
    if error_pos > 0 and error_pos < len(text):
        # Look around the error position for common issues
        start = max(0, error_pos - 50)
        end = min(len(text), error_pos + 50)
        context = text[start:end]
        
        # Common issue: missing comma before a key
        # e.g., }"key": or ]"key": should be },"key": or ],"key":
        if error.msg == "Expecting ',' delimiter":
            # Try inserting comma at error position
            fixed = text[:error_pos] + ',' + text[error_pos:]
            try:
                json.loads(fixed)
                return fixed
            except json.JSONDecodeError:
                pass
            
            # Try inserting comma just before error position
            for offset in range(-5, 5):
                pos = error_pos + offset
                if 0 <= pos < len(text):
                    fixed = text[:pos] + ',' + text[pos:]
                    try:
                        json.loads(fixed)
                        return fixed
                    except json.JSONDecodeError:
                        continue
    
    # ── Strategy 2: Global regex fixes ──
    
    # Add missing comma between } and {
    text = re.sub(r'(\})\s*\n\s*(\{)', r'\1,\n\2', text)
    
    # Add missing comma between ] and [
    text = re.sub(r'(\])\s*\n\s*(\[)', r'\1,\n\2', text)
    
    # Add missing comma between } and "
    text = re.sub(r'(\})\s*\n\s*(")', r'\1,\n\2', text)
    
    # Add missing comma between ] and "
    text = re.sub(r'(\])\s*\n\s*(")', r'\1,\n\2', text)
    
    # Add missing comma between " and " (end of string value, start of key)
    # This pattern: "value"\n"key": → "value",\n"key":
    text = re.sub(r'(")\s*\n(\s*"[^"]+"\s*:)', r'\1,\n\2', text)
    
    # Add missing comma between number and "
    text = re.sub(r'(\d)\s*\n\s*(")', r'\1,\n\2', text)
    
    # Add missing comma between true/false/null and "
    text = re.sub(r'(true|false|null)\s*\n\s*(")', r'\1,\n\2', text)
    
    # Add missing comma between } and [ (object followed by array)
    text = re.sub(r'(\})\s*\n\s*(\[)', r'\1,\n\2', text)
    
    # Add missing comma between ] and { (array followed by object)
    text = re.sub(r'(\])\s*\n\s*(\{)', r'\1,\n\2', text)
    
    # ── Strategy 3: Fix inline missing commas (no newline) ──
    # Pattern: }"key" or ]"key" without comma
    text = re.sub(r'(\})(\s*)("[^"]+"\s*:)', r'\1,\2\3', text)
    text = re.sub(r'(\])(\s*)("[^"]+"\s*:)', r'\1,\2\3', text)
    
    # ── Strategy 4: Trailing commas ──
    # Remove trailing comma before }
    text = re.sub(r',(\s*\})', r'\1', text)
    
    # Remove trailing comma before ]
    text = re.sub(r',(\s*\])', r'\1', text)
    
    # ── Strategy 5: Try to close incomplete JSON ──
    # Count brackets
    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')
    
    if open_braces > 0 or open_brackets > 0:
        # JSON is truncated, try to close it
        # First, try to find a clean cut point (after a complete value)
        
        # Remove any trailing incomplete content after last complete element
        # Look for last complete object or array
        last_complete = max(
            text.rfind('}'),
            text.rfind(']'),
            text.rfind('"'),  # end of string
        )
        
        if last_complete > 0:
            # Check if we're in the middle of a string
            after_last = text[last_complete + 1:].strip()
            if after_last and not after_last.startswith((',', '}', ']')):
                # Truncate at last complete element
                text = text[:last_complete + 1]
        
        # Close remaining brackets
        open_braces = text.count('{') - text.count('}')
        open_brackets = text.count('[') - text.count(']')
        
        # Close in reverse order of expected nesting
        text = text.rstrip()
        if text.endswith(','):
            text = text[:-1]
        
        text += ']' * max(0, open_brackets)
        text += '}' * max(0, open_braces)
    
    # ── Verify the repair worked ──
    if text != json_str:
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError as e2:
            # Iterative repair: keep fixing until it parses or we hit max attempts
            # Large LLM outputs (20k+ chars) often have multiple missing commas
            current = text
            for attempt in range(10):
                if e2.pos == error.pos or e2.pos <= 0:
                    break
                logger.debug(f"Repair round {attempt+2}: error moved to pos {e2.pos}, retrying...")
                repaired = _repair_json(current, e2)
                if repaired is None or repaired == current:
                    break
                try:
                    json.loads(repaired)
                    logger.info(f"JSON repair succeeded after {attempt+2} rounds")
                    return repaired
                except json.JSONDecodeError as e3:
                    current = repaired
                    e2 = e3
    
    return None


def _validate_and_fix_topology(topology: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and auto-fix common topology issues from LLM output.

    Fixes:
      - Missing/empty children
      - Nodes without width/height
      - Nodes without labels
      - Duplicate node IDs
      - Edges referencing non-existent nodes → AUTO-CREATE missing nodes
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

    # ── RESOLVE DOT-NOTATION in edges before orphan detection ──────────
    # LLMs frequently generate edges with dot-notation IDs like
    # "stage_1_graph_construction.adjacency_matrix_w" to reference
    # child node "adjacency_matrix_w" inside compound parent
    # "stage_1_graph_construction". Resolve these first.
    dot_index = _build_dot_notation_index(fixed_children)
    resolved_count = 0
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        for key in ("sources", "targets", "source", "target"):
            refs = edge.get(key)
            if not refs:
                continue
            if isinstance(refs, str):
                refs = [refs]
                edge[key] = refs
            if not isinstance(refs, list):
                continue
            for i, ref in enumerate(refs):
                if isinstance(ref, str) and ref not in all_valid_ids and "." in ref:
                    resolved = _resolve_dot_notation(ref, dot_index, all_valid_ids)
                    if resolved and resolved != ref:
                        refs[i] = resolved
                        resolved_count += 1

    if resolved_count > 0:
        logger.info(
            f"Resolved {resolved_count} dot-notation edge references "
            f"to nested child node IDs"
        )

    # ── AUTO-CREATE missing nodes referenced by edges ─────────────────
    orphan_ids: set = set()
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        for src in (edge.get("sources") or edge.get("source") or []):
            s = src if isinstance(src, str) else ""
            if s and s not in all_valid_ids:
                orphan_ids.add(s)
        for tgt in (edge.get("targets") or edge.get("target") or []):
            t = tgt if isinstance(tgt, str) else ""
            if t and t not in all_valid_ids:
                orphan_ids.add(t)

    if orphan_ids:
        logger.warning(
            f"Found {len(orphan_ids)} orphan node IDs referenced in edges "
            f"but missing from children: {orphan_ids}. Auto-creating them."
        )

        # Build lookup: top-level node ID → node dict
        parent_lookup: Dict[str, Dict[str, Any]] = {}
        for child in fixed_children:
            if isinstance(child, dict) and child.get("id"):
                parent_lookup[child["id"]] = child

        resolved_orphan_count = 0
        for orphan_id in orphan_ids:
            child_id = orphan_id
            placed_inside_parent = False

            # DOT-NOTATION ORPHAN PLACEMENT: "parent_name.child_name" →
            # create child_name INSIDE parent compound node
            if "." in orphan_id:
                parts = orphan_id.split(".")
                for end_idx in range(1, len(parts)):
                    parent_candidate = ".".join(parts[:end_idx])
                    child_id = ".".join(parts[end_idx:])

                    if parent_candidate in parent_lookup:
                        parent_node = parent_lookup[parent_candidate]
                        if not isinstance(parent_node.get("children"), list):
                            parent_node["children"] = []

                        existing_child_ids = {
                            c.get("id") for c in parent_node["children"]
                            if isinstance(c, dict)
                        }
                        if child_id not in existing_child_ids:
                            label_text = child_id.replace("_", " ").title()
                            parent_node["children"].append({
                                "id": child_id,
                                "width": 160,
                                "height": 50,
                                "labels": [{"text": label_text}],
                            })
                            all_valid_ids.add(child_id)
                            logger.info(
                                f"Placed orphan '{orphan_id}' as child '{child_id}' "
                                f"inside parent '{parent_candidate}'"
                            )
                        else:
                            all_valid_ids.add(child_id)

                        placed_inside_parent = True
                        resolved_orphan_count += 1
                        break

                # Rewrite edge references: orphan_id → child_id
                if placed_inside_parent:
                    for edge in edges:
                        if not isinstance(edge, dict):
                            continue
                        for key in ("sources", "targets", "source", "target"):
                            refs = edge.get(key)
                            if not refs:
                                continue
                            if isinstance(refs, list):
                                for i, r in enumerate(refs):
                                    if r == orphan_id:
                                        refs[i] = child_id
                            elif isinstance(refs, str) and refs == orphan_id:
                                edge[key] = child_id

            # Fallback: create at top level if no parent match
            if not placed_inside_parent:
                label_text = orphan_id.split(".")[-1].replace("_", " ").title()
                new_node = {
                    "id": orphan_id,
                    "width": 160,
                    "height": 50,
                    "labels": [{"text": label_text}],
                }
                fixed_children.append(new_node)
                all_valid_ids.add(orphan_id)

        if resolved_orphan_count > 0:
            logger.info(
                f"Placed {resolved_orphan_count}/{len(orphan_ids)} orphan nodes "
                f"inside their parent compound nodes (dot-notation resolution)"
            )

        topology["children"] = fixed_children

    fixed_edges = _fix_edge_list(edges, all_valid_ids, edge_id_prefix="e")
    topology["edges"] = fixed_edges

    # Also check nested edges inside compound nodes for orphan references
    _fix_orphan_refs_recursive(fixed_children, all_valid_ids)

    original_child_count = len(children)
    original_edge_count = len(edges)

    logger.info(
        f"Topology validated: {len(fixed_children)} top-level nodes "
        f"({len(all_valid_ids)} total including nested), "
        f"{len(fixed_edges)} top-level edges "
        f"(dropped {original_child_count - len(fixed_children)} nodes, "
        f"{original_edge_count - len(fixed_edges)} edges)"
        f"{f', auto-created {len(orphan_ids)} missing nodes' if orphan_ids else ''}"
    )

    return topology


def _fix_orphan_refs_recursive(
    children: List[Dict[str, Any]],
    all_valid_ids: set,
) -> None:
    """Recursively check nested edges for orphan references and auto-create missing nodes."""
    for node in children:
        if not isinstance(node, dict):
            continue
        nested_edges = node.get("edges", [])
        nested_children = node.get("children", [])

        if isinstance(nested_edges, list) and nested_edges:
            orphan_ids: set = set()
            for edge in nested_edges:
                if not isinstance(edge, dict):
                    continue
                for src in (edge.get("sources") or []):
                    if isinstance(src, str) and src not in all_valid_ids:
                        orphan_ids.add(src)
                for tgt in (edge.get("targets") or []):
                    if isinstance(tgt, str) and tgt not in all_valid_ids:
                        orphan_ids.add(tgt)

            if orphan_ids:
                if not isinstance(nested_children, list):
                    nested_children = []
                    node["children"] = nested_children
                for orphan_id in orphan_ids:
                    nested_children.append({
                        "id": orphan_id,
                        "width": 150,
                        "height": 50,
                        "labels": [{"text": orphan_id.replace("_", " ").title()}],
                    })
                    all_valid_ids.add(orphan_id)
                logger.warning(
                    f"Auto-created {len(orphan_ids)} missing nodes inside "
                    f"compound node '{node.get('id', '?')}': {orphan_ids}"
                )

        if isinstance(nested_children, list):
            _fix_orphan_refs_recursive(nested_children, all_valid_ids)


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

        # Ensure labels (handle null labels from LLM)
        labels = node.get("labels")
        if isinstance(labels, list) and len(labels) > 0:
            # Clean label objects: remove null text values
            clean_labels = []
            for lbl in labels:
                if isinstance(lbl, dict) and lbl.get("text"):
                    clean_labels.append({"text": lbl["text"]})
                elif isinstance(lbl, str) and lbl:
                    clean_labels.append({"text": lbl})
            fixed_node["labels"] = clean_labels if clean_labels else [{"text": node_id.replace("_", " ").title()}]
        else:
            fixed_node["labels"] = [{"text": node_id.replace("_", " ").title()}]

        # Preserve compound node fields (skip null/None values)
        if node.get("group"):
            fixed_node["group"] = True
        if node.get("borderless"):
            fixed_node["borderless"] = True
        if node.get("labelOnly"):
            fixed_node["labelOnly"] = True
        if node.get("iconHint"):
            fixed_node["iconHint"] = node["iconHint"]
        # Only preserve layoutOptions if it's a non-null dict
        lo = node.get("layoutOptions")
        if lo and isinstance(lo, dict):
            fixed_node["layoutOptions"] = lo

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
    """
    Fix a list of edges, validating references against valid node IDs.

    CRITICAL: Decompose hyperedges (multiple sources or targets) into
    simple 1-to-1 edges. ELK's layered algorithm throws
    "Hyperedges are not supported" when an edge has >1 source or >1 target.
    """
    seen_edge_ids: set = set()
    fixed_edges = []

    for i, edge in enumerate(edges):
        if not isinstance(edge, dict):
            continue

        sources = edge.get("sources") or []
        targets = edge.get("targets") or []

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

        # Decompose hyperedges (M sources × N targets) into M*N simple edges
        # Each simple edge has exactly 1 source and 1 target to avoid ELK error
        for si, src_id in enumerate(valid_sources):
            for ti, tgt_id in enumerate(valid_targets):
                base_id = edge.get("id", f"{edge_id_prefix}_{i}")
                needs_suffix = len(valid_sources) > 1 or len(valid_targets) > 1
                edge_id = f"{base_id}_s{si}_t{ti}" if needs_suffix else base_id
                if edge_id in seen_edge_ids:
                    edge_id = f"{edge_id}_{i}"
                seen_edge_ids.add(edge_id)

                fixed_edge = {
                    "id": edge_id,
                    "sources": [src_id],
                    "targets": [tgt_id],
                }

                # Preserve advanced edge properties only on first decomposed edge
                if si == 0 and ti == 0:
                    adv = edge.get("advanced")
                    if adv and isinstance(adv, dict):
                        fixed_edge["advanced"] = adv
                    lbls = edge.get("labels")
                    if lbls and isinstance(lbls, list) and len(lbls) > 0:
                        fixed_edge["labels"] = lbls

                fixed_edges.append(fixed_edge)

    return fixed_edges


def create_example_topology(
    name: str = "transformer",
    algorithm: ElkAlgorithm = ElkAlgorithm.LAYERED,
    direction: ElkDirection = ElkDirection.RIGHT,
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


# ============================================================================
# Dot-notation resolution helpers
# ============================================================================

def _build_dot_notation_index(
    children: List[Dict[str, Any]],
    parent_prefix: str = "",
) -> Dict[str, str]:
    """
    Build an index mapping dotted path strings to resolved node IDs.
    E.g., "stage_1.adjacency_matrix_w" → "adjacency_matrix_w"
    """
    index: Dict[str, str] = {}
    for node in children:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id", "")
        if not node_id:
            continue

        nested = node.get("children", [])
        if isinstance(nested, list):
            for child in nested:
                if not isinstance(child, dict):
                    continue
                child_id = child.get("id", "")
                if child_id:
                    dotted_key = f"{node_id}.{child_id}"
                    index[dotted_key] = child_id
                    # Recurse for deeper nesting
                    deeper = _build_dot_notation_index([child], parent_prefix=dotted_key)
                    index.update(deeper)

    return index


def _resolve_dot_notation(
    ref: str,
    dot_index: Dict[str, str],
    all_valid_ids: set,
) -> Optional[str]:
    """
    Try to resolve a dot-notation reference to an actual nested child ID.

    Strategies:
    1. Direct lookup in dot_index
    2. Last segment match: if "child" part exists in all_valid_ids, use it
    3. Progressive suffix matching
    """
    # Strategy 1: direct lookup
    if ref in dot_index:
        resolved = dot_index[ref]
        if resolved in all_valid_ids:
            return resolved

    # Strategy 2: try suffixes
    parts = ref.split(".")
    for start_idx in range(1, len(parts)):
        suffix = ".".join(parts[start_idx:])
        if suffix in all_valid_ids:
            return suffix
        if suffix in dot_index:
            resolved = dot_index[suffix]
            if resolved in all_valid_ids:
                return resolved

    # Strategy 3: last segment only
    last = parts[-1]
    if last in all_valid_ids:
        return last

    return None