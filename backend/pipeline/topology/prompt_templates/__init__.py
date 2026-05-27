"""prompt_templates/ — The PolicySelector.

CCCL f984c90 uses PolicySelector to map GPU architecture to optimal
tuning parameters for each kernel launch:

    static constexpr topk_policy policy =
        PolicySelector{}(::cuda::arch_id{CUB_PTX_ARCH / 10});

Different architectures (sm_70, sm_80, sm_90) get different block sizes,
items per thread, and radix pass counts.  The kernel code is identical;
only the policy differs.

We do the same for LLM prompts.  Different diagram types (ARCHITECTURE,
FLOWCHART, ENGINEERING_FLOW) get different system prompts, user templates,
and JSON schemas.  The pipeline code is identical; only the prompt differs.

Usage:
    from backend.pipeline.topology.prompt_templates import select_prompts
    system, user_tpl = select_prompts(DiagramType.ARCHITECTURE, 'plan')
    user = user_tpl.format(width=900, height=500, text=text, ...)

Stages:
    'plan'     — region planning (histogram kernel)
    'generate' — per-region ELK generation (fused kernel)
    'refine'   — refinement pass (second radix pass, M12)
"""
from __future__ import annotations

from typing import Dict, Tuple

from backend.pipeline.topology.user_intent_parser import DiagramType
from backend.pipeline.topology.prompt_templates.plan_prompts import (
    PLAN_SYSTEMS, PLAN_USER,
)
from backend.pipeline.topology.prompt_templates.generate_prompts import (
    GEN_SYSTEM_BASE, GEN_TYPE_SUPPLEMENTS, GEN_USER,
)
from backend.pipeline.topology.prompt_templates.icon_registry import (
    resolve_icon_hint,
    fetch_icons_for_subgraph,
    fetch_icons_for_canvas,
)


def select_prompts(
    diagram_type: DiagramType,
    stage: str,
) -> Tuple[str, str]:
    """Select (system_prompt, user_template) for a pipeline stage.

    Like CCCL's PolicySelector: same kernel, different tuning.

    Args:
        diagram_type: Classified diagram type from intent parser
        stage: 'plan' | 'generate' | 'refine'

    Returns:
        (system_prompt, user_template_with_placeholders)
    """
    if stage == "plan":
        system = PLAN_SYSTEMS.get(diagram_type, PLAN_SYSTEMS[DiagramType.UNKNOWN])
        return system, PLAN_USER

    if stage == "generate":
        system = GEN_SYSTEM_BASE
        supplement = GEN_TYPE_SUPPLEMENTS.get(diagram_type, "")
        if supplement:
            system += supplement
        return system, GEN_USER

    if stage == "refine":
        # M12 will add refinement prompts; for now reuse generate
        system = GEN_SYSTEM_BASE + "\n\n=== REFINEMENT MODE ===\n" \
            "You are refining an existing subgraph, not creating from scratch.\n" \
            "Adjust node sizes for visual balance with neighboring regions.\n" \
            "Fix edge alignment issues. Keep all existing nodes.\n"
        return system, GEN_USER

    raise ValueError(f"Unknown stage: {stage}")


__all__ = [
    "select_prompts",
    "resolve_icon_hint",
    "fetch_icons_for_subgraph",
    "fetch_icons_for_canvas",
]
