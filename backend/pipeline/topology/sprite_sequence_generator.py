"""sprite_sequence_generator.py — Identity-locked family sequences (M215).

This repo already solved "keep the subject identical across a sequence, change
only one named thing": frame_generator.py.  Its design is exactly what a sprite
*family* needs —

    IDENTITY LOCK (frame_generator §2):
        every prompt carries an identity header forbidding visual drift
        ("DO NOT change colors, proportions, style").
    IMAGE EDITING (not text-to-image):
        request = [{inline_data: base_image}, {text: edit_instruction}]
        → the model EDITS the base instead of inventing a new picture.
    thought_signature:
        threaded frame→frame so later variants stay anchored to earlier ones.
    EXACT GREEN SCREEN #00FF00:
        every output is transparent-ready (M213 lifts it with the same chroma).

We follow that pattern.  For a sprite family (node_classifier.SpriteFamily)
whose members differ only along `variation_axis` — your

    input feature (C×H×W) → frequency decompose → Decomposed feats
        → ⊗ → selection map → ⊕ → c×h×w

chain, where the feature-map-like members share one base block and only the
channel/decomposition density advances — we:

  1. generate ONE base sprite (the family's base_concept), green-screen, via
     the same single-image surface M212 already calls;
  2. for each remaining member, EDIT the base, instructing the model to change
     ONLY the variation_axis to that variant's concrete delta (from M211's
     _AXIS_DELTAS), threading the thought_signature for series coherence.

This is the *sequence* path.  Small families (<= SINGLE_SHOT_MAX members) are
better served by M212's one-shot grid sheet (a single pass keeps adjacent
cells consistent for free); the strong-consistency families your chain
describes go through this edit-based path.  `choose_family_strategy()` picks.

Output is a list of green-screen sprite frames (base64), one per member, in
member order — handed to M213's split_and_clean for background removal, then
M216 for intra-family bbox alignment.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  §1  Strategy selection — sequence (edit-based) vs single-shot (grid sheet)
# ═══════════════════════════════════════════════════════════════════════════

# Families at or below this size are cheaper/cleaner via M212's grid sheet:
# a single model pass keeps adjacent cells consistent.  Above it, the
# edit-based sequence enforces stronger member-to-member identity lock.
SINGLE_SHOT_MAX = 2

FamilyStrategy = Literal["sequence", "single_shot"]


def choose_family_strategy(family: Any) -> FamilyStrategy:
    """Pick the generation path for a family.

    Singletons and pairs → "single_shot" (M212 grid). Larger families, or any
    family whose axis demands a monotone progression (the chain case), →
    "sequence" (edit-based identity lock, this module).
    """
    members = list(getattr(family, "member_node_ids", []))
    if len(members) <= SINGLE_SHOT_MAX:
        return "single_shot"
    return "sequence"


# ═══════════════════════════════════════════════════════════════════════════
#  §2  Prompt construction — base + per-variant edit instruction
# ═══════════════════════════════════════════════════════════════════════════

# Mirror frame_generator's exact contracts so M213 removal is identical.
GREEN_SCREEN_HEX = "#00FF00"
_GREEN_PREAMBLE = (
    f"The ENTIRE background MUST be solid {GREEN_SCREEN_HEX} green — "
    "no gradients, no shadows, no other background color. "
)
_IDENTITY_LOCK = (
    "VISUAL IDENTITY LOCK — DO NOT MODIFY THESE ATTRIBUTES: overall shape, "
    "stroke weight, color palette, line style, composition or scale. "
)


def _base_sprite_prompt(family: Any) -> str:
    """The single-image prompt that creates the family's BASE sprite."""
    base = getattr(family, "base_concept", "abstract concept block")
    from backend.pipeline.topology.sprite_prompt_designer import STYLE_SUFFIX
    return (
        f"A small {base}, {STYLE_SUFFIX}. {_GREEN_PREAMBLE}"
        "This is the BASE reference for a consistent series; render the "
        "canonical, neutral form."
    )


def _variant_edit_instruction(family: Any, variant_index: int) -> str:
    """Edit instruction: change ONLY the variation axis to this variant."""
    from backend.pipeline.topology.sprite_prompt_designer import _delta_for
    axis = getattr(family, "variation_axis", "appearance")
    delta = _delta_for(axis, variant_index)
    return (
        f"{_IDENTITY_LOCK}{_GREEN_PREAMBLE}"
        f"Using the provided image, change ONLY the {axis}: {delta}. "
        "Keep everything else pixel-identical to the provided image."
    )


# ═══════════════════════════════════════════════════════════════════════════
#  §3  Result dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FamilySequence:
    """A generated family sequence: one green-screen frame per member.

    frames[k] corresponds to member_node_ids[k]; None where a frame failed
    (caller falls that node back to text via M214).
    """
    family_id: str
    member_node_ids: List[str]
    frames_b64: List[Optional[str]]
    strategy: FamilyStrategy
    base_node_id: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family_id": self.family_id,
            "member_node_ids": list(self.member_node_ids),
            "frames_present": [f is not None for f in self.frames_b64],
            "strategy": self.strategy,
            "base_node_id": self.base_node_id,
            "diagnostics": self.diagnostics,
            "success": self.success,
            "error": self.error,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  §4  The generator
# ═══════════════════════════════════════════════════════════════════════════

async def generate_family_sequence(
    family: Any,
    *,
    settings: Optional[Any] = None,
    model: str = "gemini-3-pro-image-preview",
    base_image_callable=None,
    edit_callable=None,
) -> FamilySequence:
    """Generate a family as an identity-locked sequence (M215).

    Args:
        family: a node_classifier.SpriteFamily (duck-typed: family_id,
                member_node_ids, base_concept, variation_axis).
        settings: backend Settings (forwarded to the generators).
        model: image model name.
        base_image_callable: async fn(prompt, settings, model) -> dict with
            success/image_b64. Defaults to gemini_image_gen path; tests inject
            a mock. Used to make the BASE sprite (single image, green screen).
        edit_callable: async fn(base_image_b64, edit_instruction,
            previous_thought_signature, settings, model) -> (frame_b64,
            thought_sig). Defaults to frame_generator's image-edit primitive;
            tests inject a mock. Used to make each VARIANT.

    Returns:
        FamilySequence with one frame per member (in member order).
    """
    members = list(getattr(family, "member_node_ids", []))
    fam_id = getattr(family, "family_id", "")
    strategy: FamilyStrategy = "sequence"
    seq = FamilySequence(
        family_id=fam_id, member_node_ids=members,
        frames_b64=[None] * len(members), strategy=strategy,
        base_node_id=members[0] if members else "",
    )
    if not members:
        seq.error = "empty family"
        return seq

    # Resolve generators (real or injected).
    if base_image_callable is None:
        base_image_callable = _default_base_image
    if edit_callable is None:
        edit_callable = _default_edit

    # ── Step 1: BASE sprite (member 0) ──
    base_prompt = _base_sprite_prompt(family)
    try:
        base_res = await base_image_callable(
            prompt=base_prompt, settings=settings, model=model,
        )
    except Exception as e:  # degrade — whole family falls back to text
        logger.exception("base sprite generation raised for %s", fam_id)
        seq.error = f"base generation failed: {e}"
        return seq

    base_b64 = base_res.get("image_b64") if base_res and base_res.get("success") else None
    if not base_b64:
        seq.error = "base sprite empty"
        seq.diagnostics["base_error"] = (base_res or {}).get("error")
        return seq

    seq.frames_b64[0] = base_b64
    produced = 1

    # ── Step 2: EDIT base into each subsequent variant ──
    thought_sig: Optional[str] = None
    for idx in range(1, len(members)):
        instruction = _variant_edit_instruction(family, idx)
        try:
            frame_b64, thought_sig = await edit_callable(
                base_image_b64=base_b64,
                edit_instruction=instruction,
                previous_thought_signature=thought_sig,
                settings=settings,
                model=model,
            )
        except Exception as e:
            logger.warning("variant %d edit failed for %s: %s", idx, fam_id, e)
            frame_b64 = None  # this member falls back to text at M214
        if frame_b64:
            seq.frames_b64[idx] = frame_b64
            produced += 1

    seq.diagnostics["members"] = len(members)
    seq.diagnostics["frames_produced"] = produced
    seq.success = produced > 0
    logger.info("Family %s sequence: %d/%d frames produced",
                fam_id, produced, len(members))
    return seq


# ═══════════════════════════════════════════════════════════════════════════
#  §5  Default generator adapters — wire to the real infra
# ═══════════════════════════════════════════════════════════════════════════

async def _default_base_image(*, prompt: str, settings, model: str) -> Dict[str, Any]:
    """BASE sprite via the same surface M212 uses (generate_image_with_gemini),
    with no skeleton structure (a standalone small illustration)."""
    from backend.pipeline.gemini_image_gen import generate_image_with_gemini
    return await generate_image_with_gemini(
        svg_content="", prompt=prompt, settings=settings, model=model,
        aspect_ratio="1:1", image_size="1K",
    )


async def _default_edit(
    *, base_image_b64: str, edit_instruction: str,
    previous_thought_signature: Optional[str], settings, model: str,
) -> Tuple[Optional[str], Optional[str]]:
    """VARIANT via frame_generator's image-edit primitive (identity lock +
    thought_signature threading)."""
    from backend.ai_engine import AIEngine
    from backend.pipeline.frame_generator import _generate_frame_with_image_edit
    engine = AIEngine(settings) if settings is not None else AIEngine()
    return await _generate_frame_with_image_edit(
        engine=engine, model=model, original_image_b64=base_image_b64,
        edit_instruction=edit_instruction,
        previous_thought_signature=previous_thought_signature,
        settings=engine._settings,
    )
