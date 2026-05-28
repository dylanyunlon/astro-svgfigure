"""sprite_prompt_designer.py — Structured-input → controlled prompt (M211).

We start from this repo's own good example,
gemini_image_gen.py::generate_prompt_with_grok() (line 147): it already turns
"SVG structure + method text" into an engineered, tiered image prompt, with
_analyze_svg_complexity, tier extraction, and think-tag cleaning.  It is the
proof that controlled prompts come from structured inputs, not free text.

We follow that pattern at a finer grain.  Instead of one prompt for a whole
figure, we design ONE prompt per sprite node — and, crucially, a *coherent set*
of prompts for a whole family where the base is identical and only one named
axis varies.

This is the hard part of your idea ("同类型不同含义、只有微小差别").  The risk
is that a generative model, asked for "a slightly different feature map", gives
an *uncontrolled* large difference plus style drift.  Two defenses, both here:

  1. series-consistency (NOT frame_generator's identity-lock):
     every family prompt shares one base description verbatim, and explicitly
     names "variant k of N, identical to the others except <axis>".  Adjacent
     wording forces the model toward intra-family coherence.

  2. variation_axis → concrete visual delta:
     we map the family's abstract axis ("channel decomposition density") to a
     concrete, monotone visual instruction per variant ("grid density: sparse →
     medium → dense"), so the micro-difference is *parameterised*, not left to
     the model's imagination.

Negative-prompt library mirrors frame_generator's exact green-screen contract
(#00FF00, no gradients/shadows) so M213 background removal is trivial.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  §1  Style + negative-prompt constants — the shared visual contract
# ═══════════════════════════════════════════════════════════════════════════

# Academic-figure house style.  Monochrome / minimal so sprites blend with the
# vector skeleton from to-svg.ts (M203-M207).
STYLE_SUFFIX = (
    "flat 2D vector illustration, academic paper figure style, "
    "clean thin outlines, monochrome with subtle grayscale shading, "
    "no perspective, no 3D, centered composition, generous margin"
)

# Background contract — identical to frame_generator's exact green screen so
# the same removebg tier chain (remove-bg.io → rembg → chroma) lifts it.
BG_GREEN = "#00FF00"
NEGATIVE_PROMPT = (
    "no text, no labels, no caption, no watermark, no title, "
    "no drop shadow, no gradient background, no photographic texture, "
    "no realistic lighting, no border frame, "
    f"solid pure {BG_GREEN} green background, no other background color"
)

# Default sprite raster size (square).  Small — these are node-sized glyphs,
# not full figures.  Upscaled/contained into the ELK bbox at M214 injection.
DEFAULT_SPRITE_SIZE = 256


# ═══════════════════════════════════════════════════════════════════════════
#  §2  variation_axis → monotone visual deltas
# ═══════════════════════════════════════════════════════════════════════════

# For each family axis (from node_classifier._FAMILY_RULES), an ordered list of
# concrete visual instructions.  Variant k picks delta[k % len].  Monotone so
# the series reads as a progression (sparse → dense), matching how a paper's
# "input feature → decomposed feats → c×h×w" chain reads left to right.
_AXIS_DELTAS: Dict[str, List[str]] = {
    "spatial resolution and channel count": [
        "few channels, coarse spatial grid",
        "more channels, medium spatial grid",
        "many stacked channels, fine spatial grid",
    ],
    "frequency band emphasis": [
        "low-frequency smooth bands emphasized",
        "balanced frequency bands",
        "high-frequency fine stripes emphasized",
    ],
    "decomposition depth": [
        "single undivided block",
        "block split into a few sub-bands",
        "block split into many fine sub-bands",
    ],
    "highlighted region": [
        "highlight in the top-left region",
        "highlight in the center region",
        "highlight spread across the map",
    ],
    "patch density": [
        "coarse 2x2 patch grid",
        "medium 4x4 patch grid",
        "fine 8x8 patch grid",
    ],
    "tensor shape": [
        "thin single-slice tensor block",
        "medium multi-slice tensor block",
        "deep many-slice tensor block",
    ],
    "appearance": [  # generic fallback axis for misc singleton families
        "default appearance",
    ],
}


def _delta_for(axis: str, variant_index: int) -> str:
    deltas = _AXIS_DELTAS.get(axis, _AXIS_DELTAS["appearance"])
    return deltas[variant_index % len(deltas)]


# ═══════════════════════════════════════════════════════════════════════════
#  §3  Result dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SpritePrompt:
    """A fully-specified prompt for one sprite, plus cache + family metadata.

    cache_key is a SHA256 over every field that affects the pixels (prompt,
    negative, size) — mirroring triton ASTSource.hash().  M220 uses it so
    identical sprites (⊗, "feature map", common blocks) are generated once.
    """
    node_id: str
    prompt: str
    negative: str = NEGATIVE_PROMPT
    family_id: str = ""
    variant_index: int = 0
    family_size: int = 1
    target_size: int = DEFAULT_SPRITE_SIZE

    @property
    def cache_key(self) -> str:
        h = hashlib.sha256()
        h.update(self.prompt.encode("utf-8"))
        h.update(b"\x00")
        h.update(self.negative.encode("utf-8"))
        h.update(f"\x00{self.target_size}".encode("utf-8"))
        return h.hexdigest()[:32]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "prompt": self.prompt,
            "negative": self.negative,
            "family_id": self.family_id,
            "variant_index": self.variant_index,
            "family_size": self.family_size,
            "target_size": self.target_size,
            "cache_key": self.cache_key,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  §4  Label/iconHint → base concept phrasing
# ═══════════════════════════════════════════════════════════════════════════

def _node_label(node: Dict[str, Any]) -> str:
    labels = node.get("labels") or []
    if labels and isinstance(labels[0], dict):
        return (labels[0].get("text") or "").strip()
    return (node.get("id") or "").strip()


def _base_subject(node: Dict[str, Any], family_base: Optional[str]) -> str:
    """The noun phrase the illustration depicts.

    Prefer the family's shared base concept (so all members say the same
    subject verbatim — the core of series-consistency); otherwise derive from
    iconHint, then label.  Tensor-shape labels become "feature map tensor"
    rather than literal "C×H×W" (we draw the object, not the text)."""
    if family_base and not family_base.startswith("misc:"):
        return family_base
    hint = (node.get("iconHint") or "").strip()
    if hint:
        return hint
    label = _node_label(node)
    # Strip a bare tensor shape into a generic subject.
    if re.search(r"\b[bcnhwd0-9]+\s*[x×]\s*[bcnhwd0-9]+", label, re.I):
        return "feature map tensor block"
    return label or "abstract concept block"


# ═══════════════════════════════════════════════════════════════════════════
#  §5  Single-node prompt
# ═══════════════════════════════════════════════════════════════════════════

def design_sprite_prompt(
    node: Dict[str, Any],
    family: Optional["object"] = None,
    variant_index: int = 0,
    family_size: int = 1,
) -> SpritePrompt:
    """Design the prompt for ONE sprite node.

    Args:
        node: the ELK leaf node (already classified renderMode == "sprite").
        family: optional SpriteFamily (duck-typed: .base_concept,
                .variation_axis, .family_id, .member_node_ids).
        variant_index: this node's position within its family (0-based).
        family_size: total members in the family.

    Returns:
        A SpritePrompt.  For singleton families the prompt is a plain subject;
        for multi-member families it carries the series-consistency clause and
        the concrete axis delta for this variant.
    """
    family_base = getattr(family, "base_concept", None) if family else None
    family_axis = getattr(family, "variation_axis", "appearance") if family else "appearance"
    family_id = getattr(family, "family_id", "") if family else ""

    subject = _base_subject(node, family_base)

    if family_size > 1:
        delta = _delta_for(family_axis, variant_index)
        # Series-consistency: name the base, the variant, and ONLY-difference.
        prompt = (
            f"A small {subject}, {STYLE_SUFFIX}. "
            f"This is variant {variant_index + 1} of {family_size} in a "
            f"consistent series: every variant is visually identical in style, "
            f"color, stroke weight and overall shape — the ONLY difference is "
            f"the {family_axis}. For this variant specifically: {delta}."
        )
    else:
        prompt = f"A small {subject}, {STYLE_SUFFIX}."

    return SpritePrompt(
        node_id=node.get("id", ""),
        prompt=prompt,
        family_id=family_id,
        variant_index=variant_index,
        family_size=family_size,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  §6  Family-aware batch — design a coherent set of prompts
# ═══════════════════════════════════════════════════════════════════════════

def design_prompts_for_classified(
    elk_graph: Dict[str, Any],
    families: List["object"],
) -> List[SpritePrompt]:
    """Walk the classified ELK graph and produce one SpritePrompt per sprite
    node, wiring family membership / variant order so M212 (sheet) and M215
    (sequence) get series-consistent prompts.

    Args:
        elk_graph: graph already passed through classify_nodes (nodes carry
                   renderMode and familyId).
        families: the SpriteFamily list from classify_nodes' report.

    Returns:
        SpritePrompt list, ordered family-by-family (so adjacent prompts in a
        sheet belong to the same family — adjacency aids model coherence).
    """
    fam_by_id = {getattr(f, "family_id", ""): f for f in families}
    # Build node lookup.
    nodes_by_id: Dict[str, Dict[str, Any]] = {}

    def _walk(n: Dict[str, Any]) -> None:
        for c in n.get("children", []) or []:
            if isinstance(c, dict):
                if c.get("children"):
                    _walk(c)
                else:
                    nid = c.get("id", "")
                    if nid:
                        nodes_by_id[nid] = c

    _walk(elk_graph)

    prompts: List[SpritePrompt] = []
    for fam in families:
        members = list(getattr(fam, "member_node_ids", []))
        size = len(members)
        for idx, nid in enumerate(members):
            node = nodes_by_id.get(nid)
            if node is None:
                continue
            prompts.append(
                design_sprite_prompt(
                    node, family=fam, variant_index=idx, family_size=size
                )
            )

    logger.info(
        "Designed %d sprite prompts across %d families",
        len(prompts), len(families),
    )
    return prompts
