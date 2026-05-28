"""node_classifier.py — The compile-time type dispatcher (M210).

google/jax's dispatch.py decides, at trace time, which lowering path each
primitive takes — based on the primitive's type and its argument avals — so
the runtime never re-branches:

    def xla_primitive_callable(prim, *avals, **params):
        # The dispatch decision is made ONCE here, keyed on (prim, avals).
        # Runtime just invokes the compiled callable; zero re-classification.
        ...

We do the same.  Every leaf node in the composed ELK graph is classified
ONCE into a `renderMode` — and the renderer (to-svg.ts) then just dispatches
on that field with zero classification logic of its own.  Four modes:

    text   — naked label, no AI.  (operators handled by M207 vector path,
             plain semantic labels like "vit_encoder", "loss_function")
    icon   — standard concept, fetched from Iconify via svg_icon_fetcher
             (database / encoder / transformer / … → shared/icon-aliases.json)
    sprite — paper-specific visual object that is neither a standard icon nor
             a plain label ("feature map C×H×W", "frequency spectrum",
             "decomposed feats") — these get AI-generated micro-illustrations
    (operator is a sub-case of `text`: ⊗ ⊕ ⊙ ∘ etc. → M207 renders them as
     pure SVG circle+path; we tag them text + is_operator so M207 picks up)

The classifier is WHITE-BOX: the LLM decided WHAT the nodes are (upstream);
pure rules decide HOW each is rendered.  No LLM call here.

Like topology/CLAUDE.md's reject-if-missing discipline: every node is
assigned to exactly one mode.  Low-confidence sprite candidates are
conservatively demoted to `text` — we would rather render a plain label than
generate a wrong picture.

Sprite families
───────────────
The second job is `detect_sprite_families`: group sprite nodes that share a
concept and differ only along one axis.  Your motivating example —

    input feature (C×H×W) → frequency decompose → Decomposed feats
        → ⊗ → selection map → ⊕ → c×h×w

— the feature-map-like nodes form one *family*: same base concept ("feature
map"), differing only along a `variation_axis` ("channel decomposition
density").  M215 then generates the family as a *sequence* (identity-locked,
only the axis varies) rather than N independent pictures.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

logger = logging.getLogger(__name__)

RenderMode = Literal["text", "icon", "sprite"]


# ═══════════════════════════════════════════════════════════════════════════
#  §1  Operator detection — the highest-priority, zero-cost path
# ═══════════════════════════════════════════════════════════════════════════

# Math operators that M207 (renderMathOperator) draws as pure SVG.  These are
# matched in node labels.  Kept deliberately tight: a lone glyph or the glyph
# with whitespace, not e.g. a "+" inside a longer word.
_OPERATOR_GLYPHS = "⊗⊕⊙⊘⊖∘×·⨂⨁∑∏∫"
_OPERATOR_RE = re.compile(rf"^\s*[{_OPERATOR_GLYPHS}]\s*$")
# Plain-ASCII operator-only labels ("+", "x", "*", "concat") also qualify.
_ASCII_OPERATOR_RE = re.compile(r"^\s*([+\-×*]|concat|concatenate|sum|prod|dot)\s*$", re.I)
_OPERATOR_ICON_HINTS = (
    "operator", "multiply", "multiplication", "add", "addition", "sum",
    "concat", "concatenation", "elementwise", "element-wise", "hadamard",
    "dot product", "cross product",
)


# ═══════════════════════════════════════════════════════════════════════════
#  §2  Icon-alias registry — the `icon` decision source
# ═══════════════════════════════════════════════════════════════════════════

_ALIASES_CACHE: Optional[Dict[str, str]] = None


def _load_icon_aliases() -> Dict[str, str]:
    """Load shared/icon-aliases.json — the single source of truth shared with
    svg_icon_fetcher.py and to-svg-icons.ts.  Cached after first read."""
    global _ALIASES_CACHE
    if _ALIASES_CACHE is not None:
        return _ALIASES_CACHE
    # Walk up from this file to the repo root, then shared/icon-aliases.json
    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] / "shared" / "icon-aliases.json",   # repo_root/shared
        here.parents[2] / "shared" / "icon-aliases.json",
        Path("shared/icon-aliases.json"),
    ]
    for p in candidates:
        try:
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                _ALIASES_CACHE = {
                    k.lower(): v for k, v in data.get("aliases", {}).items()
                }
                logger.debug("Loaded %d icon aliases from %s",
                             len(_ALIASES_CACHE), p)
                return _ALIASES_CACHE
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Failed reading icon aliases at %s: %s", p, e)
    _ALIASES_CACHE = {}
    logger.warning("icon-aliases.json not found; icon classification disabled")
    return _ALIASES_CACHE


# ═══════════════════════════════════════════════════════════════════════════
#  §3  Sprite cues — the `sprite` decision source
# ═══════════════════════════════════════════════════════════════════════════

# iconHint substrings that signal a paper-specific visual object: something a
# generic Iconify glyph cannot convey, but that benefits from a small drawn
# illustration.  These are matched against the iconHint (preferred) and label.
_SPRITE_CUES = (
    "feature map", "feature maps", "featuremap",
    "frequency", "spectrum", "spectral", "fourier", "wavelet",
    "decompose", "decomposed", "decomposition", "decomp",
    "selection map", "attention map", "saliency", "heat map", "heatmap",
    "tensor", "c x h x w", "c×h×w", "cxhxw", "h x w", "channel",
    "patch grid", "patches", "tokens grid",
    "spatial", "receptive field", "kernel grid", "filter bank",
    "latent map", "embedding map", "activation map", "response map",
)

# Tensor-shape pattern, e.g. "C×H×W", "1 x H x W", "B×C×H×W", "256x256".
_SHAPE_RE = re.compile(
    r"\b[bcnhwd0-9]+\s*[x×]\s*[bcnhwd0-9]+(\s*[x×]\s*[bcnhwd0-9]+)*\b",
    re.I,
)


# ═══════════════════════════════════════════════════════════════════════════
#  §4  Result dataclasses
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SpriteFamily:
    """A group of sprite nodes that share a concept and differ along one axis.

    M215 generates the members as an identity-locked sequence: one base image,
    then N variants where only `variation_axis` changes.
    """
    family_id: str
    member_node_ids: List[str]
    base_concept: str               # e.g. "feature map"
    variation_axis: str             # e.g. "channel decomposition density"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family_id": self.family_id,
            "member_node_ids": list(self.member_node_ids),
            "base_concept": self.base_concept,
            "variation_axis": self.variation_axis,
        }


@dataclass
class ClassificationReport:
    """Diagnostics written to PipelineReport.diagnostics so the frontend can
    show "40 nodes → 6 sprite, 3 families, rest vector"."""
    total_leaves: int = 0
    mode_counts: Dict[str, int] = field(default_factory=dict)
    operator_count: int = 0
    families: List[SpriteFamily] = field(default_factory=list)
    demoted_node_ids: List[str] = field(default_factory=list)  # sprite→text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_leaves": self.total_leaves,
            "mode_counts": dict(self.mode_counts),
            "operator_count": self.operator_count,
            "families": [f.to_dict() for f in self.families],
            "family_count": len(self.families),
            "demoted_node_ids": list(self.demoted_node_ids),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  §5  The classifier — one decision per leaf
# ═══════════════════════════════════════════════════════════════════════════

def _node_label(node: Dict[str, Any]) -> str:
    labels = node.get("labels") or []
    if labels and isinstance(labels[0], dict):
        return (labels[0].get("text") or "").strip()
    return (node.get("id") or "").strip()


def _is_operator(label: str, icon_hint: str) -> bool:
    if _OPERATOR_RE.match(label) or _ASCII_OPERATOR_RE.match(label):
        return True
    h = icon_hint.lower()
    return any(cue in h for cue in _OPERATOR_ICON_HINTS)


def _icon_alias_hit(label: str, icon_hint: str, aliases: Dict[str, str]) -> bool:
    """True if the hint or label maps to a known standard Iconify concept.

    Mirrors svg_icon_fetcher._normalize_query's lookup order: prefer iconHint,
    fall back to label; match full string then first word."""
    for raw in (icon_hint, label):
        s = raw.lower().strip()
        if not s:
            continue
        if s in aliases:
            return True
        first = s.split()[0] if " " in s else s
        if first in aliases:
            return True
    return False


def _is_sprite_candidate(label: str, icon_hint: str) -> bool:
    blob = f"{icon_hint} {label}".lower()
    if any(cue in blob for cue in _SPRITE_CUES):
        return True
    # A bare tensor-shape label ("C×H×W") with no standard-icon meaning is a
    # sprite candidate: it wants a drawn feature-map block, not a glyph.
    if _SHAPE_RE.search(blob):
        return True
    return False


def classify_node(node: Dict[str, Any], aliases: Dict[str, str]) -> Tuple[RenderMode, bool]:
    """Classify ONE leaf node.  Returns (render_mode, is_operator).

    Priority order (highest first), so each node lands in exactly one bucket:
        1. operator  → ("text", True)   [M207 vector path]
        2. sprite    → ("sprite", False)
        3. icon      → ("icon", False)
        4. default   → ("text", False)
    """
    label = _node_label(node)
    icon_hint = (node.get("iconHint") or "").strip()

    # 1. Operators: highest priority, zero-cost vector (M207).
    if _is_operator(label, icon_hint):
        return "text", True

    # 2. Sprite candidates: paper-specific visual objects.
    #    Checked BEFORE icon so "feature map" doesn't get swallowed by a
    #    weak first-word alias match.
    if _is_sprite_candidate(label, icon_hint):
        return "sprite", False

    # 3. Standard concepts with a known Iconify alias.
    if _icon_alias_hit(label, icon_hint, aliases):
        return "icon", False

    # 4. Everything else: plain text label.
    return "text", False


# ═══════════════════════════════════════════════════════════════════════════
#  §6  Family detection — group sprite nodes by shared concept
# ═══════════════════════════════════════════════════════════════════════════

# Map a sprite node to its (base_concept, variation_axis) by cue.
_FAMILY_RULES: List[Tuple[Tuple[str, ...], str, str]] = [
    (("feature map", "featuremap", "feature maps", "activation map",
      "response map", "latent map"),
     "feature map", "spatial resolution and channel count"),
    (("frequency", "spectrum", "spectral", "fourier", "wavelet"),
     "frequency representation", "frequency band emphasis"),
    (("decompose", "decomposed", "decomposition", "decomp"),
     "decomposed feature", "decomposition depth"),
    (("selection map", "saliency", "attention map", "heatmap", "heat map"),
     "selection / attention map", "highlighted region"),
    (("patch", "tokens grid", "patch grid"),
     "patch grid", "patch density"),
]


def _family_signature(label: str, icon_hint: str) -> Optional[Tuple[str, str]]:
    blob = f"{icon_hint} {label}".lower()
    for cues, base, axis in _FAMILY_RULES:
        if any(c in blob for c in cues):
            return base, axis
    # Tensor-shape-only sprites with no named concept form a generic family.
    if _SHAPE_RE.search(blob):
        return "tensor block", "tensor shape"
    return None


def detect_sprite_families(
    sprite_nodes: List[Tuple[str, Dict[str, Any]]],
) -> List[SpriteFamily]:
    """Group sprite nodes into families by shared (base_concept, axis).

    Args:
        sprite_nodes: list of (node_id, node_dict) for nodes classified sprite.

    Returns:
        One SpriteFamily per concept that has >= 1 member.  Singletons are
        still returned as a 1-member family so M215 can decide per-family
        whether to use the sequence path (multi-member) or single-shot
        (singleton) generation.
    """
    buckets: Dict[Tuple[str, str], List[str]] = {}
    for node_id, node in sprite_nodes:
        label = _node_label(node)
        icon_hint = (node.get("iconHint") or "").strip()
        sig = _family_signature(label, icon_hint)
        if sig is None:
            # Sprite but unfamiliar — its own singleton family keyed on id.
            sig = (f"misc:{node_id}", "appearance")
        buckets.setdefault(sig, []).append(node_id)

    families: List[SpriteFamily] = []
    for i, ((base, axis), members) in enumerate(sorted(buckets.items())):
        families.append(
            SpriteFamily(
                family_id=f"fam_{i}_{re.sub(r'[^a-z0-9]+', '_', base.lower())[:16]}",
                member_node_ids=members,
                base_concept=base,
                variation_axis=axis,
            )
        )
    return families


# ═══════════════════════════════════════════════════════════════════════════
#  §7  Top-level — walk the ELK graph, mutate renderMode, return report
# ═══════════════════════════════════════════════════════════════════════════

def _walk_leaves(node: Dict[str, Any], out: List[Dict[str, Any]]) -> None:
    children = node.get("children")
    if isinstance(children, list) and children:
        for c in children:
            if isinstance(c, dict):
                _walk_leaves(c, out)
    else:
        out.append(node)


def classify_nodes(elk_graph: Dict[str, Any]) -> ClassificationReport:
    """Classify every leaf node IN PLACE and return a ClassificationReport.

    Mutates each leaf node with:
        node["renderMode"]  : "text" | "icon" | "sprite"
        node["isOperator"]  : bool  (only meaningful when renderMode == "text")
        node["familyId"]    : str   (only set when renderMode == "sprite")

    This is the M206 contract written onto the graph; to-svg.ts reads
    renderMode and dispatches, the sprite pipeline (M211→M214) reads
    familyId / renderMode to know what to generate.
    """
    aliases = _load_icon_aliases()
    report = ClassificationReport()

    leaves: List[Dict[str, Any]] = []
    for child in elk_graph.get("children", []):
        if isinstance(child, dict):
            _walk_leaves(child, leaves)

    report.total_leaves = len(leaves)
    sprite_nodes: List[Tuple[str, Dict[str, Any]]] = []

    for node in leaves:
        mode, is_op = classify_node(node, aliases)
        node["renderMode"] = mode
        node["isOperator"] = is_op
        report.mode_counts[mode] = report.mode_counts.get(mode, 0) + 1
        if is_op:
            report.operator_count += 1
        if mode == "sprite":
            sprite_nodes.append((node.get("id", ""), node))

    # Group sprite nodes into families and stamp familyId onto each member.
    families = detect_sprite_families(sprite_nodes)
    id_to_family = {
        nid: fam.family_id for fam in families for nid in fam.member_node_ids
    }
    for node_id, node in sprite_nodes:
        node["familyId"] = id_to_family.get(node_id, "")
    report.families = families

    logger.info(
        "Classified %d leaves: %s (%d operators, %d sprite families)",
        report.total_leaves, report.mode_counts,
        report.operator_count, len(families),
    )
    return report
