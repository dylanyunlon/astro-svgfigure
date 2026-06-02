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

RenderMode = Literal["text", "icon", "sprite", "kernel"]


# ═══════════════════════════════════════════════════════════════════════════
#  §1  Operator detection — the highest-priority, zero-cost path
# ═══════════════════════════════════════════════════════════════════════════

# Math operators that M207 (renderMathOperator) draws as pure SVG.  These are
# matched in node labels.  Kept deliberately tight: a lone glyph or the glyph
# with whitespace, not e.g. a "+" inside a longer word.
_OPERATOR_GLYPHS = "⊗⊕⊙⊘⊖⊛∘×·⨂⨁∑∏∫"
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
#
# Design principle: in academic figures (FreqSelect, AdaDR, Pix2Struct),
# almost every node on the operator chain is a VISUAL object — feature maps,
# encoder outputs, hidden states, kernel weights, probability distributions.
# These are best conveyed as drawn illustrations (3D tensor slabs, heatmaps,
# grid visualizations), not as text labels or generic Iconify icons.
#
# The classifier is intentionally BROAD here: it's better to generate a
# small illustration for "ViT Encoder" than to show a generic gear icon.
# The sprite pipeline can always fall back to blob if generation fails.
_SPRITE_CUES = (
    # Feature maps / tensor volumes
    "feature map", "feature maps", "featuremap",
    "feature", "features",  # "Input feature", "Output feature"
    "tensor", "volume", "c x h x w", "c×h×w", "cxhxw", "h x w",
    "channel", "channels",
    # Frequency / spectral
    "frequency", "spectrum", "spectral", "fourier", "wavelet",
    "decompose", "decomposed", "decomposition", "decomp",
    "low-freq", "high-freq", "band",
    # Attention / selection / activation
    "selection map", "attention map", "saliency", "heat map", "heatmap",
    "activation map", "response map", "latent map", "embedding map",
    "dilation map", "dilation rate",
    # Patches / grids / kernels
    "patch grid", "patches", "tokens grid", "patch",
    "spatial", "receptive field", "kernel grid", "filter bank",
    "kernel", "convolution kernel", "adaptive kernel",
    # Encoders / decoders / model components (visual blocks in academic figs)
    "encoder", "decoder", "transformer", "vit",
    "hidden state", "hidden states", "state vector",
    "embedding", "embeddings", "vector embedding",
    # Neural network layer outputs
    "softmax", "sigmoid", "relu", "activation",
    "prediction", "token prediction", "next-token",
    "probability", "distribution",
    # Visual outputs
    "output feature", "input feature", "global feature",
    "dom tree", "dom layout", "coarse dom",
    "code generation", "code agent",
    # Image / screenshot inputs
    "screenshot", "image", "photograph", "resolution",
    "leaf node image", "leaf image",
    # Weight matrices
    "weight matrix", "weight", "matrix",
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


def _is_kernel_node(label: str, icon_hint: str) -> bool:
    """M410: Detect kernel/filter/weight nodes that need grid rendering.

    These get renderMode='kernel' → NxN weighted cell grid (AdaKern reference).
    Distinct from sprite because they use deterministic geometry, not AI images.
    """
    blob = f"{icon_hint} {label}".lower()
    _KERNEL_KEYWORDS = (
        "static kernel", "low-freq kernel", "high-freq kernel",
        "adaptive kernel", "conv kernel", "convolution kernel",
        "kernel grid", "filter bank", "weight matrix",
        "low-freq kern", "hi-freq kern",
    )
    return any(kw in blob for kw in _KERNEL_KEYWORDS)


def _is_heatmap_node(label: str, icon_hint: str) -> bool:
    """M410: Detect heatmap/selection-map nodes.

    These are single-channel spatial maps (1×H×W) rendered as color gradients.
    They get renderMode='sprite' with a specialized heatmap prompt.
    """
    blob = f"{icon_hint} {label}".lower()
    _HEATMAP_KEYWORDS = (
        "selection map", "dilation map", "attention map",
        "saliency map", "heat map", "heatmap", "activation map",
        "response map",
    )
    return any(kw in blob for kw in _HEATMAP_KEYWORDS)


def classify_node(node: Dict[str, Any], aliases: Dict[str, str]) -> Tuple[RenderMode, bool]:
    """Classify ONE leaf node.  Returns (render_mode, is_operator).

    M410 enhanced priority order (highest first):
        1. operator  → ("text", True)   [M207 vector path: ⊗ ⊕ ⊛ ⊖]
        2. kernel    → ("kernel", False) [M402 grid path: static/adaptive kernel]
        3. sprite    → ("sprite", False) [AI illustration: feature maps, heatmaps]
        4. icon      → ("icon", False)   [Iconify standard glyph]
        5. default   → ("text", False)   [plain label]

    M410 additions:
        - kernel renderMode for kernel/filter/weight nodes
        - familyId auto-inheritance from parent group (set in classify_nodes)
        - heatmap nodes get sprite + specialized prompt hint
    """
    label = _node_label(node)
    icon_hint = (node.get("iconHint") or "").strip()

    # 1. Operators: highest priority, zero-cost vector (M207).
    if _is_operator(label, icon_hint):
        return "text", True

    # 2. Kernel nodes: NxN grid rendering (M402 renderKernelGrid).
    if _is_kernel_node(label, icon_hint):
        return "kernel", False

    # 3. Sprite candidates: paper-specific visual objects.
    #    Checked BEFORE icon so "feature map" doesn't get swallowed by a
    #    weak first-word alias match.
    #    M410: heatmap nodes also land here but get a prompt hint.
    if _is_sprite_candidate(label, icon_hint):
        # M410: tag heatmap nodes with a prompt hint for Gemini
        if _is_heatmap_node(label, icon_hint):
            node["_spriteHint"] = "heatmap"
        return "sprite", False

    # 4. Standard concepts with a known Iconify alias.
    if _icon_alias_hit(label, icon_hint, aliases):
        return "icon", False

    # 5. Everything else: plain text label.
    return "text", False


# ═══════════════════════════════════════════════════════════════════════════
#  §6  Family detection — group sprite nodes by shared concept
# ═══════════════════════════════════════════════════════════════════════════

# Map a sprite node to its (base_concept, variation_axis) by cue.
_FAMILY_RULES: List[Tuple[Tuple[str, ...], str, str]] = [
    (("feature map", "featuremap", "feature maps", "activation map",
      "response map", "latent map", "feature", "output feature",
      "input feature", "global feature"),
     "feature map", "spatial resolution and channel count"),
    (("frequency", "spectrum", "spectral", "fourier", "wavelet",
      "low-freq", "high-freq", "band"),
     "frequency representation", "frequency band emphasis"),
    (("decompose", "decomposed", "decomposition", "decomp"),
     "decomposed feature", "decomposition depth"),
    (("selection map", "saliency", "attention map", "heatmap", "heat map",
      "dilation map", "dilation"),
     "selection / attention map", "highlighted region"),
    (("patch", "tokens grid", "patch grid", "image patch"),
     "patch grid", "patch density"),
    (("encoder", "vit", "vision transformer"),
     "encoder block", "layer depth"),
    (("decoder", "transformer decoder"),
     "decoder block", "decoding stage"),
    (("hidden state", "state vector", "embedding", "vector"),
     "hidden representation", "representation layer"),
    (("kernel", "filter bank", "convolution"),
     "convolution kernel", "kernel configuration"),
    (("softmax", "sigmoid", "relu", "activation", "probability"),
     "activation function", "nonlinearity type"),
    (("prediction", "token prediction", "next-token"),
     "prediction output", "sequence position"),
    (("screenshot", "image", "photograph", "resolution"),
     "input image", "resolution"),
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
    """Group sprite nodes into families by shared visual concept.

    Three-tier strategy:
      1. Try LLM-powered grouping (Claude) — semantic similarity.
      2. Fall back to rule-based _family_signature().
      3. Post-process: enforce MAX_FAMILY_SIZE cap — any family bigger
         than the Gemini batch limit gets split into sub-families,
         so each Gemini call stays small and focused.

    This is the fix for "gemini一次性接受太多内容了" — even when the LLM
    lumps 15 nodes into one family, Gemini only sees ≤6 at a time.
    """
    MAX_FAMILY_SIZE = 6  # Gemini sweet spot: ≤6 images per call

    # ── Diagnostic: dump input state ──
    logger.info(
        "┌─ detect_sprite_families: %d sprite nodes entering",
        len(sprite_nodes),
    )
    for i, (nid, nd) in enumerate(sprite_nodes):
        logger.info(
            "│  [%02d] id=%-24s label=%-20s hint=%-12s",
            i, nid, _node_label(nd)[:20], (nd.get("iconHint") or "")[:12],
        )

    # ── Tier 1: Try LLM-powered grouping ──
    raw_families: Optional[List[SpriteFamily]] = None
    try:
        from backend.config import get_settings
        settings = get_settings()
        api_key = settings.ANTHROPIC_API_KEY or settings.CLAUDE_COMPATIBLE_API_KEY
        if api_key and len(sprite_nodes) >= 2:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    raw_families = pool.submit(
                        lambda: asyncio.run(_llm_detect_families(sprite_nodes, settings))
                    ).result(timeout=30)
            else:
                raw_families = asyncio.run(_llm_detect_families(sprite_nodes, settings))

            if raw_families:
                logger.info(
                    "│  LLM returned %d raw families", len(raw_families),
                )
    except Exception as e:
        logger.warning("│  LLM family detection failed, falling back to rules: %s", e)

    # ── Tier 2: Rule-based fallback ──
    if not raw_families:
        raw_families = _rule_based_families(sprite_nodes)
        logger.info("│  Rule-based fallback produced %d families", len(raw_families))

    # ── Tier 3: Enforce MAX_FAMILY_SIZE — split oversized families ──
    # This is the key fix: a 15-member family becomes three 5-member
    # sub-families, each getting its own Gemini batch call.
    capped_families: List[SpriteFamily] = []
    for fam in raw_families:
        members = fam.member_node_ids
        if len(members) <= MAX_FAMILY_SIZE:
            capped_families.append(fam)
        else:
            # Split into sub-families of ≤MAX_FAMILY_SIZE
            n_splits = (len(members) + MAX_FAMILY_SIZE - 1) // MAX_FAMILY_SIZE
            logger.info(
                "│  ✂ Splitting oversized family %s (%d members) → %d sub-families",
                fam.family_id, len(members), n_splits,
            )
            for k in range(0, len(members), MAX_FAMILY_SIZE):
                chunk = members[k:k + MAX_FAMILY_SIZE]
                sub_id = f"{fam.family_id}_s{k // MAX_FAMILY_SIZE}"
                capped_families.append(SpriteFamily(
                    family_id=sub_id,
                    member_node_ids=chunk,
                    base_concept=fam.base_concept,
                    variation_axis=fam.variation_axis,
                ))

    # ── Diagnostic: print final family layout ──
    logger.info("├─ Final family layout: %d families", len(capped_families))
    total_assigned = 0
    for fam in capped_families:
        logger.info(
            "│  fam=%-28s members=%d  concept=%s  axis=%s  ids=%s",
            fam.family_id, len(fam.member_node_ids),
            fam.base_concept[:30], fam.variation_axis[:20],
            fam.member_node_ids,
        )
        total_assigned += len(fam.member_node_ids)
    logger.info(
        "└─ %d/%d nodes assigned to families (%.0f%% coverage)",
        total_assigned, len(sprite_nodes),
        100 * total_assigned / max(len(sprite_nodes), 1),
    )

    return capped_families


def _rule_based_families(
    sprite_nodes: List[Tuple[str, Dict[str, Any]]],
) -> List[SpriteFamily]:
    """Original rule-based family detection (kept as fallback)."""
    buckets: Dict[Tuple[str, str], List[str]] = {}
    for node_id, node in sprite_nodes:
        label = _node_label(node)
        icon_hint = (node.get("iconHint") or "").strip()
        sig = _family_signature(label, icon_hint)
        if sig is None:
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


async def _llm_detect_families(
    sprite_nodes: List[Tuple[str, Dict[str, Any]]],
    settings: Any,
) -> Optional[List[SpriteFamily]]:
    """Call Claude to semantically group sprite nodes into families.

    Prompt: give Claude the list of (id, label, iconHint), ask it to return
    JSON grouping them into families of visually similar concepts.
    Each family gets one Gemini batch call for sequence-consistent sprites.
    """
    import json as _json

    # Build compact node list for the prompt
    node_list = []
    for node_id, node in sprite_nodes:
        label = _node_label(node)
        hint = (node.get("iconHint") or "").strip()
        node_list.append({"id": node_id, "label": label, "iconHint": hint})

    prompt = (
        "You are grouping nodes from a scientific paper figure into visual families "
        "for batch sprite generation. Nodes in the same family will be drawn with "
        "the same visual style (same shape, color palette, composition) but differ "
        "in one dimension.\n\n"
        "Rules:\n"
        "- Group nodes that represent the SAME visual concept (e.g. all 'input images' "
        "together, all 'code/document outputs' together, all 'neural network layers' together)\n"
        "- HARD LIMIT: each family MUST have ≤6 members (Gemini generates "
        "one batch per family; >6 images per call causes timeout)\n"
        "- Aim for 2-6 members per family; singletons OK for unique concepts\n"
        "- A node can only belong to ONE family\n"
        "- Aim for 4-8 families total, not 30+ singletons\n\n"
        f"Nodes ({len(node_list)}):\n"
    )
    for n in node_list:
        prompt += f'  - id="{n["id"]}", label="{n["label"]}", iconHint="{n["iconHint"]}"\n'

    prompt += (
        "\nRespond with ONLY a JSON array, no markdown, no explanation:\n"
        '[\n'
        '  {"family": "short_name", "concept": "what they look like", '
        '"axis": "how members differ", "members": ["id1", "id2", ...]},\n'
        '  ...\n'
        ']\n'
    )

    # Call Claude
    api_key = settings.ANTHROPIC_API_KEY or settings.CLAUDE_COMPATIBLE_API_KEY
    api_base = settings.ANTHROPIC_API_BASE if settings.ANTHROPIC_API_KEY else settings.CLAUDE_COMPATIBLE_API_BASE
    model = "claude-sonnet-4-20250514"

    import httpx
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{api_base}/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": model,
                "max_tokens": 2048,
                "temperature": 0.2,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        data = resp.json()

    # Parse response
    text = ""
    for block in data.get("content", []):
        if block.get("type") == "text":
            text += block.get("text", "")

    # Extract JSON from response (handle markdown fences)
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```\w*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)
        text = text.strip()

    groups = _json.loads(text)
    if not isinstance(groups, list):
        return None

    # Convert to SpriteFamily objects
    valid_ids = {nid for nid, _ in sprite_nodes}
    families: List[SpriteFamily] = []
    claimed: set = set()

    for i, g in enumerate(groups):
        members = [m for m in g.get("members", []) if m in valid_ids and m not in claimed]
        if not members:
            continue
        claimed.update(members)
        fam_name = g.get("family", f"group_{i}")
        families.append(SpriteFamily(
            family_id=f"fam_{i}_{re.sub(r'[^a-z0-9]+', '_', fam_name.lower())[:16]}",
            member_node_ids=members,
            base_concept=g.get("concept", fam_name),
            variation_axis=g.get("axis", "appearance"),
        ))

    # Catch unclaimed nodes — put them in a misc family
    unclaimed = [nid for nid, _ in sprite_nodes if nid not in claimed]
    if unclaimed:
        families.append(SpriteFamily(
            family_id=f"fam_{len(families)}_unclaimed",
            member_node_ids=unclaimed,
            base_concept="miscellaneous",
            variation_axis="appearance",
        ))

    logger.info(
        "LLM grouped %d sprite nodes into %d families: %s",
        len(sprite_nodes), len(families),
        [(f.family_id, len(f.member_node_ids)) for f in families],
    )
    return families if families else None


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
        node["renderMode"]  : "text" | "icon" | "sprite" | "kernel"
        node["isOperator"]  : bool  (only meaningful when renderMode == "text")
        node["familyId"]    : str   (only set when renderMode == "sprite")

    M410 enhancements:
        - kernel renderMode for kernel/filter/weight nodes (AdaKern pattern)
        - familyId auto-inheritance: sprite nodes under same parent group
          share a familyId based on the parent group's ID
        - heatmap nodes tagged with _spriteHint for specialized Gemini prompts
    """
    aliases = _load_icon_aliases()
    report = ClassificationReport()

    leaves: List[Dict[str, Any]] = []
    # M410: also track parent group context for familyId inheritance
    parent_map: Dict[str, str] = {}  # node_id → parent_group_id

    def _walk_with_parent(node: Dict[str, Any], parent_group_id: str) -> None:
        children = node.get("children")
        if isinstance(children, list) and children:
            # This is a group node — use its ID as parent for children
            group_id = node.get("id", parent_group_id)
            for c in children:
                if isinstance(c, dict):
                    _walk_with_parent(c, group_id)
        else:
            leaves.append(node)
            node_id = node.get("id", "")
            if node_id and parent_group_id:
                parent_map[node_id] = parent_group_id

    for child in elk_graph.get("children", []):
        if isinstance(child, dict):
            _walk_with_parent(child, "")

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
        fam_id = id_to_family.get(node_id, "")
        # M410: If no family detected, inherit from parent group
        if not fam_id and node_id in parent_map:
            fam_id = f"group_{parent_map[node_id]}"
        node["familyId"] = fam_id
    report.families = families

    logger.info(
        "Classified %d leaves: %s (%d operators, %d sprite families)",
        report.total_leaves, report.mode_counts,
        report.operator_count, len(families),
    )
    return report


# ═══════════════════════════════════════════════════════════════════════════
#  §8  Layer consolidation — keep 2–3 largest parent groups, merge the rest
# ═══════════════════════════════════════════════════════════════════════════
#
# Academic figures like AdaKern / FreqSelect have a clear visual hierarchy:
#   - 2–3 major regions (FreqSelect, AdaDR, AdaKern) with colored backgrounds
#   - Many small sub-elements (operators, feature maps, kernels)
#
# The problem: the topology generator often creates too many nested groups,
# producing visual clutter.  This pass consolidates:
#   1. Count descendant leaves per top-level group
#   2. Keep the TOP_K (default 3) largest groups as parent regions with borders
#   3. Merge smaller sibling groups into the nearest large parent
#   4. Flatten single-child groups (remove wrapper if only 1 child)
#
# This mirrors the user's request: "只保留 2–3 个最大的父类元素群仍保留外部框,
# 其余小子元素群分到大父类" — keep only 2–3 largest parent element groups
# with their outer border, merge remaining small sub-groups into the large parents.

TOP_K_PARENTS = 3  # max parent regions to keep as bordered groups


def _count_leaves(node: Dict[str, Any]) -> int:
    """Count leaf descendants of a node (including itself if it's a leaf)."""
    children = node.get("children")
    if not isinstance(children, list) or not children:
        return 1
    return sum(_count_leaves(c) for c in children if isinstance(c, dict))


def _flatten_single_child_groups(node: Dict[str, Any]) -> None:
    """Recursively flatten groups that have exactly one child group.

    If a group has one child that is also a group, promote the grandchildren
    up and merge labels (keeps the deeper label to preserve semantic info).
    """
    children = node.get("children")
    if not isinstance(children, list):
        return

    i = 0
    while i < len(children):
        child = children[i]
        if not isinstance(child, dict):
            i += 1
            continue

        # Recurse first
        _flatten_single_child_groups(child)

        # Check: is this a single-child group whose only child is also a group?
        cc = child.get("children")
        if (isinstance(cc, list) and len(cc) == 1
                and isinstance(cc[0], dict)
                and isinstance(cc[0].get("children"), list)
                and len(cc[0].get("children", [])) > 0):
            grandchild = cc[0]
            # Promote grandchild's children into this child
            child["children"] = grandchild.get("children", [])
            # Keep grandchild's label if it's more specific
            gc_labels = grandchild.get("labels")
            if gc_labels and isinstance(gc_labels, list) and gc_labels:
                child.setdefault("labels", gc_labels)
            # Re-process the same index
            continue
        i += 1


def consolidate_layers(
    elk_graph: Dict[str, Any],
    top_k: int = TOP_K_PARENTS,
) -> Dict[str, Any]:
    """Consolidate the ELK graph's top-level groups into at most `top_k` parents.

    Steps:
      1. Flatten single-child wrapper groups
      2. Rank top-level groups by descendant count
      3. Keep top_k largest as bordered parent regions
      4. Merge smaller groups' children into the nearest large parent
         (nearest = smallest Euclidean distance between bounding-box centers)
      5. Mark kept parents with `_isTopRegion: true` for the renderer

    Returns the mutated elk_graph.
    """
    children = elk_graph.get("children")
    if not isinstance(children, list) or len(children) <= top_k:
        # Nothing to consolidate
        return elk_graph

    # Step 1: flatten single-child wrappers
    _flatten_single_child_groups(elk_graph)

    # Step 2: rank by descendant count
    groups = []
    leaves_at_top = []
    for c in children:
        if not isinstance(c, dict):
            continue
        cc = c.get("children")
        if isinstance(cc, list) and len(cc) > 0:
            groups.append((c, _count_leaves(c)))
        else:
            leaves_at_top.append(c)

    if len(groups) <= top_k:
        return elk_graph

    # Sort descending by leaf count
    groups.sort(key=lambda pair: pair[1], reverse=True)
    kept = groups[:top_k]
    merged = groups[top_k:]

    # Mark kept parents
    for g, _ in kept:
        g["_isTopRegion"] = True

    # Step 3: merge smaller groups into nearest kept parent
    def _bbox_center(node: Dict[str, Any]):
        x = node.get("x", 0)
        y = node.get("y", 0)
        w = node.get("width", 160)
        h = node.get("height", 60)
        return (x + w / 2, y + h / 2)

    kept_centers = [(_bbox_center(g), g) for g, _ in kept]

    for small_group, _ in merged:
        scx, scy = _bbox_center(small_group)
        # Find nearest kept parent
        best_dist = float("inf")
        best_parent = kept[0][0]
        for (kcx, kcy), kgroup in kept_centers:
            d = ((scx - kcx) ** 2 + (scy - kcy) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_parent = kgroup

        # Move small group's children into best_parent
        small_children = small_group.get("children", [])
        if isinstance(small_children, list):
            best_parent.setdefault("children", []).extend(small_children)
        else:
            # It's a leaf masquerading as a group — add the node itself
            best_parent.setdefault("children", []).append(small_group)

        logger.debug(
            "Merged group '%s' (%d leaves) into '%s'",
            small_group.get("id", "?"),
            _count_leaves(small_group),
            best_parent.get("id", "?"),
        )

    # Rebuild elk_graph.children: kept groups + top-level leaves
    new_children = [g for g, _ in kept] + leaves_at_top
    elk_graph["children"] = new_children

    logger.info(
        "Layer consolidation: %d groups → %d parents (%d merged, %d top-level leaves)",
        len(groups), len(kept), len(merged), len(leaves_at_top),
    )
    return elk_graph