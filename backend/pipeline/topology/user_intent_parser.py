"""user_intent_parser.py — The extract_bin_op equivalent.

CCCL f984c90 has `extract_bin_op`: a pure function that maps a key to its
radix bin.  No side effects, no global state, no kernel launch.  It runs
inside every thread of every block of every pass — the most-called function
in the entire TopK pipeline.

We need the same: a pure function that maps raw user text to structured
intent.  No LLM calls, no network, no async.  It runs before anything
else in the pipeline and its output feeds every subsequent stage.

Why this matters:
  OLD: user text → LLM prompt → pray the LLM understood "git clone → tree"
  NEW: user text → intent_parser (THIS) → structured intent → LLM prompt
       designed for that specific intent type

The parser detects:
  1. Diagram type (architecture / flowchart / engineering / comparison / recursive)
  2. Entity mentions (technical nouns, proper names, abbreviations)
  3. Containment relationships ("A contains B, C, D" / "inside X there is Y")
  4. Action chains ("clone → inspect → modify → test → deploy")
  5. Style cues ("like mastergo" / "dense" / "minimal" / "nested")
  6. Complexity signals (word count, entity count, nesting depth)

Algorithm: multi-pass regex + heuristic scoring, O(n) in text length.
No ML models, no embeddings, no API calls.  Fast enough to run on every
keystroke in a frontend autocomplete (not that we do, but we could).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


# ═══════════════════════════════════════════════════════════════════════════
#  §1  Intent Types — exhaustive enum of diagram families
# ═══════════════════════════════════════════════════════════════════════════

class DiagramType(str, Enum):
    """Detected diagram family.  Each maps to a different prompt template."""
    ARCHITECTURE = "architecture"      # nested modules, hierarchical
    FLOWCHART = "flowchart"            # sequential steps, pipeline
    ENGINEERING_FLOW = "engineering"   # git clone, tree, experiment
    COMPARISON = "comparison"          # X vs Y vs Z
    RECURSIVE_CHAIN = "recursive"      # "from C start D, let E do F..."
    DATA_FLOW = "data_flow"            # ETL, streaming, data pipeline
    UNKNOWN = "unknown"


class ComplexityLevel(str, Enum):
    SIMPLE = "simple"      # ≤10 entities
    MEDIUM = "medium"      # 11-25 entities
    COMPLEX = "complex"    # 26-50 entities
    DENSE = "dense"        # 51+ entities


# ═══════════════════════════════════════════════════════════════════════════
#  §2  Parsed structures — the output of intent parsing
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EntityMention:
    """A technical noun extracted from user text."""
    name: str
    span: Tuple[int, int]            # (start, end) char positions
    entity_type: str = "unknown"     # module, resource, operation, data, io
    parent_hint: Optional[str] = None  # "inside X" → parent = X


@dataclass
class Relationship:
    """A detected relationship between two entities."""
    source: str
    target: str
    rel_type: str = "flow"  # flow, contains, uses, produces, compares


@dataclass
class ActionStep:
    """One step in an action chain (engineering flow)."""
    verb: str          # clone, inspect, build, test, deploy
    object: str        # the thing being acted on
    tool: Optional[str] = None  # git, tree, make, pytest


@dataclass
class StyleCue:
    """User's layout/visual preference signal."""
    cue_type: str      # density, nesting, direction, reference
    value: str         # "dense", "3+ levels", "LEFT_RIGHT", "like mastergo"


@dataclass
class UserIntent:
    """Complete parsed intent — the contract between parser and pipeline.

    This is the equivalent of CCCL's bucket assignment: every element in
    the user's text has been classified into a bin, and downstream passes
    can operate on the classified data without re-reading the raw text.
    """
    diagram_type: DiagramType
    complexity: ComplexityLevel
    entities: List[EntityMention] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    action_chain: List[ActionStep] = field(default_factory=list)
    style_cues: List[StyleCue] = field(default_factory=list)
    containment_groups: Dict[str, List[str]] = field(default_factory=dict)
    raw_text: str = ""
    estimated_regions: int = 1
    estimated_entities: int = 0
    confidence: float = 0.0  # 0-1, how confident in diagram_type

    def summary(self) -> Dict[str, Any]:
        return {
            "type": self.diagram_type.value,
            "complexity": self.complexity.value,
            "entities": len(self.entities),
            "relationships": len(self.relationships),
            "action_steps": len(self.action_chain),
            "containment_groups": len(self.containment_groups),
            "style_cues": [c.value for c in self.style_cues],
            "estimated_regions": self.estimated_regions,
            "confidence": round(self.confidence, 2),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  §3  Pattern libraries — the radix bins
# ═══════════════════════════════════════════════════════════════════════════

# Engineering/DevOps action verbs (signals ENGINEERING_FLOW)
_ENGINEERING_VERBS = {
    "clone", "fork", "pull", "push", "commit", "merge", "rebase",
    "build", "compile", "link", "install", "deploy", "ship",
    "test", "benchmark", "profile", "debug", "trace",
    "inspect", "tree", "ls", "cat", "grep", "find",
    "configure", "setup", "init", "scaffold",
    "experiment", "iterate", "prototype", "refactor",
}

# Academic/ML keywords (signals ARCHITECTURE or FLOWCHART)
_ACADEMIC_NOUNS = {
    "encoder", "decoder", "attention", "transformer", "embedding",
    "convolution", "pooling", "normalization", "dropout", "softmax",
    "backbone", "head", "neck", "feature", "loss", "gradient",
    "optimizer", "scheduler", "checkpoint", "epoch", "batch",
    "layer", "block", "module", "network", "model",
    "input", "output", "hidden", "latent", "residual",
    "diffusion", "gan", "vae", "autoencoder", "classifier",
}

# Containment markers
_CONTAINMENT_PATTERNS = [
    re.compile(r"(\w[\w\s]{1,30}?)\s+(?:contains?|has|includes?|comprises?)\s+(.+?)(?:\.|;|$)", re.I),
    re.compile(r"(?:inside|within|in)\s+(\w[\w\s]{1,30}?)\s*[,:]\s*(.+?)(?:\.|;|$)", re.I),
    re.compile(r"(\w[\w\s]{1,30}?)\s+(?:consists?\s+of|is\s+made\s+(?:up\s+)?of)\s+(.+?)(?:\.|;|$)", re.I),
]

# Comparison markers
_COMPARISON_PATTERNS = [
    re.compile(r"(\w+)\s+vs\.?\s+(\w+)", re.I),
    re.compile(r"compare\s+(.+?)\s+(?:with|to|and|against)\s+(.+?)(?:\.|;|$)", re.I),
    re.compile(r"differences?\s+between\s+(.+?)\s+and\s+(.+?)(?:\.|;|$)", re.I),
]

# Recursive chain pattern: "from X start, then Y, let Z do W..."
_RECURSIVE_PATTERN = re.compile(
    r"从\s*(\S+)\s*(?:这个好?例子)?开始|from\s+(\S+)\s+start",
    re.I,
)

# Action chain: "verb + object" sequences
_ACTION_PATTERN = re.compile(
    r"\b(clone|fork|pull|build|test|deploy|inspect|tree|configure|install|"
    r"experiment|prototype|refactor|implement|create|generate|run|execute|"
    r"git\s+\w+)\s+([\w\-./]+)",
    re.I,
)

# Technical noun extractor (CamelCase, ALLCAPS, dot.notation, hyphenated)
_TECH_NOUN_PATTERN = re.compile(
    r"\b(?:"
    r"[A-Z][a-z]+(?:[A-Z][a-z]+)+"   # CamelCase: ResNet, LayerNorm
    r"|[A-Z]{2,}(?:\d+)?"             # ALLCAPS: GPU, CPU, NCCL, NVLINK
    r"|[a-z]+(?:[-_][a-z]+){1,}"      # hyphenated: self-attention, feed-forward
    r"|[a-z]+\.[a-z]+(?:\.[a-z]+)*"   # dotted: megatron.core, torch.nn
    r")\b"
)

# Style cue patterns
_STYLE_PATTERNS = {
    "density": re.compile(r"\b(dense|detailed|comprehensive|exhaustive|minimal|simple|sparse)\b", re.I),
    "nesting": re.compile(r"\b(\d+)\s*(?:层|levels?|layers?\s+deep|nesting)", re.I),
    "direction": re.compile(r"\b(left.?to.?right|top.?to.?bottom|horizontal|vertical|DOWN|RIGHT|LEFT|UP)\b", re.I),
    "reference": re.compile(r"(?:like|像|参考|similar\s+to)\s+(mastergo|figma|miro|draw\.io|excalidraw)", re.I),
}


# ═══════════════════════════════════════════════════════════════════════════
#  §4  Parser — the main extraction pipeline
# ═══════════════════════════════════════════════════════════════════════════

def parse_user_intent(text: str) -> UserIntent:
    """Parse raw user text into structured intent.  Zero LLM calls.

    Algorithm (3-pass, O(n)):
      Pass 1 — Entity extraction: scan for technical nouns, proper names
      Pass 2 — Relationship extraction: scan for containment, flow, comparison
      Pass 3 — Type classification: score each DiagramType, pick highest

    Like CCCL's extract_bin_op: pure function, deterministic, fast.
    Called before any LLM interaction to shape the prompts.

    Args:
        text: Raw user input, any language, any length

    Returns:
        UserIntent with classified type, extracted entities, relationships
    """
    if not text or not text.strip():
        return UserIntent(
            diagram_type=DiagramType.UNKNOWN,
            complexity=ComplexityLevel.SIMPLE,
            raw_text=text or "",
        )

    intent = UserIntent(
        diagram_type=DiagramType.UNKNOWN,
        complexity=ComplexityLevel.SIMPLE,
        raw_text=text,
    )

    # ── Pass 1: Entity extraction ──
    _extract_entities(text, intent)

    # ── Pass 2: Relationship extraction ──
    _extract_relationships(text, intent)
    _extract_action_chain(text, intent)
    _extract_containment(text, intent)
    _extract_style_cues(text, intent)

    # ── Pass 3: Type classification (barycenter scoring) ──
    _classify_diagram_type(text, intent)

    # ── Post-processing ──
    _estimate_complexity(intent)
    _estimate_regions(intent)

    return intent


# ═══════════════════════════════════════════════════════════════════════════
#  §5  Pass 1 — Entity extraction
# ═══════════════════════════════════════════════════════════════════════════

def _extract_entities(text: str, intent: UserIntent) -> None:
    """Extract technical nouns from text.

    Strategy: multi-pattern scan, dedup by normalized name.
    Like CCCL's per-thread histogram accumulation — each pattern
    contributes to the entity count, then we merge.
    """
    seen: Set[str] = set()

    # Pattern A: CamelCase, ALLCAPS, hyphenated, dotted
    for m in _TECH_NOUN_PATTERN.finditer(text):
        name = m.group()
        normalized = name.lower().replace("-", "_").replace(".", "_")
        if normalized not in seen and len(name) > 1:
            seen.add(normalized)
            entity_type = _classify_entity_type(name)
            intent.entities.append(EntityMention(
                name=name,
                span=(m.start(), m.end()),
                entity_type=entity_type,
            ))

    # Pattern B: Known academic/ML terms (lowercase match)
    text_lower = text.lower()
    for term in _ACADEMIC_NOUNS:
        if term in text_lower and term not in seen:
            seen.add(term)
            idx = text_lower.index(term)
            intent.entities.append(EntityMention(
                name=term.title(),
                span=(idx, idx + len(term)),
                entity_type="module",
            ))

    # Pattern C: Hardware resources (special handling — always their own entity)
    hw_keywords = {"cpu", "gpu", "ram", "ssd", "hdd", "simd", "nvlink", "nccl"}
    for hw in hw_keywords:
        if hw in text_lower and hw not in seen:
            seen.add(hw)
            idx = text_lower.index(hw)
            intent.entities.append(EntityMention(
                name=hw.upper(),
                span=(idx, idx + len(hw)),
                entity_type="resource",
            ))

    intent.estimated_entities = len(intent.entities)


def _classify_entity_type(name: str) -> str:
    """Classify an entity name into a type bucket.

    Like CCCL's `candidate_class` enum: selected, candidate, or neither.
    """
    lower = name.lower()
    if any(kw in lower for kw in ("input", "source", "request", "query", "schema", "config")):
        return "input"
    if any(kw in lower for kw in ("output", "result", "response", "target", "sink")):
        return "output"
    if any(kw in lower for kw in ("cpu", "gpu", "ram", "ssd", "hdd", "cache", "memory")):
        return "resource"
    if any(kw in lower for kw in ("join", "filter", "sort", "aggregate", "scan", "merge", "split")):
        return "operation"
    if any(kw in lower for kw in ("table", "index", "column", "store", "queue", "buffer", "stack")):
        return "data"
    return "module"


# ═══════════════════════════════════════════════════════════════════════════
#  §6  Pass 2 — Relationship and pattern extraction
# ═══════════════════════════════════════════════════════════════════════════

def _extract_relationships(text: str, intent: UserIntent) -> None:
    """Extract flow relationships from sequential patterns.

    Detect: "A → B → C", "A then B", "A followed by B", "A produces B"
    """
    # Arrow patterns: → , ->, =>, --
    arrow_pattern = re.compile(
        r"(\w[\w\s]{0,25}?)\s*(?:→|->|=>|──>|—>)\s*(\w[\w\s]{0,25}?)"
        r"(?=\s*(?:→|->|=>|──>|—>|[.,;]|$))",
    )
    for m in arrow_pattern.finditer(text):
        src, tgt = m.group(1).strip(), m.group(2).strip()
        if src and tgt and len(src) > 1 and len(tgt) > 1:
            intent.relationships.append(Relationship(
                source=src, target=tgt, rel_type="flow",
            ))

    # Sequential patterns: "first A, then B", "A followed by B"
    seq_pattern = re.compile(
        r"(?:first|then|next|followed\s+by|after\s+that|finally|subsequently)"
        r"\s*,?\s*(?:a\s+|the\s+)?(\w[\w\s]{1,30}?)\s*(?:[.,;]|$)",
        re.I,
    )
    previous: Optional[str] = None
    for m in seq_pattern.finditer(text):
        current = m.group(1).strip()
        if previous and current:
            intent.relationships.append(Relationship(
                source=previous, target=current, rel_type="flow",
            ))
        previous = current

    # Numbered steps: (1) A  (2) B  (3) C
    step_pattern = re.compile(r"[\(\[（]?(\d+)[）\)\].]?\s*(?:a\s+|the\s+)?(\w[\w\s]{1,40}?)(?=[.;,\n]|$)")
    steps: List[str] = []
    for m in step_pattern.finditer(text):
        steps.append(m.group(2).strip())
    for i in range(len(steps) - 1):
        intent.relationships.append(Relationship(
            source=steps[i], target=steps[i + 1], rel_type="flow",
        ))


def _extract_action_chain(text: str, intent: UserIntent) -> None:
    """Extract engineering action chains.

    Detect: "git clone X, tree Y, build Z, test W"
    These are sequential imperative actions — the user is describing
    a workflow, not a static architecture.
    """
    for m in _ACTION_PATTERN.finditer(text):
        verb_raw = m.group(1).strip().lower()
        obj = m.group(2).strip()

        # Split "git clone" into tool="git", verb="clone"
        tool = None
        if verb_raw.startswith("git "):
            tool = "git"
            verb_raw = verb_raw[4:]

        intent.action_chain.append(ActionStep(
            verb=verb_raw,
            object=obj,
            tool=tool,
        ))


def _extract_containment(text: str, intent: UserIntent) -> None:
    """Extract parent-child containment relationships.

    "Transformer contains encoder, decoder, attention"
    → containment_groups["Transformer"] = ["encoder", "decoder", "attention"]
    """
    for pattern in _CONTAINMENT_PATTERNS:
        for m in pattern.finditer(text):
            parent = m.group(1).strip()
            children_raw = m.group(2)
            # Split on commas, "and", Chinese 、
            children = re.split(r"[,、]\s*(?:and\s+)?|\s+and\s+", children_raw)
            children = [c.strip() for c in children if c.strip() and len(c.strip()) > 1]
            if parent and children:
                if parent not in intent.containment_groups:
                    intent.containment_groups[parent] = []
                intent.containment_groups[parent].extend(children)

                # Also create relationships
                for child in children:
                    intent.relationships.append(Relationship(
                        source=parent, target=child, rel_type="contains",
                    ))


def _extract_style_cues(text: str, intent: UserIntent) -> None:
    """Extract user style preferences."""
    for cue_type, pattern in _STYLE_PATTERNS.items():
        m = pattern.search(text)
        if m:
            intent.style_cues.append(StyleCue(
                cue_type=cue_type,
                value=m.group(1) if m.lastindex else m.group(0),
            ))


# ═══════════════════════════════════════════════════════════════════════════
#  §7  Pass 3 — Type classification (barycenter scoring)
# ═══════════════════════════════════════════════════════════════════════════

def _classify_diagram_type(text: str, intent: UserIntent) -> None:
    """Score each diagram type and pick the highest.

    Like CCCL's barycenter heuristic in Sugiyama: compute a weighted
    average position for each node, then sort.  Here we compute a
    weighted score for each diagram type, then pick the max.

    Scoring signals:
      ENGINEERING_FLOW: engineering verbs, git/cli tools, action chains
      ARCHITECTURE: containment groups, nesting keywords, "module/component"
      FLOWCHART: sequential markers, numbered steps, "pipeline/process"
      COMPARISON: "vs", "compare", "difference"
      RECURSIVE_CHAIN: "from X start" pattern, chained implementations
      DATA_FLOW: ETL keywords, streaming, data pipeline
    """
    text_lower = text.lower()
    scores: Dict[DiagramType, float] = {t: 0.0 for t in DiagramType}

    # ── Engineering signals ──
    eng_verb_count = sum(1 for v in _ENGINEERING_VERBS if v in text_lower)
    scores[DiagramType.ENGINEERING_FLOW] += eng_verb_count * 3.0
    scores[DiagramType.ENGINEERING_FLOW] += len(intent.action_chain) * 5.0
    if re.search(r"git\s+(clone|branch|checkout|pull)", text_lower):
        scores[DiagramType.ENGINEERING_FLOW] += 10.0

    # ── Architecture signals ──
    arch_keywords = ["module", "component", "layer", "block", "subsystem",
                     "architecture", "hierarchy", "nested", "contains"]
    scores[DiagramType.ARCHITECTURE] += sum(
        2.0 for kw in arch_keywords if kw in text_lower
    )
    scores[DiagramType.ARCHITECTURE] += len(intent.containment_groups) * 8.0

    # ── Flowchart signals ──
    flow_keywords = ["step", "stage", "phase", "pipeline", "process",
                     "workflow", "sequence", "first", "then", "finally"]
    scores[DiagramType.FLOWCHART] += sum(
        2.0 for kw in flow_keywords if kw in text_lower
    )
    numbered_steps = len(re.findall(r"[\(\[（]\d+[）\)\]]", text))
    scores[DiagramType.FLOWCHART] += numbered_steps * 3.0

    # ── Comparison signals ──
    for pattern in _COMPARISON_PATTERNS:
        scores[DiagramType.COMPARISON] += len(pattern.findall(text)) * 8.0

    # ── Recursive chain signals ──
    if _RECURSIVE_PATTERN.search(text):
        scores[DiagramType.RECURSIVE_CHAIN] += 15.0
    # Detect chained "implement X, let Y do Z" patterns
    chain_count = len(re.findall(
        r"(?:实现|implement|create|build|let|使|令)\s+\w+\s+(?:可以|能够|can|do|support)",
        text_lower,
    ))
    scores[DiagramType.RECURSIVE_CHAIN] += chain_count * 4.0

    # ── Data flow signals ──
    data_keywords = ["etl", "streaming", "kafka", "queue", "pipeline",
                     "ingest", "transform", "load", "data flow"]
    scores[DiagramType.DATA_FLOW] += sum(
        3.0 for kw in data_keywords if kw in text_lower
    )

    # ── Pick the winner ──
    best_type = max(scores, key=lambda t: scores[t])
    best_score = scores[best_type]
    total_score = sum(scores.values()) or 1.0

    # Fall back to UNKNOWN if no signal is strong enough
    if best_score < 3.0:
        intent.diagram_type = DiagramType.UNKNOWN
        intent.confidence = 0.0
    else:
        intent.diagram_type = best_type
        intent.confidence = min(1.0, best_score / total_score * 2.0)


# ═══════════════════════════════════════════════════════════════════════════
#  §8  Post-processing — complexity and region estimation
# ═══════════════════════════════════════════════════════════════════════════

def _estimate_complexity(intent: UserIntent) -> None:
    """Estimate complexity from extracted entity count."""
    n = len(intent.entities) + len(intent.containment_groups) * 3
    if n <= 10:
        intent.complexity = ComplexityLevel.SIMPLE
    elif n <= 25:
        intent.complexity = ComplexityLevel.MEDIUM
    elif n <= 50:
        intent.complexity = ComplexityLevel.COMPLEX
    else:
        intent.complexity = ComplexityLevel.DENSE


def _estimate_regions(intent: UserIntent) -> None:
    """Estimate how many independent regions the figure needs.

    Like CCCL's `num_tiles`: how many thread blocks to launch.
    More regions = more parallel LLM calls = better per-region quality.

    Heuristics:
      - Each top-level containment group → 1 region
      - Each action chain segment (3-5 steps) → 1 region
      - Minimum 1, maximum 8 (diminishing returns beyond 8)
    """
    regions = max(1, len(intent.containment_groups))

    # Action chains: every 4 steps is a region
    if intent.action_chain:
        regions = max(regions, (len(intent.action_chain) + 3) // 4)

    # Entity-based: every 8-10 entities is a region
    if intent.estimated_entities > 10:
        entity_regions = (intent.estimated_entities + 7) // 8
        regions = max(regions, entity_regions)

    # Complexity floor
    if intent.complexity == ComplexityLevel.COMPLEX:
        regions = max(regions, 4)
    elif intent.complexity == ComplexityLevel.DENSE:
        regions = max(regions, 6)

    intent.estimated_regions = min(8, max(1, regions))
