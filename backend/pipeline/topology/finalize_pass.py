"""finalize_pass.py — The finalize_pass() template.

CCCL f984c90's crown jewel is the `finalize_pass()` template extraction.
Before the PR, every call site (histogram-only pass, fused filter+histogram
pass) duplicated 40 lines of identical coordination logic:

    __threadfence();
    bool is_last_block = false;
    if (threadIdx.x == 0) {
        unsigned int finished = atomicInc(&counter->finished_block_cnt, ...);
        is_last_block = (finished == (gridDim.x - 1));
    }
    if (__syncthreads_or(is_last_block)) {
        // ... prefix sum, choose_bucket, init_histograms
    }

After the PR, all of that becomes:

    finalize_pass(counter, histogram, current_k, pass, is_last_pass,
                  [counter, ...] { /* caller-specific counter update */ });

The caller-supplied lambda is the only part that differs.  Everything
else — threadfence, last-block detection, prefix sum, bucket selection,
histogram reset — is shared.

We apply the exact same pattern to our pipeline.  Every LLM pass
(region planning, entity extraction, per-region generation) ends with:
  1. Parse raw LLM output (JSON extraction + repair)
  2. Validate structure (schema check)
  3. Fix common errors (edge integrity, bbox bounds)
  4. Update pipeline counters (entity count, edge count, pass number)

The only part that differs per pass is step 2's schema and step 4's
counter update logic.  So we template it.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════
#  §1  Pass Context — the `Counter` struct equivalent
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PassContext:
    """Tracks cumulative state across pipeline passes.

    Like CCCL's `Counter<key_in_t, OffsetT, OutOffsetT>`:
      - pass_number: which radix pass we're on
      - total_entities: running count (like filter_cnt)
      - total_edges: running count
      - total_regions: how many regions planned
      - errors: accumulated validation errors
      - warnings: non-fatal issues
    """
    pass_number: int = 0
    pass_name: str = ""
    total_entities: int = 0
    total_edges: int = 0
    total_regions: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    raw_outputs: List[str] = field(default_factory=list)
    is_last_pass: bool = False

    def advance(self, name: str, is_last: bool = False) -> None:
        """Advance to the next pass."""
        self.pass_number += 1
        self.pass_name = name
        self.is_last_pass = is_last

    def record_error(self, msg: str) -> None:
        self.errors.append(f"[pass {self.pass_number} {self.pass_name}] {msg}")

    def record_warning(self, msg: str) -> None:
        self.warnings.append(f"[pass {self.pass_number} {self.pass_name}] {msg}")

    def stats(self) -> Dict[str, Any]:
        return {
            "passes_completed": self.pass_number,
            "total_entities": self.total_entities,
            "total_edges": self.total_edges,
            "total_regions": self.total_regions,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  §2  Validated Output — the result of finalize_pass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ValidatedOutput:
    """Result of a finalized pass.  Always has parsed data or an error."""
    success: bool
    data: Any = None                # Parsed and validated output
    raw: str = ""                   # Raw LLM output
    errors: List[str] = field(default_factory=list)
    repairs_applied: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
#  §3  Schema validators — the caller-supplied lambdas
# ═══════════════════════════════════════════════════════════════════════════

# Type alias for validator functions: (parsed_data) -> (ok, errors)
Validator = Callable[[Any], Tuple[bool, List[str]]]

# Type alias for counter update functions: (parsed_data, context) -> None
CounterUpdater = Callable[[Any, PassContext], None]


def validate_region_list(data: Any) -> Tuple[bool, List[str]]:
    """Validate region planner output."""
    errors = []
    if not isinstance(data, list):
        return False, ["Expected JSON array of regions"]
    if len(data) == 0:
        return False, ["Empty region list"]
    for i, region in enumerate(data):
        if not isinstance(region, dict):
            errors.append(f"Region {i}: not a dict")
            continue
        if "id" not in region:
            errors.append(f"Region {i}: missing 'id'")
        if "bbox" not in region and not all(k in region for k in ("x", "y", "width", "height")):
            errors.append(f"Region {i}: missing bbox")
        if "name" not in region and "label" not in region:
            errors.append(f"Region {i}: missing name/label")
    return len(errors) == 0, errors


def validate_entity_list(data: Any) -> Tuple[bool, List[str]]:
    """Validate entity extraction output."""
    errors = []
    if not isinstance(data, list):
        return False, ["Expected JSON array of entities"]
    if len(data) < 3:
        errors.append(f"Only {len(data)} entities extracted, expected ≥3")
    for i, entity in enumerate(data):
        if not isinstance(entity, dict):
            errors.append(f"Entity {i}: not a dict")
            continue
        if "name" not in entity:
            errors.append(f"Entity {i}: missing 'name'")
    return len(errors) == 0, errors


def validate_elk_subgraph(data: Any) -> Tuple[bool, List[str]]:
    """Validate ELK subgraph output (per-region)."""
    errors = []
    if not isinstance(data, dict):
        return False, ["Expected JSON object (ELK subgraph)"]
    children = data.get("children", [])
    if not isinstance(children, list):
        errors.append("'children' must be an array")
    edges = data.get("edges", [])
    # Validate edge integrity
    all_ids = _collect_node_ids(data)
    for edge in edges:
        for src in edge.get("sources", []):
            if src not in all_ids:
                errors.append(f"Edge references missing source: {src}")
        for tgt in edge.get("targets", []):
            if tgt not in all_ids:
                errors.append(f"Edge references missing target: {tgt}")
    return len(errors) == 0, errors


def _collect_node_ids(node: Dict[str, Any]) -> set:
    """Recursively collect all node IDs from an ELK graph."""
    ids = set()
    node_id = node.get("id")
    if node_id and node_id != "root":
        ids.add(node_id)
    for child in node.get("children", []):
        if isinstance(child, dict):
            ids.update(_collect_node_ids(child))
    return ids


# ═══════════════════════════════════════════════════════════════════════════
#  §4  The Template — finalize_pass()
# ═══════════════════════════════════════════════════════════════════════════

def finalize_pass(
    raw_output: str,
    context: PassContext,
    validator: Validator,
    counter_updater: CounterUpdater,
    *,
    expect_array: bool = False,
    min_items: int = 0,
) -> ValidatedOutput:
    """Finalize a pipeline pass: parse → validate → fix → update counters.

    This is the direct analogue of CCCL's `finalize_pass()` template.
    The four steps mirror the CUDA version:

    CCCL finalize_pass():
      1. __threadfence()              → we do: JSON parse + repair
      2. atomicInc → is_last_block    → we do: structural validation
      3. compute_bin_offsets()         → we do: edge integrity fix
      4. counter_update_fn()          → we do: caller-supplied updater

    Args:
        raw_output: Raw LLM response string
        context: Mutable pipeline counter state
        validator: Schema-specific validation function
        counter_updater: Caller-supplied state update (the lambda)
        expect_array: If True, parse as JSON array; else JSON object
        min_items: Minimum items in array (if expect_array)

    Returns:
        ValidatedOutput with parsed data or errors
    """
    result = ValidatedOutput(success=False, raw=raw_output)
    context.raw_outputs.append(raw_output[:500] if raw_output else "(empty)")

    # ── Step 1: Parse (the __threadfence equivalent — ensure data is visible) ──
    parsed = _parse_llm_json(raw_output, expect_array=expect_array)
    if parsed is None:
        msg = "Failed to parse JSON from LLM output"
        context.record_error(msg)
        result.errors.append(msg)
        return result

    # ── Step 2: Validate (the is_last_block check — are we done?) ──
    ok, validation_errors = validator(parsed)
    if not ok:
        # Try auto-repair before failing
        repaired, repair_notes = _auto_repair(parsed, validation_errors)
        if repaired is not None:
            result.repairs_applied.extend(repair_notes)
            context.record_warning(f"Applied {len(repair_notes)} auto-repairs")
            parsed = repaired
            ok, validation_errors = validator(parsed)

    if not ok:
        for err in validation_errors:
            context.record_error(err)
        result.errors.extend(validation_errors)
        # Still return partial data — better than nothing
        result.data = parsed
        return result

    # ── Step 3: Min-items check (the prefix sum — enough candidates?) ──
    if expect_array and isinstance(parsed, list) and len(parsed) < min_items:
        msg = f"Got {len(parsed)} items, need ≥{min_items}"
        context.record_warning(msg)
        result.errors.append(msg)
        # Don't fail — partial data is usable

    # ── Step 4: Counter update (the caller-supplied lambda) ──
    try:
        counter_updater(parsed, context)
    except Exception as e:
        context.record_warning(f"Counter updater error: {e}")

    result.success = True
    result.data = parsed
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  §5  JSON Parse + Repair — the __threadfence
# ═══════════════════════════════════════════════════════════════════════════

def _parse_llm_json(
    raw: str,
    *,
    expect_array: bool = False,
) -> Any:
    """Parse LLM output into JSON, handling common LLM mistakes.

    Handles:
      - Markdown fences (```json ... ```)
      - Leading/trailing prose
      - Trailing commas
      - Missing commas between elements
      - Truncated output (unclosed brackets)

    Returns parsed JSON or None on failure.
    """
    if not raw or not raw.strip():
        return None

    text = raw.strip()

    # Remove markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # Remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try direct parse
    try:
        result = json.loads(text)
        if expect_array and isinstance(result, list):
            return result
        if not expect_array and isinstance(result, dict):
            return result
        if isinstance(result, (list, dict)):
            return result  # Accept either if the type check fails
    except json.JSONDecodeError:
        pass

    # Find JSON boundaries
    if expect_array:
        start = text.find("[")
        end = text.rfind("]") + 1
    else:
        start = text.find("{")
        end = text.rfind("}") + 1

    if start < 0 or end <= start:
        # Try the other bracket type as fallback
        start = text.find("[" if not expect_array else "{")
        end = text.rfind("]" if not expect_array else "}") + 1

    if start < 0 or end <= start:
        return None

    candidate = text[start:end]

    # Direct parse
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Repair: trailing commas
    repaired = re.sub(r",(\s*[}\]])", r"\1", candidate)

    # Repair: missing commas between elements
    repaired = re.sub(r"(\})\s*\n\s*(\{)", r"\1,\n\2", repaired)
    repaired = re.sub(r"(\])\s*\n\s*(\[)", r"\1,\n\2", repaired)
    repaired = re.sub(r"(\})\s*\n\s*(\")", r"\1,\n\2", repaired)
    repaired = re.sub(r"(\])\s*\n\s*(\")", r"\1,\n\2", repaired)
    repaired = re.sub(r"(\")\s*\n(\s*\"[^\"]+\"\s*:)", r"\1,\n\2", repaired)

    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Repair: unclosed brackets
    open_b = repaired.count("{") - repaired.count("}")
    open_a = repaired.count("[") - repaired.count("]")
    if open_b > 0 or open_a > 0:
        repaired = repaired.rstrip()
        if repaired.endswith(","):
            repaired = repaired[:-1]
        repaired += "]" * max(0, open_a)
        repaired += "}" * max(0, open_b)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    return None


# ═══════════════════════════════════════════════════════════════════════════
#  §6  Auto-repair — the edge integrity fix
# ═══════════════════════════════════════════════════════════════════════════

def _auto_repair(
    data: Any,
    errors: List[str],
) -> Tuple[Any, List[str]]:
    """Attempt to auto-repair validation errors.

    Like CCCL's edge filtering: items that don't pass the candidate
    check are dropped, not errored.  We apply the same principle:
    fix what we can, drop what we can't, continue.

    Returns:
        (repaired_data, list_of_repair_notes) or (None, []) if unrepairable
    """
    notes: List[str] = []

    if isinstance(data, list):
        # Remove malformed entries from arrays
        cleaned = []
        for item in data:
            if isinstance(item, dict) and ("name" in item or "id" in item or "label" in item):
                cleaned.append(item)
            else:
                notes.append(f"Dropped malformed entry: {str(item)[:80]}")
        if cleaned:
            return cleaned, notes
        return None, []

    if isinstance(data, dict):
        # Fix missing node IDs in ELK graphs
        children = data.get("children", [])
        if isinstance(children, list):
            all_ids = set()
            for child in children:
                if isinstance(child, dict):
                    cid = child.get("id")
                    if not cid:
                        # Generate an ID from name
                        name = child.get("labels", [{}])[0].get("text", "") if child.get("labels") else ""
                        if name:
                            cid = re.sub(r"[^a-zA-Z0-9_]", "_", name.lower()).strip("_")[:50]
                            child["id"] = cid
                            notes.append(f"Generated ID '{cid}' from label")
                    if cid:
                        all_ids.add(cid)
                        # Recurse into nested children
                        for sub in child.get("children", []):
                            if isinstance(sub, dict) and sub.get("id"):
                                all_ids.add(sub["id"])

            # Remove edges referencing missing nodes
            edges = data.get("edges", [])
            if isinstance(edges, list):
                valid_edges = []
                for edge in edges:
                    sources = edge.get("sources", [])
                    targets = edge.get("targets", [])
                    if (all(s in all_ids for s in sources) and
                            all(t in all_ids for t in targets)):
                        valid_edges.append(edge)
                    else:
                        notes.append(f"Dropped edge with missing refs: {edge.get('id', '?')}")
                data["edges"] = valid_edges

            return data, notes

    return None, []
