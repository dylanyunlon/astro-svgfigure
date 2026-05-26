"""dense_extractor.py — Two-pass entity extraction for mastergo-quality density.

Problem: Single LLM call extracts ~15 high-level modules. GenDB figure has ~70
         visual elements (icons, labels, data objects, resource chips, arrows).

Solution: Two-pass extraction.
  Pass 1 — Module scan: extract major components (agents, pipelines, I/O blocks)
  Pass 2 — Element drill: for each module, extract its visual sub-elements
           (icons, labels, data items, operation nodes)

Design reference: Google's MapReduce pattern.
  Pass 1 = Map phase: partition the text into module-scoped chunks.
  Pass 2 = Reduce phase: within each chunk, enumerate every drawable element.

From NVIDIA Megatron-Core's tensor-parallel attention pattern:
  The attention block is one "module" but contains Q/K/V projections, softmax,
  dropout, output projection — each a separate visual element.  We apply the
  same decomposition principle: every module gets drilled into its sub-parts.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  §1  Pass 1 — Module-level extraction (coarse)
# ═══════════════════════════════════════════════════════════════════════════

MODULE_SYSTEM = """\
You are a scientific figure architect. Given a system description, enumerate
EVERY major module, pipeline stage, input source, output target, and resource
type as a structured JSON array.

CRITICAL: The goal is to produce a COMPLETE architecture diagram. Every
component mentioned in the text must appear. If the text says "X uses Y and Z",
that's 3 entities minimum (X, Y, Z).

Output ONLY a JSON array. Each element:
{
  "name": "Short Label (max 4 words)",
  "type": "module|input|output|resource|data_store|operation|annotation",
  "children_hint": ["sub-component names mentioned in text"],
  "visual_hint": "icon or visual representation description",
  "text_span": "exact quote from input that mentions this"
}

Rules:
- Hardware resources (CPU, GPU, RAM, SSD, HDD) → separate entities, type="resource"
- Data stores (Table, Index, Column, Storage) → separate entities, type="data_store"
- I/O (Schema, SQL, Data, Config, REQ) → separate entities, type="input" or "output"
- Agents/modules → type="module" with children_hint listing sub-components
- Operations (Join, Filter, Aggregate, Scan) → type="operation"
- "..." / "etc" / enumeration markers → type="annotation"
- Output >= 30 entities for any non-trivial system description"""

MODULE_USER = """\
Extract ALL visual entities from this system description.
Target: >= {min_entities} entities. Be EXHAUSTIVE — every noun that would
appear in the figure is an entity.

System Description:
---
{text}
---

Output ONLY JSON array:"""


# ═══════════════════════════════════════════════════════════════════════════
#  §2  Pass 2 — Sub-element drill (fine-grained)
# ═══════════════════════════════════════════════════════════════════════════

DRILL_SYSTEM = """\
You are breaking down a system module into its individual VISUAL elements
for a detailed architecture diagram.

For the given module and its context, list every visual sub-element that
would appear INSIDE this module's box in the figure:
- Internal processing steps
- Data objects it produces/consumes
- Icons representing hardware/tools it uses
- Labels and annotations
- Sub-arrows or internal data flow

Output ONLY a JSON array:
[{"name": "sub_label", "type": "submodule|icon|label|data_object|operation",
  "visual_hint": "description for icon generation"}]"""

DRILL_USER = """\
Module: "{module_name}" — {module_type}
Children mentioned: {children}
Context: {context}

List ALL visual sub-elements (aim for 3-8 per module):"""


# ═══════════════════════════════════════════════════════════════════════════
#  §3  Orchestrator — two-pass extraction
# ═══════════════════════════════════════════════════════════════════════════

async def extract_dense(
    text: str,
    ai_engine=None,
    model: str = "",
    min_entities: int = 30,
    drill_modules: bool = True,
    max_retries: int = 3,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Two-pass entity extraction for mastergo-quality density.

    Returns:
        (entities, diagnostics) where entities is a flat list and
        diagnostics tracks pass1/pass2 counts and errors.
    """
    if ai_engine is None:
        from backend.config import get_settings
        from backend.ai_engine import AIEngine
        ai_engine = AIEngine(get_settings())
    if not model:
        s = ai_engine._settings
        model = s.ANTHROPIC_DEFAULT_MODEL or s.DEFAULT_MODEL

    diag = {"pass1": {}, "pass2": {}, "errors": []}

    # ── Pass 1: Module-level scan ──
    modules = []
    for attempt in range(max_retries):
        prompt = MODULE_USER.format(text=text[:12000], min_entities=min_entities)
        if attempt > 0:
            prompt += f"\n\nRETRY #{attempt+1}: previous had {len(modules)}. Need >= {min_entities}."

        try:
            provider = ai_engine._select_provider(model)
            resp = await provider.get_completion(
                messages=[
                    {"role": "system", "content": MODULE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                temperature=0.3 + attempt * 0.1,
                max_tokens=8192,
            )
            parsed = _parse_json_array(resp.get("content", ""))
            if len(parsed) >= min_entities:
                modules = parsed
                break
            if len(parsed) > len(modules):
                modules = parsed
        except Exception as e:
            diag["errors"].append(f"pass1 attempt {attempt+1}: {e}")

    diag["pass1"] = {"count": len(modules), "attempts": attempt + 1}

    if not modules:
        return [], diag

    # ── Pass 2: Sub-element drill ──
    all_entities = list(modules)  # Start with pass1 entities
    drill_count = 0

    if drill_modules:
        # Only drill modules that have children_hint
        drillable = [
            m for m in modules
            if m.get("type") in ("module", "data_store")
            and m.get("children_hint")
            and len(m.get("children_hint", [])) > 0
        ]

        for module in drillable[:15]:  # Cap at 15 drills to manage API calls
            try:
                children = module.get("children_hint", [])
                context_span = module.get("text_span", "")

                resp = await ai_engine._select_provider(model).get_completion(
                    messages=[
                        {"role": "system", "content": DRILL_SYSTEM},
                        {"role": "user", "content": DRILL_USER.format(
                            module_name=module["name"],
                            module_type=module["type"],
                            children=json.dumps(children),
                            context=context_span[:500] if context_span else "See main description",
                        )},
                    ],
                    model=model,
                    temperature=0.3,
                    max_tokens=2048,
                )
                subs = _parse_json_array(resp.get("content", ""))
                # Tag sub-elements with parent
                for sub in subs:
                    sub["parent_module"] = module["name"]
                    if not sub.get("type"):
                        sub["type"] = "submodule"
                all_entities.extend(subs)
                drill_count += len(subs)

            except Exception as e:
                diag["errors"].append(f"drill '{module['name']}': {e}")

    diag["pass2"] = {"drilled_modules": len([m for m in modules if m.get("children_hint")]),
                     "sub_elements": drill_count}
    diag["total"] = len(all_entities)

    # ── Dedup by name ──
    seen = set()
    deduped = []
    for e in all_entities:
        name = e.get("name", "").strip().lower()
        if name and name not in seen:
            seen.add(name)
            deduped.append(e)

    diag["after_dedup"] = len(deduped)

    return deduped, diag


# ═══════════════════════════════════════════════════════════════════════════
#  §4  Implicit element injection — add visual elements the LLM often misses
# ═══════════════════════════════════════════════════════════════════════════

# Elements that appear in architecture figures but LLMs consistently skip
IMPLICIT_ELEMENTS = {
    # Hardware resource icons (when mentioned in text)
    "cpu": {"name": "CPU", "type": "resource", "visual_hint": "microprocessor chip"},
    "gpu": {"name": "GPU", "type": "resource", "visual_hint": "GPU accelerator card"},
    "ram": {"name": "RAM", "type": "resource", "visual_hint": "memory module"},
    "ssd": {"name": "SSD", "type": "resource", "visual_hint": "solid state drive"},
    "hdd": {"name": "HDD", "type": "resource", "visual_hint": "hard disk drive"},
    "simd": {"name": "SIMD", "type": "resource", "visual_hint": "SIMD vector unit"},
    # Data structure icons
    "table": {"name": "Table", "type": "data_store", "visual_hint": "data table grid"},
    "index": {"name": "Index", "type": "data_store", "visual_hint": "B-tree index"},
    "column": {"name": "Column", "type": "data_store", "visual_hint": "vertical data column"},
    # I/O elements
    "schema": {"name": "Schema", "type": "input", "visual_hint": "database schema icon"},
    "sql": {"name": "SQL", "type": "input", "visual_hint": "SQL query icon"},
    "config": {"name": "Config.", "type": "input", "visual_hint": "settings gear icon"},
    # Common annotations
    "ellipsis": {"name": "...", "type": "annotation", "visual_hint": "ellipsis dots"},
}


def inject_implicit(
    entities: List[Dict[str, Any]],
    text: str,
) -> List[Dict[str, Any]]:
    """Add commonly-missed visual elements that the text mentions but LLM skips.

    Scans the source text for keywords, checks if the entity list already has
    them, and injects missing ones.
    """
    existing_names = {e.get("name", "").lower() for e in entities}
    text_lower = text.lower()

    injected = list(entities)
    for keyword, template in IMPLICIT_ELEMENTS.items():
        if keyword in text_lower and template["name"].lower() not in existing_names:
            injected.append(dict(template))

    return injected


# ═══════════════════════════════════════════════════════════════════════════
#  §5  Complexity-adaptive thresholds
# ═══════════════════════════════════════════════════════════════════════════

def adaptive_thresholds(text: str) -> Dict[str, int]:
    """Compute extraction thresholds based on text complexity.

    Uses sentence count + technical term density to set min_entities and
    min_edges dynamically.
    """
    sentences = len(re.split(r'[.!?]+', text))
    # Count technical terms (Capitalized multi-word phrases)
    tech_terms = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text))
    # Count enumeration patterns ("A, B, C, and D")
    enum_count = len(re.findall(r'(?:,\s*(?:and\s+)?[A-Z])', text))

    base = len(tech_terms) + enum_count
    min_entities = max(20, min(80, base + sentences // 3))
    min_edges = max(15, min_entities - 5)

    return {
        "min_entities": min_entities,
        "min_edges": min_edges,
        "sentences": sentences,
        "tech_terms": len(tech_terms),
        "enum_patterns": enum_count,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  §6  JSON parsing (robust, handles LLM quirks)
# ═══════════════════════════════════════════════════════════════════════════

def _parse_json_array(raw: str) -> List[Dict]:
    """Parse LLM output as JSON array, handling common issues."""
    cleaned = raw.strip()

    # Strip markdown fences
    if cleaned.startswith("```"):
        cleaned = re.sub(r'^```\w*\n?', '', cleaned)
        cleaned = re.sub(r'\n?```$', '', cleaned)

    # Try direct parse
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    except json.JSONDecodeError:
        pass

    # Try extracting array
    match = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if match:
        try:
            # Fix trailing commas
            fixed = re.sub(r',\s*([}\]])', r'\1', match.group())
            data = json.loads(fixed)
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
        except json.JSONDecodeError:
            pass

    return []
