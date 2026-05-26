"""parse_utils.py — Shared JSON parsing for LLM output.

Extracted from entity_extractor.py, dense_extractor.py, vision_constraint.py
which all had identical _parse_json_array implementations (the exact
anti-pattern that CCCL f984c90 fixes by extracting finalize_pass).

LLM output has common failure modes: markdown fences, trailing commas,
partial JSON, non-array responses. This single implementation handles all.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List


def parse_json_array(raw: str) -> List[Dict[str, Any]]:
    """Parse LLM output as a JSON array of dicts.

    Handles:
        - Markdown ```json fences
        - Trailing commas before ] or }
        - Non-array root (wraps single dict in list)
        - Partial JSON (extracts first [...] match)

    Returns empty list on all failures — never raises.
    """
    cleaned = raw.strip()

    # Strip markdown fences
    if cleaned.startswith("```"):
        cleaned = re.sub(r'^```\w*\n?', '', cleaned)
        cleaned = re.sub(r'\n?```$', '', cleaned)

    # Attempt 1: direct parse
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        if isinstance(data, dict):
            return [data]
        return []
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract [...] and fix trailing commas
    match = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if match:
        try:
            fixed = re.sub(r',\s*([}\]])', r'\1', match.group())
            data = json.loads(fixed)
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
        except json.JSONDecodeError:
            pass

    return []
