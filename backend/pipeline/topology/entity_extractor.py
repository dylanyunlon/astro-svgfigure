"""entity_extractor.py - Step 1: Paper Text -> Exhaustive Entity List"""
from __future__ import annotations
import json, logging, re
from typing import Any, Dict, List, Tuple
logger = logging.getLogger(__name__)

ENTITY_TYPES = ["input","module","submodule","data_object","operation","resource","output","annotation"]

EXTRACT_SYSTEM = """You are a scientific figure entity extractor. Enumerate EVERY distinct visual element for the system architecture figure.

RULES:
1. Output ONLY a JSON array. No markdown.
2. Each entity: {"name": "short_label", "type": "one_of_8_types", "description": "1 sentence"}
3. Types: input, module, submodule, data_object, operation, resource, output, annotation
4. If text says "A has B, C, D" -> extract A, B, C, D as 4 entities.
5. Hardware (CPU, GPU, RAM, SSD) each get OWN entity.
6. Data structures (Table, Index, Column) each get OWN entity.
7. Operations (Join, Filter, Aggregate) each get OWN entity.
8. Tools (terminal, web search, file ops) -> separate entities."""

EXTRACT_USER = """Extract ALL visual entities from this system description.
I need >= 30 entities. If you return < 20 I will reject and retry.

System Description:
{text}

Output ONLY JSON array:"""


async def extract_entities(
    text: str, ai_engine=None, model: str = "",
    max_retries: int = 3, min_entities: int = 20,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    from backend.pipeline.topology.schema_validator import validate_entities
    if ai_engine is None:
        from backend.config import get_settings
        from backend.ai_engine import AIEngine
        ai_engine = AIEngine(get_settings())
    if not model:
        s = ai_engine._settings
        model = s.ANTHROPIC_DEFAULT_MODEL or s.DEFAULT_MODEL

    best, all_errors = [], []
    for attempt in range(max_retries):
        prompt = EXTRACT_USER.format(text=text[:12000])
        if attempt > 0:
            prompt += f"\n\nRETRY: previous had {len(best)}. Need >= {min_entities}. Be MORE exhaustive."
        try:
            provider = ai_engine._select_provider(model)
            resp = await provider.get_completion(
                messages=[{"role":"system","content":EXTRACT_SYSTEM},
                          {"role":"user","content":prompt}],
                model=model, temperature=0.3+attempt*0.15, max_tokens=8192,
            )
            entities = _parse_json_array(resp.get("content",""))
            if not entities:
                all_errors.append(f"attempt {attempt+1}: JSON parse failed"); continue
            ok, errors = validate_entities(entities, min_entities)
            if ok:
                logger.info("Entity extraction: %d entities on attempt %d", len(entities), attempt+1)
                return entities, []
            all_errors.extend(errors)
            if len(entities) > len(best): best = entities
        except Exception as e:
            all_errors.append(f"attempt {attempt+1}: {e}")
    return best, all_errors


def _parse_json_array(raw: str) -> List[Dict]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r'^```\w*\n?', '', cleaned)
        cleaned = re.sub(r'\n?```$', '', cleaned)
    try:
        data = json.loads(cleaned)
        if isinstance(data, list): return [d for d in data if isinstance(d, dict)]
    except json.JSONDecodeError: pass
    match = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(re.sub(r',\s*([}\]])', r'\1', match.group()))
        except json.JSONDecodeError: pass
    return []
