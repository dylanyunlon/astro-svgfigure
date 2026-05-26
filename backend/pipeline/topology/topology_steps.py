"""topology_steps.py - Steps 2-5 of the topology pipeline."""
from __future__ import annotations
import json, logging, re, math
from typing import Any, Dict, List, Tuple
logger = logging.getLogger(__name__)

EDGE_SYSTEM = """Map relationships between system components.
Output ONLY a JSON array: [{"source":"name","target":"name","label":"2-5 words","type":"data_flow|feedback|contains|uses"}]
source/target MUST be exact entity names. Every entity needs >= 1 edge."""

EDGE_USER = """Map ALL relationships:
{entities_json}
Return >= {min_edges} edges as JSON array."""

async def map_relationships(entities, ai_engine=None, model="", max_retries=3):
    from backend.pipeline.topology.schema_validator import validate_edges
    entity_names = {e["name"] for e in entities}
    min_edges = max(15, len(entities)-3)
    if ai_engine is None:
        from backend.config import get_settings
        from backend.ai_engine import AIEngine
        ai_engine = AIEngine(get_settings())
    if not model:
        model = ai_engine._settings.ANTHROPIC_DEFAULT_MODEL or ai_engine._settings.DEFAULT_MODEL
    best, errs = [], []
    for attempt in range(max_retries):
        prompt = EDGE_USER.format(
            entities_json=json.dumps([{"name":e["name"],"type":e["type"]} for e in entities], indent=1),
            min_edges=min_edges)
        if attempt > 0: prompt += f"\nRETRY: had {len(best)}, need >= {min_edges}."
        try:
            resp = await ai_engine._select_provider(model).get_completion(
                messages=[{"role":"system","content":EDGE_SYSTEM},{"role":"user","content":prompt}],
                model=model, temperature=0.3, max_tokens=4096)
            edges = _parse_json(resp.get("content",""))
            ok, e = validate_edges(edges, entity_names, min_edges)
            if ok: return edges, []
            errs.extend(e)
            if len(edges) > len(best): best = edges
        except Exception as e: errs.append(str(e))
    return best, errs

HIER_SYSTEM = """Organize entities into visual groups for architecture diagram.
Output ONLY JSON: [{"name":"group_id","label":"Title","children":["entity1","entity2"]}]
Every entity must be in exactly one group's children.

NESTING: If a group is logically inside another group, nest it:
[{"name":"outer","label":"Outer","children":["inner_group"]},
 {"name":"inner_group","label":"Inner","children":["entity_a","entity_b"]}]
Architecture figures typically need 2-4 nesting levels. Do NOT flatten everything."""

async def build_hierarchy(entities, ai_engine=None, model=""):
    from backend.pipeline.topology.schema_validator import validate_hierarchy
    entity_names = {e["name"] for e in entities}
    if ai_engine is None:
        from backend.config import get_settings
        from backend.ai_engine import AIEngine
        ai_engine = AIEngine(get_settings())
    if not model:
        model = ai_engine._settings.ANTHROPIC_DEFAULT_MODEL or ai_engine._settings.DEFAULT_MODEL
    try:
        resp = await ai_engine._select_provider(model).get_completion(
            messages=[{"role":"system","content":HIER_SYSTEM},
                      {"role":"user","content":f"Organize: {json.dumps([e['name'] for e in entities])}"}],
            model=model, temperature=0.3, max_tokens=4096)
        groups = _parse_json(resp.get("content",""))
        ok, e = validate_hierarchy(groups, entity_names)
        return groups, e
    except Exception as e: return [], [str(e)]

_ICON_MAP = {
    "cpu":"microprocessor chip","gpu":"GPU accelerator","ram":"memory module",
    "ssd":"storage drive","hdd":"hard disk","cache":"cache memory",
    "join":"merge arrows","filter":"funnel filter","aggregate":"sigma symbol",
    "scan":"sequential read","index":"tree index","table":"data table grid",
    "column":"vertical bars","terminal":"command window","web":"browser globe",
    "file":"document file","code":"code brackets","sql":"database query",
    "optimizer":"performance gauge","planner":"flowchart","analyzer":"analysis chart",
    "schema":"schema diagram","executable":"program file","exe":"program file",
    "llm":"AI brain","agent":"robot agent","ai":"neural network",
    "user":"user panel","config":"settings gear","storage":"database cylinder",
    "query":"search magnifier","compiler":"build hammer","mmap":"memory map",
}
_TYPE_DEFAULT = {
    "input":"input panel","module":"gear module","submodule":"small tool",
    "data_object":"data file","operation":"transform arrows","resource":"hardware chip",
    "output":"result document","annotation":"info badge",
}

def classify_icons(entities):
    icons = {}
    for e in entities:
        name = e.get("name","").lower()
        hit = None
        for kw, hint in _ICON_MAP.items():
            if kw in name: hit = hint; break
        icons[e["name"]] = hit or _TYPE_DEFAULT.get(e.get("type","module"), "generic block")
    return icons

def assemble_elk(entities, edges, groups, icons):
    entity_map = {e["name"]: e for e in entities}
    group_children = {}
    for g in groups:
        for c in g.get("children",[]): group_children[c] = g["name"]
    def _id(n): return re.sub(r'[^a-zA-Z0-9_]','_',n.lower()).strip('_')[:50]
    def _node(e):
        w = max(150, min(280, len(e.get("name",""))*10+40))
        h = 60 if e.get("type") in ("input","output","resource","annotation") else 80
        n = {"id":_id(e["name"]),"width":w,"height":h,"labels":[{"text":e["name"]}]}
        if icons.get(e["name"]): n["iconHint"] = icons[e["name"]]
        return n
    def _edge(e):
        adv = {"semanticType":e.get("type","data_flow"),"edgeLabels":[{"text":e.get("label","")}]}
        if e.get("type")=="feedback": adv["lineStyle"]="dashed"; adv["strokeColor"]="#9C27B0"
        return {"id":f"e_{_id(e['source'])}_{_id(e['target'])}",
                "sources":[_id(e["source"])],"targets":[_id(e["target"])],"advanced":adv}

    root_children, root_edges = [], []
    for g in groups:
        ch = [_node(entity_map[c]) for c in g.get("children",[]) if c in entity_map]
        gids = {_id(c) for c in g.get("children",[])}
        ie = [_edge(e) for e in edges if _id(e["source"]) in gids and _id(e["target"]) in gids]
        if ch:
            w = max(n["width"] for n in ch)+40
            h = sum(n["height"] for n in ch)+len(ch)*20+50
            root_children.append({
                "id":_id(g["name"]),"width":w,"height":h,
                "labels":[{"text":g.get("label",g["name"])}],
                "layoutOptions":{"elk.padding":"[top=35,left=12,bottom=12,right=12]"},
                "children":ch,"edges":ie,"group":True,"borderless":True})
    for e in entities:
        if e["name"] not in group_children and e["name"] not in {g["name"] for g in groups}:
            root_children.append(_node(e))
    for edge in edges:
        sg, tg = group_children.get(edge["source"]), group_children.get(edge["target"])
        if sg != tg or sg is None: root_edges.append(_edge(edge))
    return {"id":"root","layoutOptions":{"elk.algorithm":"layered","elk.direction":"DOWN"},
            "children":root_children,"edges":root_edges}

async def generate_rich_topology(text, ai_engine=None, model=""):
    from backend.pipeline.topology.entity_extractor import extract_entities
    diag = {"steps":{}}
    entities, e1 = await extract_entities(text, ai_engine, model)
    diag["steps"]["entities"] = {"count":len(entities),"errors":e1}
    if not entities: return {"id":"root","children":[]}, diag
    edges, e2 = await map_relationships(entities, ai_engine, model)
    diag["steps"]["edges"] = {"count":len(edges),"errors":e2}
    groups, e3 = await build_hierarchy(entities, ai_engine, model)
    diag["steps"]["hierarchy"] = {"count":len(groups),"errors":e3}
    icons = classify_icons(entities)
    diag["steps"]["icons"] = {"count":len(icons)}
    elk = assemble_elk(entities, edges, groups, icons)
    from backend.pipeline.topology.schema_validator import validate_elk
    ok, e5 = validate_elk(elk)
    diag["steps"]["assembly"] = {"valid":ok,"errors":e5}
    diag["total_entities"] = len(entities)
    diag["total_edges"] = len(edges)
    return elk, diag


async def generate_mastergo_topology(
    text: str,
    image_b64: str = "",
    ai_engine=None,
    model: str = "",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Enhanced pipeline producing mastergo-quality topology with 50+ elements.

    Pipeline steps:
      1. Dense entity extraction (two-pass: modules + sub-element drill)
      2. Implicit element injection (hardware icons, data objects LLM missed)
      3. Adaptive threshold relationship mapping
      4. Deep hierarchy building (3+ nesting levels)
      5. Icon classification with visual_hint from entities
      6. ELK assembly with sub-element generation
      7. Vision constraint alignment (if screenshot provided)
      8. MastergoLayout conversion with full element detail

    Args:
        text: Paper method description
        image_b64: Optional screenshot for vision-guided layout
        ai_engine: AIEngine instance
        model: LLM model to use

    Returns:
        (elk_graph, diagnostics) — elk_graph has mastergo-quality node density
    """
    from backend.pipeline.topology.dense_extractor import (
        extract_dense, inject_implicit, adaptive_thresholds,
    )
    from backend.pipeline.topology.mastergo_schema import (
        elk_to_mastergo_layout, estimate_figure_complexity,
    )
    from backend.pipeline.topology.schema_validator import validate_elk

    if ai_engine is None:
        from backend.config import get_settings
        from backend.ai_engine import AIEngine
        ai_engine = AIEngine(get_settings())
    if not model:
        s = ai_engine._settings
        model = s.ANTHROPIC_DEFAULT_MODEL or s.DEFAULT_MODEL

    diag = {"steps": {}, "pipeline": "mastergo_dense"}

    # ── Step 0: Complexity estimation ──
    complexity = estimate_figure_complexity(text)
    thresholds = adaptive_thresholds(text)
    min_entities = max(complexity["min_entities"], thresholds["min_entities"])
    diag["steps"]["complexity"] = complexity
    diag["steps"]["thresholds"] = thresholds

    # ── Step 1: Dense entity extraction (two-pass) ──
    entities, extract_diag = await extract_dense(
        text, ai_engine, model,
        min_entities=min_entities,
        drill_modules=True,
    )
    diag["steps"]["dense_extract"] = extract_diag

    if not entities:
        logger.warning("Dense extraction produced 0 entities, falling back to basic")
        return await generate_rich_topology(text, ai_engine, model)

    # ── Step 2: Implicit element injection ──
    entities = inject_implicit(entities, text)
    diag["steps"]["implicit_inject"] = {"count_after": len(entities)}

    # ── Step 3: Relationship mapping with adaptive thresholds ──
    min_edges = max(thresholds["min_edges"], len(entities) // 3)
    edges, e2 = await map_relationships(entities, ai_engine, model)
    diag["steps"]["edges"] = {"count": len(edges), "errors": e2}

    # ── Step 4: Deep hierarchy building ──
    groups, e3 = await build_hierarchy(entities, ai_engine, model)
    diag["steps"]["hierarchy"] = {"count": len(groups), "errors": e3}

    # ── Step 5: Icon classification (prefer visual_hint from dense extractor) ──
    icons = {}
    for e in entities:
        # Use visual_hint from dense extractor if available
        if e.get("visual_hint"):
            icons[e["name"]] = e["visual_hint"]
        else:
            name = e.get("name", "").lower()
            hit = None
            for kw, hint in _ICON_MAP.items():
                if kw in name:
                    hit = hint
                    break
            icons[e["name"]] = hit or _TYPE_DEFAULT.get(e.get("type", "module"), "generic block")
    diag["steps"]["icons"] = {"count": len(icons)}

    # ── Step 6: ELK assembly ──
    elk = assemble_elk(entities, edges, groups, icons)

    # ── Step 7: Vision constraint alignment ──
    if image_b64:
        try:
            from backend.pipeline.topology.vision_constraint import (
                vision_constrained_layout,
            )
            elk, identified_regions, vision_diag = await vision_constrained_layout(
                image_b64, elk, ai_engine, model,
            )
            diag["steps"]["vision_constraint"] = vision_diag
        except Exception as e:
            logger.warning("Vision constraint failed (non-fatal): %s", e)
            diag["steps"]["vision_constraint"] = {"error": str(e)}

    # ── Step 8: Validation ──
    ok, e5 = validate_elk(elk)
    diag["steps"]["assembly"] = {"valid": ok, "errors": e5}
    diag["total_entities"] = len(entities)
    diag["total_edges"] = len(edges)

    # ── MastergoLayout stats (for diagnostics, not returned as main output) ──
    try:
        mastergo = elk_to_mastergo_layout(elk)
        diag["mastergo_stats"] = mastergo.stats()
    except Exception:
        pass

    return elk, diag


async def generate_constrained_topology(
    text: str,
    image_b64: str = "",
    ai_engine=None,
    model: str = "",
    canvas_width: int = 900,
    canvas_height: int = 500,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """WHITE BOX topology generation.

    The LLM decides WHAT (entities, relationships, groups).
    The constraint system decides WHERE and HOW BIG — deterministically.

    This replaces the black-box approach:
        OLD: LLM → assemble_elk(entities) → hope sizes are right
        NEW: LLM → Canonicalizer → Solver → guaranteed correct sizes

    Architecture (osdk-ts pattern):
        Registry      = type → size/padding rules (like ObjectCacheKeyRegistry)
        Canonicalizer  = normalize LLM output (like GenericCanonicalizer)
        Solver         = deterministic position computation (like Store)

    Pipeline:
        1. Entity extraction (LLM — decides WHAT)
        2. Relationship mapping (LLM — decides connections)
        3. Hierarchy building (LLM — decides grouping)
        4. Canonicalization (RULES — normalizes types, computes sizes)
        5. Constraint solving (RULES — computes positions)
        6. Output in ELK or mastergo format
    """
    from backend.pipeline.topology.entity_extractor import extract_entities
    from backend.pipeline.topology.constraint import (
        ConstraintRegistry, LayoutCanonicalizer, ConstraintSolver,
    )

    if ai_engine is None:
        from backend.config import get_settings
        from backend.ai_engine import AIEngine
        ai_engine = AIEngine(get_settings())
    if not model:
        s = ai_engine._settings
        model = s.ANTHROPIC_DEFAULT_MODEL or s.DEFAULT_MODEL

    diag = {"steps": {}, "pipeline": "constrained"}

    # ── Step 1: Entity extraction (LLM decides WHAT) ──
    entities, e1 = await extract_entities(text, ai_engine, model)
    diag["steps"]["entities"] = {"count": len(entities), "errors": e1}
    if not entities:
        return {"id": "root", "children": [], "edges": []}, diag

    # ── Step 2: Relationship mapping (LLM decides connections) ──
    edges, e2 = await map_relationships(entities, ai_engine, model)
    diag["steps"]["edges"] = {"count": len(edges), "errors": e2}

    # ── Step 3: Hierarchy building (LLM decides grouping) ──
    groups, e3 = await build_hierarchy(entities, ai_engine, model)
    diag["steps"]["hierarchy"] = {"count": len(groups), "errors": e3}

    # ════════════════════════════════════════════════════════════
    # BELOW THIS LINE: NO MORE LLM CALLS. PURE DETERMINISTIC RULES.
    # ════════════════════════════════════════════════════════════

    # ── Step 4: Canonicalization (RULES decide types + sizes) ──
    registry = ConstraintRegistry()
    canon = LayoutCanonicalizer(registry)

    c_elements = canon.canonicalize_entities(entities)
    c_element_ids = {e.id for e in c_elements}
    c_edges = canon.canonicalize_edges(edges, c_element_ids)
    c_groups = canon.canonicalize_groups(groups, c_elements)

    diag["steps"]["canonicalize"] = {
        "elements": len(c_elements),
        "edges": len(c_edges),
        "groups": len(c_groups),
        "group_types": {g.id: g.layout_type for g in c_groups},
    }

    # ── Step 5: Constraint solving (RULES decide positions) ──
    solver = ConstraintSolver(registry, canvas_width, canvas_height)
    layout = solver.solve(c_elements, c_edges, c_groups)

    diag["steps"]["solver"] = layout.stats()
    diag["total_entities"] = len(c_elements)
    diag["total_edges"] = len(c_edges)

    # ── Output: ELK graph with deterministic positions ──
    elk = layout.to_elk_graph()

    # Also provide mastergo format in diagnostics
    diag["mastergo_preview"] = layout.to_mastergo_list()[:5]  # first 5 as preview
    diag["mastergo_stats"] = layout.stats()

    return elk, diag


def _parse_json(raw):
    c = raw.strip()
    if c.startswith("```"): c = re.sub(r'^```\w*\n?','',c); c = re.sub(r'\n?```$','',c)
    try:
        d = json.loads(c)
        return d if isinstance(d, list) else []
    except json.JSONDecodeError:
        m = re.search(r'\[.*\]', c, re.DOTALL)
        if m:
            try: return json.loads(re.sub(r',\s*([}\]])',r'\1',m.group()))
            except: pass
    return []
