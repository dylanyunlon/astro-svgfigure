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
Every entity must be in exactly one group's children."""

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
