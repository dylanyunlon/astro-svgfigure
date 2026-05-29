"""test_sprite_pipeline_m210_m214.py — second Claude's deliverable verification.

Exercises classify → prompt → sheet → split → inject end-to-end with MOCKED
AI image generation and background removal, so it runs in CI with no network.

Run: python3 tests/test_sprite_pipeline_m210_m214.py
"""
from __future__ import annotations

import asyncio
import base64
import importlib.util
import io
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load(mod_name: str, rel_path: str):
    """Import a single module by path WITHOUT triggering backend/__init__
    (which imports httpx etc.). Registers deps in sys.modules so dataclasses
    and intra-package imports resolve."""
    spec = importlib.util.spec_from_file_location(mod_name, REPO / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load the four standalone modules (no heavy deps).
nc = _load("node_classifier",
           "backend/pipeline/topology/node_classifier.py")
spd = _load("sprite_prompt_designer",
            "backend/pipeline/topology/sprite_prompt_designer.py")
sbg = _load("sprite_batch_generator",
            "backend/pipeline/topology/sprite_batch_generator.py")


# ── A tiny synthetic ELK graph with a clear sprite family ──
def make_graph():
    return {
        "id": "root",
        "children": [
            {
                "id": "feat_group", "group": True,
                "children": [
                    {"id": "input_feature", "x": 0, "y": 0, "width": 120, "height": 80,
                     "labels": [{"text": "Input feature C×H×W"}],
                     "iconHint": "feature map"},
                    {"id": "decomposed", "x": 0, "y": 100, "width": 120, "height": 80,
                     "labels": [{"text": "Decomposed feats"}],
                     "iconHint": "decomposed feature map"},
                    {"id": "selection", "x": 0, "y": 200, "width": 120, "height": 80,
                     "labels": [{"text": "Selection map 1×H×W"}],
                     "iconHint": "selection map"},
                    {"id": "mult", "x": 200, "y": 100, "width": 40, "height": 40,
                     "labels": [{"text": "⊗"}]},
                    {"id": "encoder", "x": 300, "y": 0, "width": 120, "height": 60,
                     "labels": [{"text": "ViT Encoder"}], "iconHint": "transformer"},
                    {"id": "loss_label", "x": 300, "y": 100, "width": 120, "height": 24,
                     "labels": [{"text": "Loss"}]},
                ],
            }
        ],
    }


def _green_grid_png(w: int, h: int, cells) -> str:
    """Build a fake sheet: solid green with a black square in each cell, so the
    mocked removebg (chroma) + alpha-bbox crop yield a real sprite per cell."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (w, h), (0, 255, 0))
    d = ImageDraw.Draw(img)
    for c in cells:
        # a centered black-ish square inside the cell
        m = 40
        d.rectangle([c.x + m, c.y + m, c.x + c.w - m, c.y + c.h - m],
                    fill=(20, 20, 20))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


async def mock_gemini(*, svg_content, prompt, settings, model,
                      aspect_ratio, image_size):
    """Stand-in for generate_image_with_gemini: returns a green grid sheet
    sized to the prompt's implied grid. We parse rows/cols from layout via the
    sheet object instead; here we just produce a generous green canvas and let
    the splitter crop by the real cell boxes passed through SpriteSheet."""
    # The caller (generate_sprite_sheet) already computed cells; but the mock
    # only sees the prompt. Return a marker; the test patches image via cells.
    return {"success": True, "image_b64": "__PATCH__", "mime_type": "image/png"}


async def mock_removebg(frames_b64, api_key="", **kw):
    """Stand-in for handle_removebg: chroma-keys pure green to transparent."""
    from PIL import Image
    import numpy as np
    out = []
    for b64 in frames_b64:
        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw)).convert("RGBA")
        arr = np.asarray(img).copy()
        g = (arr[:, :, 1] > 200) & (arr[:, :, 0] < 80) & (arr[:, :, 2] < 80)
        arr[g, 3] = 0
        res = Image.fromarray(arr, "RGBA")
        b = io.BytesIO(); res.save(b, format="PNG")
        out.append({"image_b64": base64.b64encode(b.getvalue()).decode("ascii")})
    return {"success": True, "results": out, "method": "chroma", "tier": 3}


def test_m210_classification():
    g = make_graph()
    report = nc.classify_nodes(g)
    modes = report.mode_counts
    assert modes.get("sprite", 0) >= 3, f"expected >=3 sprites, got {modes}"
    assert report.operator_count >= 1, "⊗ should be an operator"
    # encoder → icon (transformer alias), loss_label → text
    enc = next(c for c in g["children"][0]["children"] if c["id"] == "encoder")
    assert enc["renderMode"] == "icon", enc["renderMode"]
    mult = next(c for c in g["children"][0]["children"] if c["id"] == "mult")
    assert mult["renderMode"] == "text" and mult["isOperator"]
    print("  [M210] classification OK:", modes,
          "operators:", report.operator_count,
          "families:", len(report.families))
    return g, report


def test_m211_prompts(g, report):
    prompts = spd.design_prompts_for_classified(g, report.families)
    assert len(prompts) >= 3
    # multi-member family prompts must carry the series-consistency clause
    multi = [p for p in prompts if p.family_size > 1]
    assert multi, "expected a multi-member family"
    assert any("variant" in p.prompt.lower() and "only difference" in p.prompt.lower()
               for p in multi), "series-consistency clause missing"
    # cache keys are stable + differ across variants
    keys = {p.cache_key for p in prompts}
    assert len(keys) == len(prompts), "cache keys should be unique per variant"
    print("  [M211] prompts OK:", len(prompts), "prompts;",
          len(multi), "in multi-member families")
    return prompts


def test_m212_sheets(prompts):
    sheets = sbg.plan_sheets(prompts)
    assert sheets and all(len(s) <= sbg.GRID_MAX for s in sheets)
    print("  [M212] sheet planning OK:", len(sheets), "sheet(s)")
    return sheets


async def run_async_chain():
    g, report = test_m210_classification()
    prompts = test_m211_prompts(g, report)
    sheets = test_m212_sheets(prompts)

    # Patch mock_gemini to return a real green grid matching the sheet cells.
    spl = _load("sprite_sheet_splitter",
                "backend/pipeline/topology/sprite_sheet_splitter.py")

    injected_total = 0
    fell_back_total = 0
    # Minimal stand-in for ComposedCanvas (duck-typed for inject_sprites).
    class Canvas:
        def __init__(self, elk):
            self.elk_graph = elk; self.diagnostics = {}
    canvas = Canvas(g)

    all_assets = []
    fam_by_node = {p.node_id: p.family_id for p in prompts}
    for sp in sheets:
        async def patched_gemini(**kw):
            return {"success": True, "image_b64": "__defer__"}
        sheet = await sbg.generate_sprite_sheet(sp, gemini_callable=patched_gemini)
        # Replace deferred image with a real green grid sized to the cells.
        sheet.image_b64 = _green_grid_png(sheet.sheet_w, sheet.sheet_h, sheet.cells)
        sheet.success = True
        split = await spl.split_and_clean(sheet, removebg_callable=mock_removebg,
                                          run_qc=False)
        produced = [a for a in split.assets if not a.dropped]
        assert produced, "splitter produced no sprites from a non-empty sheet"
        for a in split.assets:
            a.family_id = fam_by_node.get(a.node_id, "")
        all_assets.extend(split.assets)
        print(f"  [M213] split sheet → {len(produced)} sprites, "
              f"{len(split.dropped_node_ids)} dropped")

    # M214 inject. canvas_compositor imports sibling modules via the
    # `backend.pipeline.topology.*` path. To load it without triggering
    # backend/pipeline/__init__'s heavy eager imports (httpx, pydantic_settings,
    # the AI engine), we pre-register lightweight namespace stubs for the parent
    # packages, then import the compositor module directly by file into that
    # namespace. This mirrors how the real app loads it (package path intact)
    # while keeping the test network/dep-free.
    import types as _types
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    for pkg in ("backend", "backend.pipeline", "backend.pipeline.topology"):
        if pkg not in sys.modules:
            m = _types.ModuleType(pkg)
            m.__path__ = [str(REPO / pkg.replace(".", "/"))]
            sys.modules[pkg] = m
    # Pre-seed the sibling modules the compositor imports, so its top-level
    # `from backend.pipeline.topology.X import Y` resolves to our loaded copies.
    sys.modules["backend.pipeline.topology.node_classifier"] = nc
    # region_planner / user_intent_parser / finalize_pass are dependency-light.
    _load("backend.pipeline.topology.finalize_pass",
          "backend/pipeline/topology/finalize_pass.py")
    _load("backend.pipeline.topology.user_intent_parser",
          "backend/pipeline/topology/user_intent_parser.py")
    _load("backend.pipeline.topology.region_planner",
          "backend/pipeline/topology/region_planner.py")
    cc = _load("backend.pipeline.topology.canvas_compositor",
               "backend/pipeline/topology/canvas_compositor.py")
    cc.inject_sprites(canvas, all_assets)
    diag = canvas.diagnostics
    injected_total = diag.get("sprites_injected", 0)
    fell_back_total = diag.get("sprites_fell_back", 0)
    assert injected_total >= 3, f"expected >=3 injected, got {injected_total}"
    # Every injected sprite node must now carry a spriteRef + be in manifest.
    manifest = canvas.elk_graph.get("spriteManifest", {})
    assert len(manifest) == injected_total
    for nid in manifest:
        node = next(c for c in g["children"][0]["children"] if c["id"] == nid)
        assert node.get("spriteRef") and node["renderMode"] == "sprite"
        bb = node["spriteRef"]["bbox"]
        assert bb[2] > 0 and bb[3] > 0, "true_bbox must be non-empty"
    print(f"  [M214] inject OK: {injected_total} injected, "
          f"{fell_back_total} fell back; manifest size {len(manifest)}")

    # ── Graceful-fallback check: a sheet that failed to generate ──
    class DeadSheet:
        image_b64 = None
        cells = sheets[0] and [type("C", (), {"node_id": "input_feature",
                                              "x": 0, "y": 0, "w": 1, "h": 1})()]
    dead = type("S", (), {"image_b64": None,
                          "cells": [type("C", (), {"node_id": "input_feature"})()],
                          "family_ids": []})()
    dead_split = await spl.split_and_clean(dead, removebg_callable=mock_removebg)
    assert all(a.dropped for a in dead_split.assets)
    print("  [fallback] failed sheet → all cells dropped (→ text) OK")

    # Regression (production audit): a cell whose coords exceed the sheet
    # bounds must NOT crash PIL crop (right<left) — it must drop gracefully.
    class _Sheet:
        def __init__(s, img, cells):
            s.image_b64 = img; s.cells = cells
            s.sheet_w = 300; s.sheet_h = 300; s.family_ids = []
    class _Cell:
        def __init__(s, nid, x, y, w, h):
            s.node_id = nid; s.x = x; s.y = y; s.w = w; s.h = h
    from PIL import Image as _I
    _img = _I.new("RGB", (300, 300), (0, 255, 0))
    _b = io.BytesIO(); _img.save(_b, format="PNG")
    _green = base64.b64encode(_b.getvalue()).decode()
    _oob = await spl.split_and_clean(
        _Sheet(_green, [_Cell("oob", 9000, 9000, 150, 150)]),
        removebg_callable=mock_removebg, run_qc=False)
    assert all(a.dropped for a in _oob.assets), "out-of-bounds cell must drop, not crash"
    print("  [regression] out-of-bounds cell → graceful drop (no PIL crash) OK")

    # Regression (end-to-end audit on operator chains): a node classified
    # isOperator must be RENDERABLE as a vector symbol. We can't run the TS
    # renderer here, but we assert the backend produces the contract the
    # renderer needs: classify_nodes stamps isOperator=True on ⊗ / ⊕ so
    # to-svg.ts's renderMathOperator branch fires (verified visually in env).
    _opg = {"id": "root", "children": [{"id": "g", "group": True, "children": [
        {"id": "mul", "x": 0, "y": 0, "width": 44, "height": 44, "labels": [{"text": "⊗"}]},
        {"id": "add", "x": 60, "y": 0, "width": 44, "height": 44, "labels": [{"text": "⊕"}]},
        {"id": "feat", "x": 0, "y": 60, "width": 120, "height": 90,
         "labels": [{"text": "feature C×H×W"}], "iconHint": "feature map"},
    ]}]}
    _rep = nc.classify_nodes(_opg)
    _ops = [c for c in _opg["children"][0]["children"] if c.get("isOperator")]
    assert len(_ops) == 2, f"⊗ and ⊕ must be isOperator, got {len(_ops)}"
    assert all(c["renderMode"] == "text" for c in _ops), "operators are text-mode + isOperator"
    print(f"  [M207] ⊗/⊕ classified isOperator (renderMathOperator contract) OK")


if __name__ == "__main__":
    print("Testing M210–M214 sprite pipeline (mocked AI + removebg)…")
    asyncio.run(run_async_chain())
    print("\nALL M210–M214 CHECKS PASSED ✓")
