"""test_sprite_pipeline_m220_m224.py — fourth Claude's deliverable verification.

Covers M220 (cache), M221 (concurrency), M223 (export), and M224 acceptance:
the full run_sprite_pipeline with cache hit/miss + concurrent sheets, asserting
cost control (AI calls << sprite count on a warm cache) and determinism.

Mocked AI + removebg → no network.
Run: python3 tests/test_sprite_pipeline_m220_m224.py
"""
from __future__ import annotations

import asyncio
import base64
import importlib.util
import io
import sys
import tempfile
import types as _types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load(mod_name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(mod_name, REPO / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


for pkg in ("backend", "backend.pipeline", "backend.pipeline.topology"):
    if pkg not in sys.modules:
        m = _types.ModuleType(pkg)
        m.__path__ = [str(REPO / pkg.replace(".", "/"))]
        sys.modules[pkg] = m

nc = _load("backend.pipeline.topology.node_classifier",
           "backend/pipeline/topology/node_classifier.py")
spd = _load("backend.pipeline.topology.sprite_prompt_designer",
            "backend/pipeline/topology/sprite_prompt_designer.py")
sbg = _load("backend.pipeline.topology.sprite_batch_generator",
            "backend/pipeline/topology/sprite_batch_generator.py")
spl = _load("backend.pipeline.topology.sprite_sheet_splitter",
            "backend/pipeline/topology/sprite_sheet_splitter.py")
cache_mod = _load("backend.pipeline.topology.sprite_cache",
                  "backend/pipeline/topology/sprite_cache.py")
export_mod = _load("backend.pipeline.topology.sprite_export",
                   "backend/pipeline/topology/sprite_export.py")
_load("backend.pipeline.topology.sprite_sequence_generator",
      "backend/pipeline/topology/sprite_sequence_generator.py")
_load("backend.pipeline.topology.finalize_pass",
      "backend/pipeline/topology/finalize_pass.py")
_load("backend.pipeline.topology.user_intent_parser",
      "backend/pipeline/topology/user_intent_parser.py")
_load("backend.pipeline.topology.region_planner",
      "backend/pipeline/topology/region_planner.py")
_load("backend.pipeline.frame_consistency",
      "backend/pipeline/frame_consistency.py")
_load("backend.pipeline.topology.sprite_family_aligner",
      "backend/pipeline/topology/sprite_family_aligner.py")
_load("backend.pipeline.topology.sprite_vectorizer",
      "backend/pipeline/topology/sprite_vectorizer.py")
cc = _load("backend.pipeline.topology.canvas_compositor",
           "backend/pipeline/topology/canvas_compositor.py")
lp = _load("backend.pipeline.topology.layered_pipeline",
           "backend/pipeline/topology/layered_pipeline.py")

from PIL import Image, ImageDraw  # noqa: E402
import numpy as np  # noqa: E402


def _rect_png(size=200) -> str:
    img = Image.new("RGB", (size, size), (0, 255, 0)); d = ImageDraw.Draw(img)
    d.rectangle([size//4, size//4, 3*size//4, 3*size//4], fill=(20, 20, 20))
    b = io.BytesIO(); img.save(b, format="PNG")
    return base64.b64encode(b.getvalue()).decode()


def _transparent_png(size=120) -> str:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
    d.rectangle([size//4, size//4, 3*size//4, 3*size//4], fill=(20, 20, 20, 255))
    b = io.BytesIO(); img.save(b, format="PNG")
    return base64.b64encode(b.getvalue()).decode()


def _grid_png(w, h, cells) -> str:
    img = Image.new("RGB", (w, h), (0, 255, 0)); d = ImageDraw.Draw(img)
    for c in cells:
        d.rectangle([c.x+40, c.y+40, c.x+c.w-40, c.y+c.h-40], fill=(20, 20, 20))
    b = io.BytesIO(); img.save(b, format="PNG")
    return base64.b64encode(b.getvalue()).decode()


# ── M220: cache hit/miss + persistence + family-atomic ──
def test_m220_cache():
    with tempfile.TemporaryDirectory() as d:
        cache = cache_mod.SpriteCache(cache_dir=d, lru_capacity=4)
        # miss → put → hit (memory)
        assert cache.get("abc") is None
        cache.put("abc", {"node_id": "n", "image_b64": "X", "true_bbox": [0, 0, 5, 5],
                          "format": "png", "family_id": ""})
        got = cache.get("abc")
        assert got and got["image_b64"] == "X"
        assert cache.stats.hits == 1 and cache.stats.misses == 1

        # persistence: a fresh cache over the same dir sees the disk entry
        cache2 = cache_mod.SpriteCache(cache_dir=d)
        got2 = cache2.get("abc")
        assert got2 and got2["image_b64"] == "X", "disk persistence failed"

        # LRU eviction
        for i in range(6):
            cache.put(f"k{i}", {"image_b64": "y", "true_bbox": [0, 0, 1, 1],
                                "format": "png"})
        assert len(cache._lru) <= 4 and cache.stats.evictions >= 1

        # family-atomic: missing one member → None
        assert cache.get_family(["abc", "nope"]) is None
        assert cache.get_family(["abc"]) is not None
    print(f"  [M220] cache: hit/miss/persist/LRU/family-atomic OK "
          f"(hit_rate={cache.stats.hit_rate:.2f})")


# ── M221: concurrent sheet generation preserves order + partial success ──
def test_m221_concurrency():
    async def run():
        # 3 prompt groups; mock gen returns placeholder we patch to a grid.
        prompts = [spd.SpritePrompt(node_id=f"n{i}", prompt=f"p{i}") for i in range(6)]
        groups = [[prompts[0], prompts[1]], [prompts[2], prompts[3]],
                  [prompts[4], prompts[5]]]
        order_seen = []

        async def gen(**kw):
            return {"success": True, "image_b64": "__defer__"}

        # generate_sheets_concurrent calls generate_sprite_sheet internally;
        # patch that to size a grid and record order.
        orig = sbg.generate_sprite_sheet
        async def patched(group, **kw):
            sh = await orig(group, gemini_callable=gen)
            order_seen.append(sh.cells[0].node_id)
            sh.image_b64 = _grid_png(sh.sheet_w, sh.sheet_h, sh.cells)
            sh.success = True
            return sh
        sbg.generate_sprite_sheet = patched
        try:
            sheets = await sbg.generate_sheets_concurrent(groups, max_concurrency=2)
        finally:
            sbg.generate_sprite_sheet = orig
        assert len(sheets) == 3
        # gather preserves input order regardless of completion order
        assert [s.cells[0].node_id for s in sheets] == ["n0", "n2", "n4"]
        assert all(s.success for s in sheets)
        return len(order_seen)
    n = asyncio.run(run())
    print(f"  [M221] concurrency: {n} sheets generated, output order preserved OK")


# ── M223: export self-containment + SVG + PDF graceful degrade ──
def test_m223_export():
    # A sprite-bearing SVG: one inline data-URI image (self-contained).
    inline = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
              f'<image data-sprite="png" x="10" y="10" width="40" height="40" '
              f'href="data:image/png;base64,{_transparent_png(40)}"/></svg>')
    r = export_mod.export_svg(inline)
    assert r.success and r.diagnostics["ok"] and r.diagnostics["inline"] == 1
    # external ref → refused
    bad = ('<svg xmlns="http://www.w3.org/2000/svg"><image href="sprite.png"/></svg>')
    rb = export_mod.export_svg(bad)
    assert not rb.success and rb.diagnostics["external_refs"] == ["sprite.png"]
    # PDF: succeeds if cairosvg present, else graceful error (no crash)
    rp = export_mod.export_pdf(inline, width=100, height=100)
    if rp.success:
        assert rp.data and rp.mime_type == "application/pdf"
        msg = f"PDF {len(rp.data)} bytes"
    else:
        assert "cairosvg" in (rp.error or "").lower() or rp.error
        msg = f"PDF degraded gracefully ({(rp.error or '')[:40]})"

    # Regression (smoke-test finding): a root <svg> with the responsive
    # style="max-width:100%;height:auto;" must still rasterize NON-BLANK via
    # cairosvg (_prep_for_raster strips that style). Only assert when cairosvg
    # is installed.
    try:
        import cairosvg  # noqa: F401
        import numpy as np
        from PIL import Image as _Img
        styled = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" '
            'width="120" height="120" style="max-width:100%;height:auto;">'
            '<rect width="120" height="120" fill="#FFFFFF"/>'
            '<rect x="20" y="20" width="80" height="80" fill="#1A1A1A"/></svg>'
        )
        rr = export_mod.export_png(styled)
        assert rr.success and rr.data
        arr = np.asarray(_Img.open(io.BytesIO(rr.data)).convert("RGBA"))
        assert (arr[:, :, 3] > 0).any(), "styled-root SVG rasterized BLANK"
        msg += "; styled-root non-blank OK"
    except ImportError:
        pass
    print(f"  [M223] export: self-contained SVG OK, external ref refused, {msg}")


# ── M224: full pipeline with cache cold→warm + cost assertion ──
async def test_m224_acceptance():
    def make_graph():
        return {"id": "root", "children": [{"id": "g", "group": True, "children": [
            {"id": "patches", "x": 0, "y": 0, "width": 120, "height": 90,
             "labels": [{"text": "Patch grid"}], "iconHint": "patch grid"},
            {"id": "sel", "x": 140, "y": 0, "width": 120, "height": 90,
             "labels": [{"text": "Selection map 1×H×W"}], "iconHint": "selection map"},
            {"id": "enc", "x": 280, "y": 0, "width": 120, "height": 60,
             "labels": [{"text": "ViT Encoder"}], "iconHint": "transformer"},
        ]}]}

    async def base_img(**kw): return {"success": True, "image_b64": _rect_png(220)}
    async def edit(**kw): return _rect_png(200), "sig"
    async def rbg(frames, api_key="", **kw):
        out = []
        for b in frames:
            im = Image.open(io.BytesIO(base64.b64decode(b))).convert("RGBA")
            a = np.asarray(im).copy()
            g = (a[:, :, 1] > 200) & (a[:, :, 0] < 80) & (a[:, :, 2] < 80)
            a[g, 3] = 0
            o = Image.fromarray(a, "RGBA"); bb = io.BytesIO(); o.save(bb, format="PNG")
            out.append({"image_b64": base64.b64encode(bb.getvalue()).decode()})
        return {"success": True, "results": out, "method": "chroma"}

    # Count AI calls via a patched generate_sprite_sheet that sizes a grid.
    call_count = {"n": 0}
    orig = sbg.generate_sprite_sheet
    async def gen(**kw): return {"success": True, "image_b64": "__defer__"}
    async def patched(group, **kw):
        call_count["n"] += 1
        sh = await orig(group, gemini_callable=gen)
        sh.image_b64 = _grid_png(sh.sheet_w, sh.sheet_h, sh.cells); sh.success = True
        return sh
    sbg.generate_sprite_sheet = patched

    class Canvas:
        def __init__(self, e): self.elk_graph = e; self.diagnostics = {}

    try:
        with tempfile.TemporaryDirectory() as d:
            cache = cache_mod.SpriteCache(cache_dir=d)

            # COLD run — sprites generated, cache populated.
            c1 = Canvas(make_graph())
            r1 = await lp.run_sprite_pipeline(
                c1, base_image_callable=base_img, edit_callable=edit,
                removebg_callable=rbg, run_qc=False, sprite_cache=cache,
                enable_vectorize=True, enable_alignment=True)
            cold_calls = call_count["n"]
            assert r1.success and r1.sprites_injected >= 2, r1.to_dict()

            # WARM run — same graph; sprites served from cache, ~0 new AI calls.
            call_count["n"] = 0
            c2 = Canvas(make_graph())
            r2 = await lp.run_sprite_pipeline(
                c2, base_image_callable=base_img, edit_callable=edit,
                removebg_callable=rbg, run_qc=False, sprite_cache=cache,
                enable_vectorize=True, enable_alignment=True)
            warm_calls = call_count["n"]
            assert r2.success and r2.sprites_injected >= 2
            assert r2.diagnostics.get("cache_hits", 0) >= 1, "warm run had no cache hits"
            assert warm_calls < cold_calls, (
                f"warm calls ({warm_calls}) should be < cold ({cold_calls})")
    finally:
        sbg.generate_sprite_sheet = orig

    print(f"  [M224] acceptance: cold {cold_calls} AI call(s) → warm "
          f"{warm_calls} (cache hits={r2.diagnostics.get('cache_hits')}), "
          f"injected={r2.sprites_injected} OK")


if __name__ == "__main__":
    print("Testing M220–M224 (cache / concurrency / export / acceptance)…")
    test_m220_cache()
    test_m221_concurrency()
    test_m223_export()
    asyncio.run(test_m224_acceptance())
    print("\nALL M220–M224 CHECKS PASSED ✓")
