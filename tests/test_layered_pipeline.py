"""test_layered_pipeline.py — End-to-end test for M1-M12.

Tests the complete layered topology pipeline without LLM calls,
using _fallback_subgraph as the deterministic kernel.

Like CCCL's benchmark suite: validates correctness of the dispatch
function by running the full pass sequence with known inputs and
checking the output invariants.

Run: cd astro-svgfigure && python -m pytest tests/test_layered_pipeline.py -v
  Or: python tests/test_layered_pipeline.py
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ═══════════════════════════════════════════════════════════════════════
#  Test fixtures
# ═══════════════════════════════════════════════════════════════════════

GENDB_TEXT = """\
GenDB is a database system with AI-powered query optimization.
The architecture has three input sources: Workload (Schema, SQL, Data),
Resource (CPU, GPU, RAM, SSD, HDD), and User (Config, REQ).
The core GenDB system contains an Agent Tool Set with five sequential
agents: Workload Analyzer, Storage/Index Designer, Query Planner,
Code Generator, and Query Optimizer.
The right side has Storage outputs: INDEX, Q1.exe through Qn.exe."""

UICOPILOT_TEXT = """\
UICopilot decouples webpage generation into two stages:
coarse DOM layout generation and fine-grained code synthesis.
The structure model uses a ViT encoder and transformer-based decoder.
BBox-based data pruning removes noise. An MLLM code agent generates
detailed local code for leaf nodes."""

SIMPLE_TEXT = "A simple module with two components: input and output."

CHINESE_TEXT = "从 C 这个例子开始, 实现 D, 让 E 可以 F, 融合大厂 OSDK 风格"


# ═══════════════════════════════════════════════════════════════════════
#  M2: user_intent_parser
# ═══════════════════════════════════════════════════════════════════════

def test_m2_intent_parser():
    from backend.pipeline.topology.user_intent_parser import (
        parse_user_intent, DiagramType,
    )

    intent = parse_user_intent(GENDB_TEXT)
    assert intent.diagram_type == DiagramType.ARCHITECTURE
    assert intent.confidence >= 0.5
    assert len(intent.entities) > 0
    assert intent.estimated_regions >= 2

    eng = parse_user_intent("git clone repo, build, test, deploy")
    assert eng.diagram_type == DiagramType.ENGINEERING_FLOW

    rec = parse_user_intent(CHINESE_TEXT)
    assert rec.diagram_type == DiagramType.RECURSIVE_CHAIN

    simple = parse_user_intent(SIMPLE_TEXT)
    assert simple.estimated_regions >= 1

    print("✓ M2 intent_parser: 4 cases passed")


# ═══════════════════════════════════════════════════════════════════════
#  M4: finalize_pass
# ═══════════════════════════════════════════════════════════════════════

def test_m4_finalize_pass():
    from backend.pipeline.topology.finalize_pass import (
        PassContext, _parse_llm_json,
    )

    # JSON repair
    assert len(_parse_llm_json('[{"id":"a",},{"id":"b",}]', expect_array=True)) == 2
    assert len(_parse_llm_json('```json\n[{"id":"x"}]\n```', expect_array=True)) == 1

    ctx = PassContext()
    ctx.advance("test")
    assert ctx.pass_number == 1
    assert ctx.pass_name == "test"

    print("✓ M4 finalize_pass: JSON repair + PassContext")


# ═══════════════════════════════════════════════════════════════════════
#  M1: region_planner
# ═══════════════════════════════════════════════════════════════════════

def test_m1_region_planner():
    from backend.pipeline.topology.user_intent_parser import parse_user_intent
    from backend.pipeline.topology.region_planner import (
        compute_canvas_size, layout_regions_grid, PlannedRegion,
    )

    intent = parse_user_intent(GENDB_TEXT)
    w, h = compute_canvas_size(intent)
    assert w >= 400
    assert h >= 300

    bboxes = layout_regions_grid(4, w, h)
    assert len(bboxes) == 4

    # No overlaps
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            bi, bj = bboxes[i], bboxes[j]
            x_overlap = (bi["x"] < bj["x"] + bj["width"] and
                         bj["x"] < bi["x"] + bi["width"])
            y_overlap = (bi["y"] < bj["y"] + bj["height"] and
                         bj["y"] < bi["y"] + bi["height"])
            assert not (x_overlap and y_overlap), f"Regions {i} and {j} overlap"

    print("✓ M1 region_planner: canvas size + grid layout + no overlap")


# ═══════════════════════════════════════════════════════════════════════
#  M5: canvas_compositor (DoubleBuffer)
# ═══════════════════════════════════════════════════════════════════════

def test_m5_doublebuffer():
    from backend.pipeline.topology.canvas_compositor import DoubleBuffer

    buf = DoubleBuffer({"v": 0, "children": []})
    assert buf.current()["v"] == 0
    assert buf.selector == 0

    # 5 sequential swaps
    for i in range(1, 6):
        buf.set_alternate({"v": i, "children": []})
        assert buf.current()["v"] == i

    # selector alternates: after 5 swaps (odd), selector = 1
    assert buf.selector == 1

    print("✓ M5 DoubleBuffer: 5 swaps, selector ^= 1 pattern correct")


def test_m5_compose():
    from backend.pipeline.topology.canvas_compositor import compose
    from backend.pipeline.topology.region_planner import PlannedRegion

    regions = [
        PlannedRegion(id="a", name="A",
                      bbox={"x": 0, "y": 0, "width": 300, "height": 200}),
        PlannedRegion(id="b", name="B",
                      bbox={"x": 320, "y": 0, "width": 300, "height": 200}),
    ]
    subgraphs = [
        {"id": "a", "children": [{"id": "n1", "width": 100, "height": 40}],
         "edges": []},
        {"id": "b", "children": [{"id": "n2", "width": 100, "height": 40}],
         "edges": [{"id": "e1", "sources": ["n2"], "targets": ["a.n1"]}]},
    ]

    canvas = compose(regions, subgraphs, 640, 220)
    assert canvas.width == 640
    assert canvas.height == 220
    assert len(canvas.layers) == 2
    assert len(canvas.cross_region_edges) >= 1
    assert len(canvas.elk_graph["children"]) == 2

    print("✓ M5 compose: 2 regions, cross-region edge resolved")


# ═══════════════════════════════════════════════════════════════════════
#  M6: layered_pipeline (LayeredResult)
# ═══════════════════════════════════════════════════════════════════════

def test_m6_layered_result():
    from backend.pipeline.topology.layered_pipeline import LayeredResult
    from backend.pipeline.topology.canvas_compositor import ComposedCanvas

    canvas = ComposedCanvas(width=800, height=500)
    result = LayeredResult(success=True, canvas=canvas)
    d = result.to_dict()

    assert d["success"] is True
    assert "canvas" in d
    assert d["canvas"]["width"] == 800

    print("✓ M6 LayeredResult: serialization correct")


# ═══════════════════════════════════════════════════════════════════════
#  M8: prompt_templates
# ═══════════════════════════════════════════════════════════════════════

def test_m8_prompt_templates():
    from backend.pipeline.topology.prompt_templates import select_prompts
    from backend.pipeline.topology.user_intent_parser import DiagramType

    for dtype in DiagramType:
        sys_p, usr_p = select_prompts(dtype, "plan")
        sys_g, usr_g = select_prompts(dtype, "generate")
        assert len(sys_p) > 100
        assert len(sys_g) > 100
        assert "{text}" in usr_p
        # No hardcoded icon categories
        assert "hardware/" not in sys_g

    print("✓ M8 prompt_templates: all diagram types covered, no hardcoded icons")


# ═══════════════════════════════════════════════════════════════════════
#  M9: layout_within_bbox
# ═══════════════════════════════════════════════════════════════════════

def test_m9_layout_within_bbox():
    from backend.pipeline.topology.constraint.sugiyama import (
        layout_within_bbox,
    )

    nodes = ["a", "b", "c", "d"]
    edges = [("a", "b"), ("b", "c"), ("c", "d")]
    widths = {n: 80 for n in nodes}
    heights = {n: 40 for n in nodes}
    bbox = {"x": 0, "y": 0, "width": 300, "height": 300}

    pos, diag = layout_within_bbox(nodes, edges, widths, heights, bbox)
    assert len(pos) == 4
    assert diag.get("constrained") is True

    # All positions within bbox
    for n, (x, y) in pos.items():
        assert x >= 0
        assert y >= 0
        assert x + widths[n] <= bbox["width"] + 20  # tolerance
        assert y + heights[n] <= bbox["height"] + 20

    print("✓ M9 layout_within_bbox: 4 nodes constrained within bbox")


# ═══════════════════════════════════════════════════════════════════════
#  M12: DoubleBuffer refinement
# ═══════════════════════════════════════════════════════════════════════

def test_m12_refinement_interface():
    """Test that refine_region has correct interface (without LLM call)."""
    import inspect
    from backend.pipeline.topology.per_region_generator import (
        refine_region, refine_all_regions,
    )

    sig = inspect.signature(refine_region)
    params = list(sig.parameters.keys())
    assert "current_subgraph" in params
    assert "neighbor_subgraphs" in params

    sig2 = inspect.signature(refine_all_regions)
    params2 = list(sig2.parameters.keys())
    assert "subgraphs" in params2
    assert asyncio.iscoroutinefunction(refine_all_regions)

    print("✓ M12 refinement: correct async interface")


# ═══════════════════════════════════════════════════════════════════════
#  M15: icon_extractor
# ═══════════════════════════════════════════════════════════════════════

def test_m15_icon_detection():
    """Test icon detection on the UICopilot figure if available."""
    test_img = "/mnt/user-data/uploads/test2.png"
    if not os.path.exists(test_img):
        print("⊘ M15 icon_extractor: test image not available, skipped")
        return

    from PIL import Image
    from backend.pipeline.icon_extractor import (
        pass0_detect_nodes, pass1_separate_icons,
    )

    img = Image.open(test_img)
    nodes = pass0_detect_nodes(img)
    assert len(nodes) >= 10, f"Expected ≥10 nodes, got {len(nodes)}"

    pale = [n for n in nodes if n.is_pale]
    saturated = [n for n in nodes if not n.is_pale]
    assert len(pale) >= 1
    assert len(saturated) >= 8

    # Pass 1: icon separation
    icons = pass1_separate_icons(img, nodes)
    assert len(icons) == len(nodes)

    # High-confidence icons should have reasonable sub-bbox
    confident = [ir for ir in icons if ir.confidence >= 0.8]
    for ir in confident:
        _, _, iw, ih = ir.icon_bbox
        assert iw >= 10 and ih >= 10, f"Icon too small: {iw}x{ih}"
        assert iw < ir.node.width, "Icon wider than node"

    print(f"✓ M15 icon_extractor: {len(nodes)} nodes, "
          f"{len(pale)} pale, {len(confident)} confident icons")


# ═══════════════════════════════════════════════════════════════════════
#  M16: parallel generation
# ═══════════════════════════════════════════════════════════════════════

def test_m16_parallel_interface():
    """Verify generate_all_regions has max_concurrent parameter."""
    import inspect
    from backend.pipeline.topology.per_region_generator import (
        generate_all_regions,
    )

    sig = inspect.signature(generate_all_regions)
    assert "max_concurrent" in sig.parameters
    assert asyncio.iscoroutinefunction(generate_all_regions)

    print("✓ M16 parallel: generate_all_regions has max_concurrent param")


# ═══════════════════════════════════════════════════════════════════════
#  Full pipeline e2e (no LLM)
# ═══════════════════════════════════════════════════════════════════════

def test_full_pipeline_deterministic():
    """Run the complete pipeline with fallback subgraphs (no LLM)."""
    from backend.pipeline.topology.user_intent_parser import parse_user_intent
    from backend.pipeline.topology.region_planner import (
        compute_canvas_size, layout_regions_grid, PlannedRegion,
    )
    from backend.pipeline.topology.per_region_generator import _fallback_subgraph
    from backend.pipeline.topology.canvas_compositor import compose
    from backend.pipeline.topology.layered_pipeline import LayeredResult

    # Stage 0: intent
    intent = parse_user_intent(GENDB_TEXT)
    assert intent.diagram_type.value == "architecture"

    # Stage 1: plan
    w, h = compute_canvas_size(intent)
    n = min(intent.estimated_regions, 6)
    bboxes = layout_regions_grid(n, w, h)
    entities = [e.name for e in intent.entities]
    regions = [
        PlannedRegion(
            id=f"r{i}", name=f"Region {i}",
            bbox=bboxes[i],
            entity_hints=entities[i::n],
            priority=i,
        )
        for i in range(n)
    ]

    # Stage 2: generate (fallback — no LLM)
    subgraphs = [_fallback_subgraph(r) for r in regions]
    total_nodes = sum(len(sg["children"]) for sg in subgraphs)
    assert total_nodes > 0

    # Stage 3: compose
    canvas = compose(regions, subgraphs, w, h)
    assert canvas.width == w
    assert canvas.height == h
    assert len(canvas.layers) == n
    assert len(canvas.elk_graph["children"]) == n

    # Result
    result = LayeredResult(
        success=True, canvas=canvas, intent=intent, regions=regions,
    )
    d = result.to_dict()
    assert d["success"] is True
    assert len(d["layers"]) == n
    assert "elk" in d

    print(f"✓ Full pipeline: {n} regions, {total_nodes} nodes, "
          f"{w}x{h} canvas, {len(d['layers'])} layers")


# ═══════════════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Layered Topology Pipeline — E2E Test Suite")
    print("=" * 60)
    print()

    tests = [
        test_m2_intent_parser,
        test_m4_finalize_pass,
        test_m1_region_planner,
        test_m5_doublebuffer,
        test_m5_compose,
        test_m6_layered_result,
        test_m8_prompt_templates,
        test_m9_layout_within_bbox,
        test_m12_refinement_interface,
        test_m15_icon_detection,
        test_m16_parallel_interface,
        test_full_pipeline_deterministic,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1

    print()
    print(f"{'=' * 60}")
    print(f"  {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'=' * 60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
