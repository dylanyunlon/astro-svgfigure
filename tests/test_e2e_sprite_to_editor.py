"""test_e2e_sprite_to_editor.py — M416 integration test.

End-to-end test: text input → ELK topology → classify_nodes → inject_sprites
→ sprite-enriched ELK → ready for tldraw Editor.

Tests verify:
  1. classify_nodes stamps renderMode on every leaf
  2. Operators get isOperator=True
  3. Kernel nodes get renderMode='kernel'
  4. Sprite nodes get familyId
  5. inject_sprites (dry_run) stamps spriteRef on all sprite nodes
  6. The enriched ELK graph is valid for tldraw consumption

Run: python -m pytest tests/test_e2e_sprite_to_editor.py -v
"""
from __future__ import annotations

import asyncio
import json
import logging
import pytest
from typing import Any, Dict, List

logging.basicConfig(level=logging.DEBUG)

# ═══════════════════════════════════════════════════════════════════════════
#  Test fixture: FreqSelect / AdaDR / AdaKern reference architecture
# ═══════════════════════════════════════════════════════════════════════════

def _build_freqselect_elk() -> Dict[str, Any]:
    """Build ELK graph matching the reference image's three-module architecture."""
    return {
        "id": "root",
        "children": [
            # ── FreqSelect module ──
            {
                "id": "freqselect_group",
                "labels": [{"text": "FreqSelect"}],
                "children": [
                    {"id": "input_feature", "labels": [{"text": "Input feature"}],
                     "iconHint": "feature map C×H×W", "width": 120, "height": 80},
                    {"id": "freq_decompose", "labels": [{"text": "Frequency decompose"}],
                     "iconHint": "frequency decomposition", "width": 140, "height": 50},
                    {"id": "decomposed_feat_1", "labels": [{"text": "Decomposed feats"}],
                     "iconHint": "feature map stack", "width": 100, "height": 80},
                    {"id": "decomposed_feat_2", "labels": [{"text": "Decomposed feats"}],
                     "iconHint": "feature map stack", "width": 100, "height": 80},
                    {"id": "decomposed_feat_3", "labels": [{"text": "Decomposed feats"}],
                     "iconHint": "feature map stack", "width": 100, "height": 80},
                    {"id": "conv3", "labels": [{"text": "Conv3"}],
                     "iconHint": "convolution", "width": 80, "height": 40},
                    {"id": "selection_map", "labels": [{"text": "Selection map"}],
                     "iconHint": "selection map 1×H×W", "width": 100, "height": 80},
                    {"id": "op_multiply_1", "labels": [{"text": "⊗"}], "width": 40, "height": 40},
                    {"id": "global_pooling", "labels": [{"text": "Global Pooling"}],
                     "iconHint": "pooling", "width": 120, "height": 40},
                    {"id": "global_feature", "labels": [{"text": "Global feature C×1×1"}],
                     "iconHint": "feature", "width": 100, "height": 60},
                    {"id": "op_add_1", "labels": [{"text": "⊕"}], "width": 40, "height": 40},
                ],
            },
            # ── AdaDR module ──
            {
                "id": "adadr_group",
                "labels": [{"text": "AdaDR"}],
                "children": [
                    {"id": "dilation_map", "labels": [{"text": "Dilation map"}],
                     "iconHint": "dilation map 1×H×W", "width": 100, "height": 60},
                    {"id": "conv3_relu", "labels": [{"text": "Conv3+ReLU"}],
                     "iconHint": "convolution relu", "width": 100, "height": 40},
                    {"id": "adadr_feat_1", "labels": [{"text": "C×H×W"}],
                     "iconHint": "feature map", "width": 80, "height": 60},
                    {"id": "adadr_feat_2", "labels": [{"text": "C×H×W"}],
                     "iconHint": "feature map", "width": 80, "height": 60},
                    {"id": "op_convolve_1", "labels": [{"text": "⊛"}], "width": 40, "height": 40},
                    {"id": "op_convolve_2", "labels": [{"text": "⊛"}], "width": 40, "height": 40},
                    {"id": "output_feature", "labels": [{"text": "Output feature C×H×W"}],
                     "iconHint": "output feature map", "width": 120, "height": 80},
                    {"id": "dilation_rate_3", "labels": [{"text": "Dilation rate = 3"}],
                     "width": 100, "height": 24},
                    {"id": "dilation_rate_1", "labels": [{"text": "Dilation rate = 1"}],
                     "width": 100, "height": 24},
                    {"id": "sampling_label_1", "labels": [{"text": "sampling"}],
                     "width": 60, "height": 20},
                    {"id": "sampling_label_2", "labels": [{"text": "sampling"}],
                     "width": 60, "height": 20},
                ],
            },
            # ── AdaKern module ──
            {
                "id": "adakern_group",
                "labels": [{"text": "AdaKern"}],
                "children": [
                    {"id": "conv1_relu_sig_1", "labels": [{"text": "Conv1-ReLU-Conv1-Sigmoid"}],
                     "width": 180, "height": 40},
                    {"id": "conv1_relu_sig_2", "labels": [{"text": "Conv1-ReLU-Conv1-Sigmoid"}],
                     "width": 180, "height": 40},
                    {"id": "op_multiply_2", "labels": [{"text": "⊗"}], "width": 40, "height": 40},
                    {"id": "op_multiply_3", "labels": [{"text": "⊗"}], "width": 40, "height": 40},
                    {"id": "low_freq_kernel", "labels": [{"text": "Low-freq kernel"}],
                     "iconHint": "convolution kernel grid", "width": 80, "height": 60},
                    {"id": "high_freq_kernel", "labels": [{"text": "High-freq kernel"}],
                     "iconHint": "convolution kernel grid", "width": 80, "height": 60},
                    {"id": "static_kernel", "labels": [{"text": "Static kernel"}],
                     "iconHint": "static kernel", "width": 80, "height": 60},
                    {"id": "adaptive_kernel", "labels": [{"text": "Adaptive kernel"}],
                     "iconHint": "adaptive kernel", "width": 80, "height": 60},
                    {"id": "op_subtract_1", "labels": [{"text": "⊖"}], "width": 40, "height": 40},
                    {"id": "op_add_2", "labels": [{"text": "⊕"}], "width": 40, "height": 40},
                    {"id": "average_label", "labels": [{"text": "Average"}],
                     "width": 60, "height": 24},
                ],
            },
        ],
        "edges": [
            {"id": "e1", "sources": ["input_feature"], "targets": ["freq_decompose"]},
            {"id": "e2", "sources": ["freq_decompose"], "targets": ["decomposed_feat_1"]},
            {"id": "e3", "sources": ["decomposed_feat_1"], "targets": ["op_multiply_1"]},
            {"id": "e4", "sources": ["selection_map"], "targets": ["op_multiply_1"]},
            {"id": "e5", "sources": ["op_multiply_1"], "targets": ["op_add_1"]},
            {"id": "e6", "sources": ["global_feature"], "targets": ["op_add_1"]},
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Helper to collect all leaves from ELK graph
# ═══════════════════════════════════════════════════════════════════════════

def _collect_all_leaves(node: Dict[str, Any], out: List[Dict[str, Any]]) -> None:
    children = node.get("children")
    if isinstance(children, list) and children:
        for c in children:
            if isinstance(c, dict):
                _collect_all_leaves(c, out)
    else:
        out.append(node)


# ═══════════════════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestM410ClassifyNodes:
    """M410: classify_nodes stamps renderMode on every leaf."""

    def setup_method(self):
        self.elk = _build_freqselect_elk()
        from backend.pipeline.topology.node_classifier import classify_nodes
        self.report = classify_nodes(self.elk)

    def test_all_leaves_classified(self):
        leaves: List[Dict[str, Any]] = []
        for child in self.elk["children"]:
            _collect_all_leaves(child, leaves)
        for leaf in leaves:
            assert "renderMode" in leaf, f"Node {leaf.get('id')} missing renderMode"
            assert leaf["renderMode"] in ("text", "icon", "sprite", "kernel"), \
                f"Node {leaf.get('id')} has invalid renderMode: {leaf['renderMode']}"

    def test_operators_detected(self):
        """⊗ ⊕ ⊛ ⊖ should all be operators."""
        leaves: List[Dict[str, Any]] = []
        for child in self.elk["children"]:
            _collect_all_leaves(child, leaves)

        operator_ids = {n["id"] for n in leaves if n.get("isOperator")}
        expected_operators = {"op_multiply_1", "op_add_1", "op_convolve_1",
                             "op_convolve_2", "op_multiply_2", "op_multiply_3",
                             "op_subtract_1", "op_add_2"}
        assert expected_operators.issubset(operator_ids), \
            f"Missing operators: {expected_operators - operator_ids}"

    def test_kernel_nodes_detected(self):
        """M410: kernel/filter/weight nodes get renderMode='kernel'."""
        leaves: List[Dict[str, Any]] = []
        for child in self.elk["children"]:
            _collect_all_leaves(child, leaves)

        kernel_ids = {n["id"] for n in leaves if n.get("renderMode") == "kernel"}
        expected_kernels = {"low_freq_kernel", "high_freq_kernel",
                           "static_kernel", "adaptive_kernel"}
        assert expected_kernels.issubset(kernel_ids), \
            f"Missing kernel nodes: {expected_kernels - kernel_ids}"

    def test_sprite_nodes_have_family(self):
        """Sprite nodes should have a familyId."""
        leaves: List[Dict[str, Any]] = []
        for child in self.elk["children"]:
            _collect_all_leaves(child, leaves)

        sprite_nodes = [n for n in leaves if n.get("renderMode") == "sprite"]
        assert len(sprite_nodes) > 0, "No sprite nodes found"
        for node in sprite_nodes:
            assert node.get("familyId"), \
                f"Sprite node {node.get('id')} missing familyId"

    def test_feature_maps_are_sprites(self):
        """Feature map nodes should be classified as sprites."""
        leaves: List[Dict[str, Any]] = []
        for child in self.elk["children"]:
            _collect_all_leaves(child, leaves)

        feature_ids = {"input_feature", "decomposed_feat_1", "decomposed_feat_2",
                       "decomposed_feat_3", "adadr_feat_1", "adadr_feat_2",
                       "output_feature", "global_feature"}
        sprite_ids = {n["id"] for n in leaves if n.get("renderMode") == "sprite"}
        matched = feature_ids & sprite_ids
        assert len(matched) >= 4, \
            f"Only {len(matched)} feature maps classified as sprite: {matched}"

    def test_heatmap_nodes_tagged(self):
        """M410: selection_map and dilation_map should have _spriteHint='heatmap'."""
        leaves: List[Dict[str, Any]] = []
        for child in self.elk["children"]:
            _collect_all_leaves(child, leaves)

        heatmap_ids = {n["id"] for n in leaves if n.get("_spriteHint") == "heatmap"}
        assert "selection_map" in heatmap_ids or "dilation_map" in heatmap_ids, \
            f"No heatmap-tagged nodes found. Got: {heatmap_ids}"

    def test_report_statistics(self):
        assert self.report.total_leaves > 20
        assert self.report.operator_count >= 6
        assert len(self.report.families) >= 2


class TestM411InjectSpritesDryRun:
    """M411: inject_sprites with skip_generation produces valid blob fallbacks."""

    def test_dry_run_stamps_all_nodes(self):
        elk = _build_freqselect_elk()
        from backend.pipeline.topology.node_classifier import classify_nodes
        from backend.pipeline.topology.sprite_injector import inject_sprites

        classify_nodes(elk)
        result = asyncio.run(inject_sprites(elk, skip_generation=True))

        assert result.total_sprite_nodes > 0
        assert result.fallback_to_blob == result.total_sprite_nodes
        assert result.refs_stamped == 0  # dry run = no real images

        # Verify all sprite nodes have spriteRef
        leaves: List[Dict[str, Any]] = []
        for child in elk["children"]:
            _collect_all_leaves(child, leaves)

        for leaf in leaves:
            if leaf.get("renderMode") == "sprite":
                assert "spriteRef" in leaf, \
                    f"Sprite node {leaf.get('id')} missing spriteRef after dry_run"
                ref = leaf["spriteRef"]
                assert ref["format"] == "stack", \
                    f"Dry run should produce stack format, got: {ref['format']}"


class TestM415DataUri:
    """M415: spriteRef URLs are always Data URI format."""

    def test_stamp_sprite_ref_produces_data_uri(self):
        from backend.pipeline.topology.sprite_injector import _stamp_sprite_ref
        import base64

        # Simulate a real image (1x1 red pixel PNG)
        fake_b64 = base64.b64encode(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100).decode()

        node: Dict[str, Any] = {"id": "test_node"}
        _stamp_sprite_ref(node, image_b64=fake_b64)

        ref = node["spriteRef"]
        assert ref["format"] == "png"
        assert ref["url"].startswith("data:image/png;base64,"), \
            f"URL should be Data URI, got: {ref['url'][:50]}"

    def test_stamp_sprite_ref_fallback(self):
        from backend.pipeline.topology.sprite_injector import _stamp_sprite_ref

        node: Dict[str, Any] = {"id": "test_node"}
        _stamp_sprite_ref(node, image_b64=None)

        ref = node["spriteRef"]
        assert ref["format"] == "stack"
        assert "url" not in ref


class TestM416EndToEnd:
    """M416: Full pipeline from ELK graph to editor-ready enriched graph."""

    def test_full_pipeline_dry_run(self):
        """Complete e2e: build graph → classify → inject (dry) → validate."""
        elk = _build_freqselect_elk()

        from backend.pipeline.topology.node_classifier import (
            classify_nodes, consolidate_layers,
        )
        from backend.pipeline.topology.sprite_injector import inject_sprites

        # Step 1: Classify
        report = classify_nodes(elk)
        assert report.total_leaves > 20

        # Step 2: Consolidate
        consolidate_layers(elk, top_k=3)

        # Step 3: Inject (dry run)
        result = asyncio.run(inject_sprites(elk, skip_generation=True))

        # Step 4: Validate enriched graph is tldraw-ready
        leaves: List[Dict[str, Any]] = []
        for child in elk["children"]:
            _collect_all_leaves(child, leaves)

        # Every leaf has renderMode
        for leaf in leaves:
            assert "renderMode" in leaf

        # Sprite nodes have spriteRef
        sprite_leaves = [n for n in leaves if n.get("renderMode") == "sprite"]
        for node in sprite_leaves:
            assert "spriteRef" in node

        # Operator nodes have isOperator
        operator_leaves = [n for n in leaves if n.get("isOperator")]
        assert len(operator_leaves) >= 6

        # Kernel nodes exist
        kernel_leaves = [n for n in leaves if n.get("renderMode") == "kernel"]
        assert len(kernel_leaves) >= 3

        # The graph is JSON-serializable (required for frontend transfer)
        json_str = json.dumps(elk)
        assert len(json_str) > 1000  # non-trivial graph

        print(f"\n✅ E2E pipeline validated:")
        print(f"   {report.total_leaves} leaves classified")
        print(f"   {len(sprite_leaves)} sprite nodes (all with spriteRef)")
        print(f"   {len(operator_leaves)} operators")
        print(f"   {len(kernel_leaves)} kernel grids")
        print(f"   {len(report.families)} sprite families")
        print(f"   JSON size: {len(json_str)} bytes")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
