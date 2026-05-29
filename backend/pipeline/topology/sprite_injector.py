"""sprite_injector.py — Stage 2.8: Generate sprites and stamp spriteRef.

The missing glue between the node_classifier (which marks renderMode="sprite")
and the frontend renderer (which reads spriteRef to decide what to draw).

Pipeline:
  classify_nodes()             — already done by Stage 2 (marks renderMode)
  design_sprite_prompts()      — M211: per-node prompt with family coherence
  plan_sheets() + generate()   — M212: batch into sprite sheets, one Gemini call each
  split_sheet()                — M213: crop + rembg + QC per cell
  stamp_sprite_refs()          — NEW: write spriteRef onto each node in elk_graph

On failure at any step, the affected node keeps spriteRef absent or format='stack',
which makes the frontend fall back to the organic blob (renderOrganicBlob) or
dashed-box-with-label — never a silent hole.

CCCL analogy: this is the "device_inspect" pass that CCCL runs between the
histogram kernel and the filter kernel — it reads the histogram output,
decides the bin boundaries, and writes them into shared memory so the
filter kernel can read them.  We read the classified nodes, generate
sprites, and write spriteRef so the SVG renderer can read them.
"""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SpriteInjectionResult:
    """Diagnostics for the sprite injection stage."""
    total_sprite_nodes: int = 0
    prompts_designed: int = 0
    sheets_planned: int = 0
    sheets_generated: int = 0
    sheets_succeeded: int = 0
    cells_split: int = 0
    cells_dropped: int = 0
    refs_stamped: int = 0
    fallback_to_blob: int = 0
    elapsed_ms: float = 0.0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_sprite_nodes": self.total_sprite_nodes,
            "prompts_designed": self.prompts_designed,
            "sheets_planned": self.sheets_planned,
            "sheets_generated": self.sheets_generated,
            "sheets_succeeded": self.sheets_succeeded,
            "cells_split": self.cells_split,
            "cells_dropped": self.cells_dropped,
            "refs_stamped": self.refs_stamped,
            "fallback_to_blob": self.fallback_to_blob,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "errors": list(self.errors),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  §1  Walk the ELK graph to find all sprite-classified leaf nodes
# ═══════════════════════════════════════════════════════════════════════════

def _collect_sprite_nodes(
    node: Dict[str, Any],
    out: List[Dict[str, Any]],
) -> None:
    """Recursively collect all nodes with renderMode == 'sprite'."""
    children = node.get("children")
    if isinstance(children, list) and children:
        for c in children:
            if isinstance(c, dict):
                _collect_sprite_nodes(c, out)
    else:
        # Leaf node
        if node.get("renderMode") == "sprite":
            out.append(node)


# ═══════════════════════════════════════════════════════════════════════════
#  §2  stamp_sprite_refs — write spriteRef onto each node
# ═══════════════════════════════════════════════════════════════════════════

def _stamp_sprite_ref(
    node: Dict[str, Any],
    image_b64: Optional[str],
    true_bbox: Optional[Tuple[int, int, int, int]],
    format: str = "png",
    stack_count: int = 3,
) -> None:
    """Write a spriteRef onto an ELK node dict, in-place.

    If image_b64 is None, stamp format='stack' so the frontend
    falls back to renderOrganicBlob (never an empty node).
    """
    if image_b64:
        node["spriteRef"] = {
            "format": format,
            "url": f"data:image/png;base64,{image_b64}",
            "bbox": list(true_bbox) if true_bbox else [0, 0, 256, 256],
            "fit": "contain",
            "stackCount": stack_count,
        }
    else:
        # Fallback: organic blob
        node["spriteRef"] = {
            "format": "stack",
            "stackCount": stack_count,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  §3  inject_sprites — the full pipeline
# ═══════════════════════════════════════════════════════════════════════════

async def inject_sprites(
    elk_graph: Dict[str, Any],
    *,
    settings: Optional[Any] = None,
    model: str = "gemini-2.0-flash-preview-image-generation",
    gemini_callable=None,
    skip_generation: bool = False,
) -> SpriteInjectionResult:
    """Run the full sprite pipeline on a composed ELK graph.

    This is Stage 2.8 in the layered pipeline — after classification
    and composition, before SVG rendering.

    Args:
        elk_graph: The composed ELK graph dict (mutated in place).
        settings: Backend settings (forwarded to Gemini).
        model: Image generation model name.
        gemini_callable: Injectable async fn for testing.
        skip_generation: If True, stamp all sprites as 'stack' (blob fallback)
            without calling Gemini. Useful for fast preview / offline mode.

    Returns:
        SpriteInjectionResult with diagnostics.
    """
    import time
    t0 = time.monotonic()
    result = SpriteInjectionResult()

    # ── Step 1: Collect all sprite-classified nodes ──
    sprite_nodes: List[Dict[str, Any]] = []
    for child in elk_graph.get("children", []):
        if isinstance(child, dict):
            _collect_sprite_nodes(child, sprite_nodes)

    result.total_sprite_nodes = len(sprite_nodes)
    if not sprite_nodes:
        result.elapsed_ms = (time.monotonic() - t0) * 1000
        return result

    # If skipping generation, stamp all as blob fallback
    if skip_generation:
        for i, node in enumerate(sprite_nodes):
            _stamp_sprite_ref(node, image_b64=None, true_bbox=None,
                              stack_count=2 + (i % 3))
            result.fallback_to_blob += 1
        result.elapsed_ms = (time.monotonic() - t0) * 1000
        return result

    # ── Step 2: Design prompts (M211) ──
    try:
        from backend.pipeline.topology.sprite_prompt_designer import (
            design_prompts_for_classified, SpritePrompt,
        )
        from backend.pipeline.topology.node_classifier import (
            detect_sprite_families, SpriteFamily,
        )

        # Build (node_id, node_dict) pairs for family detection
        sprite_pairs = [(n.get("id", ""), n) for n in sprite_nodes]
        families = detect_sprite_families(sprite_pairs)

        # Design prompts: walks the elk_graph, wires family membership
        prompts: List[SpritePrompt] = design_prompts_for_classified(
            elk_graph, families,
        )

        result.prompts_designed = len(prompts)
    except Exception as e:
        logger.exception("Sprite prompt design failed — falling back to blobs")
        result.errors.append(f"prompt_design: {e}")
        for i, node in enumerate(sprite_nodes):
            _stamp_sprite_ref(node, image_b64=None, true_bbox=None,
                              stack_count=2 + (i % 3))
            result.fallback_to_blob += 1
        result.elapsed_ms = (time.monotonic() - t0) * 1000
        return result

    # ── Step 3: Plan sheets (M212) ──
    try:
        from backend.pipeline.topology.sprite_batch_generator import (
            plan_sheets, generate_sheets_concurrent,
        )

        sheet_groups = plan_sheets(prompts)
        result.sheets_planned = len(sheet_groups)
    except Exception as e:
        logger.exception("Sheet planning failed — falling back to blobs")
        result.errors.append(f"sheet_plan: {e}")
        for i, node in enumerate(sprite_nodes):
            _stamp_sprite_ref(node, image_b64=None, true_bbox=None,
                              stack_count=2 + (i % 3))
            result.fallback_to_blob += 1
        result.elapsed_ms = (time.monotonic() - t0) * 1000
        return result

    # ── Step 4: Generate sheets (Gemini call) ──
    try:
        sheets = await generate_sheets_concurrent(
            sheet_groups,
            settings=settings,
            model=model,
            gemini_callable=gemini_callable,
        )
        result.sheets_generated = len(sheets)
        result.sheets_succeeded = sum(1 for s in sheets if s.success)
    except Exception as e:
        logger.exception("Sheet generation failed — falling back to blobs")
        result.errors.append(f"sheet_gen: {e}")
        for i, node in enumerate(sprite_nodes):
            _stamp_sprite_ref(node, image_b64=None, true_bbox=None,
                              stack_count=2 + (i % 3))
            result.fallback_to_blob += 1
        result.elapsed_ms = (time.monotonic() - t0) * 1000
        return result

    # ── Step 5: Split sheets → per-cell sprites (M213: crop + rembg) ──
    try:
        from backend.pipeline.topology.sprite_sheet_splitter import (
            split_and_clean, SplitResult,
        )

        # Map node_id → node dict for fast lookup
        node_by_id: Dict[str, Dict[str, Any]] = {
            n.get("id", ""): n for n in sprite_nodes
        }

        for sheet in sheets:
            if not sheet.success or not sheet.image_b64:
                # Sheet failed — stamp all its cells as blob fallback
                for cell in sheet.cells:
                    node = node_by_id.get(cell.node_id)
                    if node:
                        _stamp_sprite_ref(node, image_b64=None, true_bbox=None,
                                          stack_count=2 + (cell.col % 3))
                        result.fallback_to_blob += 1
                continue

            # Split the sheet into per-cell transparent sprites
            split_result: SplitResult = await split_and_clean(
                sheet=sheet,
            )
            result.cells_split += len(split_result.assets)
            result.cells_dropped += len(split_result.dropped_node_ids)

            # Stamp each asset onto its node
            for asset in split_result.assets:
                node = node_by_id.get(asset.node_id)
                if not node:
                    continue
                if asset.dropped or not asset.image_b64:
                    _stamp_sprite_ref(node, image_b64=None, true_bbox=None,
                                      stack_count=2)
                    result.fallback_to_blob += 1
                else:
                    _stamp_sprite_ref(
                        node,
                        image_b64=asset.image_b64,
                        true_bbox=asset.true_bbox,
                        format=asset.format,
                        stack_count=3,
                    )
                    result.refs_stamped += 1

    except Exception as e:
        logger.exception("Sheet splitting failed — falling back to blobs")
        result.errors.append(f"sheet_split: {e}")
        # Stamp remaining unstamped nodes as blob fallback
        for node in sprite_nodes:
            if "spriteRef" not in node:
                _stamp_sprite_ref(node, image_b64=None, true_bbox=None,
                                  stack_count=2)
                result.fallback_to_blob += 1

    # ── Step 6: Ensure every sprite node has a spriteRef ──
    # Safety net: any node that slipped through without a ref gets blob fallback.
    for node in sprite_nodes:
        if "spriteRef" not in node:
            _stamp_sprite_ref(node, image_b64=None, true_bbox=None,
                              stack_count=2)
            result.fallback_to_blob += 1

    result.elapsed_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "Sprite injection: %d nodes, %d stamped, %d blob fallback, %.0fms",
        result.total_sprite_nodes, result.refs_stamped,
        result.fallback_to_blob, result.elapsed_ms,
    )
    return result
