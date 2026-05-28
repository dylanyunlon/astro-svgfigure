"""layered_pipeline.py — The Complete dispatch() Function.

In CCCL f984c90, the file `dispatch_topk.cuh` contains one function:

    template <...>
    CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE
    cudaError_t dispatch(...) {
        // Allocation
        counter_t* counter = allocations[0];
        OffsetT* histogram  = allocations[1];

        // Pass 0: histogram-only kernel
        { launcher.doit(histogram_kernel, d_keys_in, ...); }

        // Passes 1..N: fused filter+histogram
        DoubleBuffer<key_in_t> key_bufs(alloc[3], alloc[2]);
        for (int pass = 1; pass < num_passes; pass++) {
            launcher.doit(topk_kernel,
                key_bufs.Current(), key_bufs.Alternate(), ...);
            key_bufs.selector ^= 1;
        }

        // Final filter
        invoke_last_filter(key_bufs.Current(), ...);

        return cudaSuccess;
    }

That's the entire dispatch: 95 lines of pure orchestration.  No
algorithm — just wiring kernels together with buffer management.
The kernels do the work.  The dispatch manages the flow.

Our `generate_layered_topology()` is the same:
  1. Parse intent (pure function, no LLM — like init code before kernels)
  2. Plan regions (histogram kernel — pass 0 over full input)
  3. Generate per-region (fused kernel loop — passes 1..N)
  4. Compose canvas (final filter — merge to unified output)
  5. Return result (return cudaSuccess)

Each step calls a function from M1-M5.  This module adds no
algorithm — it just wires them together.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from backend.pipeline.topology.user_intent_parser import (
    parse_user_intent, UserIntent, DiagramType,
)
from backend.pipeline.topology.region_planner import (
    PlannedRegion, plan_regions_with_llm,
    compute_canvas_size, layout_regions_grid,
)
from backend.pipeline.topology.per_region_generator import (
    generate_all_regions, generate_region,
)
from backend.pipeline.topology.canvas_compositor import (
    compose, recompose_region, ComposedCanvas,
)
from backend.pipeline.topology.finalize_pass import PassContext

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  §1  Result type — the cudaError_t equivalent
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LayeredResult:
    """Complete output of the layered pipeline.

    Like CCCL's dispatch() return: either success with full output,
    or error with diagnostics.
    """
    success: bool = True
    canvas: Optional[ComposedCanvas] = None
    intent: Optional[UserIntent] = None
    regions: List[PlannedRegion] = field(default_factory=list)
    context: Optional[PassContext] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "success": self.success,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }
        if self.canvas:
            d.update(self.canvas.to_dict())
        if self.intent:
            d["intent"] = self.intent.summary()
        if self.regions:
            d["regions"] = [r.to_dict() for r in self.regions]
        if self.context:
            d["pipeline_stats"] = self.context.stats()
        if self.diagnostics:
            d["diagnostics"] = self.diagnostics
        if self.error:
            d["error"] = self.error
        return d


# ═══════════════════════════════════════════════════════════════════════════
#  §2  dispatch() — the full pipeline
# ═══════════════════════════════════════════════════════════════════════════

async def generate_layered_topology(
    text: str,
    ai_engine,
    *,
    model: str = "",
    canvas_width: Optional[int] = None,
    canvas_height: Optional[int] = None,
    max_regions: int = 8,
    skip_refinement: bool = False,
) -> LayeredResult:
    """Run the complete layered topology pipeline.

    This is the dispatch() function — the single entry point that
    orchestrates all kernels.

    Pipeline stages (mapped to CCCL):
      0. Intent parse    — pre-kernel init (allocations, counter init)
      1. Region plan     — DeviceTopKHistogramKernel (pass 0)
      2. Per-region gen  — DeviceTopKKernel loop (passes 1..N)
      3. Canvas compose  — invoke_last_filter + merge to output
      4. [Optional] Refinement — second radix pass (higher bits)

    Like CCCL's dispatch():
      - PassContext flows through every stage (the counter struct)
      - Early exit if num_passes == 1 (simple input, skip refinement)
      - Error handling: return error code, don't throw

    Args:
        text: Full user text (NEVER truncated — the histogram kernel
              reads the complete input)
        ai_engine: AIEngine instance for LLM calls
        model: LLM model override
        canvas_width: Optional canvas width override
        canvas_height: Optional canvas height override
        max_regions: Maximum regions to plan (default 8)
        skip_refinement: Skip the optional refinement pass

    Returns:
        LayeredResult with composed canvas or error
    """
    t0 = time.monotonic()
    context = PassContext()
    diag: Dict[str, Any] = {"stages": {}}
    result = LayeredResult(context=context, diagnostics=diag)

    try:
        # ══════════════════════════════════════════════════════════════
        #  Stage 0: Intent parse (pre-kernel — no LLM calls)
        #  CCCL: counter_t* counter = allocations[0];
        # ══════════════════════════════════════════════════════════════

        t_stage = time.monotonic()
        intent = parse_user_intent(text)
        result.intent = intent

        # Compute canvas size if not provided
        if canvas_width is None or canvas_height is None:
            cw, ch = compute_canvas_size(intent)
            canvas_width = canvas_width or cw
            canvas_height = canvas_height or ch

        # Clamp region count
        n_regions = min(intent.estimated_regions, max_regions)
        intent.estimated_regions = n_regions

        diag["stages"]["intent_parse"] = {
            "elapsed_ms": _ms(t_stage),
            "diagram_type": intent.diagram_type.value,
            "confidence": round(intent.confidence, 2),
            "entities_detected": len(intent.entities),
            "estimated_regions": n_regions,
            "canvas_size": f"{canvas_width}x{canvas_height}",
        }

        logger.info(
            f"Intent: type={intent.diagram_type.value} "
            f"entities={len(intent.entities)} regions={n_regions} "
            f"canvas={canvas_width}x{canvas_height}"
        )

        # ══════════════════════════════════════════════════════════════
        #  Stage 1: Region planning (histogram kernel — pass 0)
        #  CCCL: { launcher.doit(histogram_kernel, d_keys_in, ...); }
        # ══════════════════════════════════════════════════════════════

        t_stage = time.monotonic()

        # Fast path: simple inputs get deterministic grid layout (no LLM)
        # Like CCCL's "if num_passes == 1" optimization
        if n_regions <= 2 and intent.confidence < 0.3:
            bboxes = layout_regions_grid(n_regions, canvas_width, canvas_height)
            regions = [
                PlannedRegion(
                    id=f"region_{i}",
                    name=f"Region {i + 1}",
                    bbox=bbox,
                    entity_hints=[e.name for e in intent.entities[i::n_regions]],
                    priority=i,
                )
                for i, bbox in enumerate(bboxes)
            ]
            diag["stages"]["region_plan"] = {
                "elapsed_ms": _ms(t_stage),
                "source": "deterministic_grid",
                "region_count": len(regions),
            }
        else:
            regions, plan_diag = await plan_regions_with_llm(
                text=text,
                intent=intent,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
                ai_engine=ai_engine,
                model=model,
                context=context,
            )
            diag["stages"]["region_plan"] = {
                "elapsed_ms": _ms(t_stage),
                **plan_diag,
            }

        result.regions = regions
        logger.info(f"Planned {len(regions)} regions")

        # ══════════════════════════════════════════════════════════════
        #  Stage 2: Per-region generation (fused kernel loop)
        #  CCCL:
        #    DoubleBuffer<key_in_t> key_bufs(alloc[3], alloc[2]);
        #    for (int pass = 1; pass < num_passes; pass++) {
        #        launcher.doit(topk_kernel,
        #            key_bufs.Current(), key_bufs.Alternate(), ...);
        #        key_bufs.selector ^= 1;
        #    }
        # ══════════════════════════════════════════════════════════════

        t_stage = time.monotonic()

        subgraphs, gen_diag = await generate_all_regions(
            regions=regions,
            text=text,
            intent=intent,
            ai_engine=ai_engine,
            model=model,
            context=context,
        )

        diag["stages"]["per_region_gen"] = {
            "elapsed_ms": _ms(t_stage),
            **gen_diag,
        }
        logger.info(
            f"Generated {len(subgraphs)} subgraphs, "
            f"{context.total_entities} nodes, {context.total_edges} edges"
        )

        # ══════════════════════════════════════════════════════════════
        #  Stage 2.5: Refinement pass (second radix pass)
        #  CCCL:
        #    // Passes 2..N refine with higher radix bits
        #    key_bufs.selector ^= 1;
        #
        #  Each region reads its pass-1 result + neighbor context,
        #  writes a refined subgraph.  DoubleBuffer swap after each.
        #  Skip if: simple input, or caller requested skip_refinement.
        # ══════════════════════════════════════════════════════════════

        if not skip_refinement and n_regions >= 2 and intent.confidence >= 0.3:
            t_stage = time.monotonic()

            from backend.pipeline.topology.per_region_generator import (
                refine_all_regions,
            )

            subgraphs, refine_diag = await refine_all_regions(
                regions=regions,
                subgraphs=subgraphs,
                ai_engine=ai_engine,
                model=model,
                context=context,
            )

            diag["stages"]["refinement"] = {
                "elapsed_ms": _ms(t_stage),
                **refine_diag,
            }
            logger.info("Refinement pass completed for %d regions", len(regions))

        # ══════════════════════════════════════════════════════════════
        #  Stage 3: Canvas composition (invoke_last_filter + merge)
        #  CCCL: invoke_last_filter(key_bufs.Current(), ...);
        # ══════════════════════════════════════════════════════════════

        t_stage = time.monotonic()

        canvas = compose(
            regions=regions,
            subgraphs=subgraphs,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            context=context,
        )

        diag["stages"]["composition"] = {
            "elapsed_ms": _ms(t_stage),
            **canvas.diagnostics,
        }

        result.canvas = canvas
        result.success = True

        logger.info(
            f"Composed canvas: {canvas.width}x{canvas.height}, "
            f"{len(canvas.layers)} layers, "
            f"{len(canvas.cross_region_edges)} cross-region edges"
        )

    except Exception as e:
        logger.exception("Layered pipeline failed")
        result.success = False
        result.error = str(e)

    result.elapsed_ms = (time.monotonic() - t0) * 1000
    diag["total_elapsed_ms"] = result.elapsed_ms
    diag["total_llm_calls"] = context.pass_number
    diag["errors"] = context.errors
    diag["warnings"] = context.warnings

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  §3  Single-region regeneration — partial re-dispatch
# ═══════════════════════════════════════════════════════════════════════════

async def regenerate_single_region(
    region_id: str,
    text: str,
    intent: UserIntent,
    regions: List[PlannedRegion],
    canvas: ComposedCanvas,
    ai_engine,
    model: str = "",
) -> LayeredResult:
    """Regenerate one region and recompose the canvas.

    Like re-running one pass of CCCL's radix loop without restarting
    the entire pipeline.  The DoubleBuffer pattern means we can swap
    in a new result for one region and re-merge without touching the
    others.

    This is the "regenerate this region" feature for the frontend.
    """
    t0 = time.monotonic()
    context = PassContext()
    result = LayeredResult(intent=intent, regions=regions, context=context)

    try:
        # Find the region to regenerate
        target_region = None
        for r in regions:
            if r.id == region_id:
                target_region = r
                break

        if target_region is None:
            result.success = False
            result.error = f"Region '{region_id}' not found"
            return result

        # Re-run per_region_gen for just this region
        new_subgraph, gen_diag = await generate_region(
            region=target_region,
            text=text,
            intent=intent,
            all_regions=regions,
            ai_engine=ai_engine,
            model=model,
            context=context,
        )

        # Recompose with the new subgraph
        new_canvas = recompose_region(
            canvas=canvas,
            region_id=region_id,
            new_subgraph=new_subgraph,
            regions=regions,
        )

        result.canvas = new_canvas
        result.success = True
        result.diagnostics = {
            "regenerated_region": region_id,
            "region_gen": gen_diag,
        }

    except Exception as e:
        logger.exception(f"Region regeneration failed: {region_id}")
        result.success = False
        result.error = str(e)

    result.elapsed_ms = (time.monotonic() - t0) * 1000
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  §4  Utility
# ═══════════════════════════════════════════════════════════════════════════

def _ms(t0: float) -> float:
    """Elapsed milliseconds since t0."""
    return round((time.monotonic() - t0) * 1000, 1)


# ═══════════════════════════════════════════════════════════════════════════
#  §4  Sprite pipeline — classify → prompt → sheet → split → inject (M210-M214)
# ═══════════════════════════════════════════════════════════════════════════
#
# This is the component-level AI-sprite chain assembled by the second Claude.
# It runs AFTER compose() (which owns layout) and fills ELK-positioned cells:
#
#   M210 classify_nodes      → stamp renderMode/familyId on every leaf
#   M211 design_prompts      → one series-consistent prompt per sprite node
#   M212 plan_sheets + gen   → pack sprites into grid sheets, 1 AI call each
#   M213 split_and_clean     → remove green bg, crop cells, tighten, QC
#   M214 inject_sprites      → attach spriteRef to nodes (contain-fit)
#
# Every stage degrades gracefully: a failure flips the affected nodes back to
# text (inject_sprites enforces this), so the figure is always renderable.
# M218 (third Claude) will wrap this in StageResult + progress callbacks and
# call it from generate_layered_topology; here it is a standalone callable so
# M219's e2e test can exercise the whole chain in isolation.


@dataclass
class SpritePipelineResult:
    """Outcome of the sprite fill pass."""
    canvas: Optional[ComposedCanvas] = None
    classification: Dict[str, Any] = field(default_factory=dict)
    sheets: int = 0
    sprites_injected: int = 0
    sprites_fell_back: int = 0
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classification": self.classification,
            "sheets": self.sheets,
            "sprites_injected": self.sprites_injected,
            "sprites_fell_back": self.sprites_fell_back,
            "diagnostics": self.diagnostics,
            "success": self.success,
            "error": self.error,
        }


async def run_sprite_pipeline(
    canvas: ComposedCanvas,
    *,
    settings: Optional[Any] = None,
    model: str = "gemini-3-pro-image-preview",
    api_key: str = "",
    gemini_callable=None,
    removebg_callable=None,
    run_qc: bool = True,
) -> SpritePipelineResult:
    """Run the M210-M214 sprite fill on an already-composed canvas.

    Args:
        canvas: ComposedCanvas from compose() (has elk_graph with layout).
        settings: backend Settings (forwarded to the image generator).
        model: image model name.
        api_key: remove-bg.io key (optional).
        gemini_callable / removebg_callable: dependency-injection seams for
            tests (mock the AI + background removal so CI needs no network).
        run_qc: run per-sprite transparency QC.

    Returns:
        SpritePipelineResult; canvas is mutated in place with spriteRefs.
    """
    from backend.pipeline.topology.node_classifier import classify_nodes
    from backend.pipeline.topology.sprite_prompt_designer import (
        design_prompts_for_classified,
    )
    from backend.pipeline.topology.sprite_batch_generator import (
        plan_sheets, generate_sprite_sheet,
    )
    from backend.pipeline.topology.sprite_sheet_splitter import split_and_clean

    res = SpritePipelineResult(canvas=canvas)
    try:
        elk = canvas.elk_graph or {}

        # M210 — classify every leaf, detect sprite families.
        report = classify_nodes(elk)
        res.classification = report.to_dict()

        if not report.families:
            # No sprite nodes — pure vector figure, nothing to generate.
            res.success = True
            res.diagnostics["note"] = "no sprite nodes; vector-only figure"
            return res

        # M211 — design series-consistent prompts, family by family.
        prompts = design_prompts_for_classified(elk, report.families)
        if not prompts:
            res.success = True
            res.diagnostics["note"] = "no prompts produced"
            return res

        # M212 — pack into sheets and generate each (one AI call per sheet).
        sheets = plan_sheets(prompts)
        res.sheets = len(sheets)
        all_assets: List[Any] = []
        # node_id → family_id, to annotate assets after split.
        fam_by_node = {p.node_id: p.family_id for p in prompts}
        for sheet_prompts in sheets:
            sheet = await generate_sprite_sheet(
                sheet_prompts, settings=settings, model=model,
                gemini_callable=gemini_callable,
            )
            # M213 — slice + clean this sheet into per-node sprites.
            split = await split_and_clean(
                sheet, api_key=api_key,
                removebg_callable=removebg_callable, run_qc=run_qc,
            )
            for a in split.assets:
                a.family_id = fam_by_node.get(a.node_id, "")
            all_assets.extend(split.assets)

        # M214 — inject sprites; dropped/failed nodes fall back to text.
        from backend.pipeline.topology.canvas_compositor import inject_sprites
        inject_sprites(canvas, all_assets)

        diag = canvas.diagnostics or {}
        res.sprites_injected = int(diag.get("sprites_injected", 0))
        res.sprites_fell_back = int(diag.get("sprites_fell_back", 0))
        res.diagnostics["total_assets"] = len(all_assets)
        res.success = True
        return res

    except Exception as e:
        logger.exception("sprite pipeline failed")
        res.success = False
        res.error = str(e)
        return res
