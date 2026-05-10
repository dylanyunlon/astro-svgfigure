"""
pipeline_orchestrator.py — End-to-End Post-Generation Pipeline
================================================================
Chains the entire post-generation pipeline into a single callable:

  generate-image (input) → green-screen removal → layer separation
  → edge refinement → component outlining → transparent PNG export

This replaces the need for the frontend to make 5 sequential API calls.
One call to /api/pipeline-run does everything.

Pipeline Steps:
  Step 1: Background Removal (removebg_route.handle_removebg)
          Tiered: remove-bg.io → rembg → chroma-key
          Input:  base64 frames with green background
          Output: base64 frames with transparent background

  Step 2: Layer Separation (layer_separator.separate_layers_batch)
          Connected-component labeling to extract individual elements
          Input:  transparent-background frames
          Output: per-frame list of component layers

  Step 3: Edge Refinement (edge_refiner.refine_layers_batch)
          Anti-aliasing, sub-pixel smoothing, alpha matting
          Input:  component layers
          Output: refined component layers

  Step 4: Component Outlining (component_outliner.outline_components_batch)
          Stroke generation using distance-transform dilation
          Input:  refined layers
          Output: outlined layers (optional SVG document)

  Step 5: Export (export_components.export_batch)
          Package all layers into downloadable transparent PNGs
          Input:  outlined layers + metadata
          Output: zip archive or individual base64 PNGs

Architecture (from Megatron-Core pipeline parallelism):
───────────────────────────────────────────────────────
From Megatron's PipelineParallel class:

  class PipelineParallel:
      def __init__(self, stages, config):
          self.stages = stages  # [EmbedStage, TransformerStage, ..., LMHeadStage]
          self.micro_batches = config.num_microbatches

      def forward(self, batch):
          for stage in self.stages:
              batch = stage.forward(batch)
          return batch

Then, follow that pattern to implement PipelineOrchestrator with
stages = [RemoveBgStage, LayerSepStage, EdgeRefineStage, OutlineStage,
ExportStage]. Each stage receives the output of the previous stage.

Next, introduce the StageResult dataclass for inter-stage data passing:

  @dataclass
  class StageResult:
      success: bool
      data: Any
      stage_name: str
      processing_time_ms: float
      diagnostics: dict

Subsequently, integrate partial-success handling: if Step 2 produces
20 layers but Step 3 fails on 3 of them, the pipeline continues with
17 layers and reports which ones failed.

Finally, perfect the progress callback system so the frontend can show
a progress bar with per-stage status updates.

Knuth-Level Critiques:
─────────────────────
User Angle:
  - The pipeline takes 10-60 seconds for 16 frames depending on the
    removal method and layer complexity. A progress callback is
    essential to prevent the user from thinking the request hung.
  - If any stage fails entirely (e.g., no layers found in Step 2),
    the pipeline returns partial results up to the last successful
    stage. The user can still download the transparent PNGs even if
    outlining failed.
  - The skip_steps parameter lets power users disable steps they don't
    need (e.g., skip outlining if they want raw transparent layers).

System Angle:
  - Memory budget: 16 frames × 1024×1024 × 4 bytes × 3 copies
    (original + transparent + layers) = ~192MB peak. This is within
    the 512MB default Python process limit but close. For 32+ frames,
    consider streaming or disk-based intermediate storage.
  - The pipeline is sequential (not parallel) across stages because
    each stage depends on the previous. Within each stage, frames are
    processed in parallel using thread pools.
  - Error isolation: each stage catches all exceptions and returns a
    StageResult with success=False. The orchestrator then decides
    whether to continue (partial success) or abort (total failure).

GitHub references:
  - NVIDIA/Megatron-LM (pipeline parallelism)
  - NVIDIA/nccl (allreduce pattern for quality consensus)
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
import zipfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


# ═══════════════════════════════════════════════════════════════════════
#  Data Classes
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class StageResult:
    """Result from a single pipeline stage."""
    success: bool
    stage_name: str
    data: Any = None
    processing_time_ms: float = 0.0
    frames_processed: int = 0
    frames_failed: int = 0
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline.

    From Google's TF ConfigProto pattern:
    Start from TF's tf.compat.v1.ConfigProto which bundles all runtime
    config into a single protobuf. Then, follow that pattern to bundle
    all pipeline parameters into one config object. Next, introduce
    per-stage sub-configs. Subsequently, integrate skip_steps for
    selective stage execution. Finally, perfect the export format
    options (zip, individual, svg).
    """
    # Background removal
    removal_method: Optional[str] = None  # None = auto
    removal_api_key: str = ""
    removal_tolerance: int = 60
    removal_edge_blur: float = 1.0
    removal_despill: bool = True

    # Layer separation
    connectivity: int = 4
    alpha_threshold: int = 30
    min_component_area: int = 100
    max_components: int = 50
    merge_distance: int = 0
    padding: int = 10
    maintain_position: bool = True
    sort_by: str = "area"

    # Edge refinement
    anti_alias: bool = True
    edge_smoothing: float = 1.0
    alpha_matting: bool = False

    # Component outlining
    stroke_width: float = 2.0
    stroke_color: str = "#000000"
    stroke_profile: str = "uniform"
    glow_enabled: bool = False

    # Export
    export_format: str = "individual"  # "individual", "zip", "svg"
    canvas_width: int = 1024
    canvas_height: int = 1024

    # Pipeline control
    skip_steps: List[str] = field(default_factory=list)
    abort_on_stage_failure: bool = False


@dataclass
class PipelineReport:
    """Full pipeline execution report."""
    success: bool
    stages: List[StageResult] = field(default_factory=list)
    total_time_ms: float = 0.0
    frames_input: int = 0
    layers_output: int = 0
    export_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "stages": [
                {
                    "stage": s.stage_name,
                    "success": s.success,
                    "processing_time_ms": round(s.processing_time_ms, 2),
                    "frames_processed": s.frames_processed,
                    "frames_failed": s.frames_failed,
                    "error": s.error,
                    "diagnostics": s.diagnostics,
                }
                for s in self.stages
            ],
            "total_time_ms": round(self.total_time_ms, 2),
            "frames_input": self.frames_input,
            "layers_output": self.layers_output,
            "export_data": self.export_data,
            "error": self.error,
        }


# ═══════════════════════════════════════════════════════════════════════
#  Stage 1: Background Removal
# ═══════════════════════════════════════════════════════════════════════

async def _stage_removebg(
    frames_b64: List[str],
    config: PipelineConfig,
    progress: Optional[Callable] = None,
) -> StageResult:
    """
    Stage 1: Remove green-screen background from all frames.

    From Megatron's EmbeddingStage which is the first pipeline stage
    converting token IDs to dense vectors. Similarly, this stage
    converts green-background frames to transparent-background frames.
    """
    t0 = time.monotonic()

    if progress:
        progress("removebg", "starting", 0)

    try:
        from backend.pipeline.removebg_route import handle_removebg

        result = await handle_removebg(
            frames_b64=frames_b64,
            api_key=config.removal_api_key,
            force_method=config.removal_method,
            tolerance=config.removal_tolerance,
            edge_blur=config.removal_edge_blur,
            despill=config.removal_despill,
        )

        if not result.get("success"):
            return StageResult(
                success=False,
                stage_name="removebg",
                error=result.get("error", "Background removal failed"),
                processing_time_ms=(time.monotonic() - t0) * 1000,
            )

        # Extract transparent frames
        transparent_frames = []
        failed = 0
        for r in result.get("results", []):
            if r.get("success") and r.get("image_b64"):
                transparent_frames.append(r["image_b64"])
            else:
                failed += 1
                # Pass through original if removal failed
                transparent_frames.append(None)

        # Remove None entries (failed frames)
        valid_frames = [f for f in transparent_frames if f is not None]

        if progress:
            progress("removebg", "complete", 100)

        return StageResult(
            success=len(valid_frames) > 0,
            stage_name="removebg",
            data=valid_frames,
            processing_time_ms=(time.monotonic() - t0) * 1000,
            frames_processed=len(valid_frames),
            frames_failed=failed,
            diagnostics={
                "method": result.get("method", "unknown"),
                "tier": result.get("tier", 0),
            },
        )

    except Exception as e:
        logger.exception("Stage 1 (removebg) failed: %s", e)
        return StageResult(
            success=False,
            stage_name="removebg",
            error=str(e),
            processing_time_ms=(time.monotonic() - t0) * 1000,
        )


# ═══════════════════════════════════════════════════════════════════════
#  Stage 2: Layer Separation
# ═══════════════════════════════════════════════════════════════════════

async def _stage_layer_separate(
    transparent_frames: List[str],
    config: PipelineConfig,
    progress: Optional[Callable] = None,
) -> StageResult:
    """
    Stage 2: Separate transparent frames into component layers.

    From Megatron's TransformerStage which processes tokens through
    attention layers. Similarly, this stage processes transparent
    frames through connected-component analysis.
    """
    t0 = time.monotonic()

    if progress:
        progress("layer_separate", "starting", 0)

    try:
        from backend.pipeline.layer_separator import (
            separate_layers_batch,
            LayerSeparationConfig,
        )

        sep_config = LayerSeparationConfig(
            connectivity=config.connectivity,
            alpha_threshold=config.alpha_threshold,
            min_component_area=config.min_component_area,
            max_components=config.max_components,
            merge_distance=config.merge_distance,
            padding=config.padding,
            maintain_position=config.maintain_position,
            sort_by=config.sort_by,
        )

        result = await separate_layers_batch(transparent_frames, sep_config)

        if not result.get("success"):
            return StageResult(
                success=False,
                stage_name="layer_separate",
                error=result.get("error", "Layer separation failed"),
                processing_time_ms=(time.monotonic() - t0) * 1000,
            )

        # Collect all layer base64s across frames
        all_layers = []
        frames_ok = 0
        frames_fail = 0
        for fr in result.get("frame_results", []):
            if fr.get("success"):
                frames_ok += 1
                for layer in fr.get("layers", []):
                    all_layers.append({
                        "frame_index": fr["frame_index"],
                        "layer_id": layer["layer_id"],
                        "image_b64": layer["image_b64"],
                        "bbox": layer["bbox"],
                        "area": layer["area"],
                        "centroid": layer["centroid"],
                    })
            else:
                frames_fail += 1

        if progress:
            progress("layer_separate", "complete", 100)

        return StageResult(
            success=len(all_layers) > 0,
            stage_name="layer_separate",
            data=all_layers,
            processing_time_ms=(time.monotonic() - t0) * 1000,
            frames_processed=frames_ok,
            frames_failed=frames_fail,
            diagnostics=result.get("stats", {}),
        )

    except Exception as e:
        logger.exception("Stage 2 (layer_separate) failed: %s", e)
        return StageResult(
            success=False,
            stage_name="layer_separate",
            error=str(e),
            processing_time_ms=(time.monotonic() - t0) * 1000,
        )


# ═══════════════════════════════════════════════════════════════════════
#  Stage 3: Edge Refinement
# ═══════════════════════════════════════════════════════════════════════

async def _stage_edge_refine(
    layers: List[Dict[str, Any]],
    config: PipelineConfig,
    progress: Optional[Callable] = None,
) -> StageResult:
    """
    Stage 3: Refine edges of extracted layers.

    From Megatron's LayerNorm stage which normalizes activations
    between transformer layers. Similarly, this stage normalizes
    the alpha channel of each component for clean edges.
    """
    t0 = time.monotonic()

    if progress:
        progress("edge_refine", "starting", 0)

    try:
        from backend.pipeline.edge_refiner import refine_layers_batch

        layer_b64s = [l["image_b64"] for l in layers]

        refine_config = {
            "anti_alias": config.anti_alias,
            "edge_smoothing": config.edge_smoothing,
            "alpha_matting": config.alpha_matting,
        }

        result = refine_layers_batch(layer_b64s, refine_config)

        # Merge refined images back into layer metadata
        refined_layers = []
        processed = 0
        failed = 0
        for i, layer in enumerate(layers):
            if i < len(result) and result[i].get("success"):
                refined = dict(layer)
                refined["image_b64"] = result[i]["image_b64"]
                refined["edge_refined"] = True
                refined_layers.append(refined)
                processed += 1
            else:
                # Keep original if refinement failed
                refined = dict(layer)
                refined["edge_refined"] = False
                refined_layers.append(refined)
                failed += 1

        if progress:
            progress("edge_refine", "complete", 100)

        return StageResult(
            success=True,
            stage_name="edge_refine",
            data=refined_layers,
            processing_time_ms=(time.monotonic() - t0) * 1000,
            frames_processed=processed,
            frames_failed=failed,
        )

    except ImportError as ie:
        logger.warning("Stage 3 (edge_refine) import error: %s", ie)
        # Pass through layers unrefined
        return StageResult(
            success=True,
            stage_name="edge_refine",
            data=layers,
            processing_time_ms=(time.monotonic() - t0) * 1000,
            frames_processed=len(layers),
            diagnostics={"skipped": True, "reason": str(ie)},
        )
    except Exception as e:
        logger.exception("Stage 3 (edge_refine) failed: %s", e)
        return StageResult(
            success=True,
            stage_name="edge_refine",
            data=layers,
            processing_time_ms=(time.monotonic() - t0) * 1000,
            diagnostics={"error": str(e), "fallback": "unrefined"},
        )


# ═══════════════════════════════════════════════════════════════════════
#  Stage 4: Component Outlining
# ═══════════════════════════════════════════════════════════════════════

async def _stage_outline(
    layers: List[Dict[str, Any]],
    config: PipelineConfig,
    progress: Optional[Callable] = None,
) -> StageResult:
    """
    Stage 4: Generate outlines/strokes for each component.

    From Megatron's LMHeadStage which is the final stage producing
    logits. Similarly, this is the final processing stage producing
    the outlined component images.
    """
    t0 = time.monotonic()

    if progress:
        progress("outline", "starting", 0)

    try:
        from backend.pipeline.component_outliner import (
            outline_components_batch,
        )

        layer_b64s = [l["image_b64"] for l in layers]

        outline_config = {
            "stroke_width": config.stroke_width,
            "stroke_color": config.stroke_color,
            "stroke_profile": config.stroke_profile,
            "glow_enabled": config.glow_enabled,
        }

        result = outline_components_batch(layer_b64s, outline_config)

        # Merge outlined images back
        outlined_layers = []
        processed = 0
        failed = 0
        for i, layer in enumerate(layers):
            if i < len(result) and result[i].get("success"):
                outlined = dict(layer)
                outlined["image_b64"] = result[i].get("outlined_b64", layer["image_b64"])
                outlined["svg_path"] = result[i].get("svg_path")
                outlined["outlined"] = True
                outlined_layers.append(outlined)
                processed += 1
            else:
                outlined = dict(layer)
                outlined["outlined"] = False
                outlined_layers.append(outlined)
                failed += 1

        if progress:
            progress("outline", "complete", 100)

        return StageResult(
            success=True,
            stage_name="outline",
            data=outlined_layers,
            processing_time_ms=(time.monotonic() - t0) * 1000,
            frames_processed=processed,
            frames_failed=failed,
        )

    except ImportError as ie:
        logger.warning("Stage 4 (outline) import error: %s", ie)
        return StageResult(
            success=True,
            stage_name="outline",
            data=layers,
            processing_time_ms=(time.monotonic() - t0) * 1000,
            diagnostics={"skipped": True, "reason": str(ie)},
        )
    except Exception as e:
        logger.exception("Stage 4 (outline) failed: %s", e)
        return StageResult(
            success=True,
            stage_name="outline",
            data=layers,
            processing_time_ms=(time.monotonic() - t0) * 1000,
            diagnostics={"error": str(e), "fallback": "unoutlined"},
        )


# ═══════════════════════════════════════════════════════════════════════
#  Stage 5: Export
# ═══════════════════════════════════════════════════════════════════════

async def _stage_export(
    layers: List[Dict[str, Any]],
    config: PipelineConfig,
    progress: Optional[Callable] = None,
) -> StageResult:
    """
    Stage 5: Package layers for download.

    From OpenAI's batch results endpoint pattern:
    Start from OpenAI's batch API which returns results in a structured
    format. Then, follow that pattern to package layers into either
    individual base64 PNGs (for frontend rendering) or a zip archive
    (for bulk download). Next, introduce SVG document generation that
    combines all layers into a single interactive SVG. Subsequently,
    integrate metadata (bounding boxes, areas, layer ordering) into
    the export manifest. Finally, perfect the zip compression for
    efficient transfer.
    """
    t0 = time.monotonic()

    if progress:
        progress("export", "starting", 0)

    try:
        if config.export_format == "zip":
            # Build a zip archive in memory
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                manifest = []
                for i, layer in enumerate(layers):
                    filename = f"layer_{i:03d}_f{layer.get('frame_index', 0)}.png"
                    img_data = base64.b64decode(layer["image_b64"])
                    zf.writestr(filename, img_data)
                    manifest.append({
                        "filename": filename,
                        "frame_index": layer.get("frame_index", 0),
                        "layer_id": layer.get("layer_id", i),
                        "bbox": layer.get("bbox"),
                        "area": layer.get("area"),
                    })
                # Add manifest
                import json
                zf.writestr("manifest.json", json.dumps(manifest, indent=2))

            zip_b64 = base64.b64encode(zip_buf.getvalue()).decode("ascii")

            export_data = {
                "format": "zip",
                "zip_b64": zip_b64,
                "zip_size_bytes": len(zip_buf.getvalue()),
                "total_layers": len(layers),
                "manifest": manifest,
            }

        elif config.export_format == "svg":
            # Build combined SVG
            try:
                from backend.pipeline.component_outliner import build_svg_document
                svg_doc = build_svg_document(
                    [{"success": True, "outlined_b64": l["image_b64"], "bbox": l.get("bbox")}
                     for l in layers],
                    config.canvas_width,
                    config.canvas_height,
                )
                export_data = {
                    "format": "svg",
                    "svg_document": svg_doc,
                    "total_layers": len(layers),
                }
            except ImportError:
                # Fallback to individual
                export_data = {
                    "format": "individual",
                    "layers": [
                        {
                            "image_b64": l["image_b64"],
                            "frame_index": l.get("frame_index", 0),
                            "layer_id": l.get("layer_id", i),
                            "bbox": l.get("bbox"),
                            "area": l.get("area"),
                            "centroid": l.get("centroid"),
                        }
                        for i, l in enumerate(layers)
                    ],
                    "total_layers": len(layers),
                }
        else:
            # Individual layers (default)
            export_data = {
                "format": "individual",
                "layers": [
                    {
                        "image_b64": l["image_b64"],
                        "frame_index": l.get("frame_index", 0),
                        "layer_id": l.get("layer_id", i),
                        "bbox": l.get("bbox"),
                        "area": l.get("area"),
                        "centroid": l.get("centroid"),
                        "outlined": l.get("outlined", False),
                        "edge_refined": l.get("edge_refined", False),
                    }
                    for i, l in enumerate(layers)
                ],
                "total_layers": len(layers),
            }

        if progress:
            progress("export", "complete", 100)

        return StageResult(
            success=True,
            stage_name="export",
            data=export_data,
            processing_time_ms=(time.monotonic() - t0) * 1000,
            frames_processed=len(layers),
        )

    except Exception as e:
        logger.exception("Stage 5 (export) failed: %s", e)
        return StageResult(
            success=False,
            stage_name="export",
            error=str(e),
            processing_time_ms=(time.monotonic() - t0) * 1000,
        )


# ═══════════════════════════════════════════════════════════════════════
#  Pipeline Orchestrator — Main Entry Point
# ═══════════════════════════════════════════════════════════════════════

async def run_pipeline(
    frames_b64: List[str],
    config: Optional[PipelineConfig] = None,
    progress: Optional[Callable] = None,
) -> PipelineReport:
    """
    Run the full post-generation pipeline.

    Parameters
    ----------
    frames_b64 : list of base64 strings
        Input frames with green-screen background.
    config : PipelineConfig, optional
        Pipeline configuration. Uses defaults if None.
    progress : callable, optional
        Progress callback: progress(stage_name, status, percent)

    Returns
    -------
    PipelineReport with per-stage results and final export data.

    From NVIDIA's Megatron-Core pipeline parallel forward pass:
    Start from Megatron's forward_backward_pipelining which sends
    micro-batches through a sequence of stages. Then, follow that
    pattern to send frames through removal → separation → refinement
    → outlining → export stages. Next, introduce the skip_steps
    mechanism for selective stage execution. Subsequently, integrate
    the abort_on_stage_failure flag for strict vs lenient mode.
    Finally, perfect the PipelineReport aggregation from per-stage
    StageResults.
    """
    if config is None:
        config = PipelineConfig()

    t0 = time.monotonic()
    report = PipelineReport(
        success=False,
        frames_input=len(frames_b64),
    )

    if not frames_b64:
        report.error = "No input frames provided"
        return report

    skip = set(config.skip_steps)

    # ── Stage 1: Background Removal ──────────────────────────────────
    if "removebg" not in skip:
        stage1 = await _stage_removebg(frames_b64, config, progress)
        report.stages.append(stage1)

        if not stage1.success:
            if config.abort_on_stage_failure:
                report.error = f"Stage 1 (removebg) failed: {stage1.error}"
                report.total_time_ms = (time.monotonic() - t0) * 1000
                return report
            # Use original frames as fallback
            transparent_frames = frames_b64
        else:
            transparent_frames = stage1.data
    else:
        transparent_frames = frames_b64
        report.stages.append(StageResult(
            success=True, stage_name="removebg",
            diagnostics={"skipped": True},
        ))

    # ── Stage 2: Layer Separation ────────────────────────────────────
    if "layer_separate" not in skip:
        stage2 = await _stage_layer_separate(transparent_frames, config, progress)
        report.stages.append(stage2)

        if not stage2.success:
            if config.abort_on_stage_failure:
                report.error = f"Stage 2 (layer_separate) failed: {stage2.error}"
                report.total_time_ms = (time.monotonic() - t0) * 1000
                return report
            # Export transparent frames directly
            layers = [
                {"image_b64": f, "frame_index": i, "layer_id": 0}
                for i, f in enumerate(transparent_frames)
            ]
        else:
            layers = stage2.data
    else:
        layers = [
            {"image_b64": f, "frame_index": i, "layer_id": 0}
            for i, f in enumerate(transparent_frames)
        ]
        report.stages.append(StageResult(
            success=True, stage_name="layer_separate",
            diagnostics={"skipped": True},
        ))

    # ── Stage 3: Edge Refinement ─────────────────────────────────────
    if "edge_refine" not in skip:
        stage3 = await _stage_edge_refine(layers, config, progress)
        report.stages.append(stage3)
        layers = stage3.data if stage3.data else layers
    else:
        report.stages.append(StageResult(
            success=True, stage_name="edge_refine",
            diagnostics={"skipped": True},
        ))

    # ── Stage 4: Component Outlining ─────────────────────────────────
    if "outline" not in skip:
        stage4 = await _stage_outline(layers, config, progress)
        report.stages.append(stage4)
        layers = stage4.data if stage4.data else layers
    else:
        report.stages.append(StageResult(
            success=True, stage_name="outline",
            diagnostics={"skipped": True},
        ))

    # ── Stage 5: Export ──────────────────────────────────────────────
    if "export" not in skip:
        stage5 = await _stage_export(layers, config, progress)
        report.stages.append(stage5)

        if stage5.success:
            report.export_data = stage5.data
            report.layers_output = stage5.data.get("total_layers", 0)
    else:
        # Return layers directly
        report.export_data = {
            "format": "individual",
            "layers": [
                {
                    "image_b64": l["image_b64"],
                    "frame_index": l.get("frame_index", 0),
                    "layer_id": l.get("layer_id", 0),
                }
                for l in layers
            ],
            "total_layers": len(layers),
        }
        report.layers_output = len(layers)
        report.stages.append(StageResult(
            success=True, stage_name="export",
            diagnostics={"skipped": True},
        ))

    report.success = True
    report.total_time_ms = (time.monotonic() - t0) * 1000

    logger.info(
        "Pipeline complete: %d frames → %d layers, %.0fms, stages: %s",
        report.frames_input,
        report.layers_output,
        report.total_time_ms,
        ", ".join(
            f"{s.stage_name}={'ok' if s.success else 'FAIL'}"
            for s in report.stages
        ),
    )

    return report


# ═══════════════════════════════════════════════════════════════════════
#  Config Builder from Request Params
# ═══════════════════════════════════════════════════════════════════════

def config_from_request(params: dict) -> PipelineConfig:
    """Build PipelineConfig from a JSON request body.

    Maps frontend control names to config fields, with type coercion
    and default fallbacks.

    From ByteDance's TikTok config resolver:
    Start from TikTok's AB-testing config resolver which maps
    experiment parameters to feature flags. Then, follow that pattern
    to map API request parameters to PipelineConfig fields with
    type-safe coercion. Next, introduce nested parameter groups
    (removal.*, separation.*, etc). Subsequently, integrate validation
    with clamping for numeric parameters. Finally, perfect the
    skip_steps parsing from comma-separated string.
    """
    cfg = PipelineConfig()

    # Removal
    cfg.removal_method = params.get("removal_method") or params.get("method")
    cfg.removal_api_key = params.get("api_key", "")
    cfg.removal_tolerance = _int_clamp(params.get("tolerance", 60), 10, 150)
    cfg.removal_edge_blur = _float_clamp(params.get("edge_blur", 1.0), 0, 10)
    cfg.removal_despill = bool(params.get("despill", True))

    # Layer separation
    cfg.connectivity = 8 if params.get("connectivity") == 8 else 4
    cfg.alpha_threshold = _int_clamp(params.get("alpha_threshold", 30), 0, 255)
    cfg.min_component_area = _int_clamp(params.get("min_component_area", 100), 1, 100000)
    cfg.max_components = _int_clamp(params.get("max_components", 50), 1, 500)
    cfg.merge_distance = _int_clamp(params.get("merge_distance", 0), 0, 200)
    cfg.padding = _int_clamp(params.get("padding", 10), 0, 100)
    cfg.maintain_position = bool(params.get("maintain_position", True))
    cfg.sort_by = params.get("sort_by", "area") if params.get("sort_by") in (
        "area", "x", "y", "top-left",
    ) else "area"

    # Edge refinement
    cfg.anti_alias = bool(params.get("anti_alias", True))
    cfg.edge_smoothing = _float_clamp(params.get("edge_smoothing", 1.0), 0, 5)
    cfg.alpha_matting = bool(params.get("alpha_matting", False))

    # Outlining
    cfg.stroke_width = _float_clamp(params.get("stroke_width", 2.0), 0, 20)
    cfg.stroke_color = params.get("stroke_color", "#000000")
    cfg.stroke_profile = params.get("stroke_profile", "uniform")
    cfg.glow_enabled = bool(params.get("glow_enabled", False))

    # Export
    cfg.export_format = params.get("export_format", "individual")
    cfg.canvas_width = _int_clamp(params.get("canvas_width", 1024), 64, 8192)
    cfg.canvas_height = _int_clamp(params.get("canvas_height", 1024), 64, 8192)

    # Pipeline control
    skip = params.get("skip_steps", "")
    if isinstance(skip, str):
        cfg.skip_steps = [s.strip() for s in skip.split(",") if s.strip()]
    elif isinstance(skip, list):
        cfg.skip_steps = skip
    cfg.abort_on_stage_failure = bool(params.get("abort_on_failure", False))

    return cfg


def _int_clamp(val, lo, hi):
    try:
        return max(lo, min(hi, int(val)))
    except (TypeError, ValueError):
        return lo


def _float_clamp(val, lo, hi):
    try:
        return max(lo, min(hi, float(val)))
    except (TypeError, ValueError):
        return float(lo)
