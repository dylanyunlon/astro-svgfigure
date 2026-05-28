"""
Pipeline Sub-package
====================
Forward SVG generation pipeline modules:

  topology_gen       — Step 1: LLM → ELK topology JSON
  scaffold_builder   — Step 2→3: ELK layouted → NanoBanana JSON scaffold
  nanobanana_bridge  — Step 3: scaffold → Gemini NanoBanana SVG
  svg_validator      — Step 4: lxml validation + LLM auto-fix
  svg_scaler         — Utility: SVG coordinate scaling
  gemini_image_gen   — Step 5: SVG → Gemini 3 Pro Image (scientific figure)

Pipeline flow:
  text → [topology_gen] → topology.json
       → [ELK.js layout] → layouted.json  (done in frontend/TS)
       → [scaffold_builder] → scaffold.json
       → [nanobanana_bridge] → raw.svg
       → [svg_validator] → validated.svg
       → [svg_scaler] → final.svg
       → [gemini_image_gen] → scientific_figure.png
"""

from .topology_gen import generate_topology
from .nanobanana_bridge import beautify_with_nanobanana
from .scaffold_builder import build_scaffold
from .svg_validator import validate_svg
from .svg_scaler import scale_svg
from .gemini_image_gen import generate_scientific_figure, generate_image_with_gemini

# Animation pipeline modules (Playground page)
from .image_analyzer import analyze_image
from .animation_prompt_designer import design_animation_prompt
from .frame_generator import generate_animation_frames


# Cloud background removal (Step 4) — remove.bg (Canva) API only
try:
    from .removebg_canva_client import (
        RemoveBgCanvaClient,
        RemoveBgCanvaKeyPool,
    )
    from .batch_rembg_orchestrator import (
        orchestrate_batch_removal,
        orchestrate_batch_removal_b64,
        config_from_params,
    )
except ImportError:
    pass  # httpx may not be installed in all environments

# Layer separation (Step 5)
try:
    from .layer_separator import separate_layers, separate_layers_batch
except ImportError:
    pass

# Edge refinement & outlining (Step 6)
try:
    from .edge_refiner import refine_layer_edges, refine_layers_batch
    from .component_outliner import (
        outline_component,
        outline_components_batch,
        build_svg_document,
    )
except ImportError:
    pass

# Transparency validation (QA)
try:
    from .transparency_validator import validate_frame, validate_batch, validate_batch_b64
except ImportError:
    pass

# Row-scan pixel engine (M001+M002: OpenCV HSV lookup + thrust::transform pipeline)
try:
    from .rowscan_engine import (
        HSVLookupTable,
        PixelTransformPipeline,
        classify_green_rowscan,
        despill_green_rowscan,
        benchmark_rowscan,
    )
except ImportError:
    pass



__all__ = [
    "generate_topology",
    "beautify_with_nanobanana",
    "build_scaffold",
    "validate_svg",
    "scale_svg",
    "generate_scientific_figure",
    "generate_image_with_gemini",
    "analyze_image",
    "design_animation_prompt",
    "generate_animation_frames",
    # Advanced pipeline modules
    "process_frames_hsv",
    "GreenScreenConfig",
    "get_grok_green_screen_requirements",
    "orchestrate_batch_removal",
    "orchestrate_batch_removal_b64",
    "config_from_params",
    "separate_layers",
    "separate_layers_batch",
    "refine_layer_edges",
    "refine_layers_batch",
    "outline_component",
    "outline_components_batch",
    "build_svg_document",
    "validate_frame",
    "validate_batch",
    "validate_batch_b64",
    # M001+M002: Row-scan engine
    "HSVLookupTable",
    "PixelTransformPipeline",
    "classify_green_rowscan",
    "despill_green_rowscan",
    "benchmark_rowscan",
    # remove-bg.io cloud API
    "RemoveBgIoClient",
    "RemoveBgIoConfig",
    "is_removebgio_available",
    "process_frame_removebgio",
]