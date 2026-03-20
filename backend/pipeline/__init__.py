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
  text_inpainter     — Step 5b: Remove dark text pixels from generated images
  prompt_compressor  — Utility: (DISABLED) was compressing prompts, now bypassed

Pipeline flow:
  text → [topology_gen] → topology.json
       → [ELK.js layout] → layouted.json  (done in frontend/TS)
       → [scaffold_builder] → scaffold.json
       → [nanobanana_bridge] → raw.svg
       → [svg_validator] → validated.svg
       → [svg_scaler] → final.svg
       → [gemini_image_gen] → scientific_figure.png
         (Grok's FULL output + skeleton PNG sent to Gemini, NO compression)
"""

from .topology_gen import generate_topology
from .nanobanana_bridge import beautify_with_nanobanana
from .scaffold_builder import build_scaffold
from .svg_validator import validate_svg
from .svg_scaler import scale_svg
from .gemini_image_gen import generate_scientific_figure, generate_image_with_gemini
from .text_inpainter import inpaint_dark_text, process_gemini_result

__all__ = [
    "generate_topology",
    "beautify_with_nanobanana",
    "build_scaffold",
    "validate_svg",
    "scale_svg",
    "generate_scientific_figure",
    "generate_image_with_gemini",
    "inpaint_dark_text",
    "process_gemini_result",
]
