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

Animation Pipeline (v2 - Reference Preserving):
  image → [reference_image_validator] → validation
        → [visual_identity_extractor] → identity profile
        → [animation_prompt_composer] → frame prompts (Grok + image)
        → [animation_frame_synthesizer / reference_preserving_frame_generator] → frames
        → [visual_consistency_enforcer] → consistency check
        → [animation_quality_validator] → quality check
        → [enhanced_green_remover] → transparent PNGs
        → [animation_output_encoder] → GIF/WebP/APNG
"""

from .topology_gen import generate_topology
from .nanobanana_bridge import beautify_with_nanobanana
from .scaffold_builder import build_scaffold
from .svg_validator import validate_svg
from .svg_scaler import scale_svg
from .gemini_image_gen import generate_scientific_figure, generate_image_with_gemini

# Animation pipeline modules (Legacy - Playground page)
from .image_analyzer import analyze_image
from .animation_prompt_designer import design_animation_prompt
from .frame_generator import generate_animation_frames

# Animation pipeline v2 modules (Reference-preserving)
from .animation_pipeline_orchestrator import AnimationPipelineOrchestrator
from .animation_prompt_composer import AnimationPromptComposer
from .animation_frame_synthesizer import AnimationFrameSynthesizer
from .reference_preserving_frame_generator import ReferencePreservingFrameGenerator
from .visual_identity_extractor import VisualIdentityExtractor
from .animation_quality_validator import AnimationQualityValidator
from .visual_consistency_enforcer import VisualConsistencyEnforcer
from .enhanced_green_remover import EnhancedGreenRemover
from .reference_image_validator import ReferenceImageValidator
from .animation_output_encoder import AnimationOutputEncoder
from .gemini_image_editor import GeminiImageEditor

# Animation pipeline v3 modules (Critique-based fix)
from .playground_animation_critique_and_solution import (
    PlaygroundAnimator,
    PlaygroundAnimationType,
    VisualIdentityLock as V3IdentityLock,
    EditInstructionGenerator,
)
from .playground_vs_generate_architecture import (
    generate_playground_animation,
    VisualIdentity,
    MotionInstructionGenerator,
    AnimationType,
    AnimationGenerationResult,
)

__all__ = [
    # SVG generation pipeline
    "generate_topology",
    "beautify_with_nanobanana",
    "build_scaffold",
    "validate_svg",
    "scale_svg",
    "generate_scientific_figure",
    "generate_image_with_gemini",
    # Animation pipeline (legacy)
    "analyze_image",
    "design_animation_prompt",
    "generate_animation_frames",
    # Animation pipeline v2 (reference-preserving)
    "AnimationPipelineOrchestrator",
    "AnimationPromptComposer",
    "AnimationFrameSynthesizer",
    "ReferencePreservingFrameGenerator",
    "VisualIdentityExtractor",
    "AnimationQualityValidator",
    "VisualConsistencyEnforcer",
    "EnhancedGreenRemover",
    "ReferenceImageValidator",
    "AnimationOutputEncoder",
    "GeminiImageEditor",
    # Animation pipeline v3 (critique-based fix)
    "PlaygroundAnimator",
    "PlaygroundAnimationType",
    "V3IdentityLock",
    "EditInstructionGenerator",
    "generate_playground_animation",
    "VisualIdentity",
    "MotionInstructionGenerator",
    "AnimationType",
    "AnimationGenerationResult",
]