"""
astro-svgfigure Backend Package
================================
AI Engine + Pipeline for the forward SVG generation workflow.

Architecture (modeled after skynetCheapBuy/app/core/ai_engine.py):
  AIProvider (abstract)
    ├── OpenAIProvider      — gpt-*, o1-*, o3-* via openai SDK
    ├── AnthropicProvider   — claude-* via anthropic SDK
    ├── GoogleProvider      — gemini-* via google-generativeai SDK
    └── ClaudeCompatibleProvider — claude-* via /v1/messages (third-party)

  AIEngine
    └── _get_provider(model) → auto-routes to correct provider

  Pipeline
    ├── topology_gen       — LLM → ELK topology JSON
    ├── nanobanana_bridge  — layouted JSON → NanoBanana SVG
    ├── scaffold_builder   — ELK layout → JSON scaffold
    ├── svg_validator      — lxml validation + LLM fix
    ├── svg_scaler         — SVG coordinate scaling
    └── gemini_image_gen   — SVG → Gemini 3 Pro Image (scientific figure)

GitHub references:
  - dylanyunlon/skynetCheapBuy (AI Engine pattern)
  - kieler/elkjs (layout engine)
  - gemini-cli-extensions/nanobanana (SVG generation)
"""

__version__ = "0.1.0"
__all__ = ["ai_engine", "config", "schemas", "pipeline"]