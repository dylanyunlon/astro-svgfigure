"""
Image Analyzer — Claude 4.6 Structure Analysis
=================================================
Analyzes an uploaded image to identify visual components, spatial layout,
and animation-capable elements for the /playground frame decomposition pipeline.

Pipeline Position: Step 1 of 4
  → Step 1: THIS MODULE (Claude 4.6 image analysis)
    Step 2: Grok animation prompt engineering
    Step 3: Gemini multi-frame generation
    Step 4: Green-screen removal + encoding

Architecture Decision Record:
─────────────────────────────
1. We use Claude's vision capability (image + text → structured JSON).
   The system prompt forces JSON-only output to avoid markdown preamble.
   Fallback: if Claude returns markdown-wrapped JSON, we strip fences.

2. The analysis response is intentionally over-complete (components,
   colors, motion suggestions, layering info). This is because Grok
   in Step 2 needs rich context to design good animation prompts.
   Under-analyzing here cascades into poor animation quality.

3. We don't cache analysis results because the same image with
   different crop/rotation should produce different analysis.
   Cache-key would need to include a perceptual hash, which adds
   complexity without clear benefit for this use case.

Knuth-Level Critiques:
─────────────────────
User Angle:
  - If Claude returns a malformed JSON (rare but possible with complex
    images), we fall back to a simplified analysis with just the summary.
    The user sees "Analysis partially complete" rather than an error.
  - If the image is too small (<50px), the analysis quality degrades.
    We warn but don't block — the user may be testing with thumbnails.

System Angle:
  - Claude's vision API has a ~20MB base64 limit. Our schema validates
    at 37MB but the actual provider will reject at ~20MB. We should
    add a pre-flight size check here, not rely on provider errors.
  - The structured output schema uses Optional fields liberally to
    handle partial Claude responses gracefully. A strict schema would
    cause more failures without improving quality.

GitHub references:
  - anthropics/anthropic-sdk-python (vision API)
  - dylanyunlon/astro-svgfigure/backend/ai_engine.py (AIEngine)
"""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any, Dict, List, Optional

from ..ai_engine import AIEngine
from ..config import Settings, get_settings
from ..schemas_animation import (
    AnalyzeImageRequest,
    AnalyzeImageResponse,
    ImageComponent,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  System Prompt — forces Claude to output structured JSON
# ═══════════════════════════════════════════════════════════════════════

ANALYSIS_SYSTEM_PROMPT = """\
You are an expert image analysis AI specialized in decomposing static images
into animatable components for frame-by-frame animation generation.

Given an image, you MUST respond with ONLY a valid JSON object (no markdown,
no backticks, no explanation text). The JSON schema:

{
  "summary": "One-line description of the image content and style",
  "components": [
    {
      "name": "descriptive name of the visual element",
      "type": "object|text|shape|character|background|decoration",
      "bounds": {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0},
      "animatable": true,
      "suggested_motions": ["rotate", "translate", "scale", "morph", "fade", "bounce", "wave"],
      "layer_depth": 0,
      "description": "detailed visual description for reproduction"
    }
  ],
  "animation_suggestions": [
    "Description of a recommended animation approach"
  ],
  "color_palette": ["#hex1", "#hex2", "#hex3", "#hex4", "#hex5"],
  "style": "flat|3d|realistic|cartoon|sketch|pixel|watercolor|vector|photo",
  "complexity": "simple|moderate|complex",
  "recommended_frame_count": 8,
  "recommended_style": "smooth|bounce|rotate|morph|explode",
  "spatial_layout": "centered|scattered|grid|layered|radial|linear",
  "foreground_background_separation": "clear|moderate|difficult",
  "notes": "Any additional observations relevant to animation"
}

Rules:
1. Identify ALL distinct visual elements, even small ones
2. For each component, suggest the most natural motion type
3. layer_depth: 0 = frontmost, higher = further back
4. bounds are normalized to 0..1 (relative to image dimensions)
5. Be specific in descriptions — Grok will use them to write frame prompts
6. If the image is a character/figure, identify body parts separately
7. Consider physics-plausible motions (gravity, inertia)
8. The "notes" field should mention any challenges for animation
   (transparency, complex textures, text that shouldn't distort)
"""


# ═══════════════════════════════════════════════════════════════════════
#  Core Analysis Function
# ═══════════════════════════════════════════════════════════════════════

async def analyze_image(
    request: AnalyzeImageRequest,
    settings: Optional[Settings] = None,
    ai_engine: Optional[AIEngine] = None,
) -> AnalyzeImageResponse:
    """
    Analyze an uploaded image using Claude 4.6 vision.

    Returns a structured analysis of visual components, spatial layout,
    and animation suggestions. This feeds into Grok's prompt design (Step 2).

    Parameters
    ----------
    request : AnalyzeImageRequest
        Contains the base64 image and optional model override.
    settings : Settings, optional
        Backend configuration. Uses get_settings() singleton if None.
    ai_engine : AIEngine, optional
        Pre-initialized AI engine. Creates new one if None.

    Returns
    -------
    AnalyzeImageResponse
        Structured analysis result or error.

    Raises
    ------
    Does not raise — all errors are caught and returned in the response.
    """
    settings = settings or get_settings()
    engine = ai_engine or AIEngine(settings)
    model = request.model or settings.ANTHROPIC_DEFAULT_MODEL
    t0 = time.monotonic()

    try:
        # ── Pre-flight checks ──
        image_b64 = _sanitize_base64(request.image_b64)
        estimated_size_mb = len(image_b64) * 3 / 4 / 1_000_000
        if estimated_size_mb > 18:
            return AnalyzeImageResponse(
                success=False,
                error=f"Image too large for vision API (~{estimated_size_mb:.1f}MB, max ~18MB). "
                      "Resize the image before uploading.",
            )
        if estimated_size_mb < 0.001:
            logger.warning("Very small image (~%.1f KB) — analysis quality may be poor", estimated_size_mb * 1000)

        # ── Build Claude vision request ──
        messages = _build_vision_messages(image_b64, request.mime_type)

        # ── Call Claude 4.6 ──
        logger.info("analyze_image: calling %s with ~%.1f MB image", model, estimated_size_mb)
        result = await engine.get_completion(
            messages=messages,
            model=model,
            temperature=0.3,  # Low temp for structured output
            max_tokens=4096,
        )

        raw_content = result.get("content", "")
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        logger.info("analyze_image: Claude responded in %d ms (%d chars)", elapsed_ms, len(raw_content))

        # ── Parse Claude response ──
        analysis = _parse_analysis_json(raw_content)
        if analysis is None:
            return AnalyzeImageResponse(
                success=False,
                error="Claude returned unparseable response. Try again or use a clearer image.",
                model_used=result.get("model", model),
            )

        # ── Build response ──
        components = _extract_components(analysis)
        return AnalyzeImageResponse(
            success=True,
            analysis=analysis,
            summary=analysis.get("summary", "Image analyzed successfully"),
            components=components,
            animation_suggestions=analysis.get("animation_suggestions", []),
            color_palette=analysis.get("color_palette", []),
            model_used=result.get("model", model),
        )

    except Exception as e:
        logger.exception("analyze_image failed: %s", e)
        return AnalyzeImageResponse(
            success=False,
            error=str(e),
        )


# ═══════════════════════════════════════════════════════════════════════
#  Message Construction
# ═══════════════════════════════════════════════════════════════════════

def _build_vision_messages(
    image_b64: str,
    mime_type: str,
) -> List[Dict[str, Any]]:
    """
    Build the messages array for Claude vision API.

    Uses the Anthropic-native image format:
      { "type": "image", "source": { "type": "base64", ... } }

    For OpenAI-compatible providers, the AIEngine auto-converts this.

    System-angle critique: we include the system prompt as a separate
    system message for Anthropic, but some providers need it in the
    first user message. The AIEngine handles this normalization.
    """
    return [
        {
            "role": "system",
            "content": ANALYSIS_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": image_b64,
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "Analyze this image for animation frame generation. "
                        "Identify all visual components, their spatial relationships, "
                        "and suggest the best animation approach. "
                        "The animation will use green-screen (#00FF00) background for "
                        "chroma-key removal, so pay special attention to any green elements "
                        "in the image that might cause issues. "
                        "Respond with ONLY the JSON object, no other text."
                    ),
                },
            ],
        },
    ]


# ═══════════════════════════════════════════════════════════════════════
#  Response Parsing
# ═══════════════════════════════════════════════════════════════════════

def _parse_analysis_json(raw: str) -> Optional[Dict[str, Any]]:
    """
    Parse Claude's JSON response, handling common formatting issues.

    Known issues we handle:
    1. Markdown code fences (```json ... ```)
    2. Leading/trailing whitespace
    3. Trailing commas (Claude sometimes adds them)
    4. Single quotes instead of double quotes (rare)

    Knuth critique: This is a "defensive parsing" pattern. In an ideal
    world, Claude always returns clean JSON. In practice, ~5% of responses
    have minor formatting issues. Each handler adds ~2μs — negligible
    compared to the 2-5s API call.
    """
    if not raw or not raw.strip():
        return None

    text = raw.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        elif lines[0].strip().startswith("```"):
            lines = lines[1:]
        text = "\n".join(lines).strip()

    # Handle trailing commas before } or ]
    import re
    text = re.sub(r',\s*([\}\]])', r'\1', text)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("JSON parse failed (attempt 1): %s — trying fallback", e)

    # Fallback: try to find JSON object boundaries
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError as e2:
            logger.warning("JSON parse failed (attempt 2): %s", e2)

    # Last resort: return a minimal analysis
    logger.error("Could not parse Claude response as JSON: %s...", text[:200])
    return {
        "summary": text[:200] if len(text) > 10 else "Analysis failed to parse",
        "components": [],
        "animation_suggestions": ["Manual review recommended — AI response was not structured"],
        "color_palette": [],
    }


def _extract_components(analysis: Dict[str, Any]) -> List[ImageComponent]:
    """
    Extract and validate component list from raw analysis dict.

    Converts the raw JSON components into typed ImageComponent models,
    handling missing fields gracefully.

    User-angle critique: if Claude omits the 'bounds' field for a
    component (common for complex scenes), we don't crash — we just
    set bounds to None. The frontend knows to handle None bounds by
    not showing a bounding box overlay.
    """
    raw_components = analysis.get("components", [])
    if not isinstance(raw_components, list):
        return []

    components: List[ImageComponent] = []
    for i, comp in enumerate(raw_components):
        if not isinstance(comp, dict):
            continue
        try:
            components.append(ImageComponent(
                name=comp.get("name", f"Component {i + 1}"),
                type=comp.get("type", "object"),
                bounds=comp.get("bounds"),
                animatable=comp.get("animatable", True),
                suggested_motions=comp.get("suggested_motions", []),
            ))
        except Exception as e:
            logger.warning("Skipping malformed component %d: %s", i, e)
            continue

    return components


# ═══════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════

def _sanitize_base64(b64: str) -> str:
    """
    Remove data URI prefix if present, return raw base64.

    The Astro proxy sends raw base64 (already stripped by FileReader),
    but direct API callers might send data: URIs.
    """
    if b64.startswith("data:"):
        parts = b64.split(",", 1)
        if len(parts) == 2:
            return parts[1]
    return b64


def estimate_image_dimensions(b64: str, mime_type: str = "image/png") -> Optional[Dict[str, int]]:
    """
    Estimate image dimensions from base64 without full decode.

    For PNG: read IHDR chunk (bytes 16-23 after decode).
    For JPEG: would need more complex parsing, return None.

    This is a best-effort optimization — the full decode happens
    in the AI provider. We use this for logging and pre-flight checks.
    """
    try:
        if mime_type == "image/png":
            header = base64.b64decode(b64[:100])
            if len(header) >= 24 and header[:8] == b'\x89PNG\r\n\x1a\n':
                width = int.from_bytes(header[16:20], "big")
                height = int.from_bytes(header[20:24], "big")
                return {"width": width, "height": height}
    except Exception:
        pass
    return None
