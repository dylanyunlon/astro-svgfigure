"""
Animation Prompt Designer — Grok Frame Decomposition (Reference-Preserving)
=============================================================================
Engineers animation frame prompts using Grok, converting image analysis
results into precise per-frame EDITING instructions (not generation descriptions).

Pipeline Position: Step 2 of 4
    Step 1: Claude 4.6 image analysis
  → Step 2: THIS MODULE (Grok prompt engineering with IMAGE)
    Step 3: Gemini multi-frame generation (IMAGE EDITING mode)
    Step 4: Green-screen removal + encoding

CRITICAL FIX (2024):
───────────────────
The previous implementation sent ONLY TEXT to Grok. Grok never saw the
actual uploaded image, so it designed prompts for an image it had never seen.

User complaint: "最终效果与原始图片完全无联系"
Root cause: Grok designed prompts based on text description, not the image itself.

SOLUTION: Send BOTH the image AND the analysis to Grok.
Grok now generates EDIT INSTRUCTIONS ("move this 10% left") instead of
generation descriptions ("a magnificent car streaking with kinetic energy").

Per Google's documentation:
"Using the provided image, change only the [specific element]"

Knuth-Level Critiques:
─────────────────────
User Angle:
  - Users wanted LITERAL motion, not "creative interpretation"
  - "让车子左右移动" → "Move the car left and right", NOT "dynamic energy"

System Angle:
  - Grok now receives multimodal input (image + text)
  - Output is edit instructions, not generation prompts
  - Green screen specification is EXACT: #00FF00, no gradients

GitHub references:
  - openai/openai-python (Grok uses OpenAI-compatible API)
  - dylanyunlon/astro-svgfigure/backend/ai_engine.py
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from ..ai_engine import AIEngine
from ..config import Settings, get_settings
from ..schemas_animation import (
    AnimatePromptRequest,
    AnimatePromptResponse,
    AnimationStyle,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════

# Exact green screen specification
GREEN_SCREEN_HEX = "#00FF00"

GREEN_SCREEN_INSTRUCTION = f"""
BACKGROUND REQUIREMENT (CRITICAL):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The ENTIRE background MUST be solid {GREEN_SCREEN_HEX} green.
- NO gradients (must be perfectly flat single color)
- NO shadows on the background
- NO textures or patterns on the background
- Subject edges must be SHARP and CLEAN against the green
- NO green spill on subject edges
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

MAX_PROMPT_CHARS = 8000  # ~2000 tokens for Gemini input budget


# ═══════════════════════════════════════════════════════════════════════
#  System Prompt for Grok
# ═══════════════════════════════════════════════════════════════════════

GROK_SYSTEM_PROMPT = f"""\
You are an expert animation director. Your job is to design frame-by-frame
EDIT INSTRUCTIONS that will be sent to Gemini to MODIFY the provided image.

CRITICAL: The user has uploaded an image. Gemini will EDIT this image, not
generate a new one. Your instructions must describe HOW TO MODIFY the existing
image, not describe what to generate.

INPUT: You receive:
1. The ACTUAL SOURCE IMAGE (you can see it)
2. An analysis of the image (components, layout, colors, style)
3. The desired animation style and frame count

OUTPUT: You MUST respond with ONLY a valid JSON object:
{{
  "animation_prompt": "The complete EDIT instruction for all frames",
  "frame_descriptions": [
    "Frame 1: EDIT: [describe what to change from the original]",
    "Frame 2: EDIT: [describe what to change from frame 1]",
    ...
  ],
  "motion_summary": "Brief description of the overall motion",
  "identity_lock": "List of things that MUST NOT change"
}}

RULES:
1. {GREEN_SCREEN_INSTRUCTION}
2. Each frame describes an EDIT to the original image, not a new generation
3. Use phrases like "Move the subject 10% left" not "A car moving left"
4. Use phrases like "Rotate 45 degrees" not "The subject at a 45 degree angle"
5. The subject's visual identity MUST remain identical across all frames
6. NEVER use creative descriptions like "dynamic", "magnificent", "kinetic energy"
7. Be LITERAL: "Move 15% right" is better than "slide gracefully to the right"
8. Include exact percentages and angles where possible
9. The animation should loop seamlessly (frame N transitions back to frame 1)
10. Never change colors, proportions, or style - only position/pose

ANIMATION STYLE GUIDE (use LITERAL motion descriptions):
- smooth/horizontal: "Move X% left/right"
- vertical: "Move X% up/down"
- bounce: "Squash/stretch by X%"
- rotate: "Rotate X degrees clockwise/counterclockwise"
- scale/pulse: "Scale up/down by X%"
- wave: "Tilt X degrees left/right"
- walk: "Walk cycle phase X: [description]"
- explode: "Components separated by X% from center"
"""


# ═══════════════════════════════════════════════════════════════════════
#  Core Prompt Design Function
# ═══════════════════════════════════════════════════════════════════════

async def design_animation_prompt(
    request: AnimatePromptRequest,
    settings: Optional[Settings] = None,
    ai_engine: Optional[AIEngine] = None,
) -> AnimatePromptResponse:
    """
    Use Grok to design animation frame prompts from image analysis.

    The prompt includes mandatory green-screen background instructions
    and per-frame descriptions for consistent multi-frame generation.

    Parameters
    ----------
    request : AnimatePromptRequest
        Image analysis + animation parameters.
    settings : Settings, optional
        Backend configuration.
    ai_engine : AIEngine, optional
        Pre-initialized AI engine.

    Returns
    -------
    AnimatePromptResponse
        Engineered prompt or error.
    """
    settings = settings or get_settings()
    engine = ai_engine or AIEngine(settings)
    model = request.model or settings.DEFAULT_PROMPT_MODEL
    t0 = time.monotonic()

    try:
        # ── Build Grok request ──
        messages = _build_grok_messages(
            analysis=request.analysis,
            image_b64=request.image_b64,
            frame_count=request.frame_count,
            style=request.animation_style,
        )

        logger.info(
            "design_animation_prompt: calling %s for %d frames, style=%s",
            model, request.frame_count, request.animation_style,
        )

        # ── Call Grok ──
        result = await engine.get_completion(
            messages=messages,
            model=model,
            temperature=0.7,  # Creative but controlled
            max_tokens=4096,
        )

        raw_content = result.get("content", "")
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        logger.info("design_animation_prompt: Grok responded in %d ms", elapsed_ms)

        # ── Parse response ──
        parsed = _parse_grok_response(raw_content)
        if parsed is None:
            # Fallback to template-based prompt
            logger.warning("Grok response unparseable, using fallback template")
            parsed = _generate_fallback_prompt(
                request.analysis, request.frame_count, request.animation_style,
            )

        # ── Enforce green-screen instruction ──
        prompt = parsed.get("animation_prompt", "")
        prompt = _enforce_green_screen(prompt)

        # ── Truncate if needed ──
        if len(prompt) > MAX_PROMPT_CHARS:
            prompt = _truncate_prompt(prompt, MAX_PROMPT_CHARS)

        frame_descriptions = parsed.get("frame_descriptions", [])

        return AnimatePromptResponse(
            success=True,
            prompt=prompt,
            frame_descriptions=frame_descriptions,
            model_used=result.get("model", model),
        )

    except Exception as e:
        logger.exception("design_animation_prompt failed: %s", e)

        # Fallback: generate a basic prompt from the analysis
        try:
            fallback = _generate_fallback_prompt(
                request.analysis, request.frame_count, request.animation_style,
            )
            prompt = _enforce_green_screen(fallback.get("animation_prompt", ""))
            return AnimatePromptResponse(
                success=True,
                prompt=prompt,
                frame_descriptions=fallback.get("frame_descriptions", []),
                model_used="fallback-template",
            )
        except Exception as e2:
            logger.exception("Fallback prompt generation also failed: %s", e2)
            return AnimatePromptResponse(
                success=False,
                error=f"Grok error: {e}. Fallback also failed: {e2}",
            )


# ═══════════════════════════════════════════════════════════════════════
#  Message Construction (NOW INCLUDES IMAGE)
# ═══════════════════════════════════════════════════════════════════════

def _build_grok_messages(
    analysis: Dict[str, Any],
    image_b64: str,
    frame_count: int,
    style: AnimationStyle,
) -> List[Dict[str, Any]]:
    """
    Build the messages array for Grok API call.
    
    CRITICAL FIX: Now includes the actual image so Grok can see what
    it's designing animation for, not just a text description.
    """

    # Extract key info from analysis for the user message
    summary = analysis.get("summary", "Unknown image")
    components = analysis.get("components", [])
    colors = analysis.get("color_palette", [])
    img_style = analysis.get("style", "unknown")
    suggestions = analysis.get("animation_suggestions", [])

    component_text = ""
    if components:
        comp_lines = []
        for c in components[:15]:  # Cap at 15 components
            name = c.get("name", "?") if isinstance(c, dict) else str(c)
            motions = c.get("suggested_motions", []) if isinstance(c, dict) else []
            comp_lines.append(f"  - {name}" + (f" (motions: {', '.join(motions)})" if motions else ""))
        component_text = "Detected components:\n" + "\n".join(comp_lines)

    color_text = f"Color palette: {', '.join(colors)}" if colors else ""
    suggestion_text = ""
    if suggestions:
        suggestion_text = "Animation suggestions from analysis:\n" + "\n".join(f"  - {s}" for s in suggestions[:5])

    user_text = f"""\
Look at the provided image and design {frame_count} EDIT INSTRUCTIONS
for a "{style.value}" animation.

Image analysis summary: {summary}
Visual style: {img_style}
{component_text}
{color_text}
{suggestion_text}

CRITICAL REQUIREMENTS:
1. Generate EXACTLY {frame_count} frame EDIT instructions
2. Each frame describes HOW TO MODIFY the original image
3. Use LITERAL language: "Move 10% left" not "glide gracefully"
4. Background MUST be solid {GREEN_SCREEN_HEX} for chroma-key
5. The subject's visual identity MUST stay IDENTICAL
6. Only position/pose should change between frames
7. Animation should loop seamlessly

DO NOT use creative language like "dynamic", "magnificent", "kinetic".
DO use literal instructions like "Move", "Rotate", "Scale", "Tilt".

Respond with ONLY the JSON object as specified in your instructions.
"""

    # Build multimodal message with image
    # Clean base64
    clean_b64 = image_b64
    if clean_b64.startswith("data:"):
        clean_b64 = clean_b64.split(",", 1)[-1]

    messages = [
        {"role": "system", "content": GROK_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                # Include the actual image so Grok can see it
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": clean_b64,
                    },
                },
                # The text instructions
                {
                    "type": "text",
                    "text": user_text.strip(),
                },
            ],
        },
    ]

    return messages


# ═══════════════════════════════════════════════════════════════════════
#  Response Parsing
# ═══════════════════════════════════════════════════════════════════════

def _parse_grok_response(raw: str) -> Optional[Dict[str, Any]]:
    """Parse Grok's JSON response with fallback strategies."""
    if not raw or not raw.strip():
        return None

    text = raw.strip()

    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        elif lines[0].strip().startswith("```"):
            lines = lines[1:]
        text = "\n".join(lines).strip()

    # Remove trailing commas
    text = re.sub(r',\s*([\}\]])', r'\1', text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from mixed content
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    # If we can't parse JSON, try to extract the prompt as plain text
    if len(text) > 50:
        return {
            "animation_prompt": text,
            "frame_descriptions": [],
            "motion_summary": "Extracted from plain text response",
        }

    return None


# ═══════════════════════════════════════════════════════════════════════
#  Green-Screen Enforcement
# ═══════════════════════════════════════════════════════════════════════

def _enforce_green_screen(prompt: str) -> str:
    """
    Ensure the green-screen instruction is present in the prompt.

    This is the CRITICAL safety net. Even if Grok forgot the green-screen
    instruction, or the user edited it out, we re-inject it.

    We check for keywords rather than exact string match to handle
    paraphrased versions of the instruction.
    """
    lower = prompt.lower()
    has_green_ref = any(kw in lower for kw in [
        "#00ff00", "green screen", "green background", "chroma key",
        "chromakey", "bright green", "solid green",
    ])

    if not has_green_ref:
        logger.info("Green-screen instruction missing from prompt — injecting")
        prompt = f"{GREEN_SCREEN_INSTRUCTION}\n\n{prompt}"

    return prompt


def _truncate_prompt(prompt: str, max_chars: int) -> str:
    """
    Truncate prompt while preserving structure.

    Strategy: keep the first 40% and last 30% of the prompt,
    replace the middle with a summary note. This preserves:
    - The green-screen instruction (usually at the start)
    - The first frame description (establishes the visual)
    - The last frame description (ensures loop closure)
    """
    if len(prompt) <= max_chars:
        return prompt

    head_size = int(max_chars * 0.4)
    tail_size = int(max_chars * 0.3)
    bridge = "\n\n[... middle frames follow the same pattern with gradual progression ...]\n\n"

    return prompt[:head_size] + bridge + prompt[-tail_size:]


# ═══════════════════════════════════════════════════════════════════════
#  Fallback Template Generator
# ═══════════════════════════════════════════════════════════════════════

def _generate_fallback_prompt(
    analysis: Dict[str, Any],
    frame_count: int,
    style: AnimationStyle,
) -> Dict[str, Any]:
    """
    Generate a template-based animation prompt when Grok fails.

    This produces functional but less creative prompts. The quality
    is significantly lower than Grok's output, but it allows the
    pipeline to continue rather than failing entirely.
    """
    summary = analysis.get("summary", "the subject")
    components = analysis.get("components", [])
    colors = analysis.get("color_palette", ["#333333", "#666666", "#999999"])

    # Build a basic subject description
    if components:
        comp_names = [
            c.get("name", "element") if isinstance(c, dict) else str(c)
            for c in components[:5]
        ]
        subject_desc = ", ".join(comp_names)
    else:
        subject_desc = summary

    # Style-specific motion descriptions
    motion_templates = {
        AnimationStyle.SMOOTH: {
            "motion": "gentle, continuous movement",
            "per_frame": lambda i, n: f"smooth position shift {(i / (n - 1)) * 100:.0f}% through the motion cycle",
        },
        AnimationStyle.BOUNCE: {
            "motion": "elastic bouncing with squash and stretch",
            "per_frame": lambda i, n: f"{'squash' if i % 2 == 0 else 'stretch'} phase at {(i / (n - 1)) * 100:.0f}%",
        },
        AnimationStyle.ROTATE: {
            "motion": "smooth rotation around the center axis",
            "per_frame": lambda i, n: f"rotated {(i / n) * 360:.0f}° from the starting position",
        },
        AnimationStyle.MORPH: {
            "motion": "gradual shape transformation",
            "per_frame": lambda i, n: f"morph state at {(i / (n - 1)) * 100:.0f}% between start and end forms",
        },
        AnimationStyle.EXPLODE: {
            "motion": "parts flying outward then reassembling",
            "per_frame": lambda i, n: (
                f"parts {'separating' if i < n // 2 else 'reassembling'} — "
                f"{'explosion' if i < n // 2 else 'convergence'} at {abs(i - n // 2) / (n // 2) * 100:.0f}%"
            ),
        },
        AnimationStyle.WALK: {
            "motion": "walking cycle with alternating steps",
            "per_frame": lambda i, n: f"walk cycle phase {i + 1}/{n}: {'left' if i % 2 == 0 else 'right'} foot forward",
        },
        AnimationStyle.WAVE: {
            "motion": "oscillating wave motion",
            "per_frame": lambda i, n: f"wave position: {'up' if i % 4 < 2 else 'down'} phase at frame {i + 1}",
        },
        AnimationStyle.PULSE: {
            "motion": "rhythmic breathing scale animation",
            "per_frame": lambda i, n: f"{'expanding' if i % 4 < 2 else 'contracting'} at {(i / (n - 1)) * 100:.0f}%",
        },
    }

    template = motion_templates.get(style, motion_templates[AnimationStyle.SMOOTH])

    # Build the prompt
    frame_descs = []
    for i in range(frame_count):
        frame_motion = template["per_frame"](i, frame_count)
        frame_descs.append(
            f"Frame {i + 1}: {subject_desc} — {frame_motion}. "
            f"Solid bright green (#00FF00) background. "
            f"Clean sharp edges, no shadows on background."
        )

    animation_prompt = (
        f"{GREEN_SCREEN_INSTRUCTION}\n\n"
        f"Generate a {frame_count}-frame animation sequence of {subject_desc}.\n"
        f"Animation type: {template['motion']}.\n"
        f"Visual style: Match the source image exactly.\n"
        f"Colors: {', '.join(colors[:5])}\n\n"
        f"Frame descriptions:\n" + "\n".join(frame_descs)
    )

    return {
        "animation_prompt": animation_prompt,
        "frame_descriptions": frame_descs,
        "motion_summary": f"{style.value} animation of {summary}",
        "technical_notes": "Generated by fallback template — Grok was unavailable",
    }
