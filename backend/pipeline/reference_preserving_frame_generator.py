"""
Reference-Preserving Frame Generator — Multi-Turn Conversational Animation
=============================================================================
This module implements the CORRECT approach for animating an uploaded image
while preserving its visual identity.

CORE INSIGHT (from Google's official documentation):
─────────────────────────────────────────────────────
Per blog.google/products/gemini/image-generation-prompting-tips:
"With updated image editing capabilities, you can make quick, highly precise
edits to your photos... Using direct, conversational commands, you can modify
specific elements within the image without needing complex software."

Per developers.googleblog.com/how-to-prompt-gemini-2-5-flash-image:
"Using the provided image, change only the [specific element] to [new element].
Keep everything else in the image exactly the same, preserving the original
style, lighting, and composition."

Per sider.ai/blog (How to Write Gemini Prompts That Keep Subject Identity):
"Start every prompt with a compact identity header. This reduces drift across
iterative edits."

THE PROBLEM WITH OUR OLD APPROACH:
─────────────────────────────────
Old: Generate 8 frames from text description → NEW images each time
New: Edit the SAME image 8 times → MODIFIED versions of the original

The key difference is:
- OLD: "Generate a car moving to the left" → Creates ANY car
- NEW: "Using the provided image, move the car 10% to the left" → EDITS the car

IMPLEMENTATION STRATEGY:
───────────────────────
1. IDENTITY HEADER
   Every prompt starts with an "identity lock" that describes the subject's
   key visual features that MUST NOT change.

2. MULTI-TURN CONVERSATION SIMULATION
   Each frame is generated as if we're having a conversation with Gemini:
   - Turn 1: "Here's my image" [original]
   - Turn 2: "Now make this small change" [frame 1]
   - Turn 3: "Now continue the motion" [frame 2]
   ...

3. INCREMENTAL EDITS
   Instead of "generate frame 5 of 8", we say:
   "Using the provided image, apply this motion step..."

4. BACKGROUND INSTRUCTION
   Every frame includes: "Replace the background with solid green #00FF00"

CRITICAL TECHNICAL DETAILS:
──────────────────────────
1. THOUGHT SIGNATURES (from ai.google.dev/gemini-api/docs/image-generation):
   "Thought signatures are encrypted representations of the model's internal
   thought process and are used to preserve reasoning context across
   multi-turn interactions."

   We capture and pass thought_signature between frames for consistency.

2. IMAGE REFERENCE IN EVERY CALL:
   Per the documentation, we must include the original image in EVERY frame
   generation call. Gemini doesn't "remember" images from previous turns.

3. ASPECT RATIO PRESERVATION:
   "When editing, Gemini 2.5 Flash Image generally preserves the input image's
   aspect ratio. If it doesn't, be explicit in your prompt."

Knuth-Level Critique Resolution:
───────────────────────────────
USER CRITIQUE 1: "The animation has no relation to my original image"
SOLUTION: Every prompt explicitly references "the provided image" and
includes an identity header that locks the subject's visual features.

USER CRITIQUE 2: "The car changes colors/shape between frames"
SOLUTION: Identity header includes hard constraints like "Do not change
facial proportions, colors, or overall shape" and we pass the original
image as reference in EVERY frame generation.

SYSTEM CRITIQUE 1: "Grok didn't see the original image"
SOLUTION: This module doesn't rely on Grok's output. It extracts visual
features from Claude's analysis and constructs the identity header directly.

SYSTEM CRITIQUE 2: "The prompts were designed for sprite sheet generation"
SOLUTION: This module uses image EDITING prompts, not generation prompts.
The key difference: "edit the provided image" vs "generate an image of".

Pipeline Position: Alternative to animation_frame_synthesizer.py
    This module can be used when maximum fidelity to the original is needed.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

# Green screen specification
GREEN_HEX = "#00FF00"
GREEN_RGB = "RGB(0, 255, 0)"

# Maximum prompt length
MAX_PROMPT_LENGTH = 4000

# Frame generation timeout (seconds)
FRAME_TIMEOUT_S = 180

# Aspect ratio specifications
ASPECT_RATIOS = {
    "1:1": {"width": 1024, "height": 1024},
    "16:9": {"width": 1344, "height": 768},
    "9:16": {"width": 768, "height": 1344},
    "4:3": {"width": 1152, "height": 896},
    "3:4": {"width": 896, "height": 1152},
}


# ═══════════════════════════════════════════════════════════════════════════
#  Identity Header Template
# ═══════════════════════════════════════════════════════════════════════════

IDENTITY_HEADER_TEMPLATE = """
IDENTITY LOCK — DO NOT MODIFY THESE ATTRIBUTES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Subject: {subject_description}
Visual Style: {visual_style}

MUST PRESERVE:
{preserve_list}

HARD NEGATIVES (DO NOT DO THESE):
- Do NOT change the subject's overall shape or proportions
- Do NOT modify colors or textures
- Do NOT add new elements to the subject
- Do NOT change the viewing angle significantly
- Do NOT morph, deform, or distort the subject
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""".strip()


# ═══════════════════════════════════════════════════════════════════════════
#  Edit Instruction Template
# ═══════════════════════════════════════════════════════════════════════════

EDIT_INSTRUCTION_TEMPLATE = """
EDITING TASK — FRAME {frame_num} OF {total_frames}:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Using the PROVIDED IMAGE, apply the following edit:

{motion_instruction}

BACKGROUND REQUIREMENT:
Replace the ENTIRE background with solid bright green ({green_hex}).
The subject must have clean, sharp edges against the green background.
No shadows or gradients on the green — it must be perfectly flat.

ASPECT RATIO: {aspect_ratio}
Keep the output image at exactly this aspect ratio.
Do not change the input aspect ratio.

OUTPUT: Generate a single image showing this edit applied.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""".strip()


# ═══════════════════════════════════════════════════════════════════════════
#  Motion Instruction Templates per Animation Style
# ═══════════════════════════════════════════════════════════════════════════

class AnimationStyle(Enum):
    SMOOTH = "smooth"
    BOUNCE = "bounce"
    ROTATE = "rotate"
    MORPH = "morph"
    EXPLODE = "explode"
    WALK = "walk"
    WAVE = "wave"
    PULSE = "pulse"


def get_motion_instruction(
    style: AnimationStyle,
    frame_index: int,
    total_frames: int,
) -> str:
    """
    Generate a motion instruction for a specific frame and animation style.

    These instructions are phrased as EDITS to the provided image,
    not as generation prompts.
    """
    # Progress through the animation (0.0 to 1.0)
    t = frame_index / (total_frames - 1) if total_frames > 1 else 0

    if style == AnimationStyle.SMOOTH:
        # Linear translation
        offset_pct = int(t * 20 - 10)  # -10% to +10%
        direction = "right" if offset_pct >= 0 else "left"
        return (
            f"Move the subject {abs(offset_pct)}% to the {direction} of its current position. "
            f"This is frame {frame_index + 1} of a smooth linear motion sequence. "
            f"The subject should appear to glide smoothly in this direction."
        )

    elif style == AnimationStyle.BOUNCE:
        # Vertical bounce with squash/stretch
        phase = (frame_index % 4)
        if phase == 0:
            return (
                "The subject is at the TOP of a bounce. "
                "Stretch it slightly vertically (elongate by ~5%). "
                "Position it at the highest point of the bounce arc."
            )
        elif phase == 1:
            return (
                "The subject is FALLING downward. "
                "Return to normal proportions. "
                "Position it between the top and bottom of the bounce."
            )
        elif phase == 2:
            return (
                "The subject is at the BOTTOM of the bounce, making contact. "
                "Squash it slightly (compress vertically by ~10%, widen horizontally by ~5%). "
                "Position it at the lowest point."
            )
        else:
            return (
                "The subject is RISING upward from the bounce. "
                "Return to normal proportions. "
                "Position it between the bottom and top of the bounce."
            )

    elif style == AnimationStyle.ROTATE:
        # Rotation around center
        angle = int(t * 360)
        return (
            f"Rotate the subject {angle}° clockwise from its original orientation. "
            f"Maintain all visual details and proportions during the rotation. "
            f"The center of rotation is the subject's center point."
        )

    elif style == AnimationStyle.MORPH:
        # Subtle shape transformation
        morph_pct = int(abs(t - 0.5) * 2 * 100)  # 100% at edges, 0% at middle
        if t < 0.5:
            return (
                f"Subtly transform the subject's shape by {100 - morph_pct}% toward a more relaxed pose. "
                f"This is a gentle morphing effect — the identity must remain clear. "
                f"Do NOT change colors or major proportions."
            )
        else:
            return (
                f"Continue the subtle shape transformation, returning {morph_pct}% toward the original pose. "
                f"The subject is morphing back to its starting shape. "
                f"Maintain identity throughout."
            )

    elif style == AnimationStyle.EXPLODE:
        # Components separate and reassemble
        if t < 0.5:
            separation = int(t * 2 * 30)  # 0% to 30% separation
            return (
                f"Separate the visible components of the subject outward by {separation}% from center. "
                f"Each distinct part should move away from the center point. "
                f"Maintain each component's individual appearance."
            )
        else:
            separation = int((1 - t) * 2 * 30)  # 30% back to 0%
            return (
                f"The components are reassembling — they are now {separation}% separated. "
                f"Move each part back toward the center. "
                f"Components should be returning to their original positions."
            )

    elif style == AnimationStyle.WALK:
        # Walking cycle
        phase = frame_index % 4
        if phase == 0:
            return (
                "Walk cycle contact pose: The leading foot is just touching the ground. "
                "The body is at its lowest point in the stride. "
                "Arms are in opposite positions to the legs."
            )
        elif phase == 1:
            return (
                "Walk cycle passing pose: One leg passes the other. "
                "The body is at its highest point in the stride. "
                "Weight is balanced over the standing leg."
            )
        elif phase == 2:
            return (
                "Walk cycle contact pose (opposite side): The other foot now contacts the ground. "
                "Mirror of the first contact pose. "
                "Arms have swapped positions."
            )
        else:
            return (
                "Walk cycle passing pose (opposite side): Legs passing again. "
                "Completing the cycle, returning toward the starting position. "
                "Natural walking motion continuation."
            )

    elif style == AnimationStyle.WAVE:
        # Oscillating wave motion
        import math
        wave_angle = int(math.sin(t * 2 * math.pi) * 15)  # ±15 degrees
        direction = "right" if wave_angle >= 0 else "left"
        return (
            f"Tilt the subject {abs(wave_angle)}° to the {direction}. "
            f"This is a gentle oscillating wave motion. "
            f"The subject rocks back and forth while maintaining its identity."
        )

    elif style == AnimationStyle.PULSE:
        # Breathing/pulsing scale
        import math
        scale_change = int(math.sin(t * 2 * math.pi) * 8)  # ±8%
        if scale_change >= 0:
            return (
                f"Scale the subject UP by {scale_change}% uniformly. "
                f"This is a breathing/pulsing effect — the subject appears to expand. "
                f"Maintain center position and all proportions."
            )
        else:
            return (
                f"Scale the subject DOWN by {abs(scale_change)}% uniformly. "
                f"This is a breathing/pulsing effect — the subject appears to contract. "
                f"Maintain center position and all proportions."
            )

    # Default fallback
    return (
        f"Generate frame {frame_index + 1} of {total_frames} showing "
        f"the subject in a natural animation sequence."
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Identity Header Builder
# ═══════════════════════════════════════════════════════════════════════════

def build_identity_header(analysis: Dict[str, Any]) -> str:
    """
    Build an identity header from Claude's image analysis.

    This header "locks" the subject's visual identity and is included
    in every frame generation prompt.
    """
    # Extract subject description
    subject = analysis.get("summary", "the subject in the image")

    # Extract visual style
    style = analysis.get("style", "realistic")

    # Build preserve list
    preserve_items = []

    # Colors
    colors = analysis.get("color_palette", [])
    if colors:
        preserve_items.append(f"- Exact colors: {', '.join(colors[:5])}")

    # Components
    components = analysis.get("components", [])
    for comp in components[:5]:
        if isinstance(comp, dict):
            name = comp.get("name", "")
            desc = comp.get("description", "")
            if name:
                preserve_items.append(f"- {name}: {desc}" if desc else f"- {name}")
        elif isinstance(comp, str):
            preserve_items.append(f"- {comp}")

    # Style-specific preservations
    preserve_items.append(f"- Visual style: {style}")
    preserve_items.append("- Overall proportions and shape")
    preserve_items.append("- Texture and surface details")

    preserve_list = "\n".join(preserve_items) if preserve_items else "- All visual characteristics"

    return IDENTITY_HEADER_TEMPLATE.format(
        subject_description=subject,
        visual_style=style,
        preserve_list=preserve_list,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Frame Generation Result
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FrameResult:
    """Result of generating a single frame."""
    success: bool
    frame_b64: Optional[str] = None
    thought_signature: Optional[str] = None
    error: Optional[str] = None
    generation_time_ms: int = 0


@dataclass
class GenerationResult:
    """Result of generating all animation frames."""
    success: bool
    frames: List[str] = field(default_factory=list)
    error: Optional[str] = None
    total_time_ms: int = 0
    frame_times_ms: List[int] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
#  Main Frame Generation Function
# ═══════════════════════════════════════════════════════════════════════════

async def generate_reference_preserving_frames(
    image_b64: str,
    analysis: Dict[str, Any],
    frame_count: int,
    animation_style: AnimationStyle,
    aspect_ratio: str = "1:1",
    custom_motion: Optional[str] = None,
    api_caller: Optional[Callable] = None,
    on_frame_complete: Optional[Callable[[int, str], None]] = None,
) -> GenerationResult:
    """
    Generate animation frames that PRESERVE the original image's identity.

    This function implements the reference-preserving approach where each
    frame is generated as an EDIT of the original image, not as a new
    generation from text.

    Parameters
    ----------
    image_b64 : str
        Base64-encoded original image
    analysis : Dict[str, Any]
        Claude's analysis of the image (used for identity header)
    frame_count : int
        Number of frames to generate (2-16)
    animation_style : AnimationStyle
        Type of animation motion
    aspect_ratio : str
        Output aspect ratio (e.g., "1:1", "16:9")
    custom_motion : Optional[str]
        Custom motion description (overrides style-based motion)
    api_caller : Optional[Callable]
        Async function to call the Gemini API
    on_frame_complete : Optional[Callable]
        Callback when each frame completes (for progress updates)

    Returns
    -------
    GenerationResult
        Contains list of base64 frames or error information
    """
    t0 = time.monotonic()
    result = GenerationResult(success=False)

    # Validate inputs
    frame_count = max(2, min(frame_count, 16))

    # Build identity header (used in every prompt)
    identity_header = build_identity_header(analysis)

    # Clean base64
    image_b64 = _sanitize_base64(image_b64)

    # Generate frames sequentially
    frames: List[str] = []
    thought_signature: Optional[str] = None

    for frame_idx in range(frame_count):
        frame_t0 = time.monotonic()

        # Get motion instruction for this frame
        if custom_motion:
            motion_instruction = f"{custom_motion} (Frame {frame_idx + 1} of {frame_count})"
        else:
            motion_instruction = get_motion_instruction(
                animation_style, frame_idx, frame_count
            )

        # Build the edit instruction
        edit_instruction = EDIT_INSTRUCTION_TEMPLATE.format(
            frame_num=frame_idx + 1,
            total_frames=frame_count,
            motion_instruction=motion_instruction,
            green_hex=GREEN_HEX,
            aspect_ratio=aspect_ratio,
        )

        # Combine identity header + edit instruction
        full_prompt = f"{identity_header}\n\n{edit_instruction}"

        # Truncate if needed
        if len(full_prompt) > MAX_PROMPT_LENGTH:
            full_prompt = _truncate_prompt(full_prompt, MAX_PROMPT_LENGTH)

        logger.info(
            "Generating frame %d/%d (prompt_len=%d, style=%s)",
            frame_idx + 1, frame_count, len(full_prompt), animation_style.value,
        )

        # Generate the frame
        try:
            frame_result = await _generate_single_frame(
                prompt=full_prompt,
                original_image_b64=image_b64,
                previous_thought_signature=thought_signature,
                api_caller=api_caller,
            )
        except asyncio.TimeoutError:
            result.error = (
                f"Frame {frame_idx + 1} generation timed out after {FRAME_TIMEOUT_S}s. "
                f"The Gemini API may be overloaded."
            )
            return result
        except Exception as e:
            result.error = f"Frame {frame_idx + 1} generation failed: {str(e)}"
            return result

        frame_time_ms = int((time.monotonic() - frame_t0) * 1000)
        result.frame_times_ms.append(frame_time_ms)

        if not frame_result.success:
            result.error = f"Frame {frame_idx + 1} failed: {frame_result.error}"
            return result

        frames.append(frame_result.frame_b64)
        thought_signature = frame_result.thought_signature

        # Callback for progress
        if on_frame_complete:
            on_frame_complete(frame_idx + 1, frame_result.frame_b64)

        logger.info(
            "Frame %d/%d completed in %d ms",
            frame_idx + 1, frame_count, frame_time_ms,
        )

    # Success
    result.success = True
    result.frames = frames
    result.total_time_ms = int((time.monotonic() - t0) * 1000)

    logger.info(
        "Generated %d frames in %d ms (avg: %d ms/frame)",
        len(frames),
        result.total_time_ms,
        result.total_time_ms // len(frames) if frames else 0,
    )

    return result


async def _generate_single_frame(
    prompt: str,
    original_image_b64: str,
    previous_thought_signature: Optional[str],
    api_caller: Optional[Callable],
) -> FrameResult:
    """
    Generate a single frame using the Gemini API.

    If no api_caller is provided, returns a placeholder for testing.
    """
    t0 = time.monotonic()

    if api_caller is None:
        # Return placeholder for testing
        logger.warning("No API caller provided — returning placeholder frame")
        return FrameResult(
            success=True,
            frame_b64=original_image_b64,  # Return original as placeholder
            generation_time_ms=int((time.monotonic() - t0) * 1000),
        )

    try:
        # Build the API request
        # This structure follows Google's Gemini API format
        request_body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": original_image_b64,
                            }
                        },
                        {"text": prompt},
                    ],
                }
            ],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
            },
        }

        # Include thought signature for multi-turn consistency
        if previous_thought_signature:
            request_body["thought_signature"] = previous_thought_signature

        # Call the API
        response = await api_caller(request_body)

        # Extract the generated image
        frame_b64 = _extract_image_from_response(response)
        thought_sig = response.get("thought_signature")

        if frame_b64:
            return FrameResult(
                success=True,
                frame_b64=frame_b64,
                thought_signature=thought_sig,
                generation_time_ms=int((time.monotonic() - t0) * 1000),
            )
        else:
            return FrameResult(
                success=False,
                error="No image in API response",
                generation_time_ms=int((time.monotonic() - t0) * 1000),
            )

    except Exception as e:
        return FrameResult(
            success=False,
            error=str(e),
            generation_time_ms=int((time.monotonic() - t0) * 1000),
        )


def _extract_image_from_response(response: Dict[str, Any]) -> Optional[str]:
    """Extract base64 image from API response."""
    # Try various response formats
    candidates = response.get("candidates", [])
    if candidates:
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        for part in parts:
            if "inline_data" in part:
                return part["inline_data"].get("data")

    # Fallback: direct image field
    if "image" in response:
        return response["image"]

    return None


def _sanitize_base64(b64: str) -> str:
    """Remove data URI prefix if present."""
    if b64.startswith("data:"):
        parts = b64.split(",", 1)
        return parts[1] if len(parts) == 2 else b64
    return b64


def _truncate_prompt(prompt: str, max_length: int) -> str:
    """Truncate prompt while preserving structure."""
    if len(prompt) <= max_length:
        return prompt

    # Keep first 60% and last 30%
    head = int(max_length * 0.6)
    tail = int(max_length * 0.3)
    bridge = "\n\n[...additional instructions...]\n\n"

    return prompt[:head] + bridge + prompt[-tail:]


# ═══════════════════════════════════════════════════════════════════════════
#  Utility: Frame Interpolation (Optional Enhancement)
# ═══════════════════════════════════════════════════════════════════════════

def calculate_interpolation_params(
    frame_index: int,
    total_frames: int,
    style: AnimationStyle,
) -> Dict[str, float]:
    """
    Calculate interpolation parameters for a frame.

    This can be used to generate smoother animations by providing
    mathematical parameters to the motion instructions.
    """
    import math

    t = frame_index / (total_frames - 1) if total_frames > 1 else 0

    # Common easing functions
    linear = t
    ease_in = t * t
    ease_out = 1 - (1 - t) * (1 - t)
    ease_in_out = (1 - math.cos(t * math.pi)) / 2
    bounce = abs(math.sin(t * math.pi * 2)) * (1 - t)

    return {
        "t": t,
        "linear": linear,
        "ease_in": ease_in,
        "ease_out": ease_out,
        "ease_in_out": ease_in_out,
        "bounce": bounce,
        "frame": frame_index,
        "total": total_frames,
    }
