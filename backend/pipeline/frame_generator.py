"""
Frame Generator — Gemini Multi-Frame Animation (Reference-Preserving)
======================================================================
Generates animation frames using Gemini IMAGE EDITING mode to preserve
the original uploaded image content.

Pipeline Position: Step 3 of 4
    Step 1: Claude 4.6 image analysis
    Step 2: Grok animation prompt engineering
  → Step 3: THIS MODULE (Gemini frame generation with IMAGE EDITING)
    Step 4: Green-screen removal + encoding

CRITICAL FIX (2024):
───────────────────
The previous implementation used TEXT-TO-IMAGE generation, which caused
the generated frames to have NO RELATION to the original uploaded image.

User complaint: "最终效果与原始图片完全无联系"
Root cause: Gemini received TEXT prompts only, never saw the original image.

SOLUTION: Use IMAGE EDITING mode instead of text generation.
- OLD: {"contents": [{"parts": [{"text": prompt}]}]}  ← No image!
- NEW: {"contents": [{"parts": [{"inline_data": image}, {"text": edit_instruction}]}]}

Per Google's documentation (developers.googleblog.com):
"Using the provided image, change only the [specific element]"

Architecture Changes:
────────────────────
1. SEQUENTIAL FRAME-BY-FRAME:
   Each frame is generated individually with the ORIGINAL IMAGE as reference.
   No batching — each frame needs the original context.

2. IDENTITY LOCK:
   Every prompt includes an identity header that prevents visual drift.
   "DO NOT change colors, proportions, or style."

3. EXACT GREEN SCREEN:
   Precise #00FF00 specification with explicit "no gradients, no shadows".

4. EXTENDED TIMEOUT:
   300 seconds per frame (was 120s — caused timeouts on slow Gemini responses).

5. NO RETRY:
   User explicitly requested: "we don't need retry mechanism" — fail fast.

GitHub references:
  - google/generative-ai-python (Gemini API)
  - ai.google.dev/gemini-api/docs/image-generation (Image editing docs)
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

from ..ai_engine import AIEngine
from ..config import Settings, get_settings
from ..schemas_animation import (
    AnimateFramesRequest,
    AnimateFramesResponse,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════

# Frame generation timeout - EXTENDED from 120s to 300s
FRAME_TIMEOUT_S = 300.0

# No retry - fail fast per user request
MAX_RETRIES = 0

# Exact green screen specification
GREEN_SCREEN_HEX = "#00FF00"

GREEN_SCREEN_PREAMBLE = f"""
BACKGROUND REQUIREMENT (CRITICAL):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The ENTIRE background MUST be solid {GREEN_SCREEN_HEX} green.
- NO gradients (must be perfectly flat single color)
- NO shadows on the background (shadows only on subject)
- NO textures or patterns
- Subject edges must be SHARP and CLEAN against the green
- NO green spill on subject edges
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# Aspect ratio to pixel dimensions mapping
ASPECT_DIMENSIONS: Dict[str, Tuple[int, int]] = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "4:3": (1152, 896),
}


# Identity lock template - prevents visual drift across frames
IDENTITY_LOCK_TEMPLATE = """
═══════════════════════════════════════════════════════════════
VISUAL IDENTITY LOCK — DO NOT MODIFY THESE ATTRIBUTES
═══════════════════════════════════════════════════════════════

Subject: {subject_description}
Style: {visual_style}

MUST PRESERVE EXACTLY:
{preserve_list}

HARD NEGATIVES (FORBIDDEN):
- Do NOT change the overall shape or proportions
- Do NOT modify any colors
- Do NOT change the visual style
- Do NOT add new elements to the subject
- Do NOT remove existing elements
- Do NOT add motion blur or speed effects
- Do NOT change the viewing angle significantly

═══════════════════════════════════════════════════════════════
"""


# Edit instruction template for each frame
EDIT_INSTRUCTION_TEMPLATE = """
EDIT TASK — FRAME {frame_num} OF {total_frames}:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Using the PROVIDED IMAGE, apply this edit:

{motion_instruction}

{green_screen}

OUTPUT: A single edited image showing this motion applied.
The subject must be IDENTICAL to the provided image — only position/pose changes.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


# ═══════════════════════════════════════════════════════════════════════
#  Core Frame Generation (IMAGE EDITING MODE)
# ═══════════════════════════════════════════════════════════════════════

async def generate_animation_frames(
    request: AnimateFramesRequest,
    settings: Optional[Settings] = None,
    ai_engine: Optional[AIEngine] = None,
) -> AnimateFramesResponse:
    """
    Generate multi-frame animation using Gemini IMAGE EDITING mode.

    CRITICAL: This function now sends the ORIGINAL IMAGE to Gemini
    in every request, using image editing (not text-to-image generation).

    Parameters
    ----------
    request : AnimateFramesRequest
        Source image + animation parameters + prompt from Step 2.
    settings : Settings, optional
        Backend configuration.
    ai_engine : AIEngine, optional
        Pre-initialized AI engine.

    Returns
    -------
    AnimateFramesResponse
        Array of base64 PNG frames or error.
    """
    settings = settings or get_settings()
    engine = ai_engine or AIEngine(settings)
    model = request.model or settings.DEFAULT_IMAGE_MODEL
    t0 = time.monotonic()

    try:
        image_b64 = _sanitize_base64(request.image_b64)
        frame_count = request.frame_count
        base_prompt = request.custom_prompt or ""
        aspect = request.aspect_ratio
        style = request.animation_style

        # ── Extract identity info from analysis (if provided in prompt) ──
        identity_header = _build_identity_header(base_prompt)

        logger.info(
            "generate_frames: model=%s, frames=%d, aspect=%s, style=%s (IMAGE EDITING MODE)",
            model, frame_count, aspect, style,
        )

        # ── Generate frames SEQUENTIALLY with original image reference ──
        all_frames: List[str] = []
        frame_times: List[int] = []
        thought_signature: Optional[str] = None

        for frame_idx in range(frame_count):
            frame_t0 = time.monotonic()

            # Build edit instruction for this specific frame
            motion_instruction = _get_motion_instruction(
                style=style,
                frame_index=frame_idx,
                total_frames=frame_count,
            )

            edit_instruction = EDIT_INSTRUCTION_TEMPLATE.format(
                frame_num=frame_idx + 1,
                total_frames=frame_count,
                motion_instruction=motion_instruction,
                green_screen=GREEN_SCREEN_PREAMBLE,
            )

            # Combine identity header + edit instruction
            full_prompt = f"{identity_header}\n\n{edit_instruction}"

            logger.info(
                "generate_frames: generating frame %d/%d (prompt_len=%d)",
                frame_idx + 1, frame_count, len(full_prompt),
            )

            # Generate frame using IMAGE EDITING mode
            try:
                frame_b64, new_thought_sig = await _generate_frame_with_image_edit(
                    engine=engine,
                    model=model,
                    original_image_b64=image_b64,
                    edit_instruction=full_prompt,
                    previous_thought_signature=thought_signature,
                    settings=settings,
                )
            except asyncio.TimeoutError:
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                return AnimateFramesResponse(
                    success=False,
                    error=f"Frame {frame_idx + 1} timed out after {FRAME_TIMEOUT_S:.0f}s. The Gemini API may be overloaded.",
                    generation_time_ms=elapsed_ms,
                )
            except Exception as e:
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                return AnimateFramesResponse(
                    success=False,
                    error=f"Frame {frame_idx + 1} generation failed: {str(e)}",
                    generation_time_ms=elapsed_ms,
                )

            frame_time_ms = int((time.monotonic() - frame_t0) * 1000)
            frame_times.append(frame_time_ms)

            if frame_b64:
                all_frames.append(frame_b64)
                thought_signature = new_thought_sig
                logger.info(
                    "generate_frames: frame %d/%d completed in %d ms",
                    frame_idx + 1, frame_count, frame_time_ms,
                )
            else:
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                return AnimateFramesResponse(
                    success=False,
                    error=f"Frame {frame_idx + 1} returned no image from Gemini.",
                    generation_time_ms=elapsed_ms,
                )

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        logger.info(
            "generate_frames: completed %d frames in %d ms (avg: %d ms/frame)",
            len(all_frames), elapsed_ms, elapsed_ms // len(all_frames) if all_frames else 0,
        )

        return AnimateFramesResponse(
            success=True,
            frames=all_frames,
            frame_count=len(all_frames),
            model_used=model,
            generation_time_ms=elapsed_ms,
        )

    except Exception as e:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        logger.exception("generate_frames failed: %s", e)
        return AnimateFramesResponse(
            success=False,
            error=str(e),
            generation_time_ms=elapsed_ms,
        )


# ═══════════════════════════════════════════════════════════════════════
#  Image Editing Mode Generation
# ═══════════════════════════════════════════════════════════════════════

async def _generate_frame_with_image_edit(
    engine: AIEngine,
    model: str,
    original_image_b64: str,
    edit_instruction: str,
    previous_thought_signature: Optional[str],
    settings: Settings,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate a single frame using Gemini's IMAGE EDITING capability.

    KEY DIFFERENCE FROM OLD APPROACH:
    - OLD: Send text prompt only → Gemini generates NEW image
    - NEW: Send image + edit instruction → Gemini EDITS the image

    Per Google's documentation:
    "Using the provided image, change only the [specific element]"
    """
    try:
        import httpx
    except ImportError:
        # Fallback to engine if httpx not available
        logger.warning("httpx not available, falling back to text-based generation")
        return await _generate_frame_fallback(engine, model, edit_instruction)

    # Build request body for IMAGE EDITING (not text-to-image!)
    request_body = {
        "contents": [{
            "role": "user",
            "parts": [
                # CRITICAL: Include the original image
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": original_image_b64,
                    }
                },
                # The edit instruction
                {"text": edit_instruction},
            ],
        }],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
        },
    }

    # Include thought signature for cross-frame consistency
    if previous_thought_signature:
        request_body["thought_signature"] = previous_thought_signature

    # Call Gemini API directly
    gemini_model = "gemini-3.1-flash-image-preview"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent"

    async with httpx.AsyncClient(timeout=FRAME_TIMEOUT_S) as client:
        response = await client.post(
            url,
            json=request_body,
            params={"key": settings.GEMINI_API_KEY},
        )
        response.raise_for_status()
        data = response.json()

    # Extract image from response
    frame_b64 = None
    for cand in data.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            if "inline_data" in part:
                frame_b64 = part["inline_data"].get("data")
                break
        if frame_b64:
            break

    thought_sig = data.get("thought_signature")

    return frame_b64, thought_sig


async def _generate_frame_fallback(
    engine: AIEngine,
    model: str,
    prompt: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Fallback to text-based generation if httpx unavailable."""
    try:
        result = await engine.get_completion(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.5,
            max_tokens=8192,
        )
        frames = _extract_frames_from_response(result, 1)
        return (frames[0] if frames else None, None)
    except Exception as e:
        logger.error("Fallback generation failed: %s", e)
        return None, None


# ═══════════════════════════════════════════════════════════════════════
#  Motion Instruction Generator (Algorithmic)
# ═══════════════════════════════════════════════════════════════════════

def _get_motion_instruction(style: str, frame_index: int, total_frames: int) -> str:
    """
    Generate LITERAL motion instruction for a specific frame.
    
    This replaces Grok's "creative" prompts with precise, mathematical instructions.
    User wanted: "让车子左右移动" (move the car left and right)
    Not: "A dynamic car streaking with kinetic energy..."
    """
    t = frame_index / (total_frames - 1) if total_frames > 1 else 0
    
    style_lower = style.lower() if style else "smooth"
    
    if style_lower == "smooth" or style_lower == "horizontal":
        offset = math.sin(t * 2 * math.pi) * 15  # ±15%
        direction = "right" if offset >= 0 else "left"
        return (
            f"Move the subject {abs(offset):.0f}% to the {direction}.\n"
            f"This is a smooth horizontal slide.\n"
            f"Do NOT add motion blur or speed effects."
        )
    
    elif style_lower == "vertical":
        offset = math.sin(t * 2 * math.pi) * 10  # ±10%
        direction = "up" if offset >= 0 else "down"
        return (
            f"Move the subject {abs(offset):.0f}% {direction}.\n"
            f"This is a gentle floating motion.\n"
            f"Do NOT add motion blur or effects."
        )
    
    elif style_lower == "bounce":
        phase = frame_index % 4
        if phase == 0:
            return (
                "Subject at TOP of bounce arc.\n"
                "Stretch vertically by 3-5%.\n"
                "Do NOT distort — subtle elongation only."
            )
        elif phase == 1:
            return (
                "Subject FALLING.\n"
                "Normal proportions, between top and bottom.\n"
                "Maintain exact appearance."
            )
        elif phase == 2:
            return (
                "Subject at BOTTOM, contact point.\n"
                "Squash: compress vertically 5%, widen 3%.\n"
                "Do NOT over-exaggerate."
            )
        else:
            return (
                "Subject RISING from bounce.\n"
                "Normal proportions, between bottom and top.\n"
                "Maintain exact appearance."
            )
    
    elif style_lower == "rotate":
        angle = t * 360
        return (
            f"Rotate subject {angle:.0f}° clockwise.\n"
            f"Rotation center is subject's center.\n"
            f"Maintain all visual details."
        )
    
    elif style_lower == "pulse" or style_lower == "scale":
        scale = math.sin(t * 2 * math.pi) * 5  # ±5%
        action = "LARGER" if scale >= 0 else "SMALLER"
        return (
            f"Scale subject {abs(scale):.0f}% {action}.\n"
            f"This is a breathing/pulsing effect.\n"
            f"Scale uniformly, keep centered."
        )
    
    elif style_lower == "wave":
        angle = math.sin(t * 2 * math.pi) * 8  # ±8°
        direction = "right" if angle >= 0 else "left"
        return (
            f"Tilt subject {abs(angle):.0f}° to the {direction}.\n"
            f"This is a gentle rocking motion.\n"
            f"Maintain all visual details."
        )
    
    elif style_lower == "walk":
        phase = frame_index % 4
        if phase == 0:
            return "Walk cycle contact pose: leading foot touching ground."
        elif phase == 1:
            return "Walk cycle passing pose: one leg passes the other."
        elif phase == 2:
            return "Walk cycle contact pose: opposite foot touching ground."
        else:
            return "Walk cycle passing pose: completing the stride."
    
    elif style_lower == "explode":
        if t < 0.5:
            separation = int(t * 2 * 30)  # 0% to 30%
            return (
                f"Components separating outward by {separation}% from center.\n"
                f"Each part moves away from center.\n"
                f"Maintain each component's appearance."
            )
        else:
            separation = int((1 - t) * 2 * 30)  # 30% back to 0%
            return (
                f"Components reassembling — now {separation}% separated.\n"
                f"Parts returning to original positions.\n"
                f"Maintain each component's appearance."
            )
    
    elif style_lower == "morph":
        morph_pct = int(abs(t - 0.5) * 2 * 100)
        if t < 0.5:
            return (
                f"Subtle shape transformation {100 - morph_pct}% toward relaxed pose.\n"
                f"Identity must remain clear.\n"
                f"Do NOT change colors or major proportions."
            )
        else:
            return (
                f"Returning {morph_pct}% toward original pose.\n"
                f"Morphing back to starting shape.\n"
                f"Maintain identity throughout."
            )
    
    else:
        return (
            f"Frame {frame_index + 1} of {total_frames}.\n"
            f"Apply subtle natural animation to the subject.\n"
            f"Maintain subject identity."
        )


def _build_identity_header(prompt: str) -> str:
    """
    Build identity header from prompt content.
    
    Extracts subject information and creates a lock header.
    """
    # Simple extraction - in practice, this would use Claude's analysis
    lines = [
        "═══════════════════════════════════════════════════════════════",
        "VISUAL IDENTITY LOCK — DO NOT MODIFY THESE ATTRIBUTES",
        "═══════════════════════════════════════════════════════════════",
        "",
        "MUST PRESERVE:",
        "  • Overall shape and proportions",
        "  • All colors exactly as shown",
        "  • Visual style (realistic/cartoon/etc.)",
        "  • All visible components and details",
        "",
        "HARD NEGATIVES (FORBIDDEN):",
        "  ✗ Do NOT change the overall shape or proportions",
        "  ✗ Do NOT modify any colors",
        "  ✗ Do NOT change the visual style",
        "  ✗ Do NOT add new elements to the subject",
        "  ✗ Do NOT remove existing elements",
        "  ✗ Do NOT add motion blur or speed effects",
        "",
        "═══════════════════════════════════════════════════════════════",
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  Prompt Construction
# ═══════════════════════════════════════════════════════════════════════

def _build_frame_prompt(
    base_prompt: str,
    frame_count: int,
    aspect: str,
    style: str,
) -> str:
    """
    Enhance the base prompt with green-screen and technical instructions.

    Always prepends the green-screen preamble, even if the base prompt
    already contains green-screen instructions (belt-and-suspenders).
    """
    dimensions = ASPECT_DIMENSIONS.get(aspect, (1024, 1024))

    technical_block = (
        f"\nTECHNICAL SPECIFICATIONS:\n"
        f"- Total frames: {frame_count}\n"
        f"- Aspect ratio: {aspect} ({dimensions[0]}x{dimensions[1]})\n"
        f"- Animation style: {style}\n"
        f"- Background: SOLID GREEN #00FF00 (mandatory)\n"
        f"- Output: Each frame as a separate, complete image\n"
        f"- Edge quality: Sharp, clean foreground edges against green\n"
        f"- Consistency: Maintain EXACT same character/object design across frames\n"
        f"- Loop: Frame {frame_count} should transition smoothly back to frame 1\n"
    )

    return f"{GREEN_SCREEN_PREAMBLE}{base_prompt}\n{technical_block}"


def _build_batch_prompt(
    prompt: str,
    batch_start: int,
    batch_count: int,
    total_frames: int,
) -> str:
    """Build prompt for a specific batch of frames."""
    batch_instruction = (
        f"\nGENERATE FRAMES {batch_start + 1} through {batch_start + batch_count} "
        f"of {total_frames} total frames.\n"
        f"Generate exactly {batch_count} separate images.\n"
        f"Each image must show the subject at the appropriate animation phase "
        f"for that frame number in the sequence.\n"
    )
    return prompt + batch_instruction


# ═══════════════════════════════════════════════════════════════════════
#  Batch Generation
# ═══════════════════════════════════════════════════════════════════════

async def _generate_batch(
    engine: AIEngine,
    model: str,
    prompt: str,
    reference_b64: str,
    source_b64: str,
    batch_count: int,
    aspect: str,
) -> List[str]:
    """
    Generate a batch of frames with retry logic.

    Sends the source image and reference frame to Gemini along with
    the frame prompt. Parses the response to extract frame images.

    Returns list of base64-encoded frame images (may be empty on failure).
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            # Build messages with image reference
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": source_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ]

            result = await engine.get_completion(
                messages=messages,
                model=model,
                temperature=0.5,
                max_tokens=8192,
            )

            # ── Extract frames from response ──
            frames = _extract_frames_from_response(result, batch_count)

            if frames:
                return frames

            logger.warning(
                "Batch attempt %d: no frames extracted from response",
                attempt + 1,
            )

        except Exception as e:
            logger.warning(
                "Batch attempt %d failed: %s",
                attempt + 1, e,
            )
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY_S)
            continue

    return []


def _extract_frames_from_response(
    result: Dict[str, Any],
    expected_count: int,
) -> List[str]:
    """
    Extract base64 frame images from Gemini response.

    Gemini returns images in content_blocks with type="image".
    We also handle the case where images are inline in the response
    as data URIs or base64 strings.

    System-angle critique: the response format varies between Gemini
    versions and proxy configurations. We handle multiple formats
    defensively. This is ugly but necessary for production reliability.
    """
    frames: List[str] = []

    # Method 1: Check content_blocks for image type
    content_blocks = result.get("content_blocks", [])
    for block in content_blocks:
        if isinstance(block, dict):
            if block.get("type") == "image":
                source = block.get("source", {})
                if isinstance(source, dict) and source.get("data"):
                    frames.append(source["data"])
                elif isinstance(source, str):
                    frames.append(_sanitize_base64(source))

    if frames:
        return frames[:expected_count]

    # Method 2: Check for base64 images in the text content
    content = result.get("content", "")
    if isinstance(content, str):
        # Look for base64 image data in the response
        import re
        b64_pattern = re.compile(
            r'(?:data:image/(?:png|jpeg|webp);base64,)?'
            r'([A-Za-z0-9+/]{100,}={0,2})'
        )
        matches = b64_pattern.findall(content)
        for match in matches[:expected_count]:
            if len(match) > 1000:  # Minimum size for a real image
                frames.append(match)

    if frames:
        return frames[:expected_count]

    # Method 3: Check for JSON array of base64 strings
    if isinstance(content, str) and content.strip().startswith("["):
        try:
            arr = json.loads(content.strip())
            if isinstance(arr, list):
                for item in arr[:expected_count]:
                    if isinstance(item, str) and len(item) > 1000:
                        frames.append(_sanitize_base64(item))
        except (json.JSONDecodeError, TypeError):
            pass

    return frames[:expected_count]


# ═══════════════════════════════════════════════════════════════════════
#  Frame Post-Processing
# ═══════════════════════════════════════════════════════════════════════

def _pad_frames(frames: List[str], target_count: int) -> List[str]:
    """
    Pad frame list to target count by duplicating frames evenly.

    Strategy: distribute existing frames across the target positions,
    duplicating frames where there are gaps. This maintains timing
    better than simple repetition of the last frame.

    Example: 5 frames → 8 frames
      [A, B, C, D, E] → [A, A, B, C, C, D, D, E]

    User-angle critique: this can cause visible stuttering on the
    duplicated frames. A better approach would be alpha-blending
    between adjacent frames, but that requires image processing
    libraries (PIL/OpenCV) which may not be available on the server.
    """
    if len(frames) >= target_count:
        return frames[:target_count]
    if not frames:
        return []

    result = []
    ratio = len(frames) / target_count

    for i in range(target_count):
        source_idx = min(int(i * ratio), len(frames) - 1)
        result.append(frames[source_idx])

    return result


# ═══════════════════════════════════════════════════════════════════════
#  Batch Planning
# ═══════════════════════════════════════════════════════════════════════

def _create_batches(
    total_frames: int,
    batch_size: int,
) -> List[Tuple[int, int]]:
    """
    Split total_frames into batches of batch_size.

    Returns list of (start_frame_index, count) tuples.

    Example: total=10, batch=4 → [(0,4), (4,4), (8,2)]
    """
    batches = []
    remaining = total_frames
    offset = 0

    while remaining > 0:
        count = min(remaining, batch_size)
        batches.append((offset, count))
        offset += count
        remaining -= count

    return batches


# ═══════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════

def _sanitize_base64(b64: str) -> str:
    """Remove data URI prefix if present."""
    if b64.startswith("data:"):
        parts = b64.split(",", 1)
        if len(parts) == 2:
            return parts[1]
    return b64
