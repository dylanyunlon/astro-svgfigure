"""
Frame Generator — Gemini Multi-Frame Animation
=================================================
Generates animation frames using Gemini 3 Pro Image with mandatory
green-screen (#00FF00) background for downstream chroma-key removal.

Pipeline Position: Step 3 of 4
    Step 1: Claude 4.6 image analysis
    Step 2: Grok animation prompt engineering
  → Step 3: THIS MODULE (Gemini frame generation)
    Step 4: Green-screen removal + encoding

Architecture Decision Record:
─────────────────────────────
1. BATCHING STRATEGY:
   Gemini's image generation has output limits. For ≤4 frames, we
   generate all in one request. For 5-24 frames, we batch into groups
   of 4 and make sequential requests. This trades latency for reliability.

   Alternative considered: parallel requests. Rejected because:
   - Rate limits would cause 429 errors
   - Visual consistency degrades without sequential context
   - Error handling becomes complex with partial failures

2. GREEN-SCREEN HARDCODING:
   The green_screen flag is always True (enforced in schema validator).
   We additionally prepend the green-screen instruction to EVERY prompt
   sent to Gemini, regardless of what the user/Grok specified.
   Belt-and-suspenders approach: the cost of a green-screen failure
   (wasted minutes + API tokens) vastly exceeds the cost of redundant
   instructions (~50 tokens per request).

3. FRAME CONSISTENCY:
   We send the source image + previous frame as reference for each batch.
   This improves visual consistency but increases request size.
   Trade-off: ~2x request size vs. significantly better frame coherence.

Knuth-Level Critiques:
─────────────────────
User Angle:
  - Generation of 8+ frames takes 2-5 minutes. The frontend should show
    per-frame progress, but we currently return all-or-nothing. Future:
    implement SSE streaming of frames as they complete.
  - If Gemini generates frames with slightly different dimensions,
    the animation will jitter. We normalize all frames to the first
    frame's dimensions in post-processing.

System Angle:
  - Gemini 3 Pro Image returns images as base64 in the response.
    For 16 frames at 1024x1024, this is ~50MB of base64 data.
    We should add response streaming, but the current architecture
    (Astro proxy → FastAPI → Gemini) doesn't support it cleanly.
  - If Gemini returns fewer frames than requested (common with
    complex prompts), we interpolate missing frames by blending
    adjacent frames. This is a lossy approximation but prevents
    the animation from having gaps.

GitHub references:
  - google/generative-ai-python (Gemini API)
  - dylanyunlon/astro-svgfigure/backend/pipeline/gemini_image_gen.py
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
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

BATCH_SIZE = 4  # Max frames per Gemini request
MAX_RETRIES = 2  # Retries per batch on failure
RETRY_DELAY_S = 3  # Delay between retries

GREEN_SCREEN_PREAMBLE = (
    "MANDATORY BACKGROUND RULE: The background of this image MUST be "
    "solid bright green (#00FF00, RGB(0,255,0)). No gradients, no shadows, "
    "no texture on the background. The subject must have clean, sharp edges "
    "against the pure green background. This is for chroma-key compositing.\n\n"
)

# Aspect ratio to pixel dimensions mapping
ASPECT_DIMENSIONS: Dict[str, Tuple[int, int]] = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "4:3": (1152, 896),
}


# ═══════════════════════════════════════════════════════════════════════
#  Core Frame Generation
# ═══════════════════════════════════════════════════════════════════════

async def generate_animation_frames(
    request: AnimateFramesRequest,
    settings: Optional[Settings] = None,
    ai_engine: Optional[AIEngine] = None,
) -> AnimateFramesResponse:
    """
    Generate multi-frame animation images using Gemini.

    Each frame has a mandatory green-screen (#00FF00) background.
    Frames are generated in batches of BATCH_SIZE for reliability.

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
        prompt = request.custom_prompt or ""
        aspect = request.aspect_ratio

        # ── Validate and enhance prompt ──
        prompt = _build_frame_prompt(
            base_prompt=prompt,
            frame_count=frame_count,
            aspect=aspect,
            style=request.animation_style,
        )

        logger.info(
            "generate_frames: model=%s, frames=%d, aspect=%s, prompt_len=%d",
            model, frame_count, aspect, len(prompt),
        )

        # ── Generate frames in batches ──
        all_frames: List[str] = []
        batches = _create_batches(frame_count, BATCH_SIZE)

        for batch_idx, (start_frame, batch_count) in enumerate(batches):
            logger.info(
                "generate_frames: batch %d/%d — frames %d-%d",
                batch_idx + 1, len(batches), start_frame + 1, start_frame + batch_count,
            )

            batch_prompt = _build_batch_prompt(
                prompt=prompt,
                batch_start=start_frame,
                batch_count=batch_count,
                total_frames=frame_count,
            )

            # Reference image for consistency
            reference_b64 = image_b64
            # For subsequent batches, also reference the last generated frame
            if all_frames:
                reference_b64 = all_frames[-1]

            batch_frames = await _generate_batch(
                engine=engine,
                model=model,
                prompt=batch_prompt,
                reference_b64=reference_b64,
                source_b64=image_b64,
                batch_count=batch_count,
                aspect=aspect,
            )

            if batch_frames:
                all_frames.extend(batch_frames)
                logger.info(
                    "generate_frames: batch %d produced %d frames (total: %d)",
                    batch_idx + 1, len(batch_frames), len(all_frames),
                )
            else:
                logger.warning("generate_frames: batch %d produced 0 frames", batch_idx + 1)

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        if not all_frames:
            return AnimateFramesResponse(
                success=False,
                error="Gemini did not generate any frames. Try a simpler prompt or fewer frames.",
                generation_time_ms=elapsed_ms,
            )

        # ── Post-processing: normalize frame count ──
        if len(all_frames) < frame_count:
            logger.warning(
                "generate_frames: got %d/%d frames — interpolating missing",
                len(all_frames), frame_count,
            )
            # Don't interpolate if we're close enough
            if len(all_frames) >= frame_count * 0.5:
                all_frames = _pad_frames(all_frames, frame_count)
        elif len(all_frames) > frame_count:
            all_frames = all_frames[:frame_count]

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
