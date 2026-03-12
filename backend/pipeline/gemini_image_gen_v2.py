"""
Gemini Image Generation v2 — Knuth-Critical Rewrite
=====================================================
Root Cause Analysis (as Donald Knuth would frame it):
------------------------------------------------------
The original system has 5 compounding failures that produce the
"prompt echo" symptom. These are not independent bugs — they form
a causal chain where each failure AMPLIFIES the next:

  Cause 1: FORMAT MISMATCH (structural)
    Grok 4 outputs "Point 1: ... Point 2: ..." numbered lists.
    Gemini's image model is trained on NARRATIVE PARAGRAPHS.
    Google's official guidance (2026-02-27 blog): "Describe the scene,
    don't just list keywords. A narrative paragraph will almost always
    produce a better image than a list of disconnected words."
    → Numbered lists trigger Gemini's TEXT-COMPLETION mode instead
      of IMAGE-GENERATION mode.

  Cause 2: PAYLOAD BLOAT (quantitative)
    TIER-80 generates ~6000 chars of numbered points.
    The rule-based compressor (_distill_prompt_rule_based) aims for
    1500 chars but keeps the numbered-list FORMAT, just shorter.
    Combined with svg_summary (800 chars) + wrapper text (~200 chars),
    the total payload reaches ~2500 chars of LIST-FORMAT text.
    → Even "compressed", the format is still wrong for image gen.

  Cause 3: PROXY SSE INTERFERENCE (infrastructure)
    tryallai.com proxy returns SSE (text/event-stream) instead of
    JSON for generateContent. The reassembly logic works for IMAGE
    chunks but when the model returns TEXT-ONLY (because of Causes
    1-2), the SSE reassembly produces a text-only response that
    looks like the prompt was echoed.
    → The proxy may not faithfully forward responseModalities when
      the prompt triggers text-completion mode.

  Cause 4: MISSING IMAGE-TRIGGER PHRASE (semantic)
    Google's best practice doc (2026-03-05): "Use phrases such as
    'create an image of' or 'generate an image of'. Otherwise, the
    multimodal model might respond with text instead of the image."
    The original code starts with "Generate a publication-quality
    scientific figure IMAGE." — but the word "IMAGE" at the end of
    a long sentence is not as strong a trigger as "Generate an image
    of..." at the very start.
    → Weak image-trigger phrase lets the model default to text mode.

  Cause 5: cairosvg NOT INSTALLED (environmental)
    The log shows "cairosvg not installed — cannot render SVG to PNG".
    Without the skeleton PNG, Gemini has NO visual reference — it only
    sees a LONG TEXT describing a layout. This makes it even more likely
    to treat the entire input as a text-completion task.
    → Missing visual input removes the strongest IMAGE-mode signal.

SOLUTION: A 3-stage pipeline that addresses ALL 5 causes without retry.
------------------------------------------------------------------------
  Stage 1: Grok 4 generates verbose design spec (existing, unchanged)
  Stage 2: NARRATIVE COMPRESSOR converts numbered-list → dense paragraph
           (the prompt_compressor.py module, now integrated properly)
  Stage 3: Gemini receives:
           (a) A SHORT narrative paragraph (800-1200 chars)
           (b) Starting with "Generate an image of..." (strong trigger)
           (c) responseModalities: ["IMAGE", "TEXT"] (IMAGE first)
           (d) Skeleton PNG if available (strongest image-mode signal)
           (e) No numbered lists, no TIER headers, no "Point N:" format

NO RETRY MECHANISM. If Stage 2 compression fails, we use the fallback
compressor. If Gemini returns text instead of image, we report the
error clearly and let the user adjust.

GitHub: dylanyunlon/astro-svgfigure
"""

from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from ..config import Settings, get_settings
from .prompt_compressor import structural_compress, to_gemini_narrative

logger = logging.getLogger(__name__)


# ============================================================================
# Stage 3: Gemini Image Generation (rewritten)
# ============================================================================

async def generate_image_with_gemini_v2(
    svg_content: str,
    design_spec: str,
    method_text: str = "",
    settings: Optional[Settings] = None,
    model: str = "gemini-3-pro-image-preview",
    aspect_ratio: str = "16:9",
    image_size: str = "4K",
    elk_graph: Optional[Dict] = None,
    skeleton_png_b64: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a scientific figure image using Gemini, with proper prompt
    formatting that avoids the prompt-echo failure.

    Key differences from v1:
      1. Converts design_spec to NARRATIVE format (no numbered lists)
      2. Starts with strong image-trigger phrase
      3. Keeps total text payload under 1500 chars
      4. No retry mechanism — fail cleanly on first attempt

    Args:
        svg_content: The ELK-generated SVG (for structure extraction)
        design_spec: The detailed design spec (from Grok 4 or user)
        method_text: Paper method description (for narrative context)
        settings: Backend settings
        model: Gemini image model name
        aspect_ratio: Image aspect ratio
        image_size: Output size
        elk_graph: Optional structured graph data
        skeleton_png_b64: Pre-rendered skeleton PNG (base64), if available

    Returns:
        dict with success, image_b64, mime_type, text_response, model_used
    """
    if settings is None:
        settings = get_settings()

    api_key = settings.GEMINI_API_KEY
    api_base = settings.GEMINI_API_BASE

    if not api_key:
        return {"success": False, "error": "GEMINI_API_KEY not configured"}

    # ── Stage 2: Convert design spec to Gemini-optimized narrative ──
    # This is the CRITICAL fix for Cause 1 (format mismatch) and
    # Cause 2 (payload bloat).
    narrative = to_gemini_narrative(
        prompt=design_spec,
        method_text=method_text,
        svg_summary="",  # We'll handle layout info separately
    )

    # ── Build the final prompt — SHORT, NARRATIVE, IMAGE-TRIGGER ──
    # Fix for Cause 4: start with "Generate an image of..." explicitly.
    # Google's best practice: this phrase triggers image generation mode.
    #
    # Total target: ≤ 1200 chars. This is well within the sweet spot
    # identified by Google's own guidance (narrative paragraph, not list).
    final_prompt = narrative  # Already starts with "Generate an image of..."

    # Safety: hard cap at 1500 chars (beyond this, risk of text-echo increases)
    if len(final_prompt) > 1500:
        # Truncate at a sentence boundary
        truncated = final_prompt[:1500]
        last_period = truncated.rfind('.')
        if last_period > 800:
            final_prompt = truncated[:last_period + 1]
        else:
            final_prompt = truncated + "."

    logger.info(
        f"Gemini v2 prompt: {len(final_prompt)} chars "
        f"(narrative format, no numbered lists)"
    )

    # ── Build request parts ──
    request_parts: List[Dict[str, Any]] = []

    # Fix for Cause 5: attach skeleton PNG if available
    # The PNG is the STRONGEST signal to Gemini that we want IMAGE output.
    if skeleton_png_b64:
        request_parts.append({
            "inlineData": {
                "mimeType": "image/png",
                "data": skeleton_png_b64,
            }
        })
        # Instruction referencing the image
        request_parts.append({
            "text": (
                "Reference layout image above. "
                "Enhance this diagram into a professional scientific figure. "
                "Preserve node positions and connections."
            )
        })
        logger.info("Attached skeleton PNG as visual reference")

    # The main narrative prompt
    request_parts.append({"text": final_prompt})

    # ── Build endpoint URL ──
    if api_base:
        base_url = api_base.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        endpoint = f"{base_url}/v1beta/models/{model}:generateContent/"
    else:
        endpoint = (
            f"https://generativelanguage.googleapis.com"
            f"/v1beta/models/{model}:generateContent"
        )

    # ── Request body ──
    # Fix for Cause 3: IMAGE listed first in responseModalities.
    # Also set Accept: application/json to discourage SSE from proxy.
    request_body = {
        "contents": [
            {
                "role": "user",
                "parts": request_parts,
            }
        ],
        "generationConfig": {
            "responseModalities": ["IMAGE", "TEXT"],
            "imageConfig": {
                "aspectRatio": aspect_ratio,
                "imageSize": image_size,
            },
        },
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "X-Response-Modality": "IMAGE",
    }

    params = {}
    if not api_base:
        headers.pop("Authorization", None)
        params["key"] = api_key

    try:
        logger.info(
            f"Gemini v2 image gen: model={model}, "
            f"aspect={aspect_ratio}, size={image_size}, "
            f"prompt_chars={len(final_prompt)}, "
            f"has_skeleton_png={skeleton_png_b64 is not None}"
        )

        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(
                endpoint,
                json=request_body,
                headers=headers,
                params=params if params else None,
            )

        if response.status_code != 200:
            error_text = response.text[:500]
            logger.error(f"Gemini v2 API error {response.status_code}: {error_text}")
            return {
                "success": False,
                "error": f"Gemini API error {response.status_code}: {error_text}",
                "model_used": model,
            }

        # ── Parse response (handle SSE, JSON array, standard JSON) ──
        raw_text = response.text.strip()
        content_type = response.headers.get("content-type", "")

        is_sse = (
            raw_text.startswith("data: ")
            or raw_text.startswith("data:")
            or "text/event-stream" in content_type
            or "\ndata: " in raw_text[:200]
        )

        if is_sse:
            logger.info("Gemini v2: SSE stream detected — reassembling")
            data = _reassemble_sse(raw_text)
        else:
            try:
                data = json.loads(raw_text)
                if isinstance(data, list):
                    logger.info(f"Gemini v2: JSON array ({len(data)} items) — merging")
                    data = _merge_json_array(data)
            except json.JSONDecodeError as e:
                logger.error(f"Gemini v2: not JSON: {e}. First 300: {raw_text[:300]}")
                return {
                    "success": False,
                    "error": f"Non-JSON response: {raw_text[:200]}",
                    "model_used": model,
                }

        # ── Extract image from response ──
        result = _extract_image_from_response(data, model)

        # ── Diagnose prompt-echo if no image returned ──
        if not result.get("success"):
            text_resp = result.get("text_response", "")
            if text_resp:
                # Even with v2 fixes, if the proxy doesn't support
                # responseModalities properly, this can still happen.
                logger.error(
                    f"Gemini v2 returned text instead of image. "
                    f"Prompt was {len(final_prompt)} chars (narrative format). "
                    f"This indicates the proxy may not support image generation. "
                    f"Text preview: {text_resp[:200]}"
                )
                result["error"] = (
                    f"Proxy returned text instead of image. "
                    f"Verify that the proxy (GEMINI_API_BASE) supports "
                    f"responseModalities=[IMAGE,TEXT] for model={model}. "
                    f"Prompt was {len(final_prompt)} chars in narrative format."
                )

        return result

    except httpx.TimeoutException:
        return {
            "success": False,
            "error": "Gemini image generation timed out (600s).",
            "model_used": model,
        }
    except Exception as e:
        logger.error(f"Gemini v2 image gen failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "model_used": model,
        }


# ============================================================================
# Response parsing (simplified, no retry)
# ============================================================================

def _extract_image_from_response(data: dict, model: str) -> Dict[str, Any]:
    """Extract image data from Gemini generateContent response."""
    candidates = data.get("candidates", [])
    if not candidates:
        if "content" in data and "parts" in data.get("content", {}):
            candidates = [data]
        else:
            return {
                "success": False,
                "error": f"No candidates. Keys: {list(data.keys())}",
                "model_used": model,
            }

    content = candidates[0].get("content", {})
    parts = content.get("parts", [])

    text_response = ""
    image_b64 = None
    mime_type = "image/png"

    for part in parts:
        if "text" in part:
            text_response += part["text"]

        inline = part.get("inlineData") or part.get("inline_data")
        if inline:
            image_b64 = inline.get("data", "")
            mime_type = (
                inline.get("mimeType")
                or inline.get("mime_type")
                or "image/png"
            )

        if "image" in part and isinstance(part["image"], dict):
            img = part["image"]
            image_b64 = img.get("data", "") or img.get("base64", "")
            mime_type = img.get("mimeType") or img.get("mime_type") or "image/png"

    # Check for Markdown data URI embedded in text
    if not image_b64 and text_response:
        match = re.search(
            r'!\[.*?\]\(data:(image/[a-zA-Z]+);base64,([A-Za-z0-9+/=\s]+)\)',
            text_response, re.DOTALL,
        )
        if match:
            mime_type = match.group(1)
            image_b64 = match.group(2).replace("\n", "").replace("\r", "").replace(" ", "")
            text_response = (text_response[:match.start()] + text_response[match.end():]).strip()

    if not image_b64:
        return {
            "success": False,
            "error": (
                f"No image in response. Text: {text_response[:300]}"
                if text_response
                else "Empty response"
            ),
            "text_response": text_response[:500],
            "model_used": model,
        }

    # Clean and validate
    image_b64 = image_b64.replace("\n", "").replace("\r", "").replace(" ", "")

    if len(image_b64) < 100:
        return {
            "success": False,
            "error": f"Image data too short ({len(image_b64)} chars)",
            "model_used": model,
        }

    # Strip data URI prefix if present
    if "," in image_b64[:100]:
        image_b64 = image_b64.split(",", 1)[1]

    logger.info(f"Gemini v2 image: {len(image_b64)} base64 chars, mime={mime_type}")

    return {
        "success": True,
        "image_b64": image_b64,
        "mime_type": mime_type,
        "text_response": text_response,
        "model_used": model,
    }


def _reassemble_sse(raw_text: str) -> dict:
    """Reassemble Gemini SSE stream into a single response."""
    all_text: List[str] = []
    image_chunks: Dict[str, List[str]] = {}
    last_candidate = {}

    for line in raw_text.split("\n"):
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data_str = line.split(":", 1)[1].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        candidates = chunk.get("candidates", [])
        if not candidates:
            continue

        candidate = candidates[0]
        last_candidate = candidate
        parts = candidate.get("content", {}).get("parts", [])

        for part in parts:
            if not part:
                continue
            if "text" in part:
                all_text.append(part["text"])
            inline = part.get("inlineData") or part.get("inline_data")
            if inline:
                mime = inline.get("mimeType") or inline.get("mime_type") or "image/png"
                b64 = inline.get("data", "")
                if b64:
                    image_chunks.setdefault(mime, []).append(b64)
            if "image" in part and isinstance(part["image"], dict):
                img = part["image"]
                b64 = img.get("data", "") or img.get("base64", "")
                mime = img.get("mimeType") or img.get("mime_type") or "image/png"
                if b64:
                    image_chunks.setdefault(mime, []).append(b64)

    merged_parts: List[dict] = []
    full_text = "".join(all_text)
    if full_text:
        merged_parts.append({"text": full_text})

    for mime, chunks in image_chunks.items():
        merged_b64 = "".join(
            c.replace("\n", "").replace("\r", "").replace(" ", "")
            for c in chunks
        )
        if merged_b64:
            merged_parts.append({
                "inlineData": {"mimeType": mime, "data": merged_b64}
            })

    return {
        "candidates": [{
            "content": {"role": "model", "parts": merged_parts},
            **{k: v for k, v in last_candidate.items() if k != "content"},
        }]
    }


def _merge_json_array(items: list) -> dict:
    """Merge JSON array response into single response."""
    all_parts: List[dict] = []
    last_candidate = {}

    for item in items:
        if not isinstance(item, dict):
            continue
        candidates = item.get("candidates", [])
        if candidates:
            candidate = candidates[0]
            last_candidate = candidate
            parts = candidate.get("content", {}).get("parts", [])
            all_parts.extend(p for p in parts if p)

    if not all_parts:
        return items[-1] if items else {}

    return {
        "candidates": [{
            "content": {"role": "model", "parts": all_parts},
            **{k: v for k, v in last_candidate.items() if k != "content"},
        }]
    }


# ============================================================================
# Combined Pipeline v2
# ============================================================================

async def generate_scientific_figure_v2(
    ai_engine,
    method_text: str,
    svg_content: str,
    reference_image_b64: Optional[str] = None,
    prompt_model: Optional[str] = None,
    image_model: str = "gemini-3-pro-image-preview",
    aspect_ratio: str = "16:9",
    image_size: str = "4K",
    custom_prompt: Optional[str] = None,
    elk_graph: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Combined pipeline v2:
      Stage 1: Grok 4 generates verbose design spec (unchanged)
      Stage 2: Narrative compressor → dense paragraph (NEW)
      Stage 3: Gemini generates image from narrative (REWRITTEN)

    No retry mechanism. Fail cleanly.
    """
    settings = get_settings()

    # ── Stage 1: Generate design spec ──
    if custom_prompt:
        design_spec = custom_prompt
        prompt_model_used = "custom"
    else:
        # Import the existing Grok prompt generator
        from .gemini_image_gen import generate_prompt_with_grok

        prompt_result = await generate_prompt_with_grok(
            ai_engine=ai_engine,
            method_text=method_text,
            svg_content=svg_content,
            model=prompt_model,
            reference_image_b64=reference_image_b64,
        )

        if not prompt_result.get("success"):
            from .gemini_image_gen import _build_fallback_prompt
            design_spec = _build_fallback_prompt(method_text, svg_content)
            prompt_model_used = "fallback"
            logger.warning(
                f"Grok prompt failed ({prompt_result.get('error')}), using fallback"
            )
        else:
            design_spec = prompt_result["prompt"]
            prompt_model_used = prompt_result.get("model_used", "unknown")

    # ── Try to render skeleton PNG (fix for Cause 5) ──
    skeleton_png_b64 = None
    try:
        from .gemini_image_gen import _svg_to_png_b64
        skeleton_png_b64 = _svg_to_png_b64(svg_content, target_width=1024)
    except Exception as e:
        logger.warning(f"Skeleton PNG render failed: {e}")

    # ── Stage 2 + 3: Narrative compression + Gemini image gen ──
    image_result = await generate_image_with_gemini_v2(
        svg_content=svg_content,
        design_spec=design_spec,
        method_text=method_text,
        settings=settings,
        model=image_model,
        aspect_ratio=aspect_ratio,
        image_size=image_size,
        elk_graph=elk_graph,
        skeleton_png_b64=skeleton_png_b64,
    )

    return {
        "success": image_result.get("success", False),
        "image_b64": image_result.get("image_b64"),
        "mime_type": image_result.get("mime_type", "image/png"),
        "text_response": image_result.get("text_response", ""),
        "prompt": design_spec,
        "prompt_model_used": prompt_model_used,
        "image_model_used": image_result.get("model_used", image_model),
        "error": image_result.get("error"),
    }