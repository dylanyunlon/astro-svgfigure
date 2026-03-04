"""
Gemini 3 Pro Image Generation — Step 5 of the Forward Pipeline
================================================================
After user confirms the ELK SVG rendering has all components,
this module takes the SVG as input and requests Gemini 3 Pro Image
model to generate a publication-quality scientific figure (PNG/image).

Flow:
  1. (Optional) Grok 4 reverse-engineers a professional prompt from a
     reference image + paper method text
  2. Gemini 3 Pro Image generates the final scientific figure

API format: Gemini native v1beta/models/{model}:generateContent
  - Uses tryallai.com proxy: GEMINI_API_BASE
  - responseModalities: ["TEXT", "IMAGE"]
  - imageConfig: { aspectRatio, imageSize }

GitHub references:
  - gemini-cli-extensions/nanobanana
  - ZeroLu/awesome-nanobanana-pro
"""

from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from .edge_routing_prompts import get_grok_prompt_with_edge_routing
from ..ai_engine import AIEngine
from ..config import Settings, get_settings

logger = logging.getLogger(__name__)


# ============================================================================
# Grok 4 Prompt Engineering — Step a
# ============================================================================

GROK_PROMPT_ENGINEER_SYSTEM = """\
You are an expert AI art director specializing in academic/scientific figure design.
Your task: analyze the provided SVG layout and paper method description,
then generate a DETAILED prompt for an AI image generator (Gemini 3 Pro Image)
to create a publication-quality scientific figure.

Your prompt must cover:
1. **Visual Style**: Clean, professional academic illustration style
   (vector-like, flat design with subtle gradients, Nature/Science quality)
2. **Layout & Composition**: Describe spatial arrangement matching the SVG structure
3. **Core Elements**: Each component/module with shape, color, label
4. **Color Palette**: Professional scientific palette
   (blues, teals, soft oranges for accents, white background)
5. **Typography**: Clean sans-serif labels, readable at print size
6. **Connections**: Arrows, flow lines with clear directionality
7. **Details**: Shadows, borders, rounded corners, padding
8. **Do NOT include**: Any text formatting, markdown, or code fences

Output ONLY the prompt text, nothing else. The prompt should be 200-400 words,
highly specific, and directly usable as input to Gemini 3 Pro Image.
"""

GROK_PROMPT_ENGINEER_USER = """\
Based on this paper method description and SVG layout, generate a detailed
AI image generation prompt for creating a publication-quality scientific figure.

Paper Method:
{method_text}

SVG Layout (describes component positions and connections):
{svg_content}

Generate the detailed prompt now:
"""


async def generate_prompt_with_grok(
    ai_engine: AIEngine,
    method_text: str,
    svg_content: str,
    model: Optional[str] = None,
    reference_image_b64: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Step a: Use Grok 4 (or specified model) to reverse-engineer a professional
    prompt from the SVG + method text (+ optional reference image).

    Args:
        ai_engine: Initialized AIEngine
        method_text: Paper method description
        svg_content: The ELK-generated SVG content
        model: Model to use (defaults to grok-4 via OpenAI-compatible)
        reference_image_b64: Optional base64-encoded reference image

    Returns:
        dict with success, prompt, model_used
    """
    settings = get_settings()
    use_model = model or settings.DEFAULT_PROMPT_MODEL  # grok-4

    user_content = GROK_PROMPT_ENGINEER_USER.format(
        method_text=method_text,
        svg_content=svg_content[:8000],  # Truncate SVG if too long
    )

    # If reference image provided, add it to context
    if reference_image_b64:
        user_content = (
            "I have a reference scientific figure (attached as image) that I want "
            "the generated figure to emulate in style. Analyze its visual style, "
            "color scheme, layout approach, and quality level.\n\n"
            + user_content
        )

    try:
        logger.info(f"Grok prompt engineering: model={use_model}")

        result = await ai_engine.get_completion(
            messages=[
                {"role": "system", "content": get_grok_prompt_with_edge_routing(GROK_PROMPT_ENGINEER_SYSTEM)},
                {"role": "user", "content": user_content},
            ],
            model=use_model,
            temperature=0.7,
            max_tokens=2048,
        )

        prompt = result.get("content", "").strip()
        if not prompt:
            return {
                "success": False,
                "error": "Grok returned empty prompt",
                "model_used": use_model,
            }

        logger.info(f"Grok prompt generated: {len(prompt)} chars")
        return {
            "success": True,
            "prompt": prompt,
            "model_used": result.get("model", use_model),
        }

    except Exception as e:
        logger.error(f"Grok prompt engineering failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "model_used": use_model,
        }


# ============================================================================
# Gemini 3 Pro Image Generation — Step b
# ============================================================================

async def generate_image_with_gemini(
    svg_content: str,
    prompt: str,
    settings: Optional[Settings] = None,
    model: str = "gemini-3-pro-image-preview",
    aspect_ratio: str = "16:9",
    image_size: str = "4K",
) -> Dict[str, Any]:
    """
    Step b: Use Gemini 3 Pro Image model (via tryallai.com proxy)
    to generate a publication-quality scientific figure.

    Uses Gemini native format: v1beta/models/{model}:generateContent
    with responseModalities: ["TEXT", "IMAGE"]

    Args:
        svg_content: The ELK-generated SVG (as context for the figure)
        prompt: The detailed prompt (from Grok 4 or user)
        settings: Backend settings
        model: Gemini image model name
        aspect_ratio: Image aspect ratio
        image_size: Output size (e.g. "4K")

    Returns:
        dict with success, image_b64, mime_type, text_response, model_used
    """
    if settings is None:
        settings = get_settings()

    api_key = settings.GEMINI_API_KEY
    api_base = settings.GEMINI_API_BASE

    if not api_key:
        return {"success": False, "error": "GEMINI_API_KEY not configured"}

    # Determine endpoint
    # tryallai.com proxy: /v1beta/models/{model}:generateContent/ (trailing slash per spec)
    if api_base:
        base_url = api_base.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        endpoint = f"{base_url}/v1beta/models/{model}:generateContent/"
    else:
        # Direct Google API
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    # Build the request body (Gemini native format)
    # Extract meaningful layout info instead of dumping raw SVG XML
    svg_summary = _extract_svg_structure(svg_content)
    combined_prompt = (
        f"Generate an image: {prompt}\n\n"
        f"Layout structure reference:\n{svg_summary}\n\n"
        f"IMPORTANT: You MUST generate and return an IMAGE. "
        f"Create a high-quality, publication-ready scientific figure image."
    )

    request_body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": combined_prompt}
                ],
            }
        ],
        "generationConfig": {
            "responseModalities": ["IMAGE", "TEXT"],
        },
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # For direct Google API (no api_base), use query param instead
    params = {}
    if not api_base:
        headers.pop("Authorization", None)
        params["key"] = api_key

    try:
        logger.info(
            f"Gemini image gen: model={model}, "
            f"aspect={aspect_ratio}, size={image_size}, "
            f"endpoint={endpoint[:60]}..."
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
            logger.error(f"Gemini image API error {response.status_code}: {error_text}")
            return {
                "success": False,
                "error": f"Gemini API error {response.status_code}: {error_text}",
                "model_used": model,
            }

        raw_text = response.text.strip()
        content_type = response.headers.get("content-type", "")

        # tryallai proxy may force SSE streaming even for generateContent
        # Detect and reassemble SSE into a single JSON response
        if raw_text.startswith("data: ") or "text/event-stream" in content_type:
            logger.warning("Gemini image API returned SSE stream — reassembling")
            data = _reassemble_sse_gemini(raw_text)
        else:
            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError as e:
                logger.error(f"Gemini response not JSON: {e}. First 300 chars: {raw_text[:300]}")
                return {
                    "success": False,
                    "error": f"Gemini returned non-JSON response: {raw_text[:200]}",
                    "model_used": model,
                }

        # Debug: log response structure to diagnose proxy issues
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            part_keys = [list(p.keys()) for p in parts[:5]]
            logger.info(f"Gemini response: {len(candidates)} candidates, parts keys: {part_keys}")
        else:
            logger.warning(f"Gemini response has no candidates. Top keys: {list(data.keys())[:10]}")

        result = _parse_gemini_image_response(data, model)

        # If no image returned, retry with IMAGE-only modality
        if not result.get("success") and "no image data" in result.get("error", "").lower():
            logger.warning(
                f"Gemini returned text-only, retrying with IMAGE-only modality. "
                f"Text response: {result.get('text_response', '')[:200]}"
            )
            request_body["generationConfig"]["responseModalities"] = ["IMAGE"]
            # Also simplify the prompt to be more direct
            request_body["contents"][0]["parts"][0]["text"] = (
                f"Generate an image of: {prompt}\n\n"
                f"Output ONLY an image, no text."
            )

            async with httpx.AsyncClient(timeout=600.0) as client2:
                response2 = await client2.post(
                    endpoint,
                    json=request_body,
                    headers=headers,
                    params=params if params else None,
                )

            if response2.status_code == 200:
                raw2 = response2.text.strip()
                ct2 = response2.headers.get("content-type", "")
                if raw2.startswith("data: ") or "text/event-stream" in ct2:
                    data2 = _reassemble_sse_gemini(raw2)
                else:
                    try:
                        data2 = json.loads(raw2)
                    except json.JSONDecodeError:
                        data2 = {}
                result2 = _parse_gemini_image_response(data2, model)
                if result2.get("success"):
                    return result2
                else:
                    logger.error(f"Gemini IMAGE-only retry also failed: {result2.get('error')}")
                    # Return original error with retry info
                    result["error"] += f" (IMAGE-only retry also failed: {result2.get('error', '')})"
            else:
                logger.error(f"Gemini IMAGE-only retry HTTP error: {response2.status_code}")

        return result

    except httpx.TimeoutException:
        return {
            "success": False,
            "error": "Gemini image generation timed out (120s). Try a simpler prompt.",
            "model_used": model,
        }
    except Exception as e:
        logger.error(f"Gemini image gen failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "model_used": model,
        }


def _reassemble_sse_gemini(raw_text: str) -> dict:
    """
    Reassemble Gemini SSE streaming response into a single generateContent response.

    SSE format from proxy:
      data: {"candidates":[{"content":{"parts":[{"text":"..."}]}}]}
      data: {"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"image/png","data":"base64chunk"}}]}}]}
      data: [DONE]

    We need to merge all parts across chunks into one response.
    """
    all_parts: List[dict] = []
    last_candidate = {}

    for line in raw_text.split("\n"):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
            candidates = chunk.get("candidates", [])
            if candidates:
                candidate = candidates[0]
                last_candidate = candidate
                content = candidate.get("content", {})
                parts = content.get("parts", [])
                for part in parts:
                    # Merge: if this is a continuation of inline_data, append to last
                    if part:
                        all_parts.append(part)
        except json.JSONDecodeError:
            continue

    # Rebuild a single response
    merged = {
        "candidates": [{
            "content": {
                "role": "model",
                "parts": all_parts,
            },
            **{k: v for k, v in last_candidate.items() if k != "content"},
        }]
    }

    logger.info(f"SSE reassembled: {len(all_parts)} parts")
    return merged


def _parse_gemini_image_response(data: dict, model: str) -> Dict[str, Any]:
    """
    Parse Gemini generateContent response which may contain both text and image parts.

    Response format (standard):
    {
      "candidates": [{
        "content": {
          "role": "model",
          "parts": [
            { "text": "..." },
            { "inlineData": { "mimeType": "image/png", "data": "base64..." } }
          ]
        }
      }]
    }

    Some proxies may use snake_case: inline_data, mime_type
    """
    candidates = data.get("candidates", [])
    if not candidates:
        # Some proxies wrap in a different structure
        if "content" in data and "parts" in data.get("content", {}):
            candidates = [data]
        else:
            return {
                "success": False,
                "error": f"Gemini returned no candidates. Response keys: {list(data.keys())}",
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

        # Handle both camelCase and snake_case
        inline = part.get("inlineData") or part.get("inline_data")
        if inline:
            image_b64 = inline.get("data", "")
            mime_type = inline.get("mimeType") or inline.get("mime_type") or "image/png"

        # Some formats nest under "image" key
        if "image" in part and isinstance(part["image"], dict):
            image_b64 = part["image"].get("data", "") or part["image"].get("base64", "")
            mime_type = part["image"].get("mimeType") or part["image"].get("mime_type") or "image/png"

    # Format 3: tryallai proxy embeds image as Markdown data URI in text
    # e.g. ![image](data:image/jpeg;base64,/9j/4AAQ...)
    if not image_b64 and text_response:
        match = re.search(
            r'!\[.*?\]\(data:(image/[a-zA-Z]+);base64,([A-Za-z0-9+/=\s]+)\)',
            text_response,
            re.DOTALL,
        )
        if match:
            mime_type = match.group(1)
            image_b64 = match.group(2).replace("\n", "").replace("\r", "").replace(" ", "")
            # Remove the image markdown from text_response
            text_response = text_response[:match.start()] + text_response[match.end():]
            text_response = text_response.strip()
            logger.info(f"Extracted image from Markdown data URI: mime={mime_type}, {len(image_b64)} chars")

    if not image_b64:
        logger.error(f"Gemini no image. Text response: {text_response[:500]}")
        return {
            "success": False,
            "error": f"Gemini returned no image. Model said: {text_response[:300]}" if text_response else "Gemini returned empty response",
            "text_response": text_response[:500],
            "raw_parts_keys": [list(p.keys()) for p in parts[:5]],
            "model_used": model,
        }

    logger.info(
        f"Gemini image generated: {len(image_b64)} base64 chars, "
        f"mime={mime_type}"
    )

    return {
        "success": True,
        "image_b64": image_b64,
        "mime_type": mime_type,
        "text_response": text_response,
        "model_used": model,
    }


# ============================================================================
# Combined Pipeline: Grok 4 prompt + Gemini 3 image
# ============================================================================

async def generate_scientific_figure(
    ai_engine: AIEngine,
    method_text: str,
    svg_content: str,
    reference_image_b64: Optional[str] = None,
    prompt_model: Optional[str] = None,
    image_model: str = "gemini-3-pro-image-preview",
    aspect_ratio: str = "16:9",
    image_size: str = "4K",
    custom_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Combined Step 5 pipeline:
      a) Grok 4 generates detailed prompt from SVG + method text
      b) Gemini 3 Pro Image generates the scientific figure

    Args:
        ai_engine: Initialized AIEngine
        method_text: Paper method description
        svg_content: ELK-generated SVG
        reference_image_b64: Optional reference image (base64)
        prompt_model: Model for prompt engineering (default: grok-4)
        image_model: Gemini image model
        aspect_ratio: Image aspect ratio
        image_size: Image size
        custom_prompt: Skip Grok and use this prompt directly

    Returns:
        dict with success, image_b64, mime_type, prompt, prompt_model, image_model
    """
    settings = get_settings()

    # Step a: Generate prompt (or use custom)
    if custom_prompt:
        prompt = custom_prompt
        prompt_model_used = "custom"
    else:
        prompt_result = await generate_prompt_with_grok(
            ai_engine=ai_engine,
            method_text=method_text,
            svg_content=svg_content,
            model=prompt_model,
            reference_image_b64=reference_image_b64,
        )

        if not prompt_result.get("success"):
            # Fallback: build a basic prompt ourselves
            prompt = _build_fallback_prompt(method_text, svg_content)
            prompt_model_used = "fallback"
            logger.warning(
                f"Grok prompt failed ({prompt_result.get('error')}), using fallback"
            )
        else:
            prompt = prompt_result["prompt"]
            prompt_model_used = prompt_result.get("model_used", "unknown")

    # Step b: Generate image
    image_result = await generate_image_with_gemini(
        svg_content=svg_content,
        prompt=prompt,
        settings=settings,
        model=image_model,
        aspect_ratio=aspect_ratio,
        image_size=image_size,
    )

    return {
        "success": image_result.get("success", False),
        "image_b64": image_result.get("image_b64"),
        "mime_type": image_result.get("mime_type", "image/png"),
        "text_response": image_result.get("text_response", ""),
        "prompt": prompt,
        "prompt_model_used": prompt_model_used,
        "image_model_used": image_result.get("model_used", image_model),
        "error": image_result.get("error"),
    }


def _build_fallback_prompt(method_text: str, svg_content: str) -> str:
    """Build a basic prompt when Grok is unavailable."""
    # Extract labels from SVG to understand components
    labels = re.findall(r'>([^<]{2,50})<', svg_content)
    unique_labels = list(dict.fromkeys(labels))[:20]  # Deduplicate, max 20
    components = ", ".join(unique_labels) if unique_labels else "neural network components"

    return (
        f"Create a high-quality scientific architecture diagram for an academic paper. "
        f"The figure should show: {method_text[:500]}. "
        f"Key components include: {components}. "
        f"Style: Clean, professional vector illustration suitable for a top-tier "
        f"machine learning conference (NeurIPS, ICLR, CVPR). "
        f"Use a professional color palette with soft blues, teals, and warm accents. "
        f"White background, clean sans-serif text labels, rounded rectangles for modules, "
        f"smooth directional arrows for data flow. "
        f"The figure should be immediately understandable and visually striking."
    )


def _extract_svg_structure(svg_content: str) -> str:
    """
    Extract meaningful layout info from SVG for the image generation prompt.
    Converts SVG XML into human-readable description (image models work better
    with natural language than raw markup).
    """
    labels = re.findall(r'>([^<]{2,50})<', svg_content)
    unique_labels = list(dict.fromkeys(l.strip() for l in labels if l.strip()))[:30]

    rects = re.findall(
        r'<rect[^>]*x="([^"]*)"[^>]*y="([^"]*)"[^>]*width="([^"]*)"[^>]*height="([^"]*)"',
        svg_content
    )

    viewbox = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg_content)
    canvas_w = viewbox.group(1) if viewbox else "800"
    canvas_h = viewbox.group(2) if viewbox else "600"

    arrow_count = svg_content.count('marker-end')
    node_count = max(0, len(rects) - 1)  # subtract background rect

    parts = [
        f"Canvas: {canvas_w}x{canvas_h} pixels",
        f"Components ({node_count}): {', '.join(unique_labels[:15])}",
        f"Connections: {arrow_count} directional arrows",
    ]

    if rects and len(rects) > 2:
        ys = sorted(set(float(r[1]) for r in rects[1:]))
        if len(ys) > 1:
            parts.append(f"Layout: {len(ys)} layers arranged vertically (top to bottom flow)")

    return "\n".join(parts)