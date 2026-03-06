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
then generate a DETAILED, NUMBERED-POINT prompt for an AI image generator
(Gemini 3 Pro Image) to create a publication-quality scientific figure.

=== STEP 1: COMPLEXITY ANALYSIS ===
First, analyze the input to determine the COMPLEXITY TIER.
Count identifiable components from the SVG and method text:
- nodes, modules, layers, sub-components, connections, annotations, icons, labels

Based on your count, select EXACTLY ONE tier:
  TIER-20: Simple (≤15 identifiable components). Output exactly 20 numbered design points.
  TIER-40: Medium (16-30 components). Output exactly 40 numbered design points.
  TIER-60: Complex (31-50 components). Output exactly 60 numbered design points.
  TIER-80: Very Complex (51+ components, or deep nesting ≥3 levels). Output exactly 80 numbered design points.

You MUST state your chosen tier at the very top of your output, e.g.:
  [TIER-40: 40 design points]

=== STEP 2: ARCHITECTURE vs FLOWCHART DETECTION ===
Detect the diagram type from the input:

(A) ARCHITECTURE DIAGRAM (hierarchical, nested, spatial grouping):
  - Parent-child node relationships MUST use nested compound groups
  - Sibling nodes at the same level MUST share a BORDERLESS background region
    (e.g., Father node contains Child-1, Child-2, Child-3 → all children share
     a subtle, borderless, semi-transparent background to visually group them)
  - Grandchild / deep nesting: each nesting level gets its own slightly different
    background tint (no hard borders) to show hierarchy depth
  - Neural-network level multi-layer nesting is expected: describe EACH level explicitly

(B) FLOWCHART / PIPELINE (sequential, mostly linear):
  - Emphasize directional flow (arrows, step numbers)
  - Use clear lane separation if parallel paths exist

=== STEP 3: NUMBERED DESIGN POINTS ===
Generate EXACTLY the number of design points matching your chosen tier.
Each point must be specific and actionable for the image generator.

Cover these categories across your points:
  - Global style, canvas, background
  - EACH component: shape, size, position, color, label text, icon/illustration
  - EACH parent-child / sibling / grandchild grouping relationship
  - EACH connection: arrow type (straight/bent/orthogonal/curved), style (solid/dashed),
    color, label, direction
  - Typography, spacing, shadows, padding
  - Borderless grouping backgrounds for hierarchical layers
  - Dynamic visual elements: describe what each icon/illustration SHOULD DEPICT
    using natural language (e.g., "a small microscope illustration", "a DNA helix icon",
    "a brain-shaped motif") so the image model generates them from the description.
    NEVER use hardcoded emoji or Unicode symbols — always describe visuals in words
    so Gemini 3 Pro Image can generate them natively.

=== OUTPUT FORMAT ===
[TIER-{N}: {N} design points]
Point 1: {specific design instruction}
Point 2: {specific design instruction}
...
Point {N}: {specific design instruction}

Output ONLY the numbered points with the tier header. No markdown, no code fences,
no explanations outside the points. Each point should be 1-3 sentences.
"""

GROK_PROMPT_ENGINEER_USER = """\
Analyze this paper method + SVG layout. Determine the complexity tier (20/40/60/80),
detect whether this is an architecture diagram or flowchart, then generate the
corresponding number of precise, numbered design points for Gemini 3 Pro Image.

Paper Method:
{method_text}

SVG Layout (component positions and connections):
{svg_content}

IMPORTANT RULES:
- Count all identifiable components/modules/connections to pick the right tier
- If architecture diagram: describe parent→child→grandchild nesting, sibling groupings
  with BORDERLESS background regions, and multi-level hierarchy explicitly
- For EVERY visual element (icons, illustrations, avatars): describe what it depicts
  in natural language words so the image model generates it. NEVER write emoji or
  Unicode symbols. Example: instead of a microscope emoji, write "a small detailed
  microscope illustration in flat vector style".
- Bent/orthogonal arrows for complex routing, curved arrows for skip connections
- Each numbered point must be concrete and directly actionable

Generate your tier-tagged numbered design points now:"""


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

    Enhanced with:
      - Complexity tier detection (20/40/60/80 design points)
      - Architecture vs flowchart auto-detection
      - Hierarchical relationship preservation (parent-child-grandchild)
      - Dynamic visual descriptions (no hardcoded emoji)
      - Numbered-point structured output

    Args:
        ai_engine: Initialized AIEngine
        method_text: Paper method description
        svg_content: The ELK-generated SVG content
        model: Model to use (defaults to grok-4 via OpenAI-compatible)
        reference_image_b64: Optional base64-encoded reference image

    Returns:
        dict with success, prompt, model_used, tier, component_count
    """
    settings = get_settings()
    use_model = model or settings.DEFAULT_PROMPT_MODEL  # grok-4

    # ── Pre-analysis: estimate complexity from SVG structure ────────────
    complexity_info = _analyze_svg_complexity(svg_content)
    suggested_tier = complexity_info["suggested_tier"]
    component_count = complexity_info["component_count"]
    is_architecture = complexity_info["is_architecture"]
    nesting_depth = complexity_info["nesting_depth"]

    # Inject complexity hint into user prompt
    complexity_hint = (
        f"\n\n[COMPLEXITY HINT from pre-analysis: ~{component_count} components detected, "
        f"nesting depth={nesting_depth}, "
        f"diagram type={'ARCHITECTURE (hierarchical)' if is_architecture else 'FLOWCHART (sequential)'}. "
        f"Suggested tier: TIER-{suggested_tier}. You may adjust if your analysis differs.]\n"
    )

    user_content = GROK_PROMPT_ENGINEER_USER.format(
        method_text=method_text,
        svg_content=svg_content[:8000],  # Truncate SVG if too long
    ) + complexity_hint

    # If reference image provided, add it to context
    if reference_image_b64:
        user_content = (
            "I have a reference scientific figure (attached as image) that I want "
            "the generated figure to emulate in style. Analyze its visual style, "
            "color scheme, layout approach, and quality level.\n\n"
            + user_content
        )

    try:
        logger.info(
            f"Grok prompt engineering: model={use_model}, "
            f"pre-analysis: {component_count} components, tier={suggested_tier}, "
            f"arch={is_architecture}, depth={nesting_depth}"
        )

        result = await ai_engine.get_completion(
            messages=[
                {"role": "system", "content": get_grok_prompt_with_edge_routing(GROK_PROMPT_ENGINEER_SYSTEM)},
                {"role": "user", "content": user_content},
            ],
            model=use_model,
            temperature=0.7,
            max_tokens=4096,  # Increased for larger tier outputs
        )

        prompt = result.get("content", "").strip()
        if not prompt:
            return {
                "success": False,
                "error": "Grok returned empty prompt",
                "model_used": use_model,
            }

        # ── Post-process: validate tier and extract metadata ───────────
        detected_tier = _extract_tier_from_output(prompt)
        if detected_tier is None:
            detected_tier = suggested_tier
            logger.warning(
                f"Grok output missing tier header, using pre-analysis tier={suggested_tier}"
            )

        logger.info(
            f"Grok prompt generated: {len(prompt)} chars, "
            f"tier={detected_tier}, points={_count_numbered_points(prompt)}"
        )

        return {
            "success": True,
            "prompt": prompt,
            "model_used": result.get("model", use_model),
            "tier": detected_tier,
            "component_count": component_count,
            "is_architecture": is_architecture,
            "nesting_depth": nesting_depth,
        }

    except Exception as e:
        logger.error(f"Grok prompt engineering failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "model_used": use_model,
        }


def _analyze_svg_complexity(svg_content: str) -> Dict[str, Any]:
    """
    Pre-analyze SVG content to estimate complexity for tier selection.

    Returns:
        dict with component_count, suggested_tier, is_architecture, nesting_depth
    """
    # Count nodes (rect elements, excluding background)
    rects = re.findall(r'<rect[^>]*>', svg_content)
    rect_count = max(0, len(rects) - 1)  # subtract background

    # Count text labels
    labels = re.findall(r'>([^<]{2,50})<', svg_content)
    unique_labels = list(dict.fromkeys(l.strip() for l in labels if l.strip()))
    label_count = len(unique_labels)

    # Count arrows/connections
    arrow_count = svg_content.count('marker-end')

    # Count groups (potential hierarchy)
    group_count = svg_content.count('<g ')

    # Detect nesting depth from nested <g> tags
    nesting_depth = 0
    depth = 0
    for char_idx in range(len(svg_content) - 2):
        if svg_content[char_idx:char_idx+2] == '<g':
            depth += 1
            nesting_depth = max(nesting_depth, depth)
        elif svg_content[char_idx:char_idx+4] == '</g>':
            depth -= 1

    # Architecture detection: nested groups, compound nodes
    is_architecture = (nesting_depth >= 3) or (group_count > rect_count * 0.5)

    # Total component estimate
    component_count = rect_count + arrow_count + max(0, label_count - rect_count)

    # Tier selection
    if component_count <= 15:
        suggested_tier = 20
    elif component_count <= 30:
        suggested_tier = 40
    elif component_count <= 50:
        suggested_tier = 60
    else:
        suggested_tier = 80

    # Deep nesting always bumps to at least TIER-60
    if nesting_depth >= 3 and suggested_tier < 60:
        suggested_tier = 60

    return {
        "component_count": component_count,
        "suggested_tier": suggested_tier,
        "is_architecture": is_architecture,
        "nesting_depth": nesting_depth,
        "rect_count": rect_count,
        "arrow_count": arrow_count,
        "label_count": label_count,
    }


def _extract_tier_from_output(prompt: str) -> Optional[int]:
    """Extract the tier number from Grok's output header."""
    match = re.search(r'\[TIER-(\d+)', prompt)
    if match:
        tier = int(match.group(1))
        if tier in (20, 40, 60, 80):
            return tier
    return None


def _count_numbered_points(prompt: str) -> int:
    """Count how many numbered points Grok actually generated."""
    points = re.findall(r'(?:^|\n)\s*(?:Point\s+)?(\d+)[.:：]', prompt)
    return len(points)


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
            "imageConfig": {
                "aspectRatio": aspect_ratio,
                "imageSize": image_size,
            },
        },
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",  # Signal we prefer non-SSE response
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
        # Check both prefix and content-type; also handle \r\n line endings
        is_sse = (
            raw_text.startswith("data: ")
            or raw_text.startswith("data:")
            or "text/event-stream" in content_type
            or "\ndata: " in raw_text[:200]  # SSE after initial blank lines
        )
        if is_sse:
            logger.warning("Gemini image API returned SSE stream — reassembling")
            data = _reassemble_sse_gemini(raw_text)
        else:
            try:
                data = json.loads(raw_text)
                # Some proxies return an array of chunks instead of a single object
                if isinstance(data, list):
                    logger.warning(f"Gemini returned JSON array ({len(data)} items) — merging")
                    data = _merge_json_array_response(data)
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
        # Match against multiple possible error patterns
        error_lower = result.get("error", "").lower()
        should_retry = (
            not result.get("success")
            and (
                "no image" in error_lower
                or "returned no image" in error_lower
                or "no image data" in error_lower
                or "empty response" in error_lower
            )
        )
        if should_retry:
            logger.warning(
                f"Gemini returned text-only, retrying with IMAGE-only modality. "
                f"Text response: {result.get('text_response', '')[:200]}"
            )
            request_body["generationConfig"]["responseModalities"] = ["IMAGE"]
            # Keep imageConfig for the retry
            request_body["generationConfig"]["imageConfig"] = {
                "aspectRatio": aspect_ratio,
                "imageSize": image_size,
            }
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
            "error": "Gemini image generation timed out (600s). Try a simpler prompt.",
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

    Key challenges:
      - base64 image data may be split across multiple SSE chunks
      - text parts may also be split across chunks
      - Some proxies send incremental deltas, not full parts each time
    We need to merge all parts across chunks into one coherent response.
    """
    all_text_parts: List[str] = []
    image_data_chunks: Dict[str, List[str]] = {}  # mime_type -> [base64 chunks]
    last_candidate = {}
    current_image_mime = None

    for line in raw_text.split("\n"):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        data_str = line[6:]
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
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        for part in parts:
            if not part:
                continue

            # Text parts
            if "text" in part:
                all_text_parts.append(part["text"])

            # Image parts — handle both camelCase and snake_case
            inline = part.get("inlineData") or part.get("inline_data")
            if inline:
                mime = (
                    inline.get("mimeType")
                    or inline.get("mime_type")
                    or "image/png"
                )
                b64_data = inline.get("data", "")
                if b64_data:
                    current_image_mime = mime
                    if mime not in image_data_chunks:
                        image_data_chunks[mime] = []
                    image_data_chunks[mime].append(b64_data)

            # Some proxies nest under "image" key
            if "image" in part and isinstance(part["image"], dict):
                img = part["image"]
                b64_data = img.get("data", "") or img.get("base64", "")
                mime = img.get("mimeType") or img.get("mime_type") or "image/png"
                if b64_data:
                    current_image_mime = mime
                    if mime not in image_data_chunks:
                        image_data_chunks[mime] = []
                    image_data_chunks[mime].append(b64_data)

    # Rebuild merged parts
    merged_parts: List[dict] = []

    # Add text if present
    full_text = "".join(all_text_parts)
    if full_text:
        merged_parts.append({"text": full_text})

    # Merge all base64 chunks for each image into a single inlineData part
    for mime, chunks in image_data_chunks.items():
        # Concatenate all base64 fragments (strip whitespace that may be present)
        merged_b64 = "".join(
            c.replace("\n", "").replace("\r", "").replace(" ", "")
            for c in chunks
        )
        if merged_b64:
            merged_parts.append({
                "inlineData": {
                    "mimeType": mime,
                    "data": merged_b64,
                }
            })

    # Rebuild a single response
    merged = {
        "candidates": [{
            "content": {
                "role": "model",
                "parts": merged_parts,
            },
            **{k: v for k, v in last_candidate.items() if k != "content"},
        }]
    }

    logger.info(
        f"SSE reassembled: {len(all_text_parts)} text chunks, "
        f"{sum(len(v) for v in image_data_chunks.values())} image chunks, "
        f"{len(merged_parts)} merged parts"
    )
    return merged


def _merge_json_array_response(items: list) -> dict:
    """
    Merge a JSON array of Gemini response chunks into a single response.
    Some proxies return [chunk1, chunk2, ...] instead of SSE or single JSON.
    """
    all_parts: List[dict] = []
    last_candidate = {}

    for item in items:
        if not isinstance(item, dict):
            continue
        candidates = item.get("candidates", [])
        if candidates:
            candidate = candidates[0]
            last_candidate = candidate
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            all_parts.extend(p for p in parts if p)

    if not all_parts:
        # Return the last item as-is if we couldn't parse
        return items[-1] if items else {}

    return {
        "candidates": [{
            "content": {
                "role": "model",
                "parts": all_parts,
            },
            **{k: v for k, v in last_candidate.items() if k != "content"},
        }]
    }


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

    # Clean and validate base64 data
    image_b64 = image_b64.replace("\n", "").replace("\r", "").replace(" ", "")

    # Validate base64 — quick sanity check
    if len(image_b64) < 100:
        logger.error(f"Gemini image base64 too short ({len(image_b64)} chars), likely corrupt")
        return {
            "success": False,
            "error": f"Gemini returned corrupted image data ({len(image_b64)} chars)",
            "text_response": text_response[:500],
            "model_used": model,
        }

    # Try to decode a small sample to verify it's valid base64
    try:
        base64.b64decode(image_b64[:100] + "==")  # test first ~75 bytes
    except Exception:
        # Try removing potential data URI prefix
        if "," in image_b64[:100]:
            image_b64 = image_b64.split(",", 1)[1]

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
    """Build a complexity-aware fallback prompt when Grok is unavailable."""
    # Extract labels from SVG to understand components
    labels = re.findall(r'>([^<]{2,50})<', svg_content)
    unique_labels = list(dict.fromkeys(l.strip() for l in labels if l.strip()))[:30]
    components = ", ".join(unique_labels) if unique_labels else "neural network components"

    # Analyze complexity for tier selection
    complexity = _analyze_svg_complexity(svg_content)
    tier = complexity["suggested_tier"]
    is_arch = complexity["is_architecture"]

    hierarchy_instruction = ""
    if is_arch:
        hierarchy_instruction = (
            "This is an ARCHITECTURE diagram with hierarchical nesting. "
            "Parent nodes contain child nodes visually. Sibling nodes at the same "
            "hierarchical level share a subtle, borderless semi-transparent background "
            "region to show they belong to the same group. "
            "Use different background tints for different nesting depths. "
        )

    return (
        f"[TIER-{tier}: {tier} design points]\n"
        f"Create a high-quality scientific architecture diagram for an academic paper. "
        f"{hierarchy_instruction}"
        f"The figure should show: {method_text[:500]}. "
        f"Key components include: {components}. "
        f"Style: Clean, professional vector illustration suitable for a top-tier "
        f"machine learning conference (NeurIPS, ICLR, CVPR). "
        f"Use a professional color palette with soft blues, teals, and warm accents. "
        f"White background, clean sans-serif text labels, rounded rectangles for modules, "
        f"smooth directional arrows for data flow. Bent orthogonal arrows for complex routing. "
        f"For every visual element (icons, avatars, illustrations), describe what it depicts "
        f"in natural language — never use emoji or Unicode symbols. "
        f"The figure should be immediately understandable and visually striking."
    )


def _extract_svg_structure(svg_content: str) -> str:
    """
    Extract meaningful layout info from SVG for the image generation prompt.
    Converts SVG XML into human-readable description (image models work better
    with natural language than raw markup).

    Enhanced to detect:
      - Hierarchical nesting depth
      - Group/compound node relationships
      - Architecture vs flowchart patterns
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

    # Detect nesting depth from <g> tags
    group_count = svg_content.count('<g ')
    nesting_depth = 0
    depth = 0
    for idx in range(len(svg_content) - 2):
        if svg_content[idx:idx+2] == '<g':
            depth += 1
            nesting_depth = max(nesting_depth, depth)
        elif svg_content[idx:idx+4] == '</g>':
            depth -= 1

    is_architecture = nesting_depth >= 3

    parts = [
        f"Canvas: {canvas_w}x{canvas_h} pixels",
        f"Components ({node_count}): {', '.join(unique_labels[:15])}",
        f"Connections: {arrow_count} directional arrows",
        f"Diagram type: {'Architecture (hierarchical, nested groups)' if is_architecture else 'Flowchart (sequential pipeline)'}",
    ]

    if nesting_depth >= 2:
        parts.append(
            f"Hierarchy depth: {nesting_depth} levels. "
            f"Parent nodes contain child nodes. Sibling nodes share borderless background regions."
        )

    if rects and len(rects) > 2:
        ys = sorted(set(float(r[1]) for r in rects[1:]))
        if len(ys) > 1:
            parts.append(f"Layout: {len(ys)} layers arranged vertically (top to bottom flow)")

    parts.append(
        "IMPORTANT: For every visual element (icons, illustrations, avatars), "
        "describe what it depicts in natural language words — NEVER use emoji or Unicode symbols. "
        "The image generator will create visuals from text descriptions."
    )

    return "\n".join(parts)