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

=== VISUAL VOCABULARY HINTS ===
The SVG layout may contain RICH visual hints beyond basic boxes. Translate these
into specific drawing instructions in your numbered points:
- "visualShape: circle" + "mathSymbol: tensor_product" → draw a CIRCLE with ⊗ inside
- "fillPattern: hatching" → draw the region with DIAGONAL LINE FILL (like technical drawings)
- "contentGrid: {{rows:3,cols:4,shape:circle}}" → draw a 3×4 grid of small dots INSIDE the node
- "embeddedImage: photo of a dog" → draw/depict the described image inside the node area
- "textOrientation: vertical" → rotate the text label 90 degrees
- "nodeStyle: label" → just text, NO surrounding box
- "nodeStyle: tag" → small colored pill/badge
- "visualEmphasis: highlighted" → make this node visually prominent (thicker border, accent color)
These create REAL academic paper figure quality — not generic flowcharts.

Generate your tier-tagged numbered design points now:"""


async def generate_prompt_with_grok(
    ai_engine: AIEngine,
    method_text: str,
    svg_content: str,
    model: Optional[str] = None,
    reference_image_b64: Optional[str] = None,
    elk_graph: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Step a: Use Grok 4 (or specified model) to reverse-engineer a professional
    prompt from the SVG + method text (+ optional reference image).

    Enhanced with:
      - elk_graph support: user skeleton edits (added/renamed nodes) are
        preserved via _extract_svg_structure()
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
        elk_graph: Optional structured graph data from interactive editor

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

    # Build SVG layout description — use elk_graph if available for richer data
    svg_layout_text = _extract_svg_structure(svg_content, elk_graph=elk_graph)
    if not svg_layout_text.strip():
        svg_layout_text = svg_content[:8000]

    user_content = GROK_PROMPT_ENGINEER_USER.format(
        method_text=method_text,
        svg_content=svg_layout_text,
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

        raw_prompt = result.get("content", "").strip()
        # Strip <think> reasoning blocks, JSON wrappers, code fences
        prompt = _strip_think_and_clean(raw_prompt)
        if not prompt:
            logger.error(
                f"Grok returned empty prompt after cleaning. "
                f"Raw length={len(raw_prompt)}, first 300 chars: {raw_prompt[:300]}"
            )
            return {
                "success": False,
                "error": "Grok returned empty prompt (reasoning-only output with no design points)",
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


def _strip_think_and_clean(raw_output: str) -> str:
    """
    Strip <think>...</think> reasoning blocks and other noise from LLM output.

    Reasoning models (DeepSeek-R1, Grok with thinking, QwQ, etc.) wrap their
    chain-of-thought in <think>...</think> tags. This must be removed before
    using the output as a prompt for another model.

    Also strips:
      - ```json ... ``` code fences that some models wrap output in
      - { "prompt": "..." } JSON wrappers
      - Leading/trailing whitespace
    """
    text = raw_output

    # 1. Remove <think>...</think> blocks (greedy, handles multiline)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # 2. Also handle unclosed <think> (model output truncated mid-thought)
    #    Remove everything from <think> to end of string if no closing tag
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)

    # 3. Strip ```json ... ``` code fence wrappers
    text = re.sub(r'^```(?:json)?\s*\n?', '', text.strip())
    text = re.sub(r'\n?```\s*$', '', text.strip())

    # 4. If the entire output is a JSON object with a "prompt" key, extract it
    text = text.strip()
    if text.startswith('{') and text.endswith('}'):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and 'prompt' in parsed:
                text = parsed['prompt']
        except json.JSONDecodeError:
            pass  # Not valid JSON, keep as-is

    return text.strip()


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
# SVG → PNG Rendering (for Gemini image input)
# ============================================================================

async def _svg_to_png_b64_playwright(svg_content: str, target_width: int = 1024) -> Optional[str]:
    """
    Render SVG string to PNG using Playwright headless Chromium.

    This replaces cairosvg for the critical path because cairosvg CANNOT render
    <image href="data:image/png;base64,..."> tags — it silently drops them.
    Chromium renders everything a browser would: embedded images, CSS, fonts.

    FIX for Break 1: Gemini now sees the sprite-enriched skeleton (not empty boxes).

    Debug: logs sprite count and PNG size so you can verify sprites are in the output.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        logger.warning(
            "playwright not installed — run: pip install playwright && playwright install chromium\n"
            "Falling back to cairosvg (which CANNOT render embedded sprite images)"
        )
        return _svg_to_png_b64_cairosvg(svg_content, target_width)

    # Debug: count sprites in input SVG
    sprite_count = svg_content.count('data:image/png;base64,')
    image_tag_count = svg_content.count('<image ')
    logger.info(
        f"[_svg_to_png_b64_playwright] Input SVG: {len(svg_content)} chars, "
        f"{image_tag_count} <image> tags, {sprite_count} base64 sprites"
    )

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(viewport={"width": target_width, "height": 800})

            # Wrap SVG in minimal HTML — white background, no margin
            html = (
                f'<!DOCTYPE html><html><head><style>'
                f'body{{margin:0;background:#fff;display:flex;justify-content:center}}'
                f'svg{{max-width:{target_width}px;height:auto}}'
                f'</style></head><body>{svg_content}</body></html>'
            )
            await page.set_content(html, wait_until="networkidle")

            # Wait for all <image> elements to load (sprites are data: URIs, should be instant)
            await page.wait_for_timeout(500)

            # Screenshot the SVG element directly for tight cropping
            svg_el = await page.query_selector("svg")
            if svg_el:
                png_bytes = await svg_el.screenshot(type="png")
            else:
                # Fallback: screenshot full page
                png_bytes = await page.screenshot(type="png", full_page=True)

            await browser.close()

        if not png_bytes or len(png_bytes) < 100:
            logger.warning(f"Playwright produced tiny PNG ({len(png_bytes)} bytes)")
            return None

        b64 = base64.b64encode(png_bytes).decode("ascii")
        logger.info(
            f"[_svg_to_png_b64_playwright] Output PNG: {len(png_bytes)} bytes "
            f"({len(png_bytes)/1024:.1f} KB), base64={len(b64)} chars. "
            f"Sprites in input: {sprite_count} — if >0, they SHOULD be visible in the PNG."
        )
        return b64

    except Exception as e:
        logger.warning(f"Playwright SVG→PNG failed: {e}, falling back to cairosvg")
        return _svg_to_png_b64_cairosvg(svg_content, target_width)


def _svg_to_png_b64_cairosvg(svg_content: str, target_width: int = 1024) -> Optional[str]:
    """
    Render SVG to PNG via cairosvg. FAST but CANNOT render <image> data: URIs.

    WARNING: If svg_content contains <image href="data:image/png;base64,...">
    (i.e. sprite-enriched SVGs), those sprites will be INVISIBLE in the output.
    Use _svg_to_png_b64_playwright() instead for sprite-enriched SVGs.

    This is kept as a fast fallback for simple SVGs (no sprites, text+rects only).
    """
    try:
        import cairosvg
    except ImportError:
        logger.warning("cairosvg not installed — cannot render SVG to PNG for Gemini input")
        return None

    # Debug: warn if sprites are present (they'll be lost)
    sprite_count = svg_content.count('data:image/png;base64,')
    if sprite_count > 0:
        logger.warning(
            f"[cairosvg fallback] SVG has {sprite_count} embedded sprites — "
            f"cairosvg WILL NOT render them! Install playwright for full rendering."
        )

    try:
        png_bytes = cairosvg.svg2png(
            bytestring=svg_content.encode("utf-8"),
            output_width=target_width,
        )
        if not png_bytes or len(png_bytes) < 100:
            logger.warning(f"cairosvg produced tiny/empty PNG ({len(png_bytes) if png_bytes else 0} bytes)")
            return None

        b64 = base64.b64encode(png_bytes).decode("ascii")
        logger.info(f"SVG→PNG (cairosvg): {len(png_bytes)} bytes, {len(b64)} base64 chars (width={target_width})")
        return b64
    except Exception as e:
        logger.warning(f"SVG→PNG rendering failed: {e}")
        return None


def _svg_to_png_b64(svg_content: str, target_width: int = 1024) -> Optional[str]:
    """
    Smart router: uses Playwright if sprites are detected, cairosvg otherwise.

    This is the main entry point called by generate_image_with_gemini().
    It checks whether the SVG contains embedded sprite images:
      - If yes → Playwright (headless Chromium, renders everything)
      - If no  → cairosvg (fast, lightweight, good enough for text+rects)
    """
    has_sprites = 'data:image/png;base64,' in svg_content
    node_count = svg_content.count('data-node-id=') or svg_content.count('class="interactive-node"')

    logger.info(
        f"[_svg_to_png_b64] SVG analysis: {len(svg_content)} chars, "
        f"has_sprites={has_sprites}, ~{node_count} nodes, target_width={target_width}"
    )

    if has_sprites:
        logger.info("[_svg_to_png_b64] Sprites detected → routing to Playwright (async)")
        # Playwright is async; run in event loop
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an async context — create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    _svg_to_png_b64_playwright(svg_content, target_width)
                )
                return future.result(timeout=30)
        else:
            return asyncio.run(_svg_to_png_b64_playwright(svg_content, target_width))
    else:
        logger.info("[_svg_to_png_b64] No sprites → using cairosvg (fast path)")
        return _svg_to_png_b64_cairosvg(svg_content, target_width)


# ============================================================================
# Gemini 3 Pro Image Generation — Step b
# ============================================================================

# ============================================================================
# OpenAI Images API path (gpt-image-2, dall-e-3, etc.)
# ============================================================================

_ASPECT_TO_OPENAI_SIZE = {
    "1:1": "1024x1024", "16:9": "1536x1024", "9:16": "1024x1536",
    "4:3": "1536x1024", "3:4": "1024x1536",
}


async def _generate_image_openai_format(
    prompt: str,
    svg_content: str,
    elk_graph: Optional[Dict],
    model: str,
    api_key: str,
    api_base: str,
    aspect_ratio: str = "16:9",
    skeleton_png_b64: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate image using OpenAI /v1/images/generations endpoint.

    Used for gpt-image-2, dall-e-3, and other OpenAI-format image models
    served via tryallai.com or compatible proxies.
    """
    import time
    t0 = time.monotonic()

    base_url = (api_base or "https://api.openai.com").rstrip("/")
    if not base_url.endswith("/v1"):
        endpoint = f"{base_url}/v1/images/generations"
    else:
        endpoint = f"{base_url}/images/generations"

    svg_summary = _extract_svg_structure(svg_content, elk_graph=elk_graph)
    clean_prompt = _strip_think_and_clean(prompt)
    if len(clean_prompt) < 20 and len(prompt) > 100:
        clean_prompt = prompt

    combined_prompt = (
        f"{clean_prompt}\n\n"
        f"=== SPATIAL LAYOUT REFERENCE ===\n"
        f"{svg_summary}\n\n"
        f"IMPORTANT: Create a high-quality, publication-ready scientific figure image. "
        f"The spatial layout (node positions, sizes, connection topology) MUST closely "
        f"follow the layout reference above. Use academic paper figure style: white background, "
        f"thin dark borders, monochrome arrows, no drop shadows, print-friendly."
    )

    size = _ASPECT_TO_OPENAI_SIZE.get(aspect_ratio, "1536x1024")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    body: Dict[str, Any] = {
        "model": model,
        "prompt": combined_prompt,
        "n": 1,
        "size": size,
    }

    logger.info(
        f"OpenAI image gen: model={model}, size={size}, "
        f"prompt_len={len(combined_prompt)}, endpoint={endpoint[:60]}..."
    )

    try:
        async with httpx.AsyncClient(timeout=600.0, follow_redirects=True) as client:
            response = await client.post(endpoint, json=body, headers=headers)

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        if response.status_code != 200:
            error_text = response.text[:500]
            logger.error(f"OpenAI image API error {response.status_code}: {error_text}")
            return {
                "success": False,
                "error": f"API error {response.status_code}: {error_text}",
                "model_used": model,
                "endpoint_used": endpoint,
            }

        data = response.json()
        items = data.get("data", [])
        if not items:
            return {"success": False, "error": "API returned empty data array", "model_used": model}

        item = items[0]
        image_b64 = item.get("b64_json", "")

        if not image_b64 and item.get("url"):
            logger.info(f"Fetching image from URL: {item['url'][:80]}...")
            async with httpx.AsyncClient(timeout=60.0) as dl_client:
                img_resp = await dl_client.get(item["url"])
                if img_resp.status_code == 200:
                    import base64 as b64mod
                    image_b64 = b64mod.b64encode(img_resp.content).decode("ascii")

        if not image_b64:
            return {"success": False, "error": "No image data in API response", "model_used": model}

        usage = data.get("usage", {})
        logger.info(
            f"OpenAI image gen success: model={model}, {elapsed_ms}ms, "
            f"b64_len={len(image_b64)}, usage={usage}"
        )

        return {
            "success": True,
            "image_b64": image_b64,
            "mime_type": "image/png",
            "text_response": "",
            "model_used": model,
            "elapsed_ms": elapsed_ms,
            "usage": usage,
        }

    except httpx.TimeoutException:
        return {"success": False, "error": f"Timeout after 600s calling {model}", "model_used": model}
    except Exception as e:
        logger.exception(f"OpenAI image gen failed: {e}")
        return {"success": False, "error": str(e), "model_used": model}


def _collect_sprites_from_elk(elk_graph: Dict) -> List[tuple]:
    """
    Walk ELK graph tree and collect (nodeId, label, base64) for nodes with spriteRef.url.

    Returns list of (node_id, label, b64_data) tuples, sorted by node order.
    Used as fallback when frontend doesn't send sprite_images separately.

    Debug: prints each found sprite so developer can trace the data flow.
    """
    results = []

    def _walk(nodes):
        if not isinstance(nodes, list):
            return
        for node in nodes:
            ref = node.get("spriteRef") or {}
            url = ref.get("url", "")
            if url and isinstance(url, str) and url.startswith("data:image/"):
                b64 = url.split(",", 1)[1] if "," in url else ""
                if b64:
                    label = ""
                    labels = node.get("labels")
                    if isinstance(labels, list) and labels:
                        label = labels[0].get("text", node.get("id", ""))
                    else:
                        label = node.get("id", "unknown")
                    results.append((node.get("id", ""), label, b64))
                    logger.debug(
                        f"[_collect_sprites_from_elk] Found sprite: "
                        f"node={node.get('id')}, label='{label}', b64_len={len(b64)}"
                    )
            children = node.get("children")
            if isinstance(children, list):
                _walk(children)

    _walk(elk_graph.get("children", []))
    logger.info(f"[_collect_sprites_from_elk] Collected {len(results)} sprites from elk_graph")
    return results


async def generate_image_with_gemini(
    svg_content: str,
    prompt: str,
    settings: Optional[Settings] = None,
    model: str = "gemini-3-pro-image-preview",
    aspect_ratio: str = "16:9",
    image_size: str = "4K",
    elk_graph: Optional[Dict] = None,
    skeleton_png_b64: Optional[str] = None,
    **kwargs,  # sprite_images, skeleton_media_resolution, sprite_media_resolution
) -> Dict[str, Any]:
    """
    Step b: Use Gemini 3 Pro Image model (via tryallai.com proxy)
    to generate a publication-quality scientific figure.

    Sends Grok's FULL design specification — NO compression. User skeleton edits
    (added/renamed nodes) are preserved via _extract_svg_structure().

    Args:
        svg_content: The ELK-generated SVG (for structure extraction + PNG render)
        prompt: The detailed prompt (from Grok 4 or user)
        settings: Backend settings
        model: Gemini image model name
        aspect_ratio: Image aspect ratio
        image_size: Output size (e.g. "4K")
        elk_graph: Optional structured graph data from interactive editor
        skeleton_png_b64: Pre-rendered skeleton PNG (base64), if available

    Returns:
        dict with success, image_b64, mime_type, text_response, model_used
    """
    if settings is None:
        settings = get_settings()

    api_key = settings.GEMINI_API_KEY
    api_base = settings.GEMINI_API_BASE
    is_direct_google_api = not api_base  # Moved early: needed for media_resolution on request parts

    if not api_key:
        return {"success": False, "error": "GEMINI_API_KEY not configured"}

    # ═══════════════════════════════════════════════════════════════════
    #  OpenAI Images API path (gpt-image-2, dall-e-3, etc.)
    #  POST /v1/images/generations — prompt-only, no inline image input.
    # ═══════════════════════════════════════════════════════════════════
    is_openai_image_model = (
        model.startswith("gpt-image")
        or model.startswith("dall-e")
        or model.startswith("gpt-4o")
    )

    if is_openai_image_model:
        return await _generate_image_openai_format(
            prompt=prompt,
            svg_content=svg_content,
            elk_graph=elk_graph,
            model=model,
            api_key=api_key,
            api_base=api_base,
            aspect_ratio=aspect_ratio,
            skeleton_png_b64=skeleton_png_b64,
        )

    # Determine endpoint
    # Proxy format: {base}/v1beta/models/{model}:generateContent
    # Direct Google: https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
    if api_base:
        base_url = api_base.rstrip("/")
        # Remove trailing /v1 if present (common misconfiguration)
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
            logger.info(f"Stripped /v1 suffix from GEMINI_API_BASE: {api_base} → {base_url}")
        # Note: we use /v1beta for Gemini native format, not /v1 (OpenAI format)
        endpoint = f"{base_url}/v1beta/models/{model}:generateContent"
        logger.info(f"Using proxy endpoint: {endpoint}")
    else:
        # Direct Google API
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        logger.info(f"Using direct Google API endpoint: {endpoint}")

    # Build the request body (Gemini native format)
    # Extract RICH layout info using elk_graph (preferred) or SVG parsing (fallback)
    svg_summary = _extract_svg_structure(svg_content, elk_graph=elk_graph)
    # Safety net: strip any <think> blocks or JSON wrappers that may have
    # leaked through from reasoning models (e.g. Grok, DeepSeek-R1)
    clean_prompt = _strip_think_and_clean(prompt)
    if len(clean_prompt) < 20 and len(prompt) > 100:
        logger.warning(
            f"Prompt shrank drastically after cleaning ({len(prompt)} → {len(clean_prompt)} chars). "
            f"Original may have been reasoning-only. Using fallback."
        )
        clean_prompt = prompt  # Fall back to original if cleaning removed everything meaningful
    combined_prompt = (
        f"Generate an image: {clean_prompt}\n\n"
        f"Layout structure reference:\n{svg_summary}\n\n"
        f"IMPORTANT: You MUST generate and return an IMAGE. "
        f"Create a high-quality, publication-ready scientific figure image. "
        f"The spatial layout (node positions, sizes, connection topology) MUST closely "
        f"follow the reference layout described above and shown in the attached skeleton image."
    )

    # ── Build request parts: skeleton image first, then text ──
    request_parts: List[Dict[str, Any]] = []

    # Use pre-rendered skeleton PNG or render fresh
    if not skeleton_png_b64:
        skeleton_png_b64 = _svg_to_png_b64(svg_content, target_width=1024)

    # ═══ media_resolution support (Gemini 3 feature) ═══
    # skeleton gets HIGH (1120 tokens) — Gemini needs to see exact layout
    # per-node sprites get LOW (280 tokens) — just style/texture reference
    skeleton_res = kwargs.get("skeleton_media_resolution", "media_resolution_high")
    sprite_res = kwargs.get("sprite_media_resolution", "media_resolution_low")

    if skeleton_png_b64:
        request_parts.append({
            "text": (
                "Below is the SKELETON LAYOUT image showing the exact spatial arrangement "
                "of nodes and connections that the user has confirmed. Your generated figure "
                "MUST preserve this exact layout — same node positions, same connection "
                "topology, same spatial relationships. Enhance it into a professional, "
                "publication-quality scientific figure while keeping the structure identical."
            )
        })
        skeleton_part: Dict[str, Any] = {
            "inlineData": {
                "mimeType": "image/png",
                "data": skeleton_png_b64,
            }
        }
        # Add media_resolution for Gemini 3 (direct Google API only)
        if is_direct_google_api and skeleton_res:
            skeleton_part["media_resolution"] = {"level": skeleton_res}
        request_parts.append(skeleton_part)
        logger.info(
            f"Attached skeleton PNG as visual reference (resolution={skeleton_res}, "
            f"b64_len={len(skeleton_png_b64)})"
        )
    else:
        logger.warning("Could not render skeleton PNG — Gemini will rely on text description only")

    # ═══ Per-node sprite images (Gemini 3 Pro Image: up to 14 images/prompt) ═══
    # Each sprite is sent as a separate inlineData part with LOW resolution
    # (280 tokens each vs 1120 for HIGH). 13 sprites × 280 = 3,640 tokens.
    # This gives Gemini visual style references for each component.
    sprite_images = kwargs.get("sprite_images") or []
    if sprite_images and isinstance(sprite_images, list):
        sprite_count = min(len(sprite_images), 13)  # 13 sprites + 1 skeleton = 14 max
        logger.info(
            f"Attaching {sprite_count} per-node sprite images "
            f"(resolution={sprite_res}, max=13)"
        )

        # Add a text description of what follows
        sprite_labels = [s.get("label", s.get("nodeId", "?")) for s in sprite_images[:sprite_count]]
        request_parts.append({
            "text": (
                f"Below are {sprite_count} individual component illustrations from the diagram. "
                f"Components: {', '.join(sprite_labels)}. "
                f"Use these as style references — each component in your final figure should "
                f"maintain a similar visual style and level of detail as these sprites."
            )
        })

        for i, sprite_info in enumerate(sprite_images[:sprite_count]):
            sprite_b64 = sprite_info.get("b64", "")
            if not sprite_b64:
                continue
            sprite_part: Dict[str, Any] = {
                "inlineData": {
                    "mimeType": "image/png",
                    "data": sprite_b64,
                }
            }
            if is_direct_google_api and sprite_res:
                sprite_part["media_resolution"] = {"level": sprite_res}
            request_parts.append(sprite_part)
            logger.info(
                f"  Sprite {i+1}/{sprite_count}: "
                f"label='{sprite_info.get('label', '?')}', "
                f"b64_len={len(sprite_b64)}"
            )
    elif elk_graph and isinstance(elk_graph, dict):
        # Fallback: extract sprites from elk_graph if not sent separately
        _sprites_from_graph = _collect_sprites_from_elk(elk_graph)
        if _sprites_from_graph:
            sprite_count = min(len(_sprites_from_graph), 13)
            logger.info(f"Extracted {sprite_count} sprites from elk_graph (fallback)")
            request_parts.append({
                "text": (
                    f"Below are {sprite_count} component illustrations extracted from the graph. "
                    f"Use these as visual references for the final figure."
                )
            })
            for i, (node_id, label, b64) in enumerate(_sprites_from_graph[:sprite_count]):
                sprite_part = {"inlineData": {"mimeType": "image/png", "data": b64}}
                if is_direct_google_api and sprite_res:
                    sprite_part["media_resolution"] = {"level": sprite_res}
                request_parts.append(sprite_part)
                logger.info(f"  Sprite {i+1}/{sprite_count}: label='{label}', b64_len={len(b64)}")

    request_parts.append({"text": combined_prompt})

    # Build generationConfig — imageConfig is only supported by direct Google API,
    # most third-party proxies (avman.ai, tryallai, etc.) reject it with 400 error
    generation_config: Dict[str, Any] = {
        "responseModalities": ["IMAGE", "TEXT"],
    }

    # Only add imageConfig when using direct Google API (no proxy)
    # Proxies like api.avman.ai do not support imageConfig and return 400
    is_direct_google_api = not api_base
    if is_direct_google_api:
        generation_config["imageConfig"] = {
            "aspectRatio": aspect_ratio,
            "imageSize": image_size,
        }
        logger.info(f"Using direct Google API with imageConfig: aspect={aspect_ratio}, size={image_size}")
    else:
        logger.info(f"Using proxy API ({api_base[:30]}...), imageConfig disabled to avoid 400 errors")

    request_body = {
        "contents": [
            {
                "role": "user",
                "parts": request_parts,
            }
        ],
        "generationConfig": generation_config,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",  # Signal we prefer non-SSE response
    }

    # For direct Google API (no api_base), use query param instead of Bearer token
    params = {}
    if is_direct_google_api:
        headers.pop("Authorization", None)
        params["key"] = api_key

    try:
        logger.info(
            f"Gemini image gen: model={model}, "
            f"aspect={aspect_ratio}, size={image_size}, "
            f"endpoint={endpoint[:60]}..."
        )

        # Enable follow_redirects to handle http→https redirects (307/301/302)
        async with httpx.AsyncClient(timeout=600.0, follow_redirects=True) as client:
            response = await client.post(
                endpoint,
                json=request_body,
                headers=headers,
                params=params if params else None,
            )

        if response.status_code != 200:
            error_text = response.text[:500]
            logger.error(f"Gemini image API error {response.status_code}: {error_text}")
            
            # Provide helpful hints for common errors
            hint = None
            if response.status_code == 307:
                # 307 Temporary Redirect — usually means http→https or wrong domain
                location = response.headers.get("location", "")
                hint = (
                    f"Got 307 redirect. Check GEMINI_API_BASE in .env: "
                    f"use https:// not http://, and verify the domain. "
                    f"Redirect location: {location or 'not provided'}"
                )
            elif response.status_code == 301 or response.status_code == 302:
                location = response.headers.get("location", "")
                hint = f"Got redirect to: {location}. Update GEMINI_API_BASE to the correct URL."
            elif response.status_code == 401:
                hint = "Authentication failed. Check GEMINI_API_KEY in .env."
            elif response.status_code == 400:
                hint = "Bad request. The API may not support certain parameters."
            
            error_response = {
                "success": False,
                "error": f"Gemini API error {response.status_code}: {error_text}",
                "model_used": model,
                "endpoint_used": endpoint,
            }
            if hint:
                error_response["hint"] = hint
                logger.error(f"Hint: {hint}")
            
            return error_response

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
            # Only add imageConfig for direct Google API, proxies don't support it
            if is_direct_google_api:
                request_body["generationConfig"]["imageConfig"] = {
                    "aspectRatio": aspect_ratio,
                    "imageSize": image_size,
                }
            # Also simplify the prompt to be more direct
            request_body["contents"][0]["parts"][0]["text"] = (
                f"Generate an image of: {clean_prompt}\n\n"
                f"Output ONLY an image, no text."
            )

            async with httpx.AsyncClient(timeout=600.0, follow_redirects=True) as client2:
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
    except httpx.ConnectError as e:
        # DNS resolution failure or connection refused
        error_str = str(e)
        if "Name or service not known" in error_str or "getaddrinfo failed" in error_str:
            # DNS resolution failed — likely trying to access Google API without proxy
            hint = (
                "DNS resolution failed. If you're in China or behind a firewall, "
                "set GEMINI_API_BASE in .env to use a proxy (e.g. https://api.avman.ai or https://api.tryallai.com/v1)"
            )
            logger.error(f"DNS resolution failed for Gemini API: {e}. Hint: {hint}")
            return {
                "success": False,
                "error": f"Cannot resolve Gemini API hostname: {error_str}",
                "hint": hint,
                "model_used": model,
            }
        else:
            logger.error(f"Connection error to Gemini API: {e}")
            return {
                "success": False,
                "error": f"Connection failed: {error_str}",
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
    elk_graph: Optional[Dict] = None,
    settings: Optional[Settings] = None,
    **kwargs,  # sprite_images, skeleton_media_resolution, sprite_media_resolution
) -> Dict[str, Any]:
    """
    Combined Step 5 pipeline:
      a) Grok 4 generates detailed design spec from SVG + method text
      b) Gemini 3 Pro Image generates the scientific figure

    Sends Grok's FULL output to Gemini (no compression). User skeleton
    edits are preserved via _extract_svg_structure().

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
        elk_graph: Optional structured graph data from interactive editor
        settings: Backend settings (defaults to get_settings())

    Returns:
        dict with success, image_b64, mime_type, prompt, prompt_model, image_model
    """
    if settings is None:
        settings = get_settings()

    # Step a: Generate design spec (or use custom)
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
            elk_graph=elk_graph,
        )

        if not prompt_result.get("success"):
            prompt = _build_fallback_prompt(method_text, svg_content)
            prompt_model_used = "fallback"
            logger.warning(
                f"Grok prompt failed ({prompt_result.get('error')}), using fallback"
            )
        else:
            prompt = prompt_result["prompt"]
            prompt_model_used = prompt_result.get("model_used", "unknown")

    # Pre-render skeleton PNG once for reuse
    skeleton_png_b64 = None
    try:
        skeleton_png_b64 = _svg_to_png_b64(svg_content, target_width=1024)
    except Exception as e:
        logger.warning(f"Skeleton PNG render failed: {e}")

    # ═══════════════════════════════════════════════════════════════════
    #  Stage 2.8: Sprite injection — Gemini fills INDIVIDUAL grid cells
    #
    #  The old flow sent one big prompt to Gemini and got one big image.
    #  Now we first inject per-node sprites into the elk_graph, so Gemini
    #  gets a richer input (SVG with embedded sprites vs bare text boxes).
    #
    #  Gemini interleaved output (responseModalities: [TEXT, IMAGE]):
    #  one request → N independent images as separate parts[].inline_data.
    #  No sprite sheets needed. Natural style consistency within same call.
    #
    #  On failure at any step, affected nodes keep organic blob fallback.
    # ═══════════════════════════════════════════════════════════════════
    if elk_graph and isinstance(elk_graph, dict) and elk_graph.get("children"):
        try:
            from backend.pipeline.topology.node_classifier import classify_nodes
            from backend.pipeline.topology.sprite_injector import inject_sprites

            classify_nodes(elk_graph)

            inj_result = await inject_sprites(
                elk_graph,
                settings=settings,
                model=image_model,
            )
            logger.info(
                "Sprite injection: %d stamped, %d blob fallback (%.0fms)",
                inj_result.refs_stamped,
                inj_result.fallback_to_blob,
                inj_result.elapsed_ms,
            )
        except Exception as e:
            logger.exception("Sprite injection failed, continuing with original SVG")

    # Step b: Generate image (full prompt, no compression)
    image_result = await generate_image_with_gemini(
        svg_content=svg_content,
        prompt=prompt,
        settings=settings,
        model=image_model,
        aspect_ratio=aspect_ratio,
        image_size=image_size,
        elk_graph=elk_graph,
        skeleton_png_b64=skeleton_png_b64,
        **kwargs,  # sprite_images, media_resolution hints
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
        f"Create a high-quality scientific architecture diagram for an academic paper.\n\n"
        f"=== USER'S DETAILED DESCRIPTION (MUST reflect in the figure) ===\n"
        f"{method_text}\n\n"
        f"=== LAYOUT STRUCTURE ===\n"
        f"{hierarchy_instruction}"
        f"Key components include: {components}.\n\n"
        f"Style: Clean, professional vector illustration suitable for a top-tier "
        f"machine learning conference (NeurIPS, ICLR, CVPR). "
        f"Use a professional color palette with soft blues, teals, and warm accents. "
        f"White background, clean sans-serif text labels, rounded rectangles for modules, "
        f"smooth directional arrows for data flow. Bent orthogonal arrows for complex routing. "
        f"For every visual element (icons, avatars, illustrations), describe what it depicts "
        f"in natural language — never use emoji or Unicode symbols. "
        f"The figure should be immediately understandable and visually striking."
    )


def _node_label(node: Dict) -> str:
    """Extract display label from an ELK node dict."""
    labels = node.get("labels", [])
    if labels and isinstance(labels, list) and len(labels) > 0:
        lbl = labels[0]
        if isinstance(lbl, dict):
            return lbl.get("text", node.get("id", "?"))
        return str(lbl)
    return node.get("label", node.get("id", "?"))


def _extract_svg_structure(svg_content: str, elk_graph: Optional[Dict] = None) -> str:
    """
    Extract RICH spatial layout info from SVG + optional ELK graph data.
    Produces a detailed natural-language description of every node's position,
    size, label, and every connection — so the image model can faithfully
    reproduce the user's edited layout.

    Two data sources (combined for maximum fidelity):
      1. elk_graph (preferred): structured JSON with exact x, y, width, height
         per node and edge connection topology from the interactive editor.
      2. svg_content (fallback): parse <rect> and <text> elements from SVG XML.

    Enhanced output includes:
      - Per-node: label, position (x,y), dimensions (w×h), relative placement
      - Per-edge: source → target, with labels if present
      - Spatial relationships: which nodes are left/right/above/below each other
      - Hierarchical nesting depth
      - Overall flow direction detection
    """
    parts: List[str] = []

    # ── Strategy 1: Use elk_graph structured data (from interactive editor) ──
    if elk_graph and isinstance(elk_graph, dict):
        # Support both flat format (nodes) and ELK standard format (children)
        nodes_data = elk_graph.get("nodes", []) or elk_graph.get("children", [])
        edges_data = list(elk_graph.get("edges", []))
        canvas_w = elk_graph.get("width", 800)
        canvas_h = elk_graph.get("height", 600)

        # Collect edges from nested compound nodes too
        def _collect_nested_edges(nodes):
            for n in nodes:
                for e in n.get("edges", []):
                    edges_data.append(e)
                if n.get("children"):
                    _collect_nested_edges(n["children"])
        _collect_nested_edges(nodes_data)

        if nodes_data:
            parts.append(f"=== SPATIAL LAYOUT (canvas {canvas_w}×{canvas_h}px) ===")
            parts.append(f"Total: {len(nodes_data)} nodes, {len(edges_data)} connections")
            parts.append("")

            # ── Per-node spatial description ──
            parts.append("NODE POSITIONS AND SIZES:")
            # Sort by y then x for natural reading order (top-to-bottom, left-to-right)
            sorted_nodes = sorted(nodes_data, key=lambda n: (n.get("y", 0), n.get("x", 0)))

            # Detect layers (nodes at similar y-coordinates)
            y_threshold = 30  # nodes within 30px of each other are "same row"
            layers: List[List[Dict]] = []
            for node in sorted_nodes:
                ny = node.get("y", 0)
                placed = False
                for layer in layers:
                    if abs(layer[0].get("y", 0) - ny) < y_threshold:
                        layer.append(node)
                        placed = True
                        break
                if not placed:
                    layers.append([node])

            for li, layer in enumerate(layers):
                # Sort each layer by x (left to right)
                layer.sort(key=lambda n: n.get("x", 0))
                if len(layers) > 1:
                    parts.append(f"\n  Row/Layer {li + 1} (y ≈ {int(layer[0].get('y', 0))}px):")

                for node in layer:
                    label = _node_label(node)
                    x = int(node.get("x", 0))
                    y = int(node.get("y", 0))
                    w = int(node.get("width", 160))
                    h = int(node.get("height", 60))
                    is_group = node.get("isGroup", False) or node.get("group", False)
                    node_type = "GROUP" if is_group else "node"

                    # ── Build rich visual description ──
                    vis_parts = []
                    vis_parts.append(
                        f'    [{node_type}] "{label}" at ({x},{y}), size {w}×{h}px'
                    )

                    # Visual shape (default rect is omitted for brevity)
                    shape = node.get("visualShape", "")
                    if shape and shape != "rect":
                        vis_parts.append(f"      shape: {shape}")

                    # Node style
                    ns = node.get("nodeStyle", "")
                    if ns == "label":
                        vis_parts.append("      render as: NAKED TEXT (no box, no border)")
                    elif ns == "tag":
                        vis_parts.append("      render as: SMALL COLORED TAG/PILL")

                    # Colors
                    fc = node.get("fillColor", "")
                    sc = node.get("strokeColor", "")
                    if fc:
                        vis_parts.append(f"      fill color: {fc}")
                    if sc:
                        vis_parts.append(f"      border color: {sc}")

                    # Fill pattern (hatching, dots, etc.)
                    fp = node.get("fillPattern", "")
                    if fp and fp != "solid":
                        vis_parts.append(f"      fill pattern: {fp} (diagonal lines / crosshatch texture)")

                    # Math operator symbol
                    ms = node.get("mathSymbol", "")
                    if ms:
                        symbol_map = {
                            "tensor_product": "⊗ (circle with × inside)",
                            "direct_sum": "⊕ (circle with + inside)",
                            "composition": "○ (hollow circle)",
                            "multiply": "× (multiplication cross)",
                            "concatenate": "⊕ or [concat] (merge symbol)",
                        }
                        vis_parts.append(f"      draw math symbol: {symbol_map.get(ms, ms)}")

                    # Content grid (e.g. 3×4 grid of dots inside the node)
                    cg = node.get("contentGrid")
                    if cg and isinstance(cg, dict):
                        rows = cg.get("rows", 2)
                        cols = cg.get("cols", 2)
                        shape = cg.get("shape", "circle")
                        vis_parts.append(
                            f"      content: draw a {rows}×{cols} grid of small {shape}s inside this node"
                        )

                    # Icon description
                    icon = node.get("iconHint", "") or node.get("visualIcon", "")
                    if icon:
                        vis_parts.append(f"      icon: draw a small '{icon}' illustration inside")

                    # Embedded image
                    ei = node.get("embeddedImage", "")
                    if ei:
                        vis_parts.append(f"      embedded image: depict '{ei}'")

                    # Text orientation
                    to = node.get("textOrientation", "")
                    if to == "vertical":
                        vis_parts.append("      text: ROTATED 90° VERTICAL")

                    # Visual emphasis
                    ve = node.get("visualEmphasis", "")
                    if ve and ve != "normal":
                        vis_parts.append(f"      emphasis: {ve}")

                    # Group-specific: borderless or dashed
                    if is_group:
                        if node.get("borderless"):
                            vis_parts.append("      container style: BORDERLESS (no visible border)")
                        else:
                            vis_parts.append("      container style: dashed border rectangle")

                    # Recurse into children for hierarchical description
                    children = node.get("children", [])
                    if children:
                        vis_parts.append(f"      contains {len(children)} child nodes:")
                        for ci, child in enumerate(children):
                            cl = _node_label(child)
                            cns = child.get("nodeStyle", "")
                            cms = child.get("mathSymbol", "")
                            cfp = child.get("fillPattern", "")
                            cfc = child.get("fillColor", "")
                            cvs = child.get("visualShape", "")
                            child_desc = f'        - "{cl}"'
                            extras = []
                            if cns: extras.append(f"style={cns}")
                            if cvs and cvs != "rect": extras.append(f"shape={cvs}")
                            if cms: extras.append(f"symbol={cms}")
                            if cfp and cfp != "solid": extras.append(f"pattern={cfp}")
                            if cfc: extras.append(f"color={cfc}")
                            if child.get("iconHint"): extras.append(f"icon='{child['iconHint']}'")
                            if child.get("contentGrid"): extras.append("has content grid")
                            if extras:
                                child_desc += f" ({', '.join(extras)})"
                            vis_parts.append(child_desc)

                    parts.append("\n".join(vis_parts))

            # ── Connection topology ──
            if edges_data:
                parts.append("\nCONNECTIONS (arrows):")
                node_id_to_label = {}
                def _collect_labels(nodes):
                    for n in nodes:
                        nid = n.get("id", "")
                        nl = _node_label(n)
                        node_id_to_label[nid] = nl
                        for c in n.get("children", []):
                            _collect_labels([c])
                _collect_labels(nodes_data)

                for edge in edges_data:
                    # Support both flat (sourceId/targetId) and ELK (sources/targets)
                    src_id = edge.get("sourceId", "")
                    tgt_id = edge.get("targetId", "")
                    if not src_id and edge.get("sources"):
                        src_id = edge["sources"][0] if edge["sources"] else ""
                    if not tgt_id and edge.get("targets"):
                        tgt_id = edge["targets"][0] if edge["targets"] else ""
                    src = node_id_to_label.get(src_id, src_id or "?")
                    tgt = node_id_to_label.get(tgt_id, tgt_id or "?")

                    edge_label = edge.get("label", "")
                    adv = edge.get("advanced", {}) or {}
                    style_parts = []
                    if adv.get("lineStyle") == "dashed":
                        style_parts.append("dashed line")
                    if adv.get("lineStyle") == "dotted":
                        style_parts.append("dotted line")
                    if adv.get("semanticType"):
                        style_parts.append(adv["semanticType"].replace("_", " "))
                    if adv.get("edgeLabels"):
                        for el in adv["edgeLabels"]:
                            if el.get("text"):
                                edge_label = el["text"]

                    label_str = f' (label: "{edge_label}")' if edge_label else ""
                    style_str = f" [{', '.join(style_parts)}]" if style_parts else ""
                    parts.append(f'    "{src}" → "{tgt}"{label_str}{style_str}')

            # ── Flow direction detection ──
            if len(layers) > 1:
                avg_first_y = sum(n.get("y", 0) for n in layers[0]) / len(layers[0])
                avg_last_y = sum(n.get("y", 0) for n in layers[-1]) / len(layers[-1])
                if avg_last_y > avg_first_y + 50:
                    parts.append("\nFlow direction: TOP → BOTTOM (vertical pipeline)")
                elif avg_first_y > avg_last_y + 50:
                    parts.append("\nFlow direction: BOTTOM → TOP")
            else:
                # Check horizontal flow
                xs = [n.get("x", 0) for n in sorted_nodes]
                if len(xs) > 1 and max(xs) - min(xs) > 200:
                    parts.append("\nFlow direction: LEFT → RIGHT (horizontal pipeline)")

            parts.append("")
            parts.append(
                "IMPORTANT: Reproduce this EXACT spatial layout — node positions, sizes, "
                "and connection topology MUST match the layout described above. "
                "For every visual element (icons, illustrations), describe what it depicts "
                "in natural language — NEVER use emoji or Unicode symbols."
            )

            return "\n".join(parts)

    # ── Strategy 2: Fallback — parse SVG XML (enhanced with per-node extraction) ──
    viewbox = re.search(r'viewBox="0 0 ([0-9.]+) ([0-9.]+)"', svg_content)
    canvas_w = viewbox.group(1) if viewbox else "800"
    canvas_h = viewbox.group(2) if viewbox else "600"

    # Extract ALL rect+text pairs — each node is a rect followed by a text element
    # Pattern: match rect attributes (handle any attribute order)
    rect_pattern = re.compile(
        r'<rect\s[^>]*?(?=x=")x="([^"]*)"[^>]*?(?=y=")y="([^"]*)"'
        r'[^>]*?(?=width=")width="([^"]*)"[^>]*?(?=height=")height="([^"]*)"[^>]*?>',
        re.DOTALL
    )
    # Also handle rects where attributes are in different order
    rect_pattern_alt = re.compile(
        r'<rect\s[^>]*?width="([^"]*)"[^>]*?height="([^"]*)"'
        r'[^>]*?x="([^"]*)"[^>]*?y="([^"]*)"[^>]*?>',
        re.DOTALL
    )

    rects_found = []
    for m in rect_pattern.finditer(svg_content):
        rects_found.append({
            "x": float(m.group(1)), "y": float(m.group(2)),
            "w": float(m.group(3)), "h": float(m.group(4)),
        })
    for m in rect_pattern_alt.finditer(svg_content):
        rects_found.append({
            "x": float(m.group(3)), "y": float(m.group(4)),
            "w": float(m.group(1)), "h": float(m.group(2)),
        })

    # Deduplicate by (x, y)
    seen_coords = set()
    unique_rects = []
    for r in rects_found:
        key = (round(r["x"], 1), round(r["y"], 1))
        if key not in seen_coords:
            seen_coords.add(key)
            unique_rects.append(r)

    # Extract text labels with their positions
    text_pattern = re.compile(
        r'<text\s[^>]*?x="([^"]*)"[^>]*?y="([^"]*)"[^>]*?>([^<]+)</text>'
    )
    text_elements = []
    for m in text_pattern.finditer(svg_content):
        text_elements.append({
            "x": float(m.group(1)), "y": float(m.group(2)),
            "text": m.group(3).strip(),
        })

    # Match each text label to its nearest rect (the label belongs to that node)
    node_descriptions = []
    # Skip the first rect if it's the background (covers full canvas)
    data_rects = unique_rects
    if data_rects and data_rects[0]["w"] > float(canvas_w) * 0.9:
        data_rects = data_rects[1:]

    for rect in data_rects:
        # Find the text element closest to this rect's center
        rx_center = rect["x"] + rect["w"] / 2
        ry_center = rect["y"] + rect["h"] / 2
        best_text = None
        best_dist = float("inf")
        for te in text_elements:
            dist = abs(te["x"] - rx_center) + abs(te["y"] - ry_center)
            if dist < best_dist and dist < max(rect["w"], rect["h"]) * 1.5:
                best_dist = dist
                best_text = te["text"]

        label = best_text or "unlabeled"
        node_descriptions.append({
            "label": label,
            "x": int(rect["x"]), "y": int(rect["y"]),
            "w": int(rect["w"]), "h": int(rect["h"]),
        })

    arrow_count = svg_content.count('marker-end')

    # Detect nesting
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

    parts.append(f"=== SPATIAL LAYOUT (canvas {canvas_w}×{canvas_h}px, parsed from SVG) ===")
    parts.append(f"Total: {len(node_descriptions)} nodes, {arrow_count} connections")
    parts.append(f"Diagram type: {'Architecture (hierarchical)' if is_architecture else 'Flowchart (sequential)'}")
    parts.append("")

    if node_descriptions:
        parts.append("NODE POSITIONS AND SIZES:")
        # Sort by y then x
        node_descriptions.sort(key=lambda n: (n["y"], n["x"]))

        # Detect layers
        y_threshold = 30
        layers: List[List[Dict]] = []
        for nd in node_descriptions:
            placed = False
            for layer in layers:
                if abs(layer[0]["y"] - nd["y"]) < y_threshold:
                    layer.append(nd)
                    placed = True
                    break
            if not placed:
                layers.append([nd])

        for li, layer in enumerate(layers):
            layer.sort(key=lambda n: n["x"])
            if len(layers) > 1:
                parts.append(f"\n  Row/Layer {li + 1} (y ≈ {layer[0]['y']}px):")
            for nd in layer:
                parts.append(
                    f'    [node] "{nd["label"]}" at ({nd["x"]},{nd["y"]}), size {nd["w"]}×{nd["h"]}px'
                )

        if len(layers) > 1:
            parts.append(f"\nFlow direction: TOP → BOTTOM ({len(layers)} layers)")

    if nesting_depth >= 2:
        parts.append(
            f"\nHierarchy depth: {nesting_depth} levels. "
            f"Parent nodes contain child nodes."
        )

    parts.append("")
    parts.append(
        "IMPORTANT: Reproduce this EXACT spatial layout — node positions, sizes, "
        "and the connection topology MUST match the layout described above. "
        "For every visual element (icons, illustrations), describe what it depicts "
        "in natural language — NEVER use emoji or Unicode symbols."
    )

    return "\n".join(parts)