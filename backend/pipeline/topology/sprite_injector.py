"""sprite_injector.py — Stage 2.8: Gemini interleaved → per-node sprites.

Gemini responseModalities: ['TEXT','IMAGE'] returns interleaved multi-image
output in ONE request. We describe all sprite nodes in a single prompt, and
Gemini returns N independent images as separate parts[].inline_data — each
one corresponds to one node. No sprite sheets, no rembg, no cropping.

Pipeline:
  classify_nodes()             — already done (marks renderMode='sprite')
  design_prompts()             — M211: per-node description with family coherence
  _call_gemini_interleaved()   — ONE Gemini call → N images + text
  stamp_sprite_ref()           — write spriteRef onto each node in-place

On failure: affected node gets spriteRef.format='stack' → organic blob.
"""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


@dataclass
class SpriteInjectionResult:
    total_sprite_nodes: int = 0
    prompts_designed: int = 0
    images_received: int = 0
    refs_stamped: int = 0
    fallback_to_blob: int = 0
    elapsed_ms: float = 0.0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in [
            "total_sprite_nodes", "prompts_designed", "images_received",
            "refs_stamped", "fallback_to_blob", "elapsed_ms", "errors",
        ]}


def _collect_sprite_nodes(
    node: Dict[str, Any], out: List[Dict[str, Any]],
) -> None:
    children = node.get("children")
    if isinstance(children, list) and children:
        for c in children:
            if isinstance(c, dict):
                _collect_sprite_nodes(c, out)
    else:
        if node.get("renderMode") == "sprite":
            out.append(node)


def _stamp_sprite_ref(
    node: Dict[str, Any],
    image_b64: Optional[str],
    stack_count: int = 3,
) -> None:
    if image_b64:
        node["spriteRef"] = {
            "format": "png",
            "url": f"data:image/png;base64,{image_b64}",
            "bbox": [0, 0, 256, 256],
            "fit": "contain",
            "stackCount": stack_count,
        }
    else:
        node["spriteRef"] = {"format": "stack", "stackCount": stack_count}


def _node_label(node: Dict[str, Any]) -> str:
    labels = node.get("labels", [])
    if labels and isinstance(labels[0], dict):
        return labels[0].get("text", node.get("id", "?"))
    return node.get("id", "?")


def _build_interleaved_prompt(sprite_nodes: List[Dict[str, Any]]) -> str:
    """Build a single prompt that asks Gemini to generate one image per node.

    Gemini interleaved output: when responseModalities includes IMAGE,
    the model can return multiple images in one response, each as a
    separate part with inline_data. We ask it to generate them in order.
    """
    lines = [
        "Generate one small illustration for EACH of the following items, "
        "in the EXACT order listed. Each illustration should be a flat 2D "
        "academic-figure style icon/diagram on a pure white background. "
        "Clean thin outlines, no text labels, no 3D, centered composition. "
        "Generate EXACTLY one image per item, then name it.\n",
    ]
    for i, node in enumerate(sprite_nodes):
        label = _node_label(node)
        hint = node.get("iconHint", "")
        desc = f"{label}"
        if hint:
            desc += f" (visual: {hint})"
        lines.append(f"  {i+1}. {desc}")

    lines.append(
        f"\nGenerate {len(sprite_nodes)} images total, one per item above, "
        "in the same order. Each image should be square, ~256px, white background."
    )
    return "\n".join(lines)


async def _call_gemini_interleaved(
    prompt: str,
    n_expected: int,
    settings: Any,
    model: str,
) -> List[Optional[str]]:
    """Call Gemini with interleaved TEXT+IMAGE output, return list of base64 images.

    Returns a list of length n_expected. Each element is either a base64 PNG
    string or None (if that image was missing from the response).
    """
    from backend.config import get_settings
    if settings is None:
        settings = get_settings()

    api_key = settings.GEMINI_API_KEY
    api_base = settings.GEMINI_API_BASE

    if not api_key:
        logger.error("No GEMINI_API_KEY configured")
        return [None] * n_expected

    # Build endpoint
    if api_base:
        base_url = api_base.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        endpoint = f"{base_url}/v1beta/models/{model}:generateContent"
    else:
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    request_body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
        },
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(endpoint, json=request_body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.exception("Gemini interleaved call failed")
        return [None] * n_expected

    # Parse response: extract all inline_data images in order
    images: List[Optional[str]] = []
    candidates = data.get("candidates", [])
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        for part in parts:
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                images.append(inline["data"])

    logger.info("Gemini returned %d images (expected %d)", len(images), n_expected)

    # Pad or truncate to n_expected
    while len(images) < n_expected:
        images.append(None)
    return images[:n_expected]


async def inject_sprites(
    elk_graph: Dict[str, Any],
    *,
    settings: Optional[Any] = None,
    model: str = "gemini-2.0-flash-preview-image-generation",
    gemini_callable=None,
    skip_generation: bool = False,
) -> SpriteInjectionResult:
    """Run the full sprite pipeline on a composed ELK graph.

    New approach: Gemini interleaved output (one call → N images).
    No sprite sheets. No rembg. No sheet splitting.

    Args:
        elk_graph: The composed ELK graph dict (mutated in place).
        settings: Backend settings (forwarded to Gemini).
        model: Image generation model name.
        gemini_callable: Not used in interleaved mode (kept for API compat).
        skip_generation: If True, stamp all as blob fallback without calling Gemini.
    """
    import time
    t0 = time.monotonic()
    result = SpriteInjectionResult()

    # ── Step 1: Collect sprite nodes ──
    sprite_nodes: List[Dict[str, Any]] = []
    for child in elk_graph.get("children", []):
        if isinstance(child, dict):
            _collect_sprite_nodes(child, sprite_nodes)

    result.total_sprite_nodes = len(sprite_nodes)
    if not sprite_nodes:
        result.elapsed_ms = (time.monotonic() - t0) * 1000
        return result

    if skip_generation:
        for i, node in enumerate(sprite_nodes):
            _stamp_sprite_ref(node, image_b64=None, stack_count=2 + (i % 3))
            result.fallback_to_blob += 1
        result.elapsed_ms = (time.monotonic() - t0) * 1000
        return result

    # ── Step 2: Build interleaved prompt ──
    prompt = _build_interleaved_prompt(sprite_nodes)
    result.prompts_designed = len(sprite_nodes)

    # ── Step 3: One Gemini call → N images ──
    try:
        images = await _call_gemini_interleaved(
            prompt=prompt,
            n_expected=len(sprite_nodes),
            settings=settings,
            model=model,
        )
        result.images_received = sum(1 for img in images if img is not None)
    except Exception as e:
        logger.exception("Gemini interleaved generation failed")
        result.errors.append(str(e))
        images = [None] * len(sprite_nodes)

    # ── Step 4: Stamp each image onto its node ──
    for i, (node, img_b64) in enumerate(zip(sprite_nodes, images)):
        if img_b64:
            _stamp_sprite_ref(node, image_b64=img_b64, stack_count=3)
            result.refs_stamped += 1
        else:
            _stamp_sprite_ref(node, image_b64=None, stack_count=2 + (i % 3))
            result.fallback_to_blob += 1

    result.elapsed_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "Sprite injection: %d nodes, %d stamped, %d blob fallback, %.0fms",
        result.total_sprite_nodes, result.refs_stamped,
        result.fallback_to_blob, result.elapsed_ms,
    )
    return result
