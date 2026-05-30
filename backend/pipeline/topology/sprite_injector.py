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

    CRITICAL: background must be solid pure green #00FF00 so the existing
    chroma-key pipeline (Tier 3 in handle_removebg) can strip it without
    needing paid API keys (remove.bg / Canva).

    Image style: academic figure illustrations like those in FreqSelect,
    Pix2Struct, AdaDR papers — feature maps as colored rectangular blocks,
    encoders/decoders as stacked layer diagrams, attention maps as heatmaps,
    kernels as small grids with colored cells.
    """
    lines = [
        "You are generating illustrations for a scientific/academic paper figure. "
        "Generate one illustration for EACH item below, in EXACT order.\n\n"
        "STYLE REQUIREMENTS (CRITICAL — follow ALL of these):\n"
        "- Background: solid pure green #00FF00 (RGB 0,255,0). NO gradients, NO shadows.\n"
        "- Style: academic paper figure illustration, clean and minimal.\n"
        "- Feature maps / tensors: draw as a colored rectangular slab with subtle "
        "texture patterns (like a heatmap or activation visualization).\n"
        "- Encoders / decoders / models: draw as a stack of colored layers or blocks.\n"
        "- Attention / selection maps: draw as a red-yellow-blue heatmap.\n"
        "- Kernels / filters: draw as a small NxN grid with colored cells.\n"
        "- Input images: draw as a colorful photograph thumbnail (landscape/scene).\n"
        "- Each illustration MUST be 256x256 pixels, square, centered.\n"
        "- No text labels inside the illustration. No 3D effects. Thin clean outlines.\n\n"
        "ITEMS TO ILLUSTRATE:\n",
    ]
    for i, node in enumerate(sprite_nodes):
        label = _node_label(node)
        hint = node.get("iconHint", "")
        family_id = node.get("familyId", "")

        # Build a richer description based on the node's semantic role
        desc = f"{label}"
        if hint:
            desc += f" — visual concept: {hint}"
        if family_id:
            desc += f" [family: {family_id}]"

        lines.append(f"  {i+1}. {desc}")

    lines.append(
        f"\nGenerate EXACTLY {len(sprite_nodes)} images, one per item, in order. "
        "Each 256x256 px. Solid #00FF00 green background. Academic figure style."
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

    # Parse response: extract images from Gemini output.
    #
    # Two formats:
    #   A) Standard Gemini: N separate inlineData parts, one per image.
    #   B) Proxy (tryallai): ONE image containing all N illustrations
    #      arranged in a grid with green #00FF00 separators — a "sprite sheet".
    #      We detect this case, split the sheet into cells, chroma-key each,
    #      and return N individual base64 images.
    import re
    raw_images: List[str] = []  # base64 strings before sheet splitting
    candidates = data.get("candidates", [])
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        for part in parts:
            # Standard Gemini format: inlineData
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                raw_images.append(inline["data"])
                continue
            # Proxy format: base64 embedded in text as markdown image
            text = part.get("text", "")
            if text:
                for m in re.finditer(
                    r'data:image/(?:jpeg|png);base64,([A-Za-z0-9+/=]+)', text
                ):
                    raw_images.append(m.group(1))

    logger.info("Gemini returned %d raw image(s) (expected %d)", len(raw_images), n_expected)

    # ── Sprite sheet detection and splitting ──
    # If we got fewer images than expected (typically 1 sprite sheet for N nodes),
    # attempt to split the sheet into individual cells.
    images: List[Optional[str]] = []

    if 0 < len(raw_images) < n_expected:
        logger.info("Attempting sprite sheet split: %d image(s) → %d cells", len(raw_images), n_expected)
        for sheet_b64 in raw_images:
            cells = _split_sprite_sheet(sheet_b64, n_expected)
            images.extend(cells)
        logger.info("Sheet split produced %d cells", len(images))
    elif len(raw_images) >= n_expected:
        # Standard case: N separate images
        for b64 in raw_images:
            # Still chroma-key each individual image
            cleaned = _chroma_key_single(b64)
            images.append(cleaned)
    # else: 0 images → all None

    # Pad or truncate to n_expected
    while len(images) < n_expected:
        images.append(None)
    return images[:n_expected]


def _split_sprite_sheet(sheet_b64: str, n_expected: int) -> List[Optional[str]]:
    """Split a sprite sheet with green #00FF00 separators into individual cells.

    The sheet is a grid of illustrations separated by green bands.
    We detect green rows/columns, find non-green rectangular cells,
    chroma-key each cell (green → transparent), and return as base64 PNGs.
    """
    import base64
    import io

    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        logger.warning("Pillow/numpy not available — cannot split sprite sheet")
        return [sheet_b64]  # return the whole sheet as one image

    try:
        img_bytes = base64.b64decode(sheet_b64)
        img = Image.open(io.BytesIO(img_bytes))
        arr = np.array(img.convert("RGB"))
        h, w = arr.shape[:2]

        # Detect green pixels: R<100, G>180, B<100
        green = (arr[:, :, 0] < 100) & (arr[:, :, 1] > 180) & (arr[:, :, 2] < 100)

        # Find row and column segments that are NOT green
        row_green = green.mean(axis=1) > 0.7
        col_green = green.mean(axis=0) > 0.7

        def find_segments(mask: np.ndarray, min_size: int = 40) -> List[tuple]:
            segs = []
            start = None
            for i, v in enumerate(mask):
                if not v and start is None:
                    start = i
                elif v and start is not None:
                    if i - start >= min_size:
                        segs.append((start, i))
                    start = None
            if start is not None and len(mask) - start >= min_size:
                segs.append((start, len(mask)))
            return segs

        row_segs = find_segments(row_green)
        col_segs = find_segments(col_green)

        logger.info("Sheet %dx%d → %d row segments, %d col segments",
                     w, h, len(row_segs), len(col_segs))

        if not row_segs or not col_segs:
            logger.warning("No green grid detected — returning sheet as single image")
            cleaned = _chroma_key_single(sheet_b64)
            return [cleaned]

        # Extract cells row by row, left to right
        cells: List[Optional[str]] = []
        for ry0, ry1 in row_segs:
            for cx0, cx1 in col_segs:
                # Skip cells that are mostly green (empty)
                cell_region = green[ry0:ry1, cx0:cx1]
                if cell_region.mean() > 0.8:
                    continue

                cell = img.crop((cx0, ry0, cx1, ry1)).convert("RGBA")
                cell_arr = np.array(cell)

                # Chroma key: green → transparent
                gm = ((cell_arr[:, :, 0] < 100) &
                      (cell_arr[:, :, 1] > 180) &
                      (cell_arr[:, :, 2] < 100))
                cell_arr[gm] = [0, 0, 0, 0]

                result = Image.fromarray(cell_arr)
                buf = io.BytesIO()
                result.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                cells.append(b64)

        logger.info("Extracted %d cells from sprite sheet", len(cells))
        return cells if cells else [sheet_b64]

    except Exception as e:
        logger.exception("Sprite sheet split failed: %s", e)
        return [sheet_b64]


def _chroma_key_single(b64: str) -> Optional[str]:
    """Apply chroma key (green → transparent) to a single image."""
    import base64
    import io

    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return b64  # no Pillow → return as-is

    try:
        img_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        arr = np.array(img)

        # Green pixels → transparent
        gm = ((arr[:, :, 0] < 100) &
              (arr[:, :, 1] > 180) &
              (arr[:, :, 2] < 100))
        arr[gm] = [0, 0, 0, 0]

        result = Image.fromarray(arr)
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return b64


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

    # ── Step 3.5: Remove background from each sprite ──
    # Gemini outputs images with white/colored backgrounds, not transparent.
    # To fill into ELK node slots without visible background rectangles,
    # we need transparent PNGs. Use the existing tiered rembg pipeline:
    #   Tier 0: remove.bg (Canva) → Tier 1: remove-bg.io → Tier 2: rembg U2-Net → Tier 3: chroma
    images_with_bg = [img for img in images if img is not None]
    if images_with_bg:
        try:
            from backend.pipeline.removebg_route import handle_removebg

            rb_result = await handle_removebg(images_with_bg)
            if rb_result.get("success"):
                transparent_images = [
                    r.get("image_b64") for r in rb_result.get("results", [])
                ]
                # Map back to the full images list (None slots stay None)
                ti_iter = iter(transparent_images)
                images = [
                    next(ti_iter) if img is not None else None
                    for img in images
                ]
                logger.info(
                    "rembg: %d/%d sprites background-removed (method: %s)",
                    len(transparent_images), len(images_with_bg),
                    rb_result.get("method", "unknown"),
                )
            else:
                logger.warning("rembg failed: %s — using images with background",
                               rb_result.get("error", "unknown"))
        except Exception as e:
            logger.exception("rembg call failed — using images with background")
            result.errors.append(f"rembg: {e}")

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