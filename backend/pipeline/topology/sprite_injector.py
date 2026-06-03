"""sprite_injector.py — Stage 2.8: Gemini interleaved → per-node sprites.

Gemini responseModalities: ['TEXT','IMAGE'] returns interleaved multi-image
output in ONE request. We describe all sprite nodes in a single prompt, and
Gemini returns N independent images as separate parts[].inline_data — each
one corresponds to one node. Background removal via remove.bg (Canva) API.

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


def _build_interleaved_prompt(
    sprite_nodes: List[Dict[str, Any]],
    family_prompts: Optional[Dict[str, str]] = None,
) -> str:
    """Build a single prompt that asks Gemini to generate one image per node.

    M301 upgrade: if family_prompts is provided (from sprite_prompt_designer),
    use series-consistent per-node prompts instead of generic descriptions.
    This ensures same-family sprites share identical base descriptions and
    differ only along the variation_axis — matching "同类型不同含义、只有微小差别".

    Gemini interleaved output: when responseModalities includes IMAGE,
    the model can return multiple images in one response, each as a
    separate part with inline_data. We ask it to generate them in order.

    Background: WHITE #FFFFFF. remove.bg (Canva) API does AI-based segmentation
    to produce transparent PNGs. White produces cleaner edges than green
    because Gemini has fewer color-bleeding artifacts on white.
    """
    lines = [
        "You are generating illustrations for a scientific/academic paper figure. "
        "Generate one illustration for EACH item below, in EXACT order.\n\n"
        "STYLE REQUIREMENTS (CRITICAL — follow ALL of these):\n"
        "- Background: solid pure WHITE #FFFFFF. NO gradients, NO shadows, NO borders.\n"
        "- Style: academic paper figure illustration, clean and minimal.\n"
        "- Feature maps / tensors: draw as a colored rectangular slab with subtle "
        "texture patterns (like a heatmap or activation visualization).\n"
        "- Encoders / decoders / models: draw as a stack of colored layers or blocks.\n"
        "- Attention / selection maps: draw as a red-yellow-blue heatmap.\n"
        "- Kernels / filters: draw as a small NxN grid with colored cells.\n"
        "- Input images: draw as a colorful photograph thumbnail (landscape/scene).\n"
        "- Each illustration MUST be 512x512 pixels, square, centered on white.\n"
        "- No text labels inside the illustration. No 3D effects. Thin clean outlines.\n"
    ]

    # M301: series-consistency clause for families
    if family_prompts:
        lines.append(
            "- SERIES CONSISTENCY: some items belong to the same family. "
            "Family members MUST be visually identical except for the stated difference. "
            "Same color palette, same stroke weight, same overall shape, same composition.\n"
        )

    lines.append("ITEMS TO ILLUSTRATE:\n")

    for i, node in enumerate(sprite_nodes):
        node_id = node.get("id", "")
        label = _node_label(node)
        hint = node.get("iconHint", "")
        family_id = node.get("familyId", "")

        # ALWAYS start with the node's own identity (label + iconHint).
        # This was the bug: family_prompts replaced the node description entirely,
        # resulting in "A small nodes in stage2_group" instead of "ViT Encoder".
        desc = f"{label}"
        if hint:
            desc += f" — visual concept: {hint}"

        # M301: append family consistency constraint as SUPPLEMENT, not replacement
        if family_prompts and node_id in family_prompts:
            fam_prompt = family_prompts[node_id]
            # Extract only the style/consistency part, skip the generic description
            # Look for "variant X of Y" or "consistent series" keywords
            if "consistent series" in fam_prompt or "variant" in fam_prompt:
                # Keep only the consistency instruction, not the generic "A small nodes..." part
                import re as _re
                consistency_part = _re.search(
                    r'(This is variant.*)', fam_prompt, _re.DOTALL
                )
                if consistency_part:
                    desc += f". {consistency_part.group(1).strip()}"
            else:
                # Unknown format — append the whole thing but after the node identity
                desc += f". Style: {fam_prompt}"

        if family_id:
            desc += f" [family: {family_id}]"

        lines.append(f"  {i+1}. {desc}")

    lines.append(
        f"\nGenerate EXACTLY {len(sprite_nodes)} image(s), one per item, in order. "
        "Each 512x512 px. Solid WHITE #FFFFFF background. Academic figure style."
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
    #      arranged in a grid — a "sprite sheet".
    #      We detect this case, split the sheet into individual cells,
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

    # ── DUMP to 0531/ for debugging ──
    try:
        import base64 as _b64
        from pathlib import Path
        dump_dir = Path("0531")
        dump_dir.mkdir(exist_ok=True)
        # Use a call counter to distinguish multiple calls
        existing = list(dump_dir.glob("call_*_prompt.txt"))
        call_idx = len(existing)
        prefix = f"call_{call_idx:02d}"

        # Save prompt
        (dump_dir / f"{prefix}_prompt.txt").write_text(prompt, encoding="utf-8")

        # Save raw images (before sheet split)
        for ri, raw_b64 in enumerate(raw_images):
            try:
                raw_bytes = _b64.b64decode(raw_b64)
                (dump_dir / f"{prefix}_raw_{ri}.png").write_bytes(raw_bytes)
                # Also save dimensions
                try:
                    from PIL import Image as _Img
                    import io as _io
                    _im = _Img.open(_io.BytesIO(raw_bytes))
                    (dump_dir / f"{prefix}_raw_{ri}_dims.txt").write_text(
                        f"{_im.width}x{_im.height} mode={_im.mode}", encoding="utf-8"
                    )
                except Exception:
                    pass
            except Exception:
                pass

        logger.info("DUMP: saved %s to 0531/ (%d raw images)", prefix, len(raw_images))
    except Exception as dump_err:
        logger.warning("DUMP failed (non-fatal): %s", dump_err)

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
            # Pass through RAW — bg removal by remove.bg downstream
            images.append(b64)
    # else: 0 images → all None

    # ── DUMP split cells to 0531/ ──
    try:
        import base64 as _b64
        from pathlib import Path
        dump_dir = Path("0531")
        # Reuse the call_idx from the raw dump above
        existing_prompts = list(dump_dir.glob("call_*_prompt.txt"))
        ci = max(0, len(existing_prompts) - 1)  # last call index
        pfx = f"call_{ci:02d}"
        for si, cell_b64 in enumerate(images):
            if cell_b64:
                try:
                    cell_bytes = _b64.b64decode(cell_b64)
                    (dump_dir / f"{pfx}_cell_{si}.png").write_bytes(cell_bytes)
                except Exception:
                    pass
        logger.info("DUMP: saved %d cells as %s_cell_*.png", sum(1 for c in images if c), pfx)
    except Exception:
        pass

    # Pad or truncate to n_expected
    while len(images) < n_expected:
        images.append(None)
    return images[:n_expected]


def _split_sprite_sheet(sheet_b64: str, n_expected: int) -> List[Optional[str]]:
    """Split a proxy sprite sheet into individual cells (no bg removal)."""
    import base64, io
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return [sheet_b64]
    try:
        img_bytes = base64.b64decode(sheet_b64)
        img = Image.open(io.BytesIO(img_bytes))
        arr = np.array(img.convert("RGB"))
        h, w = arr.shape[:2]
        row_std = arr.std(axis=1).mean(axis=1)
        col_std = arr.std(axis=0).mean(axis=1)
        row_sep = row_std < 15
        col_sep = col_std < 15
        def find_segments(mask, min_size=40):
            segs, start = [], None
            for j, v in enumerate(mask):
                if not v and start is None: start = j
                elif v and start is not None:
                    if j - start >= min_size: segs.append((start, j))
                    start = None
            if start is not None and len(mask) - start >= min_size:
                segs.append((start, len(mask)))
            return segs
        row_segs = find_segments(row_sep)
        col_segs = find_segments(col_sep)
        if not row_segs or not col_segs:
            return [sheet_b64]
        cells: List[Optional[str]] = []
        for ry0, ry1 in row_segs:
            for cx0, cx1 in col_segs:
                cell = img.crop((cx0, ry0, cx1, ry1))
                buf = io.BytesIO()
                cell.save(buf, format="PNG")
                cells.append(base64.b64encode(buf.getvalue()).decode())
        return cells if cells else [sheet_b64]
    except Exception as e:
        logger.exception("Sprite sheet split failed: %s", e)
        return [sheet_b64]



async def inject_sprites(
    elk_graph: Dict[str, Any],
    *,
    settings: Optional[Any] = None,
    model: str = "gemini-2.0-flash-preview-image-generation",
    gemini_callable=None,
    skip_generation: bool = False,
) -> SpriteInjectionResult:
    """Run the full sprite pipeline on a composed ELK graph.

    M301-M304 upgrade: family-aware series-consistent pipeline.

    Pipeline:
      M300  consolidate_layers() — merge small groups into top-K parents
      M210  classify_nodes()     — stamp renderMode/familyId on every leaf
      M301  design family prompts — series-consistent per-node descriptions
      M302  Gemini interleaved   — one call → N images (white bg)
      M302  remove.bg (Canva)    — AI segmentation → transparent PNG
      M303  stamp spriteRef      — write base64 PNG into ELK node
      M304  family consistency   — validate intra-family visual coherence

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

    # ── M300: Layer consolidation ──
    # Merge small groups into top-K parents before classification.
    # This reduces the number of sprite nodes, making generation more efficient.
    try:
        from backend.pipeline.topology.node_classifier import consolidate_layers
        consolidate_layers(elk_graph, top_k=3)
    except Exception as e:
        logger.warning("consolidate_layers failed (non-fatal): %s", e)

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

    # ── M301: Design family-aware prompts ──
    # Use sprite_prompt_designer for series-consistent descriptions.
    # This ensures same-family nodes share identical base + differ on axis.
    family_prompts: Optional[Dict[str, str]] = None
    families_list = []
    try:
        from backend.pipeline.topology.node_classifier import (
            classify_nodes, ClassificationReport,
        )
        from backend.pipeline.topology.sprite_prompt_designer import (
            design_prompts_for_classified,
        )

        # classify_nodes has already run (caller did it), but we need the
        # families list.  Re-classifying is idempotent (same result).
        report = classify_nodes(elk_graph)
        families_list = report.families or []
        if report.families:
            designed = design_prompts_for_classified(elk_graph, report.families)
            if designed:
                family_prompts = {p.node_id: p.prompt for p in designed}
                logger.info(
                    "M301: %d family-aware prompts designed (%d families)",
                    len(family_prompts), len(report.families),
                )
    except Exception as e:
        logger.warning("M301 family prompt design failed (fallback to basic): %s", e)

    # ── M302: Batched Gemini generation — 1 family = 1 Gemini call ──
    # DESIGN: a family IS the batch unit. Same-family sprites must be
    # generated in ONE inference call to guarantee visual consistency
    # (same style, palette, stroke weight). Splitting = destroying consistency.
    #
    # FIX for 524 timeout: prompt写512px → Gemini实际返回1024×1024 (upscale行为)。
    # 基准测试实证: 1 sprite@512px = 20s ✓, 1 sprite@1024px = 126s ✗(524)
    # 所以prompt写512, Gemini给1024大图, 序列帧分割后每张sprite≥170px。
    # Families already capped at ≤6 members by node_classifier.
    MAX_CONCURRENT = 3  # parallel family calls (independent families)

    # Build node_id → node lookup for batch assembly
    node_id_map = {n.get("id", f"__idx_{i}"): n for i, n in enumerate(sprite_nodes)}

    # Build batches: 1 family = 1 batch (DO NOT split families)
    batches: List[List[Dict[str, Any]]] = []
    batch_labels: List[str] = []
    claimed_ids: set = set()

    if families_list:
        for fam in families_list:
            fam_nodes = [node_id_map[nid] for nid in fam.member_node_ids if nid in node_id_map]
            if not fam_nodes:
                continue
            # One family = one Gemini call = one sprite sheet / interleaved set
            batches.append(fam_nodes)
            batch_labels.append(f"{fam.family_id}({len(fam_nodes)})")
            claimed_ids.update(n.get("id", "") for n in fam_nodes)

    # Unclaimed nodes: group up to 6
    unclaimed = [n for n in sprite_nodes if n.get("id", "") not in claimed_ids]
    if unclaimed:
        for j in range(0, len(unclaimed), 6):
            chunk = unclaimed[j:j + 6]
            batches.append(chunk)
            batch_labels.append(f"unclaimed[{j}:{j+len(chunk)}]")

    result.prompts_designed = len(sprite_nodes)

    # Map: node_id → generated image (filled by batch results)
    image_map: Dict[str, Optional[str]] = {n.get("id", f"__idx_{i}"): None for i, n in enumerate(sprite_nodes)}

    # ── Diagnostic: full batch plan dump ──
    logger.info(
        "┌─ M302 FAMILY PLAN: %d sprites → %d families (%d concurrent)",
        len(sprite_nodes), len(batches), MAX_CONCURRENT,
    )
    for bi, (bnodes, blabel) in enumerate(zip(batches, batch_labels)):
        ids = [n.get("id", "?") for n in bnodes]
        labels = [_node_label(n)[:15] for n in bnodes]
        logger.info(
            "│  family[%d] %s: %d sprites  ids=%s  labels=%s",
            bi, blabel, len(bnodes), ids, labels,
        )
    logger.info("└─ Total Gemini calls: %d (max %d parallel)", len(batches), MAX_CONCURRENT)

    # ── Concurrent family calls with semaphore ──
    import asyncio as _aio
    sem = _aio.Semaphore(MAX_CONCURRENT)
    progress = {"done": 0, "ok": 0, "fail": 0}

    async def _run_family(batch_idx: int, batch_nodes: List[Dict[str, Any]], blabel: str):
        """Run one family's Gemini call. 1 family = 1 sprite sheet."""
        async with sem:
            prompt = _build_interleaved_prompt(batch_nodes, family_prompts=family_prompts)
            n_sprites = len(batch_nodes)
            logger.info(
                "│  family[%d/%d] %s: START %d sprites, prompt=%d chars",
                batch_idx + 1, len(batches), blabel, n_sprites, len(prompt),
            )
            t_call = time.monotonic()
            try:
                imgs = await _call_gemini_interleaved(
                    prompt=prompt,
                    n_expected=n_sprites,
                    settings=settings,
                    model=model,
                )
                elapsed_ms = (time.monotonic() - t_call) * 1000
                got = sum(1 for img in imgs if img)
                for node, img in zip(batch_nodes, imgs):
                    nid = node.get("id", "")
                    if img:
                        image_map[nid] = img
                        result.images_received += 1
                progress["done"] += 1
                progress["ok"] += got
                logger.info(
                    "│  family[%d/%d] ✓ %d/%d sprites in %.0fms  [%d/%d families done]",
                    batch_idx + 1, len(batches), got, n_sprites,
                    elapsed_ms, progress["done"], len(batches),
                )
            except Exception as e:
                elapsed_ms = (time.monotonic() - t_call) * 1000
                progress["done"] += 1
                progress["fail"] += 1
                logger.warning(
                    "│  family[%d/%d] ✗ FAILED %.0fms: %s  [%d/%d families done, %d failed]",
                    batch_idx + 1, len(batches), elapsed_ms, e,
                    progress["done"], len(batches), progress["fail"],
                )
                result.errors.append(f"family {batch_idx + 1} ({blabel}): {e}")

    # Launch all family calls (semaphore limits parallelism)
    await _aio.gather(*[
        _run_family(bi, bnodes, blabel)
        for bi, (bnodes, blabel) in enumerate(zip(batches, batch_labels))
    ])

    # ── Diagnostic: final image map summary ──
    got_total = sum(1 for v in image_map.values() if v)
    miss_ids = [k for k, v in image_map.items() if v is None]
    logger.info(
        "M302 DONE: %d/%d sprites received, %d missing (first 10): %s",
        got_total, len(image_map), len(miss_ids), miss_ids[:10],
    )

    # Reassemble images in sprite_nodes order
    images: List[Optional[str]] = []
    for node in sprite_nodes:
        nid = node.get("id", "")
        images.append(image_map.get(nid))

    # ── M302: Remove background via remove.bg (Canva) API ──
    # remove.bg does AI segmentation (not color matching).
    # REMOVEBG_API_KEYS: 4 keys × 50/month = 200 calls/month.
    # CRITICAL: remove.bg has MAX_FRAMES=16 per request.
    # With 35 sprites we must batch into chunks of ≤16.
    REMOVEBG_BATCH = 16

    images_with_bg = [img for img in images if img is not None]
    if images_with_bg:
        try:
            from backend.pipeline.removebg_route import handle_removebg

            # Split into chunks of REMOVEBG_BATCH
            all_transparent: List[Optional[str]] = []
            n_chunks = (len(images_with_bg) + REMOVEBG_BATCH - 1) // REMOVEBG_BATCH
            logger.info(
                "M302: sending %d sprites to remove.bg in %d chunk(s) (max %d/chunk)",
                len(images_with_bg), n_chunks, REMOVEBG_BATCH,
            )

            for ci in range(0, len(images_with_bg), REMOVEBG_BATCH):
                chunk = images_with_bg[ci:ci + REMOVEBG_BATCH]
                chunk_idx = ci // REMOVEBG_BATCH + 1
                t_rb = time.monotonic()

                rb_result = await handle_removebg(chunk)
                elapsed_rb = (time.monotonic() - t_rb) * 1000

                if rb_result.get("success"):
                    chunk_transparent = [
                        r.get("image_b64") for r in rb_result.get("results", [])
                    ]
                    all_transparent.extend(chunk_transparent)
                    logger.info(
                        "M302: removebg chunk[%d/%d] ✓ %d images in %.0fms (method: %s)",
                        chunk_idx, n_chunks, len(chunk_transparent), elapsed_rb,
                        rb_result.get("method", "?"),
                    )
                else:
                    # Chunk failed — keep originals for this chunk
                    logger.warning(
                        "M302: removebg chunk[%d/%d] ✗ %s — keeping %d originals",
                        chunk_idx, n_chunks, rb_result.get("error", "?"), len(chunk),
                    )
                    all_transparent.extend(chunk)  # originals as fallback

            # Reassemble: map transparent images back to full images list
            ti_iter = iter(all_transparent)
            images = [
                next(ti_iter) if img is not None else None
                for img in images
            ]
            logger.info(
                "M302: bg-removal complete, %d/%d sprites processed",
                len(all_transparent), len(images_with_bg),
            )
        except Exception as e:
            logger.warning("M302: removebg unavailable (%s) — keeping originals", e)

    # ── M303: Stamp each image onto its node (spriteRef embedding) ──
    for i, (node, img_b64) in enumerate(zip(sprite_nodes, images)):
        if img_b64:
            _stamp_sprite_ref(node, image_b64=img_b64, stack_count=3)
            result.refs_stamped += 1
        else:
            _stamp_sprite_ref(node, image_b64=None, stack_count=2 + (i % 3))
            result.fallback_to_blob += 1

    # ── M304: Family consistency validation ──
    # Check that sprites within the same family have similar visual properties.
    # If a sprite deviates too much, mark it for re-generation (future: trigger
    # re-gen; for now: log warning + add to diagnostics).
    try:
        _validate_family_consistency(sprite_nodes, images, result)
    except Exception as e:
        logger.warning("M304 consistency validation failed (non-fatal): %s", e)

    result.elapsed_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "Sprite injection: %d nodes, %d stamped, %d blob fallback, %.0fms",
        result.total_sprite_nodes, result.refs_stamped,
        result.fallback_to_blob, result.elapsed_ms,
    )
    return result


def _validate_family_consistency(
    sprite_nodes: List[Dict[str, Any]],
    images: List[Optional[str]],
    result: SpriteInjectionResult,
) -> None:
    """M304: Validate intra-family visual consistency.

    Groups sprite nodes by familyId, checks that all members within a family
    have similar image dimensions and non-empty content.  Logs warnings for
    families where members have inconsistent sizes (which would indicate
    Gemini produced mismatched illustrations).

    Future enhancement: compute perceptual hash (pHash) distance between
    family members and trigger re-generation if delta exceeds threshold.
    """
    import base64
    import io

    try:
        from PIL import Image
    except ImportError:
        return  # no Pillow → skip validation

    # Group by familyId
    families: Dict[str, List[tuple]] = {}
    for node, img_b64 in zip(sprite_nodes, images):
        fam_id = node.get("familyId", "")
        if fam_id and img_b64:
            families.setdefault(fam_id, []).append((node.get("id", ""), img_b64))

    inconsistent_families = []
    for fam_id, members in families.items():
        if len(members) < 2:
            continue

        # Check dimensions consistency
        sizes = []
        for node_id, b64 in members:
            try:
                img = Image.open(io.BytesIO(base64.b64decode(b64)))
                sizes.append((img.width, img.height))
            except Exception:
                sizes.append((0, 0))

        # All members should have similar dimensions (within 20% tolerance)
        if sizes:
            avg_w = sum(s[0] for s in sizes) / len(sizes)
            avg_h = sum(s[1] for s in sizes) / len(sizes)
            for node_id, (w, h) in zip([m[0] for m in members], sizes):
                if avg_w > 0 and abs(w - avg_w) / avg_w > 0.2:
                    inconsistent_families.append(fam_id)
                    logger.warning(
                        "M304: family %s member %s has inconsistent width "
                        "(%d vs avg %.0f)",
                        fam_id, node_id, w, avg_w,
                    )
                    break

    if inconsistent_families:
        result.errors.append(
            f"M304: {len(inconsistent_families)} families with size inconsistency"
        )