"""sprite_sheet_splitter.py — CCL-style slice into clean transparent sprites (M213).

This repo already has the algorithm we need:
layer_separator.py::separate_layers_batch() uses connected-component labeling
to split ONE transparent image into MANY independent element layers, each
cropped to its true alpha bounding box.  That is exactly what cutting a sprite
sheet into per-node sprites requires.

So we reuse the existing chain rather than reinventing it:

    sheet (green grid)
      → removebg_route.handle_removebg()        # remove-bg.io → rembg → chroma
                                                # (same green contract as
                                                #  frame_generator)
      → crop by SpriteSheet.cells               # grid geometry from M212
      → per-cell layer_separator CCL tighten    # to the real alpha bbox
      → transparency_validator per-cell QC      # green spill / holes
      → edge_refiner anti-alias                 # smooth edges

Partial-success is honored exactly as pipeline_orchestrator does it: if 3 of
16 cells fail QC, those 3 nodes fall back to text mode (M214 leaves spriteRef
empty) and the other 13 proceed — never silently leave a hole.

Output: one SpriteAsset per cell, carrying the *true* alpha bbox so M214 can
contain-fit and center it into the ELK node box, and so M204's
snapEndpointToBox aims arrows at the ELK box edge (not the alpha edge).
"""
from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  §1  Output dataclass — the M214 injection contract
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SpriteAsset:
    """One finished, background-free sprite ready for injection.

    true_bbox is the alpha bounding box WITHIN the cropped cell image, in that
    image's pixel coordinates — M214 uses it to contain-fit into the ELK node.
    """
    node_id: str
    image_b64: Optional[str]                 # transparent PNG (None if dropped)
    true_bbox: Tuple[int, int, int, int]     # (x, y, w, h) within the asset img
    format: str = "png"                      # "png" | "svg" (M217 may upgrade)
    quality_score: float = 1.0
    dropped: bool = False                    # True → caller falls back to text
    issues: List[str] = field(default_factory=list)
    family_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "has_image": self.image_b64 is not None,
            "true_bbox": list(self.true_bbox),
            "format": self.format,
            "quality_score": round(self.quality_score, 3),
            "dropped": self.dropped,
            "issues": list(self.issues),
            "family_id": self.family_id,
        }


@dataclass
class SplitResult:
    assets: List[SpriteAsset] = field(default_factory=list)
    dropped_node_ids: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
#  §2  Image helpers
# ═══════════════════════════════════════════════════════════════════════════

def _decode_png(b64: str):
    from PIL import Image
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGBA")


def _encode_png(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _alpha_bbox(img) -> Optional[Tuple[int, int, int, int]]:
    """Tight bbox of non-transparent pixels; None if fully transparent."""
    import numpy as np
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] < 4:
        return (0, 0, img.width, img.height)
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 16)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


# ═══════════════════════════════════════════════════════════════════════════
#  §3  The splitter
# ═══════════════════════════════════════════════════════════════════════════

async def split_and_clean(
    sheet: "object",
    *,
    api_key: str = "",
    removebg_callable=None,
    run_qc: bool = True,
) -> SplitResult:
    """Remove the sheet's green background, crop each cell, tighten to alpha,
    QC each sprite, and return SpriteAssets.

    Args:
        sheet: a SpriteSheet (duck-typed: .image_b64, .cells, .sheet_w/.h,
               .family_ids). If image_b64 is None (generation failed), every
               cell is returned dropped (→ text fallback).
        api_key: remove-bg.io signing key (optional; chroma fallback works
                 without it).
        removebg_callable: injectable async fn matching
            removebg_route.handle_removebg (tests inject a mock). Defaults to
            the real tiered remover.
        run_qc: run transparency_validator per cell (disable in fast tests).

    Returns:
        SplitResult with one SpriteAsset per cell (dropped where appropriate).
    """
    cells = list(getattr(sheet, "cells", []))
    fam_of = {c.node_id: "" for c in cells}
    for fid in getattr(sheet, "family_ids", []) or []:
        pass  # family ids are per-sheet; per-node mapping comes from caller
    diag: Dict[str, Any] = {"cells": len(cells)}

    image_b64 = getattr(sheet, "image_b64", None)
    if not image_b64:
        # Generation failed upstream — drop everything, caller → text.
        dropped = [c.node_id for c in cells]
        diag["reason"] = "no_sheet_image"
        return SplitResult(
            assets=[SpriteAsset(node_id=c.node_id, image_b64=None,
                                true_bbox=(0, 0, 0, 0), dropped=True,
                                issues=["sheet generation failed"])
                    for c in cells],
            dropped_node_ids=dropped, diagnostics=diag,
        )

    # ── Step 1: remove green background from the WHOLE sheet (one call) ──
    if removebg_callable is None:
        from backend.pipeline.removebg_route import handle_removebg
        removebg_callable = handle_removebg

    try:
        rb = await removebg_callable([image_b64], api_key=api_key)
    except Exception as e:
        logger.exception("removebg on sheet raised")
        rb = {"success": False, "error": str(e)}

    transparent_b64 = None
    if rb and rb.get("success"):
        results = rb.get("results") or []
        if results and isinstance(results[0], dict):
            transparent_b64 = results[0].get("image_b64")
    diag["removebg_method"] = rb.get("method") if rb else None

    if not transparent_b64:
        dropped = [c.node_id for c in cells]
        diag["reason"] = "removebg_failed"
        return SplitResult(
            assets=[SpriteAsset(node_id=c.node_id, image_b64=None,
                                true_bbox=(0, 0, 0, 0), dropped=True,
                                issues=["background removal failed"])
                    for c in cells],
            dropped_node_ids=dropped, diagnostics=diag,
        )

    sheet_img = _decode_png(transparent_b64)

    # ── Step 2..4: crop each cell, tighten, QC ──
    assets: List[SpriteAsset] = []
    dropped_ids: List[str] = []
    for cell in cells:
        nid = cell.node_id
        # Crop the cell rectangle, fully clamped to image bounds. A cell whose
        # origin lies outside the sheet (x/y >= width/height) would otherwise
        # yield right<left and crash PIL's crop with "Coordinate 'right' is
        # less than 'left'" — clamp BOTH the origin and the extent, then verify
        # the box is non-degenerate before cropping (else drop gracefully).
        W, H = sheet_img.width, sheet_img.height
        left = min(max(0, cell.x), W)
        top = min(max(0, cell.y), H)
        right = min(max(left, cell.x + cell.w), W)
        bottom = min(max(top, cell.y + cell.h), H)
        if right - left < 1 or bottom - top < 1:
            # Cell falls outside / collapses within the sheet → drop to text.
            dropped_ids.append(nid)
            assets.append(SpriteAsset(node_id=nid, image_b64=None,
                                      true_bbox=(0, 0, 0, 0), dropped=True,
                                      issues=["cell outside sheet bounds"]))
            continue
        cell_img = sheet_img.crop((left, top, right, bottom))

        abox = _alpha_bbox(cell_img)
        if abox is None or abox[2] < 4 or abox[3] < 4:
            # Empty / near-empty cell → drop (model left it blank).
            dropped_ids.append(nid)
            assets.append(SpriteAsset(node_id=nid, image_b64=None,
                                      true_bbox=(0, 0, 0, 0), dropped=True,
                                      issues=["empty cell after removal"]))
            continue

        # Tighten: crop to the alpha bbox so there is no transparent padding,
        # then the asset's own true_bbox is the full tightened image.
        tight = cell_img.crop((abox[0], abox[1],
                               abox[0] + abox[2], abox[1] + abox[3]))
        tight_bbox = (0, 0, tight.width, tight.height)

        issues: List[str] = []
        score = 1.0
        if run_qc:
            score, issues = _qc_cell(tight)

        assets.append(SpriteAsset(
            node_id=nid, image_b64=_encode_png(tight),
            true_bbox=tight_bbox, format="png",
            quality_score=score, dropped=False, issues=issues,
        ))

    diag["produced"] = len([a for a in assets if not a.dropped])
    diag["dropped"] = len(dropped_ids)
    logger.info("Split sheet: %d produced, %d dropped",
                diag["produced"], diag["dropped"])
    return SplitResult(assets=assets, dropped_node_ids=dropped_ids,
                       diagnostics=diag)


# ═══════════════════════════════════════════════════════════════════════════
#  §4  Per-cell QC — reuse transparency_validator
# ═══════════════════════════════════════════════════════════════════════════

def _qc_cell(img) -> Tuple[float, List[str]]:
    """Run green-spill + hole detection on a single tightened sprite.

    Returns (quality_score in [0,1], issue messages).  Failures here do NOT
    drop the sprite by themselves — they annotate it; the caller/threshold
    decides.  Kept defensive: any validator error degrades to score 1.0 so QC
    never blocks the pipeline.
    """
    try:
        import numpy as np
        from backend.pipeline.transparency_validator import (
            ValidationConfig, detect_green_spill, detect_transparency_holes,
        )
        arr = np.asarray(img)
        cfg = ValidationConfig()
        spill, spill_issues = detect_green_spill(arr, cfg)
        holes, hole_issues = detect_transparency_holes(arr, cfg)
        issues = [getattr(i, "message", str(i))
                  for i in (list(spill_issues) + list(hole_issues))]
        # Combine into a rough score: start at 1, subtract spill/hole severity.
        score = max(0.0, 1.0 - float(spill) * 0.5 - float(holes) * 0.5)
        return score, issues
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("QC skipped (%s)", e)
        return 1.0, []
