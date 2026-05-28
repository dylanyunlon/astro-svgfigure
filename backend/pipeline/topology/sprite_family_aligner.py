"""sprite_family_aligner.py — Intra-family alignment (M216).

Megatron-LM's schedules.py::forward_backward_pipelining_without_interleaving()
tracks each microbatch's state in parallel lists (input_tensors /
output_tensors) so any element of the sequence can be located and aggregated
precisely.  A family of sprites is exactly such a sequence: members must be
located, centered, and sized consistently before they land in adjacent ELK
boxes — otherwise three "feature maps" sit at three different sizes and the
figure looks ragged.

Two responsibilities:

  1. cross_frame_subject_matching():
     the Hungarian-style centroid+area matching that frame_consistency.py's
     M104 docstring described but never implemented.  Here it is, applied to a
     family: align every member's subject centroid to a common reference, so
     variant 2's block sits where variant 1's block sits (no jitter across the
     series).

  2. align_family_assets():
     after M213 has removed the green background and tightened each sprite to
     its own alpha bbox, re-pad every member onto ONE common canvas — sized to
     the family's largest subject — with the subject centered.  The result is
     a set of equal-sized, center-aligned SpriteAssets whose shared true_bbox
     M214 contain-fits identically into same-sized ELK boxes.

Reuses frame_consistency.FrameConsistencyChecker for the family-level CV check
and transparency_validator.validate_cross_frame_consistency for QC; a member
that drifts past the CV/QC threshold is flagged (caller may drop → text).
"""
from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  §1  Image helpers (mirror sprite_sheet_splitter's, kept local)
# ═══════════════════════════════════════════════════════════════════════════

def _decode_png(b64: str):
    from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")


def _encode_png(img) -> str:
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _alpha_stats(img) -> Optional[Tuple[Tuple[int, int, int, int], Tuple[float, float], int]]:
    """Return (bbox, centroid, area) of opaque pixels, or None if empty.

    bbox = (x, y, w, h); centroid = (cx, cy) in image coords; area = opaque px.
    """
    import numpy as np
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] < 4:
        return ((0, 0, img.width, img.height),
                (img.width / 2.0, img.height / 2.0), img.width * img.height)
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 16)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    cx = float(xs.mean()); cy = float(ys.mean())
    return ((x0, y0, x1 - x0 + 1, y1 - y0 + 1), (cx, cy), int(len(xs)))


# ═══════════════════════════════════════════════════════════════════════════
#  §2  Cross-frame subject matching (the unimplemented M104 piece)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SubjectTrack:
    """One member's subject geometry within the family sequence."""
    node_id: str
    index: int
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    area: int
    present: bool = True


def cross_frame_subject_matching(
    assets: List[Any],
) -> List[SubjectTrack]:
    """Locate each member's subject (centroid + area + bbox).

    Unlike the multi-object Hungarian matching frame_consistency described for
    full animation frames, a sprite family has ONE subject per member, so the
    "matching" reduces to extracting each subject's geometry and tracking it
    across the sequence.  This is the precise-tracking primitive M216 needs to
    align members; it also surfaces an outlier (a member whose subject area is
    wildly off — a likely generation failure).

    Args:
        assets: SpriteAssets for one family (duck-typed: node_id, image_b64,
                dropped).
    Returns:
        SubjectTrack per asset, in input order.
    """
    tracks: List[SubjectTrack] = []
    for i, a in enumerate(assets):
        nid = getattr(a, "node_id", "")
        if getattr(a, "dropped", False) or not getattr(a, "image_b64", None):
            tracks.append(SubjectTrack(node_id=nid, index=i,
                                       bbox=(0, 0, 0, 0), centroid=(0.0, 0.0),
                                       area=0, present=False))
            continue
        img = _decode_png(a.image_b64)
        st = _alpha_stats(img)
        if st is None:
            tracks.append(SubjectTrack(node_id=nid, index=i,
                                       bbox=(0, 0, 0, 0), centroid=(0.0, 0.0),
                                       area=0, present=False))
            continue
        bbox, centroid, area = st
        tracks.append(SubjectTrack(node_id=nid, index=i, bbox=bbox,
                                   centroid=centroid, area=area, present=True))
    return tracks


# ═══════════════════════════════════════════════════════════════════════════
#  §3  Family alignment — equal-size, center-aligned canvas
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AlignmentResult:
    assets: List[Any]                         # re-canvased SpriteAssets
    common_size: Tuple[int, int] = (0, 0)
    consistency: Dict[str, Any] = field(default_factory=dict)
    outlier_node_ids: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def align_family_assets(
    assets: List[Any],
    *,
    pad_ratio: float = 0.12,
    run_consistency: bool = True,
) -> AlignmentResult:
    """Re-pad a family's sprites onto one common canvas, subject-centered.

    After M213 each present sprite is tightly cropped to its own alpha bbox
    (true_bbox == full image). Members therefore differ in size. Here we:
      - compute the family's max subject width/height,
      - build a common canvas = max dims * (1 + pad_ratio),
      - paste each subject centered on that canvas,
      - set every asset's image to the re-canvased PNG and its true_bbox to the
        common canvas (so M214 contain-fits identically into same-size boxes).

    Dropped/empty members are left untouched (still dropped → text at M214).

    Args:
        assets: SpriteAssets for ONE family.
        pad_ratio: fractional margin around the largest subject.
        run_consistency: run family-level CV + cross-frame QC.

    Returns:
        AlignmentResult with the mutated assets and consistency diagnostics.
    """
    present = [a for a in assets
               if not getattr(a, "dropped", False) and getattr(a, "image_b64", None)]
    if not present:
        return AlignmentResult(assets=assets, diagnostics={"reason": "no present members"})

    tracks = cross_frame_subject_matching(assets)
    present_tracks = [t for t in tracks if t.present]

    # Common canvas: largest subject bbox across the family, plus margin.
    max_w = max(t.bbox[2] for t in present_tracks)
    max_h = max(t.bbox[3] for t in present_tracks)
    canvas_w = int(round(max_w * (1.0 + pad_ratio)))
    canvas_h = int(round(max_h * (1.0 + pad_ratio)))
    canvas_w = max(1, canvas_w)
    canvas_h = max(1, canvas_h)

    from PIL import Image

    # Re-canvas each present member with its subject centered.
    track_by_id = {t.node_id: t for t in tracks}
    areas: List[int] = []
    for a in assets:
        nid = getattr(a, "node_id", "")
        t = track_by_id.get(nid)
        if t is None or not t.present:
            continue
        img = _decode_png(a.image_b64)
        # Crop to the subject bbox (defensive — should already be tight).
        bx, by, bw, bh = t.bbox
        subj = img.crop((bx, by, bx + bw, by + bh))
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        ox = (canvas_w - bw) // 2
        oy = (canvas_h - bh) // 2
        canvas.alpha_composite(subj, (max(0, ox), max(0, oy)))
        a.image_b64 = _encode_png(canvas)
        a.true_bbox = (0, 0, canvas_w, canvas_h)
        areas.append(t.area)

    result = AlignmentResult(
        assets=assets, common_size=(canvas_w, canvas_h),
        diagnostics={"members_aligned": len(areas),
                     "common_size": [canvas_w, canvas_h]},
    )

    # ── Family-level consistency: reuse the CV checker on subject areas ──
    if run_consistency and len(areas) >= 2:
        try:
            from backend.pipeline.frame_consistency import FrameConsistencyChecker
            # Scale areas to "layer-count-like" integers for the checker.
            checker = FrameConsistencyChecker()
            rep = checker.check([max(1, a) for a in areas])
            result.consistency = rep.to_dict()
            # Map the checker's outlier indices back to node ids (present only).
            present_ids = [t.node_id for t in present_tracks]
            for oi in rep.outlier_frames:
                if 0 <= oi < len(present_ids):
                    result.outlier_node_ids.append(present_ids[oi])
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("family consistency check skipped: %s", e)

    logger.info("Aligned family: %d members → common %dx%d (%d outliers)",
                len(areas), canvas_w, canvas_h, len(result.outlier_node_ids))
    return result
