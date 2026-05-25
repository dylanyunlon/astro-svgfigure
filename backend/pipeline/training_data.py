"""
training_data.py — Layout Quality Scoring + Training Pair Collection
=====================================================================
Computes precision metrics between intended (ELK) and detected layouts
using Hungarian matching + IoU. Collects (screenshot, layout) pairs
for model fine-tuning.

§1  LayoutQualityScorer: per-element and aggregate quality metrics
§2  TrainingPairCollector: capture + store pipeline runs as JSONL
§3  DatasetStats: aggregate metrics across collected pairs
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
#  §1  Layout Quality Scorer
#
#  Given two layouts (intended vs detected), compute:
#    - Per-element: IoU, center error, size error
#    - Aggregate: mean IoU, precision, recall, F1
#    - Uses Hungarian matching for optimal alignment
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ElementMatch:
    """Quality metrics for a single matched element pair."""
    intended_id: str
    detected_id: str
    iou: float
    center_error_px: float      # L2 distance between centers
    width_error_px: float
    height_error_px: float
    name_match: bool            # Whether names are similar
    distance: float             # Hungarian cost

    @property
    def bbox_error_px(self) -> float:
        """Total bbox error = center error + size error."""
        return self.center_error_px + abs(self.width_error_px) + abs(self.height_error_px)


@dataclass
class LayoutQuality:
    """Aggregate quality metrics between two layouts."""
    num_intended: int
    num_detected: int
    num_matched: int
    precision: float            # matched / detected
    recall: float               # matched / intended
    f1: float
    mean_iou: float
    median_iou: float
    mean_center_error: float    # pixels
    mean_size_error: float      # pixels
    max_center_error: float
    p90_center_error: float     # 90th percentile
    unmatched_intended: List[str]   # IDs not found in detected
    unmatched_detected: List[str]   # IDs not found in intended
    matches: List[ElementMatch]
    scoring_time_ms: float


def score_layout_quality(
    intended: List[Dict[str, Any]],
    detected: List[Dict[str, Any]],
    max_distance: float = 80.0,
    iou_threshold: float = 0.1,
) -> LayoutQuality:
    """Score detected layout against intended (ground truth) layout.

    Uses Hungarian matching from layout_algorithms to find optimal
    1:1 alignment, then computes per-element and aggregate metrics.
    """
    from backend.pipeline.layout_algorithms import hungarian_match, iou

    t0 = time.monotonic()
    n_int, n_det = len(intended), len(detected)

    if n_int == 0 and n_det == 0:
        return LayoutQuality(0, 0, 0, 1.0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0, [], [], [], 0)
    if n_int == 0 or n_det == 0:
        return LayoutQuality(n_int, n_det, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             [e["id"] for e in intended], [e["id"] for e in detected], [],
                             (time.monotonic()-t0)*1000)

    raw_matches = hungarian_match(intended, detected, max_distance)

    matches: List[ElementMatch] = []
    matched_int = set()
    matched_det = set()

    for idx_i, idx_d, dist in raw_matches:
        ei, ed = intended[idx_i], detected[idx_d]
        bi, bd = ei.get("bbox", {}), ed.get("bbox", {})

        elem_iou = iou(bi, bd)
        if elem_iou < iou_threshold:
            continue

        # Center error
        ci = (bi["x"] + bi["width"]/2, bi["y"] + bi["height"]/2)
        cd = (bd["x"] + bd["width"]/2, bd["y"] + bd["height"]/2)
        center_err = math.sqrt((ci[0]-cd[0])**2 + (ci[1]-cd[1])**2)

        # Size error
        w_err = bd["width"] - bi["width"]
        h_err = bd["height"] - bi["height"]

        # Name similarity (simple prefix match)
        ni = ei.get("name", "").lower().replace(" ", "_")
        nd = ed.get("name", "").lower().replace(" ", "_")
        name_match = ni == nd or ni.startswith(nd[:8]) or nd.startswith(ni[:8]) if ni and nd else False

        matches.append(ElementMatch(
            intended_id=ei["id"], detected_id=ed["id"],
            iou=elem_iou, center_error_px=center_err,
            width_error_px=w_err, height_error_px=h_err,
            name_match=name_match, distance=dist,
        ))
        matched_int.add(idx_i)
        matched_det.add(idx_d)

    n_matched = len(matches)
    precision = n_matched / n_det if n_det > 0 else 0
    recall = n_matched / n_int if n_int > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    ious = sorted([m.iou for m in matches]) if matches else [0]
    center_errs = sorted([m.center_error_px for m in matches]) if matches else [0]
    size_errs = [abs(m.width_error_px) + abs(m.height_error_px) for m in matches] if matches else [0]

    elapsed = (time.monotonic() - t0) * 1000

    return LayoutQuality(
        num_intended=n_int,
        num_detected=n_det,
        num_matched=n_matched,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        mean_iou=round(sum(ious)/len(ious), 4),
        median_iou=round(ious[len(ious)//2], 4),
        mean_center_error=round(sum(center_errs)/len(center_errs), 2),
        mean_size_error=round(sum(size_errs)/len(size_errs), 2),
        max_center_error=round(max(center_errs), 2),
        p90_center_error=round(center_errs[int(len(center_errs)*0.9)], 2),
        unmatched_intended=[intended[i]["id"] for i in range(n_int) if i not in matched_int],
        unmatched_detected=[detected[i]["id"] for i in range(n_det) if i not in matched_det],
        matches=matches,
        scoring_time_ms=round(elapsed, 2),
    )


# ═══════════════════════════════════════════════════════════════════════
#  §2  Training Pair Collector
#
#  Captures (image_hash, elk_layout, detected_layout, quality_score)
#  from each pipeline run. Stores as JSONL for fine-tuning.
#  Image pixels stored separately as PNG to keep JSONL small.
# ═══════════════════════════════════════════════════════════════════════

TRAINING_DIR = os.environ.get("TRAINING_DATA_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "training_data"))


@dataclass
class TrainingPair:
    """One (screenshot, layout) training example."""
    image_hash: str
    timestamp: float
    prompt: str
    intended_layout: List[Dict]     # From ELK (ground truth)
    detected_layout: List[Dict]     # From vision detect
    quality: Optional[Dict] = None  # LayoutQuality as dict
    image_path: Optional[str] = None  # Path to saved PNG


def collect_training_pair(
    image_b64: str,
    elk_layout: Optional[List[Dict[str, Any]]],
    detected_layout: Optional[List[Dict[str, Any]]],
    prompt: str = "",
    save_image: bool = True,
) -> Optional[TrainingPair]:
    """Collect one training pair from a pipeline run.

    Computes quality metrics if both intended and detected exist.
    Appends to JSONL file. Saves image as PNG.
    """
    if not elk_layout and not detected_layout:
        return None

    img_hash = hashlib.md5(image_b64[:2048].encode()).hexdigest()[:12]
    ts = time.time()

    # Compute quality if we have both
    quality_dict = None
    if elk_layout and detected_layout:
        try:
            q = score_layout_quality(elk_layout, detected_layout)
            quality_dict = {
                "precision": q.precision, "recall": q.recall, "f1": q.f1,
                "mean_iou": q.mean_iou, "mean_center_error": q.mean_center_error,
                "num_matched": q.num_matched, "num_intended": q.num_intended,
                "num_detected": q.num_detected,
            }
        except Exception as e:
            logger.warning("Quality scoring failed: %s", e)

    pair = TrainingPair(
        image_hash=img_hash,
        timestamp=ts,
        prompt=prompt,
        intended_layout=elk_layout or [],
        detected_layout=detected_layout or [],
        quality=quality_dict,
    )

    # Save to disk
    try:
        os.makedirs(TRAINING_DIR, exist_ok=True)

        # Save image
        if save_image:
            import base64
            raw = image_b64.split(",", 1)[-1] if image_b64.startswith("data:") else image_b64
            img_path = os.path.join(TRAINING_DIR, f"{img_hash}.png")
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(raw))
            pair.image_path = img_path

        # Append JSONL
        jsonl_path = os.path.join(TRAINING_DIR, "pairs.jsonl")
        record = {
            "image_hash": img_hash,
            "timestamp": ts,
            "prompt": prompt,
            "intended": _strip_b64(elk_layout),
            "detected": _strip_b64(detected_layout),
            "quality": quality_dict,
            "image_file": f"{img_hash}.png",
        }
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info("Training pair saved: %s (quality: %s)", img_hash,
                     f"F1={quality_dict['f1']}" if quality_dict else "n/a")
        return pair

    except Exception as e:
        logger.warning("Failed to save training pair: %s", e)
        return pair


def _strip_b64(layout: Optional[List[Dict]]) -> Optional[List[Dict]]:
    """Remove image_b64 fields from layout to keep JSONL small."""
    if not layout:
        return layout
    return [{k: v for k, v in elem.items() if k != "image_b64"} for elem in layout]


# ═══════════════════════════════════════════════════════════════════════
#  §3  Dataset Statistics
#
#  Aggregate metrics across all collected pairs.
#  Histogram of quality scores for dataset health monitoring.
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DatasetStats:
    """Aggregate statistics for the training dataset."""
    total_pairs: int
    pairs_with_quality: int
    mean_f1: float
    mean_precision: float
    mean_recall: float
    mean_iou: float
    mean_center_error: float
    quality_histogram: Dict[str, int]   # {"excellent":N, "good":N, "poor":N}
    total_elements: int
    unique_prompts: int


def compute_dataset_stats(jsonl_path: Optional[str] = None) -> DatasetStats:
    """Compute aggregate statistics from the training JSONL file."""
    if jsonl_path is None:
        jsonl_path = os.path.join(TRAINING_DIR, "pairs.jsonl")

    if not os.path.exists(jsonl_path):
        return DatasetStats(0, 0, 0, 0, 0, 0, 0, {}, 0, 0)

    pairs = []
    prompts = set()
    total_elements = 0

    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                pairs.append(rec)
                prompts.add(rec.get("prompt", ""))
                total_elements += len(rec.get("intended", []))
            except json.JSONDecodeError:
                continue

    quality_pairs = [p for p in pairs if p.get("quality")]
    if not quality_pairs:
        return DatasetStats(len(pairs), 0, 0, 0, 0, 0, 0, {}, total_elements, len(prompts))

    f1s = [p["quality"]["f1"] for p in quality_pairs]
    precisions = [p["quality"]["precision"] for p in quality_pairs]
    recalls = [p["quality"]["recall"] for p in quality_pairs]
    ious = [p["quality"]["mean_iou"] for p in quality_pairs]
    center_errs = [p["quality"]["mean_center_error"] for p in quality_pairs]

    # Quality histogram
    hist = {"excellent": 0, "good": 0, "acceptable": 0, "poor": 0}
    for f1 in f1s:
        if f1 >= 0.9:
            hist["excellent"] += 1
        elif f1 >= 0.7:
            hist["good"] += 1
        elif f1 >= 0.5:
            hist["acceptable"] += 1
        else:
            hist["poor"] += 1

    n = len(quality_pairs)
    return DatasetStats(
        total_pairs=len(pairs),
        pairs_with_quality=n,
        mean_f1=round(sum(f1s)/n, 4),
        mean_precision=round(sum(precisions)/n, 4),
        mean_recall=round(sum(recalls)/n, 4),
        mean_iou=round(sum(ious)/n, 4),
        mean_center_error=round(sum(center_errs)/n, 2),
        quality_histogram=hist,
        total_elements=total_elements,
        unique_prompts=len(prompts),
    )


# ═══════════════════════════════════════════════════════════════════════
#  §4  Fine-Tuning Export (Gemini / GPT format)
# ═══════════════════════════════════════════════════════════════════════

def export_for_finetuning(
    jsonl_path: Optional[str] = None,
    output_path: Optional[str] = None,
    min_f1: float = 0.5,
    format: str = "gemini",
) -> Tuple[str, int]:
    """Export high-quality pairs as fine-tuning dataset.

    Gemini format: {"contents": [{"role":"user","parts":[image,text]}, {"role":"model","parts":[json]}]}
    GPT format:    {"messages": [{"role":"user","content":[image,text]}, {"role":"assistant","content":json}]}

    Only exports pairs with F1 >= min_f1 threshold.
    Returns: (output_path, num_exported)
    """
    if jsonl_path is None:
        jsonl_path = os.path.join(TRAINING_DIR, "pairs.jsonl")
    if output_path is None:
        output_path = os.path.join(TRAINING_DIR, f"finetune_{format}.jsonl")

    if not os.path.exists(jsonl_path):
        return output_path, 0

    exported = 0
    with open(jsonl_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            try:
                rec = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            q = rec.get("quality", {})
            if q.get("f1", 0) < min_f1:
                continue

            intended = rec.get("intended", [])
            if not intended:
                continue

            # Build the target output (what the model should produce)
            target_json = json.dumps(intended, ensure_ascii=False)

            prompt_text = (
                f"Analyze this UI screenshot and output a JSON array of all visible "
                f"UI elements with pixel-precise bounding boxes.\n"
                f"Context: {rec.get('prompt', 'UI dashboard')}"
            )

            if format == "gemini":
                example = {
                    "contents": [
                        {"role": "user", "parts": [
                            {"text": prompt_text},
                            {"file_data": {"file_uri": rec.get("image_file", ""), "mime_type": "image/png"}},
                        ]},
                        {"role": "model", "parts": [{"text": target_json}]},
                    ]
                }
            else:  # openai / gpt format
                example = {
                    "messages": [
                        {"role": "system", "content": "Output JSON array of UI elements with pixel bounding boxes."},
                        {"role": "user", "content": prompt_text},
                        {"role": "assistant", "content": target_json},
                    ]
                }

            fout.write(json.dumps(example, ensure_ascii=False) + "\n")
            exported += 1

    logger.info("Exported %d pairs to %s (min_f1=%.2f)", exported, output_path, min_f1)
    return output_path, exported
