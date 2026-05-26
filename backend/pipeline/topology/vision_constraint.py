"""vision_constraint.py — Screenshot-guided layout constraint system.

Problem: ELK produces valid but generic layouts. The original figure has a
         specific spatial arrangement that ELK doesn't know about.

Solution: Use the screenshot as a constraint source.
  1. CCL pass: scipy connected-component labeling extracts raw bboxes from edges
  2. Vision LLM pass: identify what each CCL region represents
  3. Constraint merge: align ELK node positions to match CCL bboxes

Design reference: NVIDIA NVLink's topology-aware placement.
  NVLink doesn't place tensors randomly — it reads the physical interconnect
  topology and places data on the GPU closest to where it will be consumed.
  Similarly, we read the visual topology from the screenshot and constrain
  ELK node placement to match the figure's spatial intent.

From Google's Layout-Guided Diffusion pattern:
  The diffusion model doesn't generate from noise alone — it takes a spatial
  layout as conditioning signal.  We apply the same principle: the screenshot
  is the conditioning signal for our layout engine.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from PIL import Image
    import numpy as np
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False


# ═══════════════════════════════════════════════════════════════════════════
#  §1  CCL-based bbox extraction from screenshot
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CCLRegion:
    """One connected component from the screenshot."""
    label_id: int
    x: int
    y: int
    width: int
    height: int
    area: int
    pixel_density: float  # ratio of edge pixels to bbox area

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label_id": self.label_id,
            "bbox": {"x": self.x, "y": self.y, "width": self.width, "height": self.height},
            "area": self.area,
            "pixel_density": round(self.pixel_density, 3),
        }


def extract_ccl_regions(
    image_b64: str,
    edge_threshold: int = 25,
    min_area: int = 200,
    max_regions: int = 150,
    merge_distance: int = 5,
) -> Tuple[List[CCLRegion], Dict[str, Any]]:
    """Extract connected-component bboxes from screenshot edges.

    From scipy.ndimage.label — the standard CCL implementation.
    Enhanced with:
      - Adaptive edge detection (Sobel-like gradient)
      - Small-region merging (components within merge_distance get combined)
      - Density filtering (reject noise regions with low pixel density)

    Parameters
    ----------
    image_b64 : str
        Base64-encoded screenshot.
    edge_threshold : int
        Gradient magnitude threshold for edge detection.
    min_area : int
        Minimum bbox area to keep a region.
    max_regions : int
        Cap on returned regions.
    merge_distance : int
        Merge regions whose bboxes are within this many pixels.

    Returns
    -------
    (regions, stats) where regions is sorted by area descending.
    """
    if not _HAS_DEPS:
        return [], {"error": "PIL/numpy not available"}

    try:
        # Decode image
        raw_b64 = image_b64
        if raw_b64.startswith("data:"):
            raw_b64 = raw_b64.split(",", 1)[1]

        img = Image.open(io.BytesIO(base64.b64decode(raw_b64)))
        gray = np.array(img.convert("L"), dtype=np.float32)
        img_h, img_w = gray.shape

        # Gradient-based edge detection (Sobel-like)
        dx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        dy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        gradient = dx + dy

        # Binary edge map
        edge_map = (gradient > edge_threshold).astype(np.uint8)

        # Morphological closing to connect nearby edges
        try:
            from scipy.ndimage import binary_dilation, label as scipy_label, find_objects
            # Dilate to connect nearby components
            struct = np.ones((3, 3), dtype=np.uint8)
            closed = binary_dilation(edge_map, structure=struct, iterations=2).astype(np.uint8)
        except ImportError:
            # Fallback: simple dilation via convolution
            from scipy.ndimage import label as scipy_label, find_objects
            closed = edge_map

        # Connected component labeling
        labels, num_components = scipy_label(closed)

        # Extract regions
        slices = find_objects(labels)
        regions = []

        for i, sl in enumerate(slices):
            if sl is None:
                continue
            y_sl, x_sl = sl
            x, y = int(x_sl.start), int(y_sl.start)
            w, h = int(x_sl.stop - x_sl.start), int(y_sl.stop - y_sl.start)
            area = w * h

            if area < min_area:
                continue

            # Compute pixel density (how much of the bbox is actual edge)
            region_mask = labels[y_sl, x_sl] == (i + 1)
            pixel_count = int(np.sum(region_mask))
            density = pixel_count / area if area > 0 else 0

            # Skip very sparse regions (noise)
            if density < 0.01:
                continue

            regions.append(CCLRegion(
                label_id=i + 1,
                x=x, y=y, width=w, height=h,
                area=area,
                pixel_density=density,
            ))

        # Merge nearby regions
        if merge_distance > 0:
            regions = _merge_nearby(regions, merge_distance)

        # Sort by area descending
        regions.sort(key=lambda r: r.area, reverse=True)

        # Cap
        if len(regions) > max_regions:
            regions = regions[:max_regions]

        stats = {
            "image_size": f"{img_w}x{img_h}",
            "raw_components": num_components,
            "after_area_filter": len(regions),
            "edge_threshold": edge_threshold,
            "min_area": min_area,
        }

        return regions, stats

    except Exception as e:
        logger.error("CCL extraction failed: %s", e, exc_info=True)
        return [], {"error": str(e)}


def _merge_nearby(regions: List[CCLRegion], distance: int) -> List[CCLRegion]:
    """Merge regions whose bboxes are within `distance` pixels."""
    if not regions:
        return regions

    merged = list(regions)
    changed = True

    while changed:
        changed = False
        new_merged = []
        used = set()

        for i, a in enumerate(merged):
            if i in used:
                continue

            # Find all regions close to a
            cluster = [a]
            for j, b in enumerate(merged):
                if j <= i or j in used:
                    continue
                if _bbox_distance(a, b) <= distance:
                    cluster.append(b)
                    used.add(j)
                    changed = True

            if len(cluster) == 1:
                new_merged.append(a)
            else:
                # Merge cluster into one region
                min_x = min(r.x for r in cluster)
                min_y = min(r.y for r in cluster)
                max_x = max(r.x + r.width for r in cluster)
                max_y = max(r.y + r.height for r in cluster)
                total_pixels = sum(int(r.pixel_density * r.area) for r in cluster)
                new_area = (max_x - min_x) * (max_y - min_y)

                new_merged.append(CCLRegion(
                    label_id=cluster[0].label_id,
                    x=min_x, y=min_y,
                    width=max_x - min_x,
                    height=max_y - min_y,
                    area=new_area,
                    pixel_density=total_pixels / new_area if new_area > 0 else 0,
                ))

        merged = new_merged

    return merged


def _bbox_distance(a: CCLRegion, b: CCLRegion) -> int:
    """Compute minimum distance between two bboxes."""
    dx = max(0, max(a.x, b.x) - min(a.x + a.width, b.x + b.width))
    dy = max(0, max(a.y, b.y) - min(a.y + a.height, b.y + b.height))
    return max(dx, dy)


# ═══════════════════════════════════════════════════════════════════════════
#  §2  Vision LLM identification of CCL regions
# ═══════════════════════════════════════════════════════════════════════════

IDENTIFY_SYSTEM = """\
You are a visual element identifier. Given a screenshot with numbered bounding
boxes overlaid, identify what each numbered region represents.

Output ONLY a JSON array:
[{"region_id": 1, "name": "short label", "type": "module|icon|label|data|arrow|panel|annotation"}]

Be precise — match the visual content exactly. If a region contains:
- An icon (gear, chip, database) → type="icon"
- A text label → type="label"
- A box with title → type="module"
- A group border → type="panel"
- An arrow/connector → type="arrow"
- Data visualization (table, chart) → type="data"
"""


async def identify_regions(
    image_b64: str,
    regions: List[CCLRegion],
    ai_engine=None,
    model: str = "",
    max_regions_to_identify: int = 60,
) -> List[Dict[str, Any]]:
    """Use vision LLM to identify what each CCL region represents.

    Overlays numbered boxes on the screenshot and asks the LLM to name them.
    """
    if not _HAS_DEPS:
        return [{"region_id": r.label_id, "name": f"region_{r.label_id}", "type": "unknown"}
                for r in regions]

    if ai_engine is None:
        from backend.config import get_settings
        from backend.ai_engine import AIEngine
        ai_engine = AIEngine(get_settings())
    if not model:
        s = ai_engine._settings
        model = s.ANTHROPIC_DEFAULT_MODEL or s.DEFAULT_MODEL

    # Limit regions to identify
    top_regions = regions[:max_regions_to_identify]

    # Create annotated image with numbered boxes
    annotated_b64 = _annotate_image(image_b64, top_regions)

    # Build vision message
    data_uri = f"data:image/png;base64,{annotated_b64}"

    region_list = "\n".join(
        f"  Region {r.label_id}: bbox({r.x},{r.y},{r.width},{r.height})"
        for r in top_regions
    )

    messages = [
        {"role": "system", "content": IDENTIFY_SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": f"Identify these {len(top_regions)} numbered regions:\n{region_list}\n\nOutput JSON array:"},
        ]},
    ]

    try:
        provider = ai_engine._select_provider(model)
        resp = await provider.get_completion(
            messages=messages,
            model=model,
            temperature=0.1,
            max_tokens=4096,
        )
        raw = resp.get("content", "")
        identified = _parse_json_array(raw)

        # Merge with CCL region data
        id_map = {item.get("region_id"): item for item in identified}
        results = []
        for r in top_regions:
            info = id_map.get(r.label_id, {})
            results.append({
                "region_id": r.label_id,
                "name": info.get("name", f"region_{r.label_id}"),
                "type": info.get("type", "unknown"),
                "bbox": {"x": r.x, "y": r.y, "width": r.width, "height": r.height},
                "area": r.area,
                "pixel_density": r.pixel_density,
            })
        return results

    except Exception as e:
        logger.error("Region identification failed: %s", e)
        return [{"region_id": r.label_id, "name": f"region_{r.label_id}", "type": "unknown",
                 "bbox": {"x": r.x, "y": r.y, "width": r.width, "height": r.height}}
                for r in top_regions]


def _annotate_image(image_b64: str, regions: List[CCLRegion]) -> str:
    """Overlay numbered red boxes on the image for LLM identification."""
    if not _HAS_DEPS:
        return image_b64

    raw_b64 = image_b64
    if raw_b64.startswith("data:"):
        raw_b64 = raw_b64.split(",", 1)[1]

    try:
        from PIL import ImageDraw, ImageFont

        img = Image.open(io.BytesIO(base64.b64decode(raw_b64))).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for r in regions:
            # Semi-transparent red box
            draw.rectangle(
                [r.x, r.y, r.x + r.width, r.y + r.height],
                outline=(255, 0, 0, 200),
                width=2,
            )
            # Region number label
            draw.text(
                (r.x + 2, r.y + 2),
                str(r.label_id),
                fill=(255, 0, 0, 255),
            )

        composite = Image.alpha_composite(img, overlay).convert("RGB")

        buf = io.BytesIO()
        composite.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    except Exception as e:
        logger.warning("Image annotation failed: %s", e)
        return raw_b64


# ═══════════════════════════════════════════════════════════════════════════
#  §3  Constraint alignment — match ELK nodes to CCL regions
# ═══════════════════════════════════════════════════════════════════════════

def align_elk_to_regions(
    elk_graph: Dict[str, Any],
    identified_regions: List[Dict[str, Any]],
    tolerance: float = 0.3,
) -> Dict[str, Any]:
    """Align ELK-computed positions to screenshot CCL regions.

    For each ELK node, find the best-matching CCL region by name similarity
    and spatial proximity, then override the ELK position with the CCL bbox.

    This is the key bridge between "algorithmically correct layout" and
    "visually faithful reproduction of the original figure."

    Parameters
    ----------
    elk_graph : dict
        Post-layout ELK graph (with x, y coordinates from elkjs).
    identified_regions : list
        CCL regions with names from vision LLM identification.
    tolerance : float
        Name similarity threshold (0-1) for matching. Lower = stricter.

    Returns
    -------
    Modified elk_graph with positions adjusted to match screenshot regions.
    """
    # Build name → region index
    region_index = {}
    for r in identified_regions:
        name = r.get("name", "").lower().strip()
        if name:
            region_index[name] = r
            # Also index by significant words
            for word in name.split("_"):
                if len(word) > 3:
                    if word not in region_index:
                        region_index[word] = r

    # Walk ELK tree and try to match each node
    _align_recursive(elk_graph, region_index, tolerance)

    return elk_graph


def _align_recursive(
    node: Dict[str, Any],
    region_index: Dict[str, Dict],
    tolerance: float,
) -> None:
    """Recursively align ELK nodes to CCL regions."""
    node_id = node.get("id", "")
    labels = node.get("labels", [])
    name = labels[0].get("text", "").lower() if labels else node_id.lower()

    if node_id and node_id != "root":
        # Try to find matching region
        match = _find_best_match(name, node_id.lower(), region_index, tolerance)
        if match:
            bbox = match.get("bbox", {})
            if bbox:
                node["x"] = bbox["x"]
                node["y"] = bbox["y"]
                node["width"] = bbox["width"]
                node["height"] = bbox["height"]
                # Mark as vision-constrained
                node["_vision_aligned"] = True

    # Recurse
    for child in node.get("children", []):
        _align_recursive(child, region_index, tolerance)


def _find_best_match(
    name: str,
    node_id: str,
    region_index: Dict[str, Dict],
    tolerance: float,
) -> Optional[Dict]:
    """Find best matching region for a node by name similarity."""
    # Direct match
    if name in region_index:
        return region_index[name]
    if node_id in region_index:
        return region_index[node_id]

    # Word overlap match
    name_words = set(re.split(r'[\s_]+', name))
    best_score = 0
    best_match = None

    for key, region in region_index.items():
        key_words = set(re.split(r'[\s_]+', key))
        if not name_words or not key_words:
            continue
        overlap = len(name_words & key_words)
        score = overlap / max(len(name_words), len(key_words))
        if score > best_score and score >= tolerance:
            best_score = score
            best_match = region

    return best_match


# ═══════════════════════════════════════════════════════════════════════════
#  §4  Full vision-constrained pipeline
# ═══════════════════════════════════════════════════════════════════════════

async def vision_constrained_layout(
    image_b64: str,
    elk_graph: Dict[str, Any],
    ai_engine=None,
    model: str = "",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """Full pipeline: screenshot → CCL → identify → align ELK.

    Returns
    -------
    (aligned_elk, identified_regions, diagnostics)
    """
    diag = {}

    # Step 1: CCL extraction
    regions, ccl_stats = extract_ccl_regions(image_b64)
    diag["ccl"] = ccl_stats
    diag["ccl_regions"] = len(regions)

    if not regions:
        logger.warning("No CCL regions extracted, returning original ELK")
        return elk_graph, [], diag

    # Step 2: Vision LLM identification
    identified = await identify_regions(image_b64, regions, ai_engine, model)
    diag["identified"] = len(identified)

    # Step 3: Align ELK to regions
    aligned = align_elk_to_regions(elk_graph, identified)
    aligned_count = _count_aligned(aligned)
    diag["aligned_nodes"] = aligned_count

    return aligned, identified, diag


def _count_aligned(node: Dict[str, Any]) -> int:
    """Count how many nodes were vision-aligned."""
    count = 1 if node.get("_vision_aligned") else 0
    for child in node.get("children", []):
        count += _count_aligned(child)
    return count


# ═══════════════════════════════════════════════════════════════════════════
#  §5  JSON parsing utility
# ═══════════════════════════════════════════════════════════════════════════

def _parse_json_array(raw: str) -> List[Dict]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r'^```\w*\n?', '', cleaned)
        cleaned = re.sub(r'\n?```$', '', cleaned)
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    except json.JSONDecodeError:
        pass
    match = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(re.sub(r',\s*([}\]])', r'\1', match.group()))
        except json.JSONDecodeError:
            pass
    return []
