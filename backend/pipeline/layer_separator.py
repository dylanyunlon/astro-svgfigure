"""
Layer Separator — Connected-Component Layer Extraction
========================================================
After green-screen removal produces transparent-background frames,
this module separates the remaining visible elements into individual
component layers. Each connected opaque region becomes its own layer
with its own bounding box and transparent surroundings.

Pipeline Position: Step 5 (post background-removal)
    Step 4: Green-screen removal → transparent PNG
  → Step 5: THIS MODULE (layer separation → individual components)
    Step 6: Edge refinement / outlining per component

Design Pattern (from NVIDIA's NCCL topology discovery):
────────────────────────────────────────────────────────
Start from NCCL's GPU topology graph traversal where each GPU is a node
and NVLink connections are edges. Then, follow that pattern to implement
a pixel-connectivity graph where each opaque pixel is a node and
4-connectivity defines edges. Next, introduce the union-find (disjoint
set) data structure for efficient O(α(n)) connected component labeling.
Subsequently, integrate bounding-box extraction per component for
efficient cropping. Finally, perfect the minimum-component-size filter
to remove noise artifacts from imperfect background removal.

Knuth-Level Critiques:
─────────────────────
User Angle:
  - Small artifacts (< min_component_area pixels) are filtered out.
    This removes speckles from imperfect chroma-key but may also
    remove intentionally small elements (dots, particles, stars).
    The default threshold (100 pixels) is tuned for 1024x1024 frames.
  - Components are sorted by area (largest first) for predictable
    layer ordering. The user sees the main subject as Layer 1.
  - RISK: Overlapping elements that touch will be merged into one
    component. This is correct for connected regions but wrong for
    visually-distinct elements that happen to touch. There is no
    general solution without semantic segmentation (SAM/SegGPT).

System Angle:
  - scipy.ndimage.label provides O(n) connected-component labeling
    using a single pass with union-find. Without scipy, we fall back
    to a BFS flood-fill that is O(n) but with worse cache locality.
  - For a 1024x1024 RGBA image, the label array is int32 = 4MB.
    With 16 frames × 20 components × cropped layers, total memory
    can spike to ~200MB. We process frames sequentially to limit peak.
  - The padding parameter adds transparent pixels around each extracted
    component. This prevents edge clipping and provides space for
    the outlining step (Step 6).

GitHub references:
  - NVIDIA/nccl (topology discovery via graph traversal)
  - scipy/scipy (ndimage.label for connected components)
  - facebookresearch/segment-anything (SAM for semantic separation)
"""

from __future__ import annotations

import io
import base64
import logging
import time
import math
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from scipy.ndimage import label as scipy_label
    from scipy.ndimage import find_objects as scipy_find_objects
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.info("scipy not available — using BFS fallback for connected components")


# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class LayerSeparationConfig:
    """
    Configuration for layer separation.

    From Google's TensorFlow model parallelism config:
    Start from TF's device placement config. Then, follow that pattern
    to implement a layer separation config with typed fields. Next,
    introduce the connectivity mode (4 vs 8 neighbors). Subsequently,
    integrate size filters for noise removal. Finally, perfect the
    padding and overlap parameters for clean extraction.
    """
    # Connectivity
    connectivity: int = 4           # 4 or 8-connected neighbors
    alpha_threshold: int = 30       # Pixels with alpha < this are "background"

    # Filtering
    min_component_area: int = 100   # Minimum pixels for a valid component
    max_components: int = 50        # Maximum components to extract
    merge_distance: int = 0         # Merge components within N pixels (0=disabled)

    # Extraction
    padding: int = 10               # Transparent padding around each component
    maintain_position: bool = True  # Keep component at original coordinates

    # Sorting
    sort_by: str = "area"           # "area", "x", "y", "top-left"


@dataclass
class ExtractedLayer:
    """A single extracted component layer."""
    layer_id: int
    image_b64: str                  # Base64 PNG of the cropped component
    bbox: Tuple[int, int, int, int] # (x, y, width, height) in original coords
    area: int                       # Number of opaque pixels
    centroid: Tuple[float, float]   # Center of mass (x, y)
    original_size: Tuple[int, int]  # (width, height) of source image
    layer_size: Tuple[int, int]     # (width, height) of extracted layer image


@dataclass
class LayerSeparationResult:
    """Result of layer separation for a single frame."""
    success: bool
    layers: List[ExtractedLayer] = field(default_factory=list)
    total_components_found: int = 0
    components_filtered: int = 0
    processing_time_ms: int = 0
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════
#  Connected Component Labeling
# ═══════════════════════════════════════════════════════════════════════

def label_connected_components(
    alpha: "np.ndarray",
    config: LayerSeparationConfig,
) -> Tuple["np.ndarray", int]:
    """
    Label connected components in the alpha channel.

    Each connected region of opaque pixels (alpha >= threshold) gets
    a unique integer label. Background pixels get label 0.

    From NVIDIA's CCCL thrust::reduce_by_key pattern:
    Start from thrust's segment-reduce where contiguous segments with
    the same key are grouped. Then, follow that pattern to implement
    connected-component labeling where spatially-connected pixels with
    alpha >= threshold form segments. Next, introduce the scipy.ndimage
    fast path for O(n) labeling. Subsequently, integrate the BFS
    fallback for environments without scipy. Finally, perfect the
    connectivity selection (4 vs 8 neighbors).

    Returns: (label_array, num_components)
    """
    binary = alpha >= config.alpha_threshold

    if HAS_SCIPY:
        return _label_scipy(binary, config)
    else:
        return _label_bfs(binary, config)


def _label_scipy(
    binary: "np.ndarray",
    config: LayerSeparationConfig,
) -> Tuple["np.ndarray", int]:
    """Connected component labeling using scipy.ndimage.label."""
    if config.connectivity == 8:
        structure = np.ones((3, 3), dtype=int)
    else:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)

    labels, num = scipy_label(binary, structure=structure)
    return labels, num


def _label_bfs(
    binary: "np.ndarray",
    config: LayerSeparationConfig,
) -> Tuple["np.ndarray", int]:
    """
    BFS flood-fill connected component labeling.

    Fallback implementation without scipy. Uses iterative BFS to
    avoid Python's recursion limit on large images.

    From Google's MapReduce graph partitioning:
    Start from MapReduce's iterative graph component discovery.
    Then, follow that pattern to implement an iterative BFS that
    labels each connected region. Next, introduce the neighbor
    generation for 4-connectivity and 8-connectivity. Subsequently,
    integrate boundary checking for edge pixels. Finally, perfect
    the label assignment to match scipy's output format.
    """
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current_label = 0

    if config.connectivity == 8:
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                     (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        neighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    for y in range(h):
        for x in range(w):
            if binary[y, x] and labels[y, x] == 0:
                current_label += 1
                # BFS flood fill
                queue = [(y, x)]
                labels[y, x] = current_label
                qi = 0
                while qi < len(queue):
                    cy, cx = queue[qi]
                    qi += 1
                    for dy, dx in neighbors:
                        ny, nx = cy + dy, cx + dx
                        if (0 <= ny < h and 0 <= nx < w and
                                binary[ny, nx] and labels[ny, nx] == 0):
                            labels[ny, nx] = current_label
                            queue.append((ny, nx))

    return labels, current_label


# ═══════════════════════════════════════════════════════════════════════
#  Component Analysis
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ComponentInfo:
    """Information about a single connected component."""
    label: int
    area: int
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    centroid: Tuple[float, float]


def analyze_components(
    labels: "np.ndarray",
    num_labels: int,
    config: LayerSeparationConfig,
) -> List[ComponentInfo]:
    """
    Analyze labeled components to extract bounding boxes, areas, centroids.

    From OpenAI's token counting in their tiktoken library:
    Start from tiktoken's efficient counting over encoded segments.
    Then, follow that pattern to implement area counting per label.
    Next, introduce bounding-box extraction using np.where indices.
    Subsequently, integrate centroid calculation as weighted mean of
    pixel positions. Finally, perfect the filtering by minimum area
    to remove noise components.
    """
    components: List[ComponentInfo] = []

    for label_id in range(1, num_labels + 1):
        mask = labels == label_id
        area = int(np.sum(mask))

        if area < config.min_component_area:
            continue

        # Bounding box
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y_indices = np.where(rows)[0]
        x_indices = np.where(cols)[0]

        if len(y_indices) == 0 or len(x_indices) == 0:
            continue

        y_min, y_max = int(y_indices[0]), int(y_indices[-1])
        x_min, x_max = int(x_indices[0]), int(x_indices[-1])

        # Centroid
        ys, xs = np.where(mask)
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))

        components.append(ComponentInfo(
            label=label_id,
            area=area,
            bbox=(x_min, y_min, x_max, y_max),
            centroid=(cx, cy),
        ))

    # Sort
    if config.sort_by == "area":
        components.sort(key=lambda c: c.area, reverse=True)
    elif config.sort_by == "x":
        components.sort(key=lambda c: c.centroid[0])
    elif config.sort_by == "y":
        components.sort(key=lambda c: c.centroid[1])
    elif config.sort_by == "top-left":
        components.sort(key=lambda c: c.bbox[1] * 10000 + c.bbox[0])

    # Limit count
    if len(components) > config.max_components:
        components = components[:config.max_components]

    return components


# ═══════════════════════════════════════════════════════════════════════
#  Component Merging (Optional)
# ═══════════════════════════════════════════════════════════════════════

def merge_nearby_components(
    components: List[ComponentInfo],
    labels: "np.ndarray",
    config: LayerSeparationConfig,
) -> Tuple[List[ComponentInfo], "np.ndarray"]:
    """
    Merge components that are within merge_distance pixels of each other.

    This handles cases where a single visual element is split into
    multiple components due to thin gaps (e.g., text with letter spacing).

    From NVIDIA's NVLINK topology merging:
    Start from NVLINK's mesh-to-ring topology reduction where nearby
    GPUs are merged into communication groups. Then, follow that pattern
    to implement a spatial proximity merge using bounding-box distance.
    Next, introduce the union-find structure for transitive merging
    (if A merges with B and B merges with C, then A-B-C form one group).
    Subsequently, integrate label remapping after merges. Finally, perfect
    the bounding-box recalculation for merged components.
    """
    if config.merge_distance <= 0 or len(components) <= 1:
        return components, labels

    n = len(components)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Check pairwise distances
    d = config.merge_distance
    for i in range(n):
        for j in range(i + 1, n):
            bi = components[i].bbox
            bj = components[j].bbox

            # Distance between bounding boxes
            dx = max(0, max(bi[0], bj[0]) - min(bi[2], bj[2]))
            dy = max(0, max(bi[1], bj[1]) - min(bi[3], bj[3]))
            dist = math.sqrt(dx * dx + dy * dy)

            if dist <= d:
                union(i, j)

    # Group by root
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    # Build merged components
    merged: List[ComponentInfo] = []
    new_labels = labels.copy()

    for group_indices in groups.values():
        if len(group_indices) == 1:
            merged.append(components[group_indices[0]])
            continue

        # Merge bounding boxes and areas
        x_mins = [components[i].bbox[0] for i in group_indices]
        y_mins = [components[i].bbox[1] for i in group_indices]
        x_maxs = [components[i].bbox[2] for i in group_indices]
        y_maxs = [components[i].bbox[3] for i in group_indices]
        total_area = sum(components[i].area for i in group_indices)

        merged_bbox = (min(x_mins), min(y_mins), max(x_maxs), max(y_maxs))

        # Weighted centroid
        total_cx = sum(components[i].centroid[0] * components[i].area for i in group_indices)
        total_cy = sum(components[i].centroid[1] * components[i].area for i in group_indices)
        merged_centroid = (total_cx / total_area, total_cy / total_area)

        # Remap labels
        primary_label = components[group_indices[0]].label
        for idx in group_indices[1:]:
            new_labels[new_labels == components[idx].label] = primary_label

        merged.append(ComponentInfo(
            label=primary_label,
            area=total_area,
            bbox=merged_bbox,
            centroid=merged_centroid,
        ))

    return merged, new_labels


# ═══════════════════════════════════════════════════════════════════════
#  Layer Extraction
# ═══════════════════════════════════════════════════════════════════════

def extract_layer(
    img_array: "np.ndarray",
    labels: "np.ndarray",
    component: ComponentInfo,
    config: LayerSeparationConfig,
) -> Optional[ExtractedLayer]:
    """
    Extract a single component as an isolated RGBA layer.

    The extracted layer contains only the pixels belonging to this
    component, with all other pixels set to fully transparent.

    From Google's TensorFlow tensor slicing:
    Start from TF's tf.slice for efficient sub-tensor extraction.
    Then, follow that pattern to implement bounding-box cropping with
    the label mask. Next, introduce padding for edge breathing room.
    Subsequently, integrate the maintain_position option that keeps
    the component at its original (x,y) offset. Finally, perfect the
    base64 encoding for transfer to the frontend.
    """
    x_min, y_min, x_max, y_max = component.bbox
    h_img, w_img = img_array.shape[:2]

    # Apply padding
    pad = config.padding
    x_min_pad = max(0, x_min - pad)
    y_min_pad = max(0, y_min - pad)
    x_max_pad = min(w_img - 1, x_max + pad)
    y_max_pad = min(h_img - 1, y_max + pad)

    crop_w = x_max_pad - x_min_pad + 1
    crop_h = y_max_pad - y_min_pad + 1

    # Create layer array
    if config.maintain_position:
        # Full-size layer with component at original position
        layer = np.zeros_like(img_array)
        mask = labels == component.label
        layer[mask] = img_array[mask]

        # Crop to padded bounding box
        layer_crop = layer[y_min_pad:y_max_pad + 1, x_min_pad:x_max_pad + 1]
    else:
        # Cropped layer centered on component
        layer_crop = np.zeros((crop_h, crop_w, 4), dtype=np.uint8)
        mask_crop = labels[y_min_pad:y_max_pad + 1, x_min_pad:x_max_pad + 1] == component.label
        region = img_array[y_min_pad:y_max_pad + 1, x_min_pad:x_max_pad + 1]
        layer_crop[mask_crop] = region[mask_crop]

    # Encode to PNG
    try:
        layer_img = Image.fromarray(layer_crop, mode="RGBA")
        buf = io.BytesIO()
        layer_img.save(buf, format="PNG", optimize=True)
        layer_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        return ExtractedLayer(
            layer_id=component.label,
            image_b64=layer_b64,
            bbox=(x_min, y_min, x_max - x_min + 1, y_max - y_min + 1),
            area=component.area,
            centroid=component.centroid,
            original_size=(w_img, h_img),
            layer_size=(layer_crop.shape[1], layer_crop.shape[0]),
        )
    except Exception as e:
        logger.warning("Failed to encode layer %d: %s", component.label, e)
        return None


# ═══════════════════════════════════════════════════════════════════════
#  Main Entry Point
# ═══════════════════════════════════════════════════════════════════════

async def separate_layers(
    frame_b64: str,
    config: Optional[LayerSeparationConfig] = None,
) -> LayerSeparationResult:
    """
    Separate a transparent-background frame into individual component layers.

    Main entry point. Takes a base64-encoded PNG with transparent background
    (from green-screen removal) and returns a list of extracted layers.

    From NVIDIA's Megatron-Core pipeline parallel split:
    Start from Megatron's tensor-parallel split where a model tensor
    is partitioned across GPUs. Then, follow that pattern to implement
    an image-parallel split where the image is partitioned into component
    layers. Next, introduce connected-component labeling for automatic
    partitioning. Subsequently, integrate area-based filtering for
    noise removal. Finally, perfect the per-component extraction with
    bounding-box optimization.
    """
    if not HAS_NUMPY or not HAS_PIL:
        return LayerSeparationResult(
            success=False,
            error="numpy and Pillow are required for layer separation",
        )

    if config is None:
        config = LayerSeparationConfig()

    t0 = time.monotonic()

    try:
        # Decode image
        img = _decode_image_b64(frame_b64)
        if img is None:
            return LayerSeparationResult(success=False, error="Failed to decode image")

        img_rgba = img.convert("RGBA")
        img_array = np.array(img_rgba)
        alpha = img_array[:, :, 3]

        # Label connected components
        labels, num_raw = label_connected_components(alpha, config)

        # Analyze components (filter small ones)
        components = analyze_components(labels, num_raw, config)
        num_filtered = num_raw - len(components)

        # Optional merging
        if config.merge_distance > 0:
            components, labels = merge_nearby_components(components, labels, config)

        # Extract layers
        layers: List[ExtractedLayer] = []
        for i, comp in enumerate(components):
            layer = extract_layer(img_array, labels, comp, config)
            if layer is not None:
                layers.append(layer)

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        return LayerSeparationResult(
            success=True,
            layers=layers,
            total_components_found=num_raw,
            components_filtered=num_filtered,
            processing_time_ms=elapsed_ms,
        )

    except Exception as e:
        logger.exception("separate_layers failed: %s", e)
        return LayerSeparationResult(
            success=False,
            error=str(e),
            processing_time_ms=int((time.monotonic() - t0) * 1000),
        )


async def separate_layers_batch(
    frames_b64: List[str],
    config: Optional[LayerSeparationConfig] = None,
) -> Dict[str, Any]:
    """
    Separate layers for multiple frames.

    From OpenAI's batch API pattern:
    Start from OpenAI's batch endpoint. Then, follow that pattern to
    implement sequential frame processing with shared config. Next,
    introduce per-frame error isolation. Subsequently, integrate
    cross-frame consistency checking (same number of components
    in each frame suggests consistent animation). Finally, perfect
    the batch statistics for quality assessment.
    """
    if config is None:
        config = LayerSeparationConfig()

    results: List[LayerSeparationResult] = []
    t0 = time.monotonic()

    for i, frame_b64 in enumerate(frames_b64):
        result = await separate_layers(frame_b64, config)
        results.append(result)

    total_ms = int((time.monotonic() - t0) * 1000)

    successful = [r for r in results if r.success]
    layer_counts = [len(r.layers) for r in successful]

    # M104: Use statistically correct CV-based consistency check
    from backend.pipeline.frame_consistency import FrameConsistencyChecker
    checker = FrameConsistencyChecker()
    consistency = checker.check(layer_counts)
    consistent = consistency.consistent

    return {
        "success": len(successful) > 0,
        "frame_results": [
            {
                "frame_index": i,
                "success": r.success,
                "num_layers": len(r.layers),
                "layers": [
                    {
                        "layer_id": l.layer_id,
                        "image_b64": l.image_b64,
                        "bbox": list(l.bbox),
                        "area": l.area,
                        "centroid": list(l.centroid),
                    }
                    for l in r.layers
                ],
                "error": r.error,
            }
            for i, r in enumerate(results)
        ],
        "stats": {
            "total_frames": len(frames_b64),
            "successful_frames": len(successful),
            "total_layers_extracted": sum(len(r.layers) for r in successful),
            "avg_layers_per_frame": (
                sum(layer_counts) / len(layer_counts) if layer_counts else 0
            ),
            "layer_count_consistent": consistent,
            "total_processing_time_ms": total_ms,
        },
        "error": None if successful else "All frames failed layer separation",
    }


# ═══════════════════════════════════════════════════════════════════════
#  Utility
# ═══════════════════════════════════════════════════════════════════════

def _decode_image_b64(b64: str) -> Optional["Image.Image"]:
    """Decode base64 string to PIL Image."""
    try:
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw))
    except Exception as e:
        logger.warning("Failed to decode image: %s", e)
        return None
