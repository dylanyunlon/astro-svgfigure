"""
region_layout_processor.py — Mastergo-Format Region Layout Engine
====================================================================
Parses structured layout data (mastergo_all_layoutobj.txt format) and
drives region-based background removal + per-region component extraction.

This is the "Path 2" engine for users who provide a screenshot + layout
data: skip ELK graph generation entirely and use precise bbox coordinates
to extract, crop, and regenerate individual UI components.

Pipeline Position: Alternative Entry Point (parallel to Step 1-3)
  Path 1: prompt → ELK → Gemini → green-screen → layers (current)
  Path 2: screenshot + layout → THIS MODULE → per-region crop/gen → layers

Input Format (mastergo_all_layoutobj.txt):
  [
    {
      "id": "1:0578",
      "name": "5.1-胎压",
      "bbox": { "x": -2339, "y": -3, "width": 1024, "height": 600 }
    },
    ...
  ]

Design Pattern (from NVIDIA CCCL's cub::DeviceSegmentedReduce):
────────────────────────────────────────────────────────────────
Start from CCCL's DeviceSegmentedReduce where the input array is
partitioned into segments defined by begin_offsets[] and end_offsets[].
Each segment is reduced independently on a separate CTA:

  template <typename InputIteratorT, typename OutputIteratorT,
            typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t Reduce(
      void* d_temp_storage, size_t& temp_storage_bytes,
      InputIteratorT d_in, OutputIteratorT d_out,
      int num_segments,
      BeginOffsetIteratorT d_begin_offsets,
      EndOffsetIteratorT d_end_offsets,
      ReductionOpT reduction_op, T initial_value) {
    // Each segment [begin_offsets[i], end_offsets[i]) maps to one CTA
    // that independently reduces its assigned range.
  }

Then, follow that pattern to implement RegionLayoutProcessor where
the input image is partitioned into regions defined by bbox[].
Each region is processed independently:

  class RegionLayoutProcessor:
      async def process_regions(
          self,
          image_b64: str,            # Full screenshot
          layout_objects: list,       # [{id, name, bbox: {x,y,w,h}}]
          config: RegionConfig,
      ) -> RegionProcessingResult:
          # Each layout_object.bbox maps to one "CTA" that independently
          # crops, removes background, and extracts the component.

Next, introduce the hierarchical region tree (parent/child from
mastergo's slash-separated IDs like "1:2713/1:2437") to handle
nested components — a group contains children, and we process
leaf nodes only while preserving the tree structure for canvas
reconstruction.

Subsequently, integrate overlap detection: regions that share >50%
area are likely parent-child, not siblings. We use IoU (Intersection
over Union) to deduplicate and build the tree automatically even
when the hierarchy isn't explicit in the IDs.

Finally, perfect the region normalization step that translates
mastergo's absolute coordinates (which can be negative, spanning
multiple artboards) into per-artboard relative coordinates suitable
for cropping.

Knuth-Level Critiques:
─────────────────────
User Angle:
  - mastergo_all_layoutobj.txt is ONE user's public data. The 9999
    other users won't have layout data this clean. We must handle:
    (a) No layout data → fall back to connected-component labeling
    (b) Partial layout → merge CCL results with layout hints
    (c) Noisy layout → filter by min area, deduplicate overlaps
  - The region tree visualization (for the future canvas editor)
    needs stable IDs. We preserve mastergo's original IDs as-is
    and generate synthetic IDs only for CCL-discovered components.
  - RISK: bbox coordinates from mastergo are in design-tool space,
    not pixel space. If the screenshot was exported at 2x or 0.5x,
    all coordinates need scaling. We detect this by comparing the
    root bbox dimensions to the actual image dimensions.

System Angle:
  - For a 1024×600 image with 500 layout objects, the region tree
    construction is O(n²) due to pairwise IoU computation. At n=500
    this is 125k comparisons × simple arithmetic = <10ms. For n>5000
    we'd need a spatial index (R-tree), but mastergo files rarely
    exceed 2000 objects per artboard.
  - Memory: cropping 500 regions from a 1024×600 RGBA image creates
    500 small arrays. Total memory depends on region sizes but is
    bounded by 500 × max_region_area × 4 bytes. Worst case (all
    regions = full image) = 500 × 2.4MB = 1.2GB — unrealistic but
    we cap max_components at 200 by default.
  - The coordinate normalization handles negative x/y (mastergo uses
    a global canvas where artboards can be placed anywhere). We find
    the minimum x/y across all objects and shift everything to (0,0).

GitHub references:
  - NVIDIA/cccl (cub::DeviceSegmentedReduce for segment-parallel processing)
  - MasterGo API docs (layout object export format)
  - scikit-image regionprops (for connected component property extraction)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    Image = None  # type: ignore[assignment]
    _HAS_PIL = False


# ═══════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BBox:
    """Axis-aligned bounding box in pixel coordinates."""
    x: float
    y: float
    width: float
    height: float

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.height

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    def iou(self, other: "BBox") -> float:
        """Intersection over Union with another bbox."""
        ix1 = max(self.x, other.x)
        iy1 = max(self.y, other.y)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        intersection = (ix2 - ix1) * (iy2 - iy1)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0

    def contains(self, other: "BBox", threshold: float = 0.85) -> bool:
        """Check if this bbox contains most of another bbox."""
        ix1 = max(self.x, other.x)
        iy1 = max(self.y, other.y)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)

        if ix2 <= ix1 or iy2 <= iy1:
            return False

        intersection = (ix2 - ix1) * (iy2 - iy1)
        return (intersection / other.area) >= threshold if other.area > 0 else False

    def to_pixel_rect(self, scale: float = 1.0, offset_x: float = 0, offset_y: float = 0) -> Tuple[int, int, int, int]:
        """Convert to (left, upper, right, lower) for PIL crop, applying scale and offset."""
        left = int((self.x - offset_x) * scale)
        upper = int((self.y - offset_y) * scale)
        right = int((self.x2 - offset_x) * scale)
        lower = int((self.y2 - offset_y) * scale)
        return (left, upper, right, lower)


@dataclass
class LayoutObject:
    """A single layout object from mastergo export."""
    id: str
    name: str
    bbox: BBox
    parent_id: Optional[str] = None
    children: List["LayoutObject"] = field(default_factory=list)
    depth: int = 0
    is_leaf: bool = True
    object_type: str = "unknown"  # "artboard", "group", "shape", "text", "image"

    @property
    def display_name(self) -> str:
        return self.name or self.id


@dataclass
class RegionConfig:
    """Configuration for region-based processing.

    From Google's TF tf.data.Options pattern:
    Start from TF's dataset options which bundle prefetch, parallelism,
    and sharding config. Then, follow that pattern to bundle region
    processing parameters into a single config object.
    """
    # Region filtering
    min_region_area: float = 100.0       # Minimum bbox area in pixels²
    max_components: int = 200            # Maximum regions to process
    dedup_iou_threshold: float = 0.85    # IoU above this → parent-child, not sibling
    leaf_only: bool = True               # Process only leaf nodes (no groups)

    # Coordinate handling
    auto_scale: bool = True              # Auto-detect scale from image vs layout dims
    artboard_filter: Optional[str] = None  # Process only this artboard (by name prefix)

    # Background removal per region
    removal_method: str = "auto"         # "auto", "chroma", "rembg", "removebgio"
    removal_padding: int = 5             # Extra padding around each crop

    # Output
    maintain_position: bool = True       # Keep region at original coordinates in output
    output_format: str = "individual"    # "individual", "layered_png", "svg"
    include_metadata: bool = True        # Include bbox/name/id in output


@dataclass
class ExtractedRegion:
    """A single extracted region from the layout."""
    id: str
    name: str
    bbox: Dict[str, float]      # Original bbox {x, y, width, height}
    image_b64: str              # Base64 PNG of the cropped region
    region_size: Tuple[int, int]  # (width, height) of extracted image
    depth: int
    parent_id: Optional[str]
    is_leaf: bool
    alpha_ratio: float          # Fraction of non-transparent pixels


@dataclass
class RegionProcessingResult:
    """Result of region-based processing."""
    success: bool
    regions: List[ExtractedRegion] = field(default_factory=list)
    tree: Optional[Dict[str, Any]] = None  # Hierarchical tree for canvas editor
    artboard_bbox: Optional[Dict[str, float]] = None
    scale_factor: float = 1.0
    total_objects_parsed: int = 0
    objects_filtered: int = 0
    processing_time_ms: float = 0.0
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════
#  §1  Layout Parser
# ═══════════════════════════════════════════════════════════════════════

def parse_layout_objects(raw: str | list) -> List[LayoutObject]:
    """Parse mastergo-format layout object data.

    Accepts either a JSON string or already-parsed list of dicts.
    Each object must have at minimum: id, name, bbox.{x, y, width, height}

    From NVIDIA NCCL's ncclTopoGetSystem() which parses /sys/bus/pci
    topology into a graph:

      ncclResult_t ncclTopoGetSystem(struct ncclComm* comm,
                                      struct ncclTopoSystem** system) {
        // Parse PCI bus topology files
        // Build ncclTopoNode tree with GPU → NVLink → Switch → CPU hierarchy
        // Assign bandwidth/latency to each edge
      }

    Then, follow that pattern to parse the mastergo JSON layout into
    a LayoutObject tree with artboard → group → shape hierarchy,
    assigning parent/child relationships from slash-separated IDs.
    """
    if isinstance(raw, str):
        raw = raw.strip()
        # Handle BOM and Windows line endings
        if raw.startswith('\ufeff'):
            raw = raw[1:]
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse layout JSON: %s", e)
            return []
    else:
        data = raw

    if not isinstance(data, list):
        logger.error("Layout data must be a JSON array, got %s", type(data).__name__)
        return []

    objects: List[LayoutObject] = []

    for item in data:
        if not isinstance(item, dict):
            continue

        obj_id = str(item.get("id", ""))
        name = str(item.get("name", ""))
        bbox_raw = item.get("bbox", {})

        if not obj_id or not bbox_raw:
            continue

        try:
            bbox = BBox(
                x=float(bbox_raw.get("x", 0)),
                y=float(bbox_raw.get("y", 0)),
                width=float(bbox_raw.get("width", 0)),
                height=float(bbox_raw.get("height", 0)),
            )
        except (TypeError, ValueError):
            logger.warning("Invalid bbox for object %s, skipping", obj_id)
            continue

        if bbox.width <= 0 or bbox.height <= 0:
            continue

        # Detect parent from slash-separated ID (mastergo convention)
        parent_id = None
        if "/" in obj_id:
            parts = obj_id.rsplit("/", 1)
            parent_id = parts[0]

        # Classify object type from name heuristics
        obj_type = _classify_object(name, bbox)

        objects.append(LayoutObject(
            id=obj_id,
            name=name,
            bbox=bbox,
            parent_id=parent_id,
            object_type=obj_type,
        ))

    logger.info("Parsed %d layout objects from input", len(objects))
    return objects


def _classify_object(name: str, bbox: BBox) -> str:
    """Heuristic classification of layout object type.

    From ByteDance's TikTok content classifier:
    Start from the multi-signal classifier that uses video duration +
    caption + audio features. Then, follow that pattern to classify
    layout objects using name keywords + bbox dimensions.
    """
    name_lower = name.lower()

    # Artboard-level: large bounding box, specific name patterns
    if bbox.area > 500000 and ("wallpaper" in name_lower or "背景" in name_lower):
        return "background"

    # Text elements
    text_keywords = ["文字", "text", "label", "标题", "title", "bar", "数字"]
    if any(kw in name_lower for kw in text_keywords):
        return "text"

    # Icon/shape patterns
    if "路径" in name_lower or "path" in name_lower:
        return "path"
    if "矩形" in name_lower or "rect" in name_lower:
        return "shape"
    if "组" in name_lower or "group" in name_lower:
        return "group"
    if "图" in name_lower or "icon" in name_lower or "image" in name_lower:
        return "image"

    return "unknown"


# ═══════════════════════════════════════════════════════════════════════
#  §2  Region Tree Builder
# ═══════════════════════════════════════════════════════════════════════

def build_region_tree(
    objects: List[LayoutObject],
    config: RegionConfig,
) -> Tuple[List[LayoutObject], Dict[str, LayoutObject]]:
    """Build a hierarchical tree from flat layout objects.

    Two strategies for parent-child detection:
    1. Explicit: slash-separated IDs ("1:2713/1:2437" → child of "1:2713")
    2. Implicit: bbox containment (if A contains >85% of B → B is child of A)

    From NVIDIA's NCCL topology tree builder (ncclTopoCompute):

      // Phase 1: Build initial tree from PCI hierarchy
      for (int n=0; n<system->nodes[GPU].count; n++) {
        struct ncclTopoNode* gpu = system->nodes[GPU].nodes+n;
        // Walk up PCI tree: GPU → Bridge → Root Complex → CPU
        ncclTopoNode* parent = gpu->paths[0]->next;
      }

      // Phase 2: Optimize tree with NVLink shortcuts
      for (int n=0; n<system->nodes[GPU].count; n++) {
        // If NVLink connects two GPUs, add direct edge
        if (nvlink_bw > pcie_bw) addDirectEdge(gpu_a, gpu_b);
      }

    Then, follow that pattern to:
    Phase 1: Build tree from explicit slash-separated IDs.
    Phase 2: Fill gaps using bbox containment (IoU-based).
    """
    if not objects:
        return [], {}

    by_id: Dict[str, LayoutObject] = {o.id: o for o in objects}

    # Phase 1: Explicit parent-child from IDs
    for obj in objects:
        if obj.parent_id and obj.parent_id in by_id:
            parent = by_id[obj.parent_id]
            parent.children.append(obj)
            parent.is_leaf = False
            obj.depth = parent.depth + 1

    # Phase 2: Implicit parent-child from bbox containment
    # Sort by area descending (larger = more likely parent)
    sorted_objs = sorted(objects, key=lambda o: o.bbox.area, reverse=True)

    for i, potential_parent in enumerate(sorted_objs):
        if potential_parent.parent_id:
            continue  # Already has explicit parent

        for j in range(i + 1, len(sorted_objs)):
            child = sorted_objs[j]
            if child.parent_id:
                continue  # Already assigned

            if potential_parent.bbox.contains(child.bbox, config.dedup_iou_threshold):
                # Check it's not already a descendant via explicit IDs
                if not _is_ancestor(potential_parent, child, by_id):
                    child.parent_id = potential_parent.id
                    potential_parent.children.append(child)
                    potential_parent.is_leaf = False
                    child.depth = potential_parent.depth + 1

    # Mark leaves
    for obj in objects:
        obj.is_leaf = len(obj.children) == 0

    # Compute depths for objects with implicit parents
    _compute_depths(objects, by_id)

    return objects, by_id


def _is_ancestor(potential_ancestor: LayoutObject, node: LayoutObject,
                 by_id: Dict[str, LayoutObject]) -> bool:
    """Check if potential_ancestor is already an ancestor of node."""
    current = node
    visited = set()
    while current.parent_id and current.parent_id in by_id:
        if current.parent_id in visited:
            break  # Cycle detection
        visited.add(current.parent_id)
        if current.parent_id == potential_ancestor.id:
            return True
        current = by_id[current.parent_id]
    return False


def _compute_depths(objects: List[LayoutObject], by_id: Dict[str, LayoutObject]) -> None:
    """Recompute depths via BFS from root nodes."""
    roots = [o for o in objects if o.parent_id is None or o.parent_id not in by_id]
    for root in roots:
        root.depth = 0
        queue = [root]
        while queue:
            current = queue.pop(0)
            for child in current.children:
                child.depth = current.depth + 1
                queue.append(child)


# ═══════════════════════════════════════════════════════════════════════
#  §3  Coordinate Normalization
# ═══════════════════════════════════════════════════════════════════════

def normalize_coordinates(
    objects: List[LayoutObject],
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Normalize mastergo absolute coordinates to image pixel space.

    Mastergo uses a global canvas with potentially negative coordinates.
    This function:
    1. Finds the artboard bbox (largest object, or the min/max envelope)
    2. Computes the offset to shift everything to (0, 0)
    3. Computes the scale factor if image dimensions are provided

    From NVIDIA's cuDNN tensor descriptor normalization:

      cudnnSetTensorNdDescriptor(desc, dataType, nbDims, dimA, strideA);
      // Converts logical tensor layout to physical memory layout
      // by computing strides from dimensions

    Then, follow that pattern to convert logical layout coordinates
    to physical pixel coordinates by computing offset and scale from
    the artboard dimensions vs image dimensions.

    Returns: (offset_x, offset_y, scale_factor)
    """
    if not objects:
        return 0.0, 0.0, 1.0

    # Find the envelope (bounding box of all objects)
    min_x = min(o.bbox.x for o in objects)
    min_y = min(o.bbox.y for o in objects)
    max_x = max(o.bbox.x2 for o in objects)
    max_y = max(o.bbox.y2 for o in objects)

    # The offset shifts everything so min coordinate = 0
    offset_x = min_x
    offset_y = min_y

    # Compute scale if image dimensions available
    layout_width = max_x - min_x
    layout_height = max_y - min_y

    scale = 1.0
    if image_width and image_height and layout_width > 0 and layout_height > 0:
        scale_x = image_width / layout_width
        scale_y = image_height / layout_height
        # Use the smaller scale to preserve aspect ratio
        scale = min(scale_x, scale_y)

        # If scales differ by >10%, the layout might not match the image
        if abs(scale_x - scale_y) / max(scale_x, scale_y) > 0.1:
            logger.warning(
                "Layout aspect ratio (%.0f×%.0f) doesn't match image (%.0f×%.0f). "
                "Using conservative scale %.3f",
                layout_width, layout_height, image_width, image_height, scale,
            )

    logger.info(
        "Coordinate normalization: offset=(%.1f, %.1f), scale=%.3f, "
        "layout=%.0f×%.0f",
        offset_x, offset_y, scale, layout_width, layout_height,
    )

    return offset_x, offset_y, scale


# ═══════════════════════════════════════════════════════════════════════
#  §4  Region Extraction (the "CTA per segment" core)
# ═══════════════════════════════════════════════════════════════════════

def extract_region_from_image(
    image: "Image.Image",
    obj: LayoutObject,
    offset_x: float,
    offset_y: float,
    scale: float,
    padding: int = 5,
) -> Optional[Tuple[str, Tuple[int, int], float]]:
    """Extract a single region from the source image.

    From CCCL's cub::DeviceSegmentedReduce per-segment kernel:

      // Each CTA processes one segment independently
      template <typename SegmentIteratorT>
      __global__ void SegmentedReduceKernel(
          SegmentIteratorT d_in, OutputT* d_out,
          int segment_begin, int segment_end) {
        // Load segment data into shared memory
        // Reduce within CTA
        // Write result to d_out[blockIdx.x]
      }

    Then, follow that pattern to process one layout region:
    Load the bbox region from the image, apply padding,
    crop, and encode to base64 PNG.

    Returns: (base64_png, (width, height), alpha_ratio) or None
    """
    if not _HAS_PIL or not _HAS_NUMPY:
        return None

    img_w, img_h = image.size

    # Convert layout coordinates to pixel coordinates
    left, upper, right, lower = obj.bbox.to_pixel_rect(scale, offset_x, offset_y)

    # Apply padding
    left = max(0, left - padding)
    upper = max(0, upper - padding)
    right = min(img_w, right + padding)
    lower = min(img_h, lower + padding)

    # Validate
    crop_w = right - left
    crop_h = lower - upper
    if crop_w <= 0 or crop_h <= 0:
        logger.warning("Region %s has zero or negative crop dimensions, skipping", obj.id)
        return None

    # Crop
    try:
        cropped = image.crop((left, upper, right, lower))
    except Exception as e:
        logger.warning("Failed to crop region %s: %s", obj.id, e)
        return None

    # Convert to RGBA for alpha analysis
    cropped_rgba = cropped.convert("RGBA")
    arr = np.array(cropped_rgba)

    # Compute alpha ratio (fraction of non-transparent pixels)
    alpha = arr[:, :, 3]
    total_px = alpha.shape[0] * alpha.shape[1]
    opaque_px = int(np.sum(alpha > 30))
    alpha_ratio = opaque_px / total_px if total_px > 0 else 0.0

    # Encode to PNG
    buf = io.BytesIO()
    cropped_rgba.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return b64, (crop_w, crop_h), alpha_ratio


# ═══════════════════════════════════════════════════════════════════════
#  §5  Background Removal Integration
# ═══════════════════════════════════════════════════════════════════════

async def remove_background_for_region(
    region_b64: str,
    method: str = "auto",
    api_key: str = "",
) -> Optional[str]:
    """Remove background from a single cropped region.

    Delegates to the existing removal pipeline (removebg_route) but
    for a single region rather than a batch of animation frames.

    From NCCL's fallback chain (NVLink → PCIe → Net):
    Try removebg.io first, then rembg, then chroma-key.
    """
    try:
        from backend.pipeline.removebg_route import handle_removebg
        result = await handle_removebg(
            frames_b64=[region_b64],
            api_key=api_key,
            force_method=method if method != "auto" else None,
        )
        if result.get("success") and result.get("results"):
            first = result["results"][0]
            if first.get("success") and first.get("image_b64"):
                return first["image_b64"]
    except ImportError:
        logger.warning("removebg_route not available, returning original")
    except Exception as e:
        logger.warning("Background removal failed for region: %s", e)

    return None


# ═══════════════════════════════════════════════════════════════════════
#  §6  Tree Serialization (for Canvas Editor)
# ═══════════════════════════════════════════════════════════════════════

def serialize_tree(
    objects: List[LayoutObject],
    by_id: Dict[str, LayoutObject],
) -> Dict[str, Any]:
    """Serialize the region tree to a JSON-compatible dict.

    Output format matches what a React canvas editor (fabric.js or
    react-konva) would consume:

    {
      "artboard": { "width": 1024, "height": 600 },
      "layers": [
        {
          "id": "1:0578",
          "name": "5.1-胎压",
          "type": "group",
          "bbox": { "x": 0, "y": 0, "width": 1024, "height": 600 },
          "children": [
            { "id": "1:2704", "name": "胎压正常", "type": "shape", ... }
          ]
        }
      ]
    }

    From OpenAI's structured output format:
    Start from OpenAI's function calling response schema. Then, follow
    that pattern to produce a typed, self-describing tree structure
    that the frontend can render directly without transformation.
    """
    # Find root nodes
    roots = [o for o in objects if o.parent_id is None or o.parent_id not in by_id]

    def _serialize_node(node: LayoutObject) -> Dict[str, Any]:
        result = {
            "id": node.id,
            "name": node.name,
            "type": node.object_type,
            "bbox": {
                "x": node.bbox.x,
                "y": node.bbox.y,
                "width": node.bbox.width,
                "height": node.bbox.height,
            },
            "depth": node.depth,
            "is_leaf": node.is_leaf,
        }
        if node.children:
            result["children"] = [_serialize_node(c) for c in node.children]
        return result

    # Determine artboard dimensions from the envelope
    if roots:
        min_x = min(r.bbox.x for r in roots)
        min_y = min(r.bbox.y for r in roots)
        max_x = max(r.bbox.x2 for r in roots)
        max_y = max(r.bbox.y2 for r in roots)
    else:
        min_x = min_y = 0
        max_x = max_y = 1024

    return {
        "artboard": {
            "x": min_x,
            "y": min_y,
            "width": max_x - min_x,
            "height": max_y - min_y,
        },
        "layers": [_serialize_node(r) for r in roots],
        "total_nodes": len(objects),
        "total_leaves": sum(1 for o in objects if o.is_leaf),
    }


# ═══════════════════════════════════════════════════════════════════════
#  §7  Main Entry Point
# ═══════════════════════════════════════════════════════════════════════

async def process_regions(
    image_b64: Optional[str],
    layout_data: str | list,
    config: Optional[RegionConfig] = None,
    remove_bg: bool = True,
    api_key: str = "",
) -> RegionProcessingResult:
    """Process an image using structured layout data for region extraction.

    Main entry point for "Path 2" — layout-guided component extraction.

    Parameters
    ----------
    image_b64 : str or None
        Base64-encoded source image. If None, only the layout tree is
        built (useful for generating region metadata without an image).
    layout_data : str or list
        Raw JSON string or parsed list of mastergo-format layout objects.
    config : RegionConfig, optional
        Processing configuration.
    remove_bg : bool
        Whether to run background removal on each extracted region.
    api_key : str
        API key for remove-bg.io cloud removal.

    Returns
    -------
    RegionProcessingResult
        Extracted regions with metadata and hierarchical tree.

    From NVIDIA Megatron-Core's pipeline parallel forward_backward_pipelining:

      def forward_backward_pipelining(
          forward_step_func, data_iterator, model,
          num_microbatches, ...):
          # Schedule: 1F1B (one forward, one backward) for steady state
          for i in range(num_microbatches):
              output = forward_step_func(data_iterator, model)
              if i >= num_warmup_microbatches:
                  backward_step_func(input_tensor, output_tensor)

    Then, follow that pattern to process regions in a steady-state loop:
    for each layout object, forward = extract + remove_bg, building
    up the result list incrementally.
    """
    if config is None:
        config = RegionConfig()

    t0 = time.monotonic()

    # ── Step 1: Parse layout objects ─────────────────────────────────
    objects = parse_layout_objects(layout_data)
    if not objects:
        return RegionProcessingResult(
            success=False,
            error="No valid layout objects found in input data",
            processing_time_ms=(time.monotonic() - t0) * 1000,
        )

    total_parsed = len(objects)

    # ── Step 2: Build region tree ────────────────────────────────────
    objects, by_id = build_region_tree(objects, config)
    tree = serialize_tree(objects, by_id)

    # ── Step 3: Filter objects ───────────────────────────────────────
    candidates = objects
    if config.leaf_only:
        candidates = [o for o in candidates if o.is_leaf]
    candidates = [o for o in candidates if o.bbox.area >= config.min_region_area]

    # Artboard filter
    if config.artboard_filter:
        prefix = config.artboard_filter.lower()
        candidates = [o for o in candidates if o.name.lower().startswith(prefix)]

    # Sort by area (largest first) and limit
    candidates.sort(key=lambda o: o.bbox.area, reverse=True)
    if len(candidates) > config.max_components:
        candidates = candidates[:config.max_components]

    filtered_count = total_parsed - len(candidates)

    # ── Step 4: If no image, return tree only ────────────────────────
    if image_b64 is None:
        return RegionProcessingResult(
            success=True,
            tree=tree,
            total_objects_parsed=total_parsed,
            objects_filtered=filtered_count,
            processing_time_ms=(time.monotonic() - t0) * 1000,
        )

    # ── Step 5: Decode image ─────────────────────────────────────────
    if not _HAS_PIL or not _HAS_NUMPY:
        return RegionProcessingResult(
            success=False,
            error="PIL and numpy required for image processing",
            tree=tree,
            processing_time_ms=(time.monotonic() - t0) * 1000,
        )

    try:
        if image_b64.startswith("data:"):
            image_b64 = image_b64.split(",", 1)[1]
        raw_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(raw_bytes))
    except Exception as e:
        return RegionProcessingResult(
            success=False,
            error=f"Failed to decode image: {e}",
            tree=tree,
            processing_time_ms=(time.monotonic() - t0) * 1000,
        )

    img_w, img_h = image.size

    # ── Step 6: Normalize coordinates ────────────────────────────────
    offset_x, offset_y, scale = normalize_coordinates(
        objects,
        img_w if config.auto_scale else None,
        img_h if config.auto_scale else None,
    )

    # Artboard bbox (for frontend canvas)
    artboard_bbox = {
        "x": 0, "y": 0,
        "width": img_w, "height": img_h,
    }

    # ── Step 7: Extract regions (segment-parallel) ───────────────────
    # Convert image to RGBA once for all crops
    image_rgba = image.convert("RGBA")

    extracted: List[ExtractedRegion] = []
    for obj in candidates:
        result = extract_region_from_image(
            image_rgba, obj, offset_x, offset_y, scale,
            padding=config.removal_padding,
        )
        if result is None:
            continue

        region_b64, region_size, alpha_ratio = result

        # Optional background removal per region
        if remove_bg and config.removal_method != "none":
            cleaned = await remove_background_for_region(
                region_b64,
                method=config.removal_method,
                api_key=api_key,
            )
            if cleaned:
                region_b64 = cleaned

        extracted.append(ExtractedRegion(
            id=obj.id,
            name=obj.name,
            bbox={
                "x": obj.bbox.x,
                "y": obj.bbox.y,
                "width": obj.bbox.width,
                "height": obj.bbox.height,
            },
            image_b64=region_b64,
            region_size=region_size,
            depth=obj.depth,
            parent_id=obj.parent_id,
            is_leaf=obj.is_leaf,
            alpha_ratio=alpha_ratio,
        ))

    elapsed_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "Region processing complete: %d parsed → %d extracted, %.0fms",
        total_parsed, len(extracted), elapsed_ms,
    )

    return RegionProcessingResult(
        success=len(extracted) > 0,
        regions=extracted,
        tree=tree,
        artboard_bbox=artboard_bbox,
        scale_factor=scale,
        total_objects_parsed=total_parsed,
        objects_filtered=filtered_count,
        processing_time_ms=elapsed_ms,
    )


# ═══════════════════════════════════════════════════════════════════════
#  §8  Layout-to-Prompt Generator
# ═══════════════════════════════════════════════════════════════════════

def generate_region_prompts(
    objects: List[LayoutObject],
    artboard_width: float = 1024,
    artboard_height: float = 600,
    style_context: str = "",
) -> List[Dict[str, Any]]:
    """Generate per-region image generation prompts from layout data.

    For regions where no source image exists (e.g., the user only
    provides layout data from mastergo without a screenshot), we
    generate prompts for each region based on its name, size, and
    position context.

    This enables "Path 2b": layout data → per-region prompts → Gemini
    → transparent components.

    From Google's Gemini spatial understanding prompts:
    Start from Gemini's bounding-box-conditioned generation where
    the prompt includes spatial constraints. Then, follow that pattern
    to generate per-region prompts that include the region's role
    (derived from its name) and spatial context (derived from its
    position relative to siblings).
    """
    prompts = []
    leaves = [o for o in objects if o.is_leaf]

    for obj in leaves:
        # Relative position in artboard
        rel_x = (obj.bbox.center[0] / artboard_width) if artboard_width > 0 else 0.5
        rel_y = (obj.bbox.center[1] / artboard_height) if artboard_height > 0 else 0.5

        # Position description
        if rel_x < 0.33:
            h_pos = "left"
        elif rel_x > 0.66:
            h_pos = "right"
        else:
            h_pos = "center"

        if rel_y < 0.33:
            v_pos = "top"
        elif rel_y > 0.66:
            v_pos = "bottom"
        else:
            v_pos = "middle"

        # Size description
        area_ratio = obj.bbox.area / (artboard_width * artboard_height) if (artboard_width * artboard_height) > 0 else 0
        if area_ratio > 0.3:
            size_desc = "large, prominent"
        elif area_ratio > 0.05:
            size_desc = "medium-sized"
        else:
            size_desc = "small, compact"

        # Build prompt
        prompt = (
            f"Generate a {size_desc} UI component for '{obj.name}' "
            f"positioned at the {v_pos}-{h_pos} of a dashboard interface. "
            f"Size: {int(obj.bbox.width)}×{int(obj.bbox.height)} pixels. "
            f"Transparent background (PNG with alpha channel). "
        )
        if style_context:
            prompt += f"Style: {style_context}. "

        prompts.append({
            "id": obj.id,
            "name": obj.name,
            "prompt": prompt,
            "bbox": {
                "x": obj.bbox.x,
                "y": obj.bbox.y,
                "width": obj.bbox.width,
                "height": obj.bbox.height,
            },
            "target_size": {
                "width": int(obj.bbox.width),
                "height": int(obj.bbox.height),
            },
            "position": f"{v_pos}-{h_pos}",
            "area_ratio": round(area_ratio, 4),
        })

    return prompts


# ═══════════════════════════════════════════════════════════════════════
#  §9  API Route Handler
# ═══════════════════════════════════════════════════════════════════════

async def handle_region_layout(
    request_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Handle /api/region-layout endpoint.

    Request body:
    {
        "image_b64": "...",                    // Optional: source screenshot
        "layout_data": [...],                  // Required: mastergo-format layout objects
        "remove_bg": true,                     // Optional: run background removal
        "api_key": "",                         // Optional: remove-bg.io key
        "config": {                            // Optional: processing config
            "min_region_area": 100,
            "max_components": 200,
            "leaf_only": true,
            "removal_method": "auto",
            ...
        }
    }

    Response:
    {
        "success": true,
        "regions": [...],
        "tree": {...},
        "stats": {...}
    }
    """
    layout_data = request_data.get("layout_data")
    if not layout_data:
        return {"success": False, "error": "layout_data is required"}

    image_b64 = request_data.get("image_b64")
    remove_bg = request_data.get("remove_bg", True)
    api_key = request_data.get("api_key", "")

    # Build config from request
    cfg_params = request_data.get("config", {})
    config = RegionConfig(
        min_region_area=float(cfg_params.get("min_region_area", 100)),
        max_components=int(cfg_params.get("max_components", 200)),
        dedup_iou_threshold=float(cfg_params.get("dedup_iou_threshold", 0.85)),
        leaf_only=bool(cfg_params.get("leaf_only", True)),
        auto_scale=bool(cfg_params.get("auto_scale", True)),
        artboard_filter=cfg_params.get("artboard_filter"),
        removal_method=cfg_params.get("removal_method", "auto"),
        removal_padding=int(cfg_params.get("removal_padding", 5)),
        maintain_position=bool(cfg_params.get("maintain_position", True)),
        output_format=cfg_params.get("output_format", "individual"),
        include_metadata=bool(cfg_params.get("include_metadata", True)),
    )

    result = await process_regions(
        image_b64=image_b64,
        layout_data=layout_data,
        config=config,
        remove_bg=remove_bg,
        api_key=api_key,
    )

    # Serialize response
    response: Dict[str, Any] = {
        "success": result.success,
        "error": result.error,
    }

    if result.regions:
        response["regions"] = [
            {
                "id": r.id,
                "name": r.name,
                "bbox": r.bbox,
                "image_b64": r.image_b64,
                "region_size": list(r.region_size),
                "depth": r.depth,
                "parent_id": r.parent_id,
                "is_leaf": r.is_leaf,
                "alpha_ratio": round(r.alpha_ratio, 4),
            }
            for r in result.regions
        ]

    if result.tree:
        response["tree"] = result.tree

    response["stats"] = {
        "total_objects_parsed": result.total_objects_parsed,
        "objects_filtered": result.objects_filtered,
        "regions_extracted": len(result.regions),
        "scale_factor": round(result.scale_factor, 4),
        "processing_time_ms": round(result.processing_time_ms, 2),
    }

    if result.artboard_bbox:
        response["artboard"] = result.artboard_bbox

    return response
