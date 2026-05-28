"""
unified_layer.py — Single Canonical Layer Type
==================================================
Replaces the format-sniffing if-else in pipeline_orchestrator.py (line 1103)
where component_extractor outputs Dict[str,Any] and layer_separator outputs
ExtractedLayer dataclass with incompatible bbox formats.

From CCCL's `cub/cub/detail/device_double_buffer.cuh`:
  device_double_buffer<T> unifies two physical buffers behind a single
  interface — callers use `.current()` / `.alternate()` without caring
  which physical buffer is active.

We unify component_extractor's {id, name, bbox:{x,y,w,h}} dict and
layer_separator's ExtractedLayer(bbox: Tuple[int,int,int,int]) behind
a single UnifiedLayer type that both producers emit and all consumers
(edge_refiner, component_outliner, export) accept.

Milestone: M100
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple


@dataclass
class UnifiedLayer:
    """Single canonical layer type for the entire pipeline.

    Every layer producer (CCL, ELK-guided, manual) outputs this.
    Every consumer (edge_refiner, outliner, export, frontend) accepts this.
    """
    layer_id: int
    image_b64: str
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    area: int = 0
    centroid_x: float = 0.0
    centroid_y: float = 0.0
    name: str = ""
    source: Literal["ccl", "elk_guided", "manual", "unknown"] = "unknown"
    confidence: float = 1.0
    original_width: int = 0
    original_height: int = 0
    frame_index: int = 0

    # ── Compatibility accessors ──────────────────────────────────────

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """(x, y, width, height) — matches ExtractedLayer.bbox semantics."""
        return (self.bbox_x, self.bbox_y, self.bbox_w, self.bbox_h)

    @property
    def bbox_dict(self) -> Dict[str, int]:
        """{'x':…, 'y':…, 'width':…, 'height':…} — matches component_extractor."""
        return {
            "x": self.bbox_x, "y": self.bbox_y,
            "width": self.bbox_w, "height": self.bbox_h,
        }

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "image_b64": self.image_b64,
            "bbox": self.bbox_dict,
            "area": self.area,
            "centroid": [self.centroid_x, self.centroid_y],
            "name": self.name,
            "source": self.source,
            "confidence": self.confidence,
            "frame_index": self.frame_index,
        }

    def to_pipeline_dict(self) -> Dict[str, Any]:
        """Format expected by pipeline_orchestrator's inter-stage passing."""
        return {
            "image_b64": self.image_b64,
            "frame_index": self.frame_index,
            "layer_id": self.layer_id,
            "bbox": self.bbox_dict,
            "area": self.area,
            "name": self.name,
            "source": self.source,
        }

    # ── Factory methods (one per producer) ───────────────────────────

    @classmethod
    def from_extracted_layer(cls, ext, frame_index: int = 0) -> "UnifiedLayer":
        """Convert layer_separator.ExtractedLayer → UnifiedLayer."""
        return cls(
            layer_id=ext.layer_id,
            image_b64=ext.image_b64,
            bbox_x=ext.bbox[0],
            bbox_y=ext.bbox[1],
            bbox_w=ext.bbox[2],
            bbox_h=ext.bbox[3],
            area=ext.area,
            centroid_x=ext.centroid[0],
            centroid_y=ext.centroid[1],
            original_width=ext.original_size[0],
            original_height=ext.original_size[1],
            source="ccl",
            frame_index=frame_index,
        )

    @classmethod
    def from_component_extractor(
        cls, layout_item: Dict[str, Any], crop_b64: str, frame_index: int = 0,
    ) -> "UnifiedLayer":
        """Convert component_extractor's {id, name, bbox:{x,y,width,height}} → UnifiedLayer."""
        b = layout_item.get("bbox", {})
        bx = int(b.get("x", 0))
        by = int(b.get("y", 0))
        bw = int(b.get("width", 0))
        bh = int(b.get("height", 0))
        idx_str = layout_item.get("id", "0")
        try:
            lid = int(idx_str.replace("c", "").replace("elk_", ""))
        except (ValueError, AttributeError):
            lid = 0
        return cls(
            layer_id=lid,
            image_b64=crop_b64,
            bbox_x=bx, bbox_y=by, bbox_w=bw, bbox_h=bh,
            area=bw * bh,
            centroid_x=bx + bw / 2.0,
            centroid_y=by + bh / 2.0,
            name=layout_item.get("name", f"component_{lid}"),
            source="elk_guided" if "elk" in str(idx_str) else "ccl",
            frame_index=frame_index,
        )

    @classmethod
    def from_raw_frame(
        cls, image_b64: str, frame_index: int = 0,
    ) -> "UnifiedLayer":
        """Wrap a whole frame as a single layer (fallback when separation fails)."""
        return cls(
            layer_id=0,
            image_b64=image_b64,
            bbox_x=0, bbox_y=0, bbox_w=0, bbox_h=0,
            name="full_frame",
            source="unknown",
            frame_index=frame_index,
        )

    # ── Equality / hashing for set/dict dedup ────────────────────────

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UnifiedLayer):
            return False
        return (self.layer_id == other.layer_id
                and self.frame_index == other.frame_index
                and self.source == other.source)

    def __hash__(self) -> int:
        return hash((self.layer_id, self.frame_index, self.source))

    def __repr__(self) -> str:
        return (
            f"UnifiedLayer(id={self.layer_id}, name={self.name!r}, "
            f"bbox=({self.bbox_x},{self.bbox_y},{self.bbox_w},{self.bbox_h}), "
            f"source={self.source!r}, frame={self.frame_index})"
        )
