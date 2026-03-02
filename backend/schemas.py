"""
Backend Schemas
===============
Pydantic models for API request/response validation.

Used by:
  - server.py API endpoints
  - Pipeline modules
  - Frontend TypeScript types (mirror these in src/lib/pipeline/types.ts)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================================
# Enums
# ============================================================================

class ElkAlgorithm(str, Enum):
    """ELK layout algorithms. Reference: eclipse.org/elk/reference/algorithms.html"""
    LAYERED = "layered"
    MRTREE = "mrtree"
    FORCE = "force"
    STRESS = "stress"
    RADIAL = "radial"


class ElkDirection(str, Enum):
    """ELK layout directions."""
    DOWN = "DOWN"
    UP = "UP"
    RIGHT = "RIGHT"
    LEFT = "LEFT"


class ExportFormat(str, Enum):
    """Supported export formats."""
    SVG = "svg"
    PNG = "png"
    PDF = "pdf"


# ============================================================================
# ELK Topology (Step 1 output / Step 2 input)
# ============================================================================

class ElkLabel(BaseModel):
    """ELK node label."""
    text: str


class ElkNode(BaseModel):
    """ELK graph node (child)."""
    id: str
    width: float = 150
    height: float = 50
    labels: List[ElkLabel] = []
    layoutOptions: Optional[Dict[str, str]] = None
    # After layout, these are populated:
    x: Optional[float] = None
    y: Optional[float] = None


class ElkEdge(BaseModel):
    """ELK graph edge."""
    id: str
    sources: List[str]
    targets: List[str]
    labels: Optional[List[ElkLabel]] = None


class ElkLayoutOptions(BaseModel):
    """ELK layout configuration."""
    algorithm: ElkAlgorithm = ElkAlgorithm.LAYERED
    direction: ElkDirection = ElkDirection.DOWN
    node_spacing: float = 80
    layer_spacing: float = 100


class ElkGraph(BaseModel):
    """
    Full ELK graph (input to elk.layout()).
    Reference: kieler/elkjs
    """
    id: str = "root"
    layoutOptions: Optional[Dict[str, str]] = None
    children: List[ElkNode] = []
    edges: List[ElkEdge] = []


# ============================================================================
# NanoBanana Scaffold (Step 2 output / Step 3 input)
# ============================================================================

class ScaffoldElement(BaseModel):
    """Single element in the NanoBanana scaffold."""
    id: str
    type: str = "box"
    label: str
    x: float
    y: float
    width: float
    height: float
    style: str = "rounded_rect"
    fill: str = "#E3F2FD"


class ScaffoldConnection(BaseModel):
    """Connection in the NanoBanana scaffold."""
    from_id: str = Field(alias="from")
    to_id: str = Field(alias="to")
    style: str = "arrow"
    points: List[Dict[str, float]] = []

    model_config = {"populate_by_name": True}


class NanoBananaScaffold(BaseModel):
    """
    JSON scaffold sent to NanoBanana for SVG generation.
    This bridges ELK layout → Gemini NanoBanana.

    Reference: gemini-cli-extensions/nanobanana
    """
    figure_type: str = "academic_architecture"
    canvas: Dict[str, float] = {"width": 800, "height": 600}
    elements: List[ScaffoldElement] = []
    connections: List[ScaffoldConnection] = []
    style_hints: Optional[Dict[str, Any]] = None


# ============================================================================
# API Request Models
# ============================================================================

class TopologyRequest(BaseModel):
    """POST /api/topology — Generate ELK topology from text."""
    text: str = Field(..., min_length=10, description="Paper method description text")
    model: Optional[str] = None
    algorithm: ElkAlgorithm = ElkAlgorithm.LAYERED
    direction: ElkDirection = ElkDirection.DOWN


class LayoutRequest(BaseModel):
    """POST /api/layout — Apply ELK layout to topology."""
    topology: ElkGraph
    options: Optional[ElkLayoutOptions] = None


class BeautifyRequest(BaseModel):
    """POST /api/beautify — Generate SVG via NanoBanana from layouted graph."""
    layouted: ElkGraph
    scaffold: Optional[NanoBananaScaffold] = None
    model: Optional[str] = None
    style: Optional[str] = "academic"


class ValidateRequest(BaseModel):
    """POST /api/validate — Validate SVG syntax."""
    svg: str
    auto_fix: bool = True
    model: Optional[str] = None


class ExportRequest(BaseModel):
    """POST /api/export — Export SVG to PNG/PDF."""
    svg: str
    format: ExportFormat = ExportFormat.SVG
    width: Optional[int] = None
    height: Optional[int] = None
    scale: float = 1.0


# ============================================================================
# API Response Models
# ============================================================================

class TopologyResponse(BaseModel):
    """Response from /api/topology."""
    success: bool
    topology: Optional[ElkGraph] = None
    raw_llm_output: Optional[str] = None
    error: Optional[str] = None
    model_used: Optional[str] = None


class LayoutResponse(BaseModel):
    """Response from /api/layout."""
    success: bool
    layouted: Optional[ElkGraph] = None
    scaffold: Optional[NanoBananaScaffold] = None
    error: Optional[str] = None


class BeautifyResponse(BaseModel):
    """Response from /api/beautify."""
    success: bool
    svg: Optional[str] = None
    error: Optional[str] = None
    model_used: Optional[str] = None


class ValidateResponse(BaseModel):
    """Response from /api/validate."""
    valid: bool
    errors: List[str] = []
    fixed_svg: Optional[str] = None
    fix_iterations: int = 0


class ExportResponse(BaseModel):
    """Response from /api/export."""
    success: bool
    file_path: Optional[str] = None
    file_url: Optional[str] = None
    format: ExportFormat = ExportFormat.SVG
    error: Optional[str] = None


# ============================================================================
# SSE Streaming Events (for frontend)
# ============================================================================

class StreamEvent(BaseModel):
    """Server-Sent Event for streaming pipeline progress."""
    type: str  # "start" | "progress" | "topology" | "layout" | "beautify" | "done" | "error"
    step: Optional[int] = None  # 1-4
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    progress: Optional[float] = None  # 0.0 - 1.0
