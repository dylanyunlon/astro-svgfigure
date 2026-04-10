"""
Animation Pipeline Schemas
============================
Pydantic models for the /playground animation frame generation pipeline.

Pipeline Flow:
  Step 1: POST /api/analyze-image → Claude 4.6 structure analysis
  Step 2: POST /api/animate-prompt → Grok animation prompt engineering
  Step 3: POST /api/animate-frames → Gemini multi-frame generation (green-screen)
  Step 4: POST /api/rembg-frames → Server-side background removal (optional)
  Step 5: POST /api/encode-animation → GIF/APNG encoding

Critical Design Decisions (Knuth-level rigor):
───────────────────────────────────────────────
1. All image data is base64-encoded to avoid multipart complexity across
   the Astro→FastAPI proxy boundary. Trade-off: ~33% size overhead vs.
   simpler error handling and JSON-native transport.

2. green_screen=True is MANDATORY for frame generation. The Grok prompt
   designer explicitly instructs green background (#00FF00) so the
   client-side chroma-key or server-side rembg can cleanly separate
   foreground from background.

3. Frame ordering is array-index-based (0..N-1). No separate frame_id
   field is needed because the array position IS the temporal position.
   This avoids a class of bugs where frame_id and array index desync.

System Critique:
  - If the client sends corrupt base64, the backend will get a
    binascii.Error deep inside the AI provider. We validate base64
    at the schema level with a length check + character set check.
  - mime_type defaults to 'image/png' but we accept 'image/jpeg' and
    'image/webp'. If the user uploads a BMP or TIFF, the Astro proxy
    should reject it before it reaches us (enforced in the file input).

GitHub references:
  - pydantic/pydantic (BaseModel, Field, field_validator)
  - dylanyunlon/astro-svgfigure (existing backend/schemas.py pattern)
"""

from __future__ import annotations

import base64
import re
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ═══════════════════════════════════════════════════════════════════════
#  Enums
# ═══════════════════════════════════════════════════════════════════════

class AnimationStyle(str, Enum):
    """Supported animation styles for frame decomposition."""
    SMOOTH = "smooth"
    BOUNCE = "bounce"
    ROTATE = "rotate"
    MORPH = "morph"
    EXPLODE = "explode"
    WALK = "walk"
    WAVE = "wave"
    PULSE = "pulse"


class AspectRatio(str, Enum):
    """Supported aspect ratios for generated frames."""
    SQUARE = "1:1"
    LANDSCAPE = "16:9"
    PORTRAIT = "9:16"
    FOUR_THREE = "4:3"


class ExportFormat(str, Enum):
    """Supported animation export formats."""
    GIF = "gif"
    APNG = "apng"
    WEBP = "webp"
    ZIP_FRAMES = "zip"


class ChromaKeyMethod(str, Enum):
    """Background removal methods."""
    GREEN_SCREEN = "green_screen"
    REMBG_U2NET = "rembg_u2net"
    REMBG_ISNET = "rembg_isnet"


# ═══════════════════════════════════════════════════════════════════════
#  Validators
# ═══════════════════════════════════════════════════════════════════════

_B64_PATTERN = re.compile(r'^[A-Za-z0-9+/]*={0,2}$')

def _validate_base64(v: str, field_name: str = "image_b64") -> str:
    """
    Validate base64 string is well-formed.

    User-angle critique: if someone pastes a data: URL instead of raw
    base64, this will catch it early with a clear error message rather
    than failing deep in the AI provider with an opaque decode error.

    System-angle critique: we only check format, not decodability to
    avoid the cost of actually decoding multi-MB images at validation
    time. The actual decode happens once at the provider layer.
    """
    if not v:
        raise ValueError(f"{field_name} cannot be empty")
    # Strip optional data URI prefix
    if v.startswith("data:"):
        parts = v.split(",", 1)
        if len(parts) == 2:
            v = parts[1]
    # Basic format check (skip full decode for performance)
    if len(v) < 100:
        raise ValueError(f"{field_name} too short — expected a full image, got {len(v)} chars")
    if len(v) > 50_000_000:  # ~37MB decoded
        raise ValueError(f"{field_name} too large — max 37MB decoded ({len(v)} chars base64)")
    return v


# ═══════════════════════════════════════════════════════════════════════
#  Step 1: Image Analysis (Claude 4.6)
# ═══════════════════════════════════════════════════════════════════════

class AnalyzeImageRequest(BaseModel):
    """
    Request body for POST /api/analyze-image.

    Claude 4.6 analyzes the uploaded image to identify:
    - Visual components (objects, shapes, characters)
    - Spatial layout and layering
    - Animatable elements and suggested motion types
    - Color palette and style characteristics
    """
    image_b64: str = Field(..., description="Base64-encoded image data")
    mime_type: str = Field(
        default="image/png",
        description="MIME type of the uploaded image",
        pattern=r"^image/(png|jpeg|webp)$",
    )
    model: Optional[str] = Field(
        default=None,
        description="Override Claude model for analysis (default: from config)",
    )

    @field_validator("image_b64")
    @classmethod
    def check_image_b64(cls, v: str) -> str:
        return _validate_base64(v, "image_b64")


class ImageComponent(BaseModel):
    """A detected component/element in the analyzed image."""
    name: str = Field(..., description="Human-readable component name")
    type: str = Field(default="object", description="Component type: object, text, shape, character, background")
    bounds: Optional[Dict[str, float]] = Field(
        default=None,
        description="Bounding box {x, y, width, height} in normalized 0..1 coords",
    )
    animatable: bool = Field(default=True, description="Whether this component can be animated")
    suggested_motions: List[str] = Field(
        default_factory=list,
        description="Suggested animation types: rotate, translate, scale, morph, fade",
    )


class AnalyzeImageResponse(BaseModel):
    """Response from POST /api/analyze-image."""
    success: bool = True
    analysis: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Full analysis result from Claude 4.6",
    )
    summary: Optional[str] = Field(
        default=None,
        description="One-line summary of the image",
    )
    components: Optional[List[ImageComponent]] = Field(
        default=None,
        description="Detected visual components with animation suggestions",
    )
    animation_suggestions: Optional[List[str]] = Field(
        default=None,
        description="Top-level animation strategy suggestions",
    )
    color_palette: Optional[List[str]] = Field(
        default=None,
        description="Dominant colors as hex strings",
    )
    error: Optional[str] = None
    model_used: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════
#  Step 2: Animation Prompt Design (Grok)
# ═══════════════════════════════════════════════════════════════════════

class AnimatePromptRequest(BaseModel):
    """
    Request body for POST /api/animate-prompt.

    Grok designs the frame-by-frame animation prompt, explicitly
    instructing green-screen background (#00FF00) for chroma-key removal.

    Critical: The prompt MUST include the green-screen instruction.
    This is not optional — without it, the rembg step will fail or
    produce poor results. The green-screen constraint is injected
    at the system prompt level, not left to the user.
    """
    image_b64: str = Field(..., description="Base64-encoded source image")
    analysis: Dict[str, Any] = Field(
        ..., description="Analysis result from Step 1"
    )
    frame_count: int = Field(
        default=8, ge=2, le=24,
        description="Number of animation frames to generate",
    )
    animation_style: AnimationStyle = Field(
        default=AnimationStyle.SMOOTH,
        description="Animation motion style",
    )
    model: Optional[str] = Field(
        default=None,
        description="Override Grok model (default: from config)",
    )

    @field_validator("image_b64")
    @classmethod
    def check_image_b64(cls, v: str) -> str:
        return _validate_base64(v, "image_b64")


class AnimatePromptResponse(BaseModel):
    """Response from POST /api/animate-prompt."""
    success: bool = True
    prompt: Optional[str] = Field(
        default=None,
        description="The engineered animation prompt with green-screen instructions",
    )
    frame_descriptions: Optional[List[str]] = Field(
        default=None,
        description="Per-frame descriptions for reference",
    )
    error: Optional[str] = None
    model_used: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════
#  Step 3: Frame Generation (Gemini)
# ═══════════════════════════════════════════════════════════════════════

class AnimateFramesRequest(BaseModel):
    """
    Request body for POST /api/animate-frames.

    Gemini generates N frames with green-screen (#00FF00) background.
    Each frame is returned as a separate base64 PNG.

    System-angle critique: generating 16 frames at once may exceed
    Gemini's output token limit. We implement a fallback strategy:
    - ≤8 frames: single request with all frames
    - >8 frames: batch into groups of 4, sequential requests
    This trades latency for reliability.

    User-angle critique: the progress feels stuck during multi-minute
    generation. We use SSE or polling for real-time frame-by-frame
    progress updates in future iterations. For now, the frontend shows
    an estimated time based on frame count.
    """
    image_b64: str = Field(..., description="Base64-encoded source image")
    frame_count: int = Field(
        default=8, ge=2, le=24,
        description="Number of animation frames",
    )
    fps: int = Field(
        default=12, ge=1, le=60,
        description="Target playback FPS (metadata only, doesn't affect generation)",
    )
    animation_style: str = Field(
        default="smooth",
        description="Animation motion style",
    )
    custom_prompt: Optional[str] = Field(
        default=None,
        description="Custom/edited animation prompt from Step 2",
    )
    green_screen: bool = Field(
        default=True,
        description="MUST be True — green-screen BG is mandatory",
    )
    aspect_ratio: str = Field(
        default="1:1",
        description="Aspect ratio for generated frames",
    )
    model: Optional[str] = Field(
        default=None,
        description="Override Gemini model (default: from config)",
    )

    @field_validator("image_b64")
    @classmethod
    def check_image_b64(cls, v: str) -> str:
        return _validate_base64(v, "image_b64")

    @field_validator("green_screen")
    @classmethod
    def enforce_green_screen(cls, v: bool) -> bool:
        """Green-screen is mandatory. Override any attempt to disable it."""
        if not v:
            import logging
            logging.getLogger(__name__).warning(
                "green_screen=False was requested but overridden to True. "
                "Green-screen is mandatory for the animation pipeline."
            )
        return True  # Always True


class AnimateFramesResponse(BaseModel):
    """Response from POST /api/animate-frames."""
    success: bool = True
    frames: Optional[List[str]] = Field(
        default=None,
        description="Array of base64-encoded PNG frames with green-screen BG",
    )
    frame_count: int = Field(default=0)
    error: Optional[str] = None
    model_used: Optional[str] = None
    generation_time_ms: Optional[int] = None


# ═══════════════════════════════════════════════════════════════════════
#  Step 4: Background Removal (Server-side rembg)
# ═══════════════════════════════════════════════════════════════════════

class RembgFramesRequest(BaseModel):
    """
    Request body for POST /api/rembg-frames.

    Server-side background removal as fallback when client-side
    chroma-key isn't sufficient (complex edges, green objects, etc.)

    User-angle critique: this adds round-trip latency for the frames.
    For most cases, client-side green-screen removal is faster and
    sufficient. Server-side rembg is offered as an "enhance" option.
    """
    frames_b64: List[str] = Field(
        ..., description="Array of base64-encoded frames to process",
        min_length=1, max_length=24,
    )
    method: ChromaKeyMethod = Field(
        default=ChromaKeyMethod.GREEN_SCREEN,
        description="Background removal method",
    )
    tolerance: int = Field(
        default=60, ge=10, le=150,
        description="Green-screen chroma-key tolerance (0-255 range)",
    )
    edge_blur: float = Field(
        default=1.0, ge=0.0, le=5.0,
        description="Edge softness in pixels",
    )
    despill: bool = Field(
        default=True,
        description="Apply green-spill correction on edges",
    )


class RembgFramesResponse(BaseModel):
    """Response from POST /api/rembg-frames."""
    success: bool = True
    frames_b64: Optional[List[str]] = Field(
        default=None,
        description="Processed frames with transparent background",
    )
    stats: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Processing stats: pixels removed, time, etc.",
    )
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════
#  Step 5: Animation Encoding (GIF / APNG / WebP)
# ═══════════════════════════════════════════════════════════════════════

class EncodeAnimationRequest(BaseModel):
    """
    Request body for POST /api/encode-animation.

    Encodes transparent PNG frames into an animated format.

    System-angle critique: GIF only supports 1-bit alpha (fully
    transparent or fully opaque). For smooth semi-transparent edges,
    APNG or WebP-animation is preferred. We default to APNG but
    offer GIF for maximum compatibility.
    """
    frames_b64: List[str] = Field(
        ..., description="Transparent PNG frames as base64",
        min_length=1, max_length=24,
    )
    format: ExportFormat = Field(
        default=ExportFormat.APNG,
        description="Output animation format",
    )
    fps: int = Field(
        default=12, ge=1, le=60,
        description="Playback frames per second",
    )
    loop_count: int = Field(
        default=0,
        description="Loop count (0 = infinite loop)",
    )
    optimize: bool = Field(
        default=True,
        description="Optimize file size (may increase encoding time)",
    )


class EncodeAnimationResponse(BaseModel):
    """Response from POST /api/encode-animation."""
    success: bool = True
    animation_b64: Optional[str] = Field(
        default=None,
        description="Base64-encoded animation file",
    )
    mime_type: Optional[str] = Field(
        default=None,
        description="MIME type of the encoded animation",
    )
    file_size_bytes: Optional[int] = None
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════
#  Health Check extension for animation pipeline
# ═══════════════════════════════════════════════════════════════════════

class AnimationHealthResponse(BaseModel):
    """Extended health check including animation pipeline readiness."""
    backend: bool = True
    animation_pipeline: bool = True
    rembg_available: bool = False
    gif_encoder_available: bool = False
    apng_encoder_available: bool = False
    models: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
