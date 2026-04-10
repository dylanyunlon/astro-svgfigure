"""
Gemini Image Editor — Direct API Integration for Image-to-Image Editing
=========================================================================
This module provides a specialized interface to Gemini's image editing
capabilities, optimized for animation frame generation.

WHY A DEDICATED MODULE:
──────────────────────
The existing ai_engine.py is a general-purpose wrapper. For animation,
we need specific features:

1. MULTI-TURN CONVERSATION SUPPORT
   Per Google's documentation, Gemini excels at "conversational editing"
   where each turn refines the previous result.

2. THOUGHT SIGNATURE PASSING
   Per ai.google.dev/gemini-api/docs/image-generation:
   "Thought signatures are encrypted representations of the model's internal
   thought process and are used to preserve reasoning context across
   multi-turn interactions."

3. IMAGE-TO-IMAGE EDITING MODE
   Not text-to-image generation, but editing of provided images.

4. CONFIGURABLE TIMEOUTS
   Frame generation needs longer timeouts than text completion.

5. GREEN SCREEN ENFORCEMENT
   Every request includes background color specification.

BASED ON RESEARCH:
─────────────────
Per blog.google/products/gemini/image-generation-prompting-tips:
"Gemini now offers... precise, conversational editing."

Per developers.googleblog.com/how-to-prompt-gemini-2-5-flash-image:
"Using the provided image, change only the [specific element]."

Per cloud.google.com/blog/gemini-2-5-flash-image-on-vertex-ai:
"Character & style consistency: Maintain the same subject or visual style
across multiple generations."

API STRUCTURE:
─────────────
This module wraps the Gemini API with animation-specific functionality:

1. GeminiImageEditor class:
   - Manages API connection
   - Handles thought signature tracking
   - Provides edit_image() method

2. ConversationContext:
   - Tracks multi-turn state
   - Stores thought signatures
   - Manages reference images

3. EditRequest/EditResponse:
   - Type-safe request/response handling
   - Validation and error handling

TIMEOUT STRATEGY:
────────────────
- Default httpx timeout: 120s (too short for image generation)
- This module: 300s per request (5 minutes)
- Total pipeline: 600s for all frames (10 minutes)

The longer timeout is critical because:
1. Image generation is compute-intensive
2. Gemini servers can be slow during peak times
3. Complex prompts take longer to process

Knuth-Level Critiques:
─────────────────────
USER CRITIQUE: "The API times out constantly"
SOLUTION: 5-minute timeout per frame, no retry (fail fast).

SYSTEM CRITIQUE: "No multi-turn support in existing code"
SOLUTION: This module implements full conversation tracking.

SYSTEM CRITIQUE: "Thought signatures not being passed"
SOLUTION: Explicit thought_signature field in all requests/responses.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Conditional imports ──
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    logger.warning("httpx not available — GeminiImageEditor disabled")


# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

# API Configuration
DEFAULT_MODEL = "gemini-3.1-flash-image-preview"  # Nano Banana 2
FALLBACK_MODEL = "gemini-2.5-flash-image"
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Timeout configuration (seconds)
REQUEST_TIMEOUT = 300.0  # 5 minutes per request
CONNECT_TIMEOUT = 30.0  # 30 seconds to establish connection

# Green screen specification
GREEN_SCREEN_HEX = "#00FF00"
GREEN_SCREEN_PROMPT_SUFFIX = (
    "\n\nBACKGROUND REQUIREMENT: "
    f"The entire background MUST be solid bright green ({GREEN_SCREEN_HEX}). "
    "No gradients, shadows, or textures on the background. "
    "The subject must have clean, sharp edges against the green."
)


# ═══════════════════════════════════════════════════════════════════════════
#  Data Classes
# ═══════════════════════════════════════════════════════════════════════════

class ThinkingLevel(Enum):
    """Gemini thinking levels for image generation."""
    NONE = "None"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


@dataclass
class EditRequest:
    """Request for editing an image."""
    prompt: str
    reference_image_b64: str
    additional_images_b64: List[str] = field(default_factory=list)
    aspect_ratio: str = "1:1"
    thinking_level: ThinkingLevel = ThinkingLevel.MEDIUM
    include_green_screen: bool = True
    previous_thought_signature: Optional[str] = None

    def validate(self) -> Optional[str]:
        """Validate the request. Returns error message or None."""
        if not self.prompt or len(self.prompt) < 5:
            return "Prompt too short"
        if not self.reference_image_b64:
            return "Reference image required"
        if len(self.reference_image_b64) < 100:
            return "Reference image data too small"
        return None


@dataclass
class EditResponse:
    """Response from an edit operation."""
    success: bool
    image_b64: Optional[str] = None
    thought_signature: Optional[str] = None
    error: Optional[str] = None
    model_used: Optional[str] = None
    generation_time_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "has_image": self.image_b64 is not None,
            "has_thought_signature": self.thought_signature is not None,
            "error": self.error,
            "model_used": self.model_used,
            "generation_time_ms": self.generation_time_ms,
        }


@dataclass
class ConversationContext:
    """
    Tracks multi-turn conversation state for consistent editing.

    Per Google's documentation, passing thought_signature between turns
    helps maintain consistency across edits.
    """
    conversation_id: str
    reference_image_b64: str
    thought_signatures: List[str] = field(default_factory=list)
    turn_count: int = 0
    created_at: float = field(default_factory=time.time)

    @property
    def latest_thought_signature(self) -> Optional[str]:
        """Get the most recent thought signature."""
        return self.thought_signatures[-1] if self.thought_signatures else None

    def add_turn(self, thought_signature: Optional[str] = None):
        """Record a new conversation turn."""
        self.turn_count += 1
        if thought_signature:
            self.thought_signatures.append(thought_signature)


# ═══════════════════════════════════════════════════════════════════════════
#  Gemini Image Editor Class
# ═══════════════════════════════════════════════════════════════════════════

class GeminiImageEditor:
    """
    Specialized Gemini API client for image editing operations.

    This class provides:
    - Multi-turn conversation tracking
    - Thought signature management
    - Animation-optimized timeouts
    - Green screen enforcement
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = API_BASE_URL,
        timeout: float = REQUEST_TIMEOUT,
    ):
        """
        Initialize the Gemini image editor.

        Parameters
        ----------
        api_key : str
            Google AI API key
        model : str
            Model to use (default: gemini-3.1-flash-image-preview)
        base_url : str
            API base URL
        timeout : float
            Request timeout in seconds
        """
        if not HAS_HTTPX:
            raise RuntimeError("httpx required for GeminiImageEditor")

        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

        # Create httpx client with extended timeout
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                timeout=timeout,
                connect=CONNECT_TIMEOUT,
            ),
            headers={
                "Content-Type": "application/json",
            },
        )

        # Track active conversations
        self._conversations: Dict[str, ConversationContext] = {}

        logger.info(
            "GeminiImageEditor initialized: model=%s, timeout=%.0fs",
            model, timeout,
        )

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    # ─────────────────────────────────────────────────────────────────────
    #  Main Edit Method
    # ─────────────────────────────────────────────────────────────────────

    async def edit_image(self, request: EditRequest) -> EditResponse:
        """
        Edit an image using Gemini's image generation capabilities.

        This method sends the reference image + prompt to Gemini and
        returns a modified version of the image.

        Parameters
        ----------
        request : EditRequest
            The edit request containing image and prompt

        Returns
        -------
        EditResponse
            The result containing the edited image or error
        """
        t0 = time.monotonic()

        # Validate request
        validation_error = request.validate()
        if validation_error:
            return EditResponse(success=False, error=validation_error)

        # Build prompt with green screen suffix
        prompt = request.prompt
        if request.include_green_screen and GREEN_SCREEN_HEX not in prompt:
            prompt += GREEN_SCREEN_PROMPT_SUFFIX

        # Build API request body
        request_body = self._build_request_body(
            prompt=prompt,
            reference_image_b64=request.reference_image_b64,
            additional_images_b64=request.additional_images_b64,
            thinking_level=request.thinking_level,
            previous_thought_signature=request.previous_thought_signature,
        )

        # Make API call
        try:
            response = await self._call_api(request_body)
            return self._parse_response(response, t0)

        except httpx.TimeoutException:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            return EditResponse(
                success=False,
                error=(
                    f"Request timed out after {self.timeout:.0f}s ({elapsed_ms}ms elapsed). "
                    "The Gemini API may be overloaded. Please try again."
                ),
                generation_time_ms=elapsed_ms,
            )

        except httpx.HTTPStatusError as e:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            return EditResponse(
                success=False,
                error=f"HTTP error {e.response.status_code}: {e.response.text[:200]}",
                generation_time_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            logger.exception("Unexpected error in edit_image: %s", e)
            return EditResponse(
                success=False,
                error=f"Unexpected error: {str(e)}",
                generation_time_ms=elapsed_ms,
            )

    # ─────────────────────────────────────────────────────────────────────
    #  Conversation Management
    # ─────────────────────────────────────────────────────────────────────

    def start_conversation(
        self,
        conversation_id: str,
        reference_image_b64: str,
    ) -> ConversationContext:
        """
        Start a new conversation for multi-turn editing.

        Parameters
        ----------
        conversation_id : str
            Unique identifier for this conversation
        reference_image_b64 : str
            The original reference image

        Returns
        -------
        ConversationContext
            The new conversation context
        """
        context = ConversationContext(
            conversation_id=conversation_id,
            reference_image_b64=reference_image_b64,
        )
        self._conversations[conversation_id] = context
        logger.info("Started conversation: %s", conversation_id)
        return context

    def get_conversation(self, conversation_id: str) -> Optional[ConversationContext]:
        """Get an existing conversation context."""
        return self._conversations.get(conversation_id)

    async def edit_in_conversation(
        self,
        conversation_id: str,
        prompt: str,
        include_green_screen: bool = True,
    ) -> EditResponse:
        """
        Edit an image within an existing conversation context.

        This method automatically:
        - Uses the conversation's reference image
        - Passes the latest thought signature
        - Updates the conversation state

        Parameters
        ----------
        conversation_id : str
            The conversation to use
        prompt : str
            The edit prompt for this turn

        Returns
        -------
        EditResponse
            The result containing the edited image
        """
        context = self.get_conversation(conversation_id)
        if context is None:
            return EditResponse(
                success=False,
                error=f"Conversation not found: {conversation_id}",
            )

        request = EditRequest(
            prompt=prompt,
            reference_image_b64=context.reference_image_b64,
            include_green_screen=include_green_screen,
            previous_thought_signature=context.latest_thought_signature,
        )

        response = await self.edit_image(request)

        # Update conversation state
        if response.success:
            context.add_turn(response.thought_signature)

        return response

    def end_conversation(self, conversation_id: str):
        """End and clean up a conversation."""
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            logger.info("Ended conversation: %s", conversation_id)

    # ─────────────────────────────────────────────────────────────────────
    #  Animation Frame Generation
    # ─────────────────────────────────────────────────────────────────────

    async def generate_animation_frames(
        self,
        reference_image_b64: str,
        frame_prompts: List[str],
        include_green_screen: bool = True,
        on_frame_complete: Optional[callable] = None,
    ) -> Tuple[List[str], List[EditResponse]]:
        """
        Generate multiple animation frames from a reference image.

        This method generates frames sequentially, passing thought
        signatures between frames for consistency.

        Parameters
        ----------
        reference_image_b64 : str
            The original image to animate
        frame_prompts : List[str]
            Prompts for each frame
        include_green_screen : bool
            Whether to enforce green background
        on_frame_complete : callable
            Optional callback(frame_index, response) after each frame

        Returns
        -------
        Tuple[List[str], List[EditResponse]]
            List of frame images (base64) and list of responses
        """
        frames: List[str] = []
        responses: List[EditResponse] = []
        thought_signature: Optional[str] = None

        for i, prompt in enumerate(frame_prompts):
            logger.info("Generating frame %d/%d", i + 1, len(frame_prompts))

            request = EditRequest(
                prompt=prompt,
                reference_image_b64=reference_image_b64,
                include_green_screen=include_green_screen,
                previous_thought_signature=thought_signature,
            )

            response = await self.edit_image(request)
            responses.append(response)

            if response.success and response.image_b64:
                frames.append(response.image_b64)
                thought_signature = response.thought_signature
            else:
                logger.warning("Frame %d failed: %s", i + 1, response.error)
                # Don't break — continue with remaining frames

            if on_frame_complete:
                on_frame_complete(i, response)

        return frames, responses

    # ─────────────────────────────────────────────────────────────────────
    #  API Request Building
    # ─────────────────────────────────────────────────────────────────────

    def _build_request_body(
        self,
        prompt: str,
        reference_image_b64: str,
        additional_images_b64: List[str],
        thinking_level: ThinkingLevel,
        previous_thought_signature: Optional[str],
    ) -> Dict[str, Any]:
        """Build the API request body."""
        # Build parts list
        parts = []

        # Add reference image
        parts.append({
            "inline_data": {
                "mime_type": "image/png",
                "data": _sanitize_base64(reference_image_b64),
            }
        })

        # Add additional images
        for img_b64 in additional_images_b64:
            parts.append({
                "inline_data": {
                    "mime_type": "image/png",
                    "data": _sanitize_base64(img_b64),
                }
            })

        # Add prompt
        parts.append({"text": prompt})

        # Build request body
        body: Dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
            },
        }

        # Add thinking configuration
        if thinking_level != ThinkingLevel.NONE:
            body["generationConfig"]["thinkingConfig"] = {
                "thinkingLevel": thinking_level.value,
                "includeThoughts": False,  # We just want the signature
            }

        # Add previous thought signature for consistency
        if previous_thought_signature:
            body["thought_signature"] = previous_thought_signature

        return body

    async def _call_api(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """Make the API call."""
        url = f"{self.base_url}/models/{self.model}:generateContent"

        response = await self._client.post(
            url,
            json=request_body,
            params={"key": self.api_key},
        )
        response.raise_for_status()

        return response.json()

    def _parse_response(
        self,
        response: Dict[str, Any],
        start_time: float,
    ) -> EditResponse:
        """Parse the API response."""
        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        # Extract image from response
        image_b64 = None
        candidates = response.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            for part in parts:
                if "inline_data" in part:
                    image_b64 = part["inline_data"].get("data")
                    break

        # Extract thought signature
        thought_signature = response.get("thought_signature")

        # Extract usage stats
        usage = response.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        completion_tokens = usage.get("candidatesTokenCount", 0)

        if image_b64:
            return EditResponse(
                success=True,
                image_b64=image_b64,
                thought_signature=thought_signature,
                model_used=self.model,
                generation_time_ms=elapsed_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        else:
            return EditResponse(
                success=False,
                error="No image in API response",
                thought_signature=thought_signature,
                model_used=self.model,
                generation_time_ms=elapsed_ms,
            )


# ═══════════════════════════════════════════════════════════════════════════
#  Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def _sanitize_base64(b64: str) -> str:
    """Remove data URI prefix if present."""
    if b64.startswith("data:"):
        parts = b64.split(",", 1)
        return parts[1] if len(parts) == 2 else b64
    return b64


def is_available() -> bool:
    """Check if the Gemini image editor is available."""
    return HAS_HTTPX


def create_editor(
    api_key: str,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
) -> GeminiImageEditor:
    """Factory function to create a GeminiImageEditor."""
    return GeminiImageEditor(
        api_key=api_key,
        model=model or DEFAULT_MODEL,
        timeout=timeout or REQUEST_TIMEOUT,
    )
