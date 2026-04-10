"""
Animation Pipeline Routes — FastAPI Endpoint Registration
============================================================
Registers all animation-related API endpoints for the /playground
page's frame generation pipeline.

Endpoints:
  POST /api/analyze-image    → Claude 4.6 image analysis (Step 1)
  POST /api/animate-prompt   → Grok animation prompt design (Step 2)
  POST /api/animate-frames   → Gemini frame generation (Step 3)
  POST /api/rembg-frames     → Server-side background removal (Step 4)
  POST /api/encode-animation → GIF/APNG/WebP encoding (Step 5)

Integration:
────────────
This module exports `register_animation_routes(app)` which should be
called from server.py after the FastAPI app is created. It adds the
routes to the existing app without modifying server.py's structure.

Usage in server.py:
    from backend.server_animation_routes import register_animation_routes
    register_animation_routes(app)

Knuth-Level Critiques:
─────────────────────
User Angle:
  - ALL endpoints return consistent JSON: { success: bool, error?: string, ... }
    This allows the frontend to use a single error handler pattern.
  - Timeout for animate-frames is 900s (15 min) to handle slow Gemini
    responses with many frames. If the user's connection drops, the
    backend still finishes (fire-and-forget) but the response is lost.
    Future: use job IDs and polling for long operations.

System Angle:
  - The AIEngine is created per-request, not shared. This is intentional:
    sharing would require thread-safety guarantees across async contexts.
    The overhead of creating AIEngine per-request is negligible (~1ms)
    because the provider instances cache their HTTP clients.
  - Request body size is limited by FastAPI/Starlette default (100MB).
    For 16 frames of base64 images, this is adequate. For 24 frames
    at 4K resolution, we might hit the limit — but 24×4K is unrealistic
    for animation frame generation.

GitHub references:
  - tiangolo/fastapi (APIRouter, HTTPException)
  - dylanyunlon/astro-svgfigure/server.py (existing route pattern)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from backend.ai_engine import AIEngine
from backend.config import get_settings
from backend.schemas_animation import (
    AnalyzeImageRequest,
    AnalyzeImageResponse,
    AnimatePromptRequest,
    AnimatePromptResponse,
    AnimateFramesRequest,
    AnimateFramesResponse,
    RembgFramesRequest,
    RembgFramesResponse,
    EncodeAnimationRequest,
    EncodeAnimationResponse,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Route Registration
# ═══════════════════════════════════════════════════════════════════════

def register_animation_routes(app: FastAPI) -> None:
    """
    Register all animation pipeline endpoints on the FastAPI app.

    Call this from server.py after creating the FastAPI instance:
        from backend.server_animation_routes import register_animation_routes
        register_animation_routes(app)
    """

    # ── Step 1: Image Analysis (Claude 4.6) ──────────────────────────

    @app.post("/api/analyze-image")
    async def api_analyze_image(request: AnalyzeImageRequest) -> JSONResponse:
        """
        Analyze an uploaded image for animation frame generation.

        Uses Claude 4.6 vision to identify visual components, spatial
        layout, and animatable elements. This feeds into Grok's prompt
        design in Step 2.
        """
        t0 = time.monotonic()
        settings = get_settings()
        engine = AIEngine(settings)

        try:
            from backend.pipeline.image_analyzer import analyze_image
            response = await analyze_image(
                request=request,
                settings=settings,
                ai_engine=engine,
            )
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            logger.info("POST /api/analyze-image → %s (%d ms)",
                       "ok" if response.success else "error", elapsed_ms)

            status_code = 200 if response.success else 500
            return JSONResponse(
                content=response.model_dump(exclude_none=True),
                status_code=status_code,
            )

        except Exception as e:
            logger.exception("POST /api/analyze-image failed: %s", e)
            return JSONResponse(
                content={"success": False, "error": str(e)},
                status_code=500,
            )

    # ── Step 2: Animation Prompt Design (Grok) ───────────────────────

    @app.post("/api/animate-prompt")
    async def api_animate_prompt(request: AnimatePromptRequest) -> JSONResponse:
        """
        Design animation frame prompts using Grok.

        Takes the image analysis from Step 1 and engineers per-frame
        animation instructions with mandatory green-screen background.
        The user can review and edit the prompt before proceeding.
        """
        t0 = time.monotonic()
        settings = get_settings()
        engine = AIEngine(settings)

        try:
            from backend.pipeline.animation_prompt_designer import design_animation_prompt
            response = await design_animation_prompt(
                request=request,
                settings=settings,
                ai_engine=engine,
            )
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            logger.info("POST /api/animate-prompt → %s (%d ms)",
                       "ok" if response.success else "error", elapsed_ms)

            status_code = 200 if response.success else 500
            return JSONResponse(
                content=response.model_dump(exclude_none=True),
                status_code=status_code,
            )

        except Exception as e:
            logger.exception("POST /api/animate-prompt failed: %s", e)
            return JSONResponse(
                content={"success": False, "error": str(e)},
                status_code=500,
            )

    # ── Step 3: Frame Generation (Gemini) ────────────────────────────

    @app.post("/api/animate-frames")
    async def api_animate_frames(request: AnimateFramesRequest) -> JSONResponse:
        """
        Generate multi-frame animation images using Gemini.

        Produces N frames with green-screen (#00FF00) background.
        Long-running operation — may take 1-5 minutes depending on
        frame count and complexity.
        """
        t0 = time.monotonic()
        settings = get_settings()
        engine = AIEngine(settings)

        try:
            from backend.pipeline.frame_generator import generate_animation_frames
            response = await generate_animation_frames(
                request=request,
                settings=settings,
                ai_engine=engine,
            )
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            frame_count = response.frame_count or 0
            logger.info(
                "POST /api/animate-frames → %s, %d frames (%d ms)",
                "ok" if response.success else "error", frame_count, elapsed_ms,
            )

            status_code = 200 if response.success else 500
            return JSONResponse(
                content=response.model_dump(exclude_none=True),
                status_code=status_code,
            )

        except Exception as e:
            logger.exception("POST /api/animate-frames failed: %s", e)
            return JSONResponse(
                content={"success": False, "error": str(e)},
                status_code=500,
            )

    # ── Step 4: Server-Side Background Removal ───────────────────────

    @app.post("/api/rembg-frames")
    async def api_rembg_frames(request: RembgFramesRequest) -> JSONResponse:
        """
        Server-side background removal from animation frames.

        Optional enhancement — the client-side green-screen removal
        (Canvas API chroma-key) handles most cases. This endpoint
        is for complex edges or when client-side results are poor.

        Methods:
        - green_screen: Fast chroma-key removal (PIL-based)
        - rembg_u2net: Accurate ML-based removal (requires rembg package)
        - rembg_isnet: Alternative ML model (requires rembg package)
        """
        t0 = time.monotonic()

        try:
            from backend.pipeline.rembg_processor import process_frames
            result = await process_frames(
                frames_b64=request.frames_b64,
                method=request.method.value,
                tolerance=request.tolerance,
                edge_blur=request.edge_blur,
                despill=request.despill,
            )
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            logger.info(
                "POST /api/rembg-frames → %s, %d frames, method=%s (%d ms)",
                "ok" if result["success"] else "error",
                len(request.frames_b64),
                request.method.value,
                elapsed_ms,
            )

            status_code = 200 if result["success"] else 500
            return JSONResponse(content=result, status_code=status_code)

        except Exception as e:
            logger.exception("POST /api/rembg-frames failed: %s", e)
            return JSONResponse(
                content={"success": False, "error": str(e)},
                status_code=500,
            )

    # ── Step 5: Animation Encoding ───────────────────────────────────

    @app.post("/api/encode-animation")
    async def api_encode_animation(request: EncodeAnimationRequest) -> JSONResponse:
        """
        Encode transparent PNG frames into an animated format.

        Formats:
        - apng: Animated PNG (best quality, full alpha, large files)
        - gif: Animated GIF (universal compat, 1-bit alpha, 256 colors)
        - webp: Animated WebP (good quality + small size, modern browsers)
        - zip: ZIP archive of individual PNG frames
        """
        t0 = time.monotonic()

        try:
            from backend.pipeline.gif_encoder import encode_animation
            result = await encode_animation(
                frames_b64=request.frames_b64,
                format=request.format.value,
                fps=request.fps,
                loop_count=request.loop_count,
                optimize=request.optimize,
            )
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            file_size = result.get("file_size_bytes", 0) or 0
            logger.info(
                "POST /api/encode-animation → %s, format=%s, %d KB (%d ms)",
                "ok" if result["success"] else "error",
                request.format.value,
                file_size // 1024,
                elapsed_ms,
            )

            status_code = 200 if result["success"] else 500
            return JSONResponse(content=result, status_code=status_code)

        except Exception as e:
            logger.exception("POST /api/encode-animation failed: %s", e)
            return JSONResponse(
                content={"success": False, "error": str(e)},
                status_code=500,
            )

    # ── Animation Pipeline Health Check ──────────────────────────────

    @app.get("/api/animation-health")
    async def api_animation_health() -> JSONResponse:
        """
        Health check for animation pipeline capabilities.

        Reports which pipeline modules are available:
        - rembg (server-side background removal)
        - GIF/APNG/WebP encoding
        - AI providers for each step
        """
        try:
            from backend.pipeline.rembg_processor import is_available as rembg_available
            from backend.pipeline.gif_encoder import is_available as encoder_available

            rembg_status = rembg_available()
            encoder_status = encoder_available()

            settings = get_settings()
            models = settings.AVAILABLE_MODELS

            return JSONResponse(content={
                "backend": True,
                "animation_pipeline": True,
                "animation_pipeline_v2": True,
                "rembg": rembg_status,
                "encoders": encoder_status,
                "models": {
                    "analyzer": settings.ANTHROPIC_DEFAULT_MODEL,
                    "prompt_designer": settings.DEFAULT_PROMPT_MODEL,
                    "frame_generator": settings.DEFAULT_IMAGE_MODEL,
                },
                "providers_configured": {
                    "anthropic": bool(settings.ANTHROPIC_API_KEY),
                    "openai": bool(settings.OPENAI_API_KEY),
                    "gemini": bool(settings.GEMINI_API_KEY),
                },
            })

        except Exception as e:
            return JSONResponse(
                content={"backend": True, "animation_pipeline": False, "error": str(e)},
                status_code=500,
            )

    # ═══════════════════════════════════════════════════════════════════════
    #  V2 API: Reference-Preserving Animation Pipeline
    # ═══════════════════════════════════════════════════════════════════════

    @app.post("/api/v2/animate")
    async def api_animate_v2(request: dict) -> JSONResponse:
        """
        V2 Animation Pipeline: Single endpoint that preserves original image.

        This is the NEW reference-preserving pipeline that solves the core
        issue of generated frames having no relation to the original image.

        Key differences from v1:
        - Grok receives BOTH the image AND the analysis (not just text)
        - Gemini edits the ORIGINAL image instead of generating new ones
        - Green screen removal uses HSV color space for accuracy
        - Cross-frame consistency validation

        Request body:
        {
            "image_b64": "base64-encoded image",
            "user_prompt": "optional animation direction",
            "frame_count": 8,
            "fps": 12,
            "output_format": "gif" | "webp" | "apng",
            "green_screen": true
        }

        Response:
        {
            "success": true,
            "animation_b64": "base64-encoded result",
            "frames_b64": ["frame1", "frame2", ...],
            "frame_count": 8,
            "format": "gif",
            "timing_ms": { ... }
        }
        """
        t0 = time.monotonic()
        settings = get_settings()

        try:
            from backend.pipeline.animation_pipeline_orchestrator import (
                AnimationPipelineOrchestrator,
                AnimationConfig,
            )

            # Parse request
            image_b64 = request.get("image_b64")
            if not image_b64:
                return JSONResponse(
                    content={"success": False, "error": "image_b64 is required"},
                    status_code=400,
                )

            # Build config
            config = AnimationConfig(
                frame_count=request.get("frame_count", 8),
                fps=request.get("fps", 12),
                output_format=request.get("output_format", "gif"),
                green_screen=request.get("green_screen", True),
                user_prompt=request.get("user_prompt"),
            )

            # Run orchestrator
            orchestrator = AnimationPipelineOrchestrator(settings)
            result = await orchestrator.run(image_b64, config)

            elapsed_ms = int((time.monotonic() - t0) * 1000)
            logger.info(
                "POST /api/v2/animate → %s, %d frames (%d ms)",
                "ok" if result.get("success") else "error",
                result.get("frame_count", 0),
                elapsed_ms,
            )

            status_code = 200 if result.get("success") else 500
            return JSONResponse(content=result, status_code=status_code)

        except Exception as e:
            logger.exception("POST /api/v2/animate failed: %s", e)
            return JSONResponse(
                content={"success": False, "error": str(e)},
                status_code=500,
            )

    @app.post("/api/v2/animate/preview")
    async def api_animate_preview_v2(request: dict) -> JSONResponse:
        """
        Quick preview: Generate just 2-3 frames to preview the animation.

        Useful for the user to verify the animation direction before
        committing to a full 8-16 frame generation.
        """
        t0 = time.monotonic()
        settings = get_settings()

        try:
            from backend.pipeline.animation_pipeline_orchestrator import (
                AnimationPipelineOrchestrator,
                AnimationConfig,
            )

            image_b64 = request.get("image_b64")
            if not image_b64:
                return JSONResponse(
                    content={"success": False, "error": "image_b64 is required"},
                    status_code=400,
                )

            # Preview mode: just 3 frames
            config = AnimationConfig(
                frame_count=3,
                fps=request.get("fps", 8),
                output_format="gif",
                green_screen=request.get("green_screen", True),
                user_prompt=request.get("user_prompt"),
            )

            orchestrator = AnimationPipelineOrchestrator(settings)
            result = await orchestrator.run(image_b64, config)

            elapsed_ms = int((time.monotonic() - t0) * 1000)
            logger.info("POST /api/v2/animate/preview → %s (%d ms)",
                       "ok" if result.get("success") else "error", elapsed_ms)

            status_code = 200 if result.get("success") else 500
            return JSONResponse(content=result, status_code=status_code)

        except Exception as e:
            logger.exception("POST /api/v2/animate/preview failed: %s", e)
            return JSONResponse(
                content={"success": False, "error": str(e)},
                status_code=500,
            )

    @app.post("/api/v2/validate-image")
    async def api_validate_image_v2(request: dict) -> JSONResponse:
        """
        Pre-validate an image before animation.

        Checks for issues that would affect animation quality:
        - Image too small/large
        - Already contains green background
        - Contains text that might get distorted
        - Low complexity (solid colors)
        """
        try:
            from backend.pipeline.reference_image_validator import (
                ReferenceImageValidator,
            )

            image_b64 = request.get("image_b64")
            if not image_b64:
                return JSONResponse(
                    content={"success": False, "error": "image_b64 is required"},
                    status_code=400,
                )

            validator = ReferenceImageValidator()
            result = await validator.validate(image_b64)

            return JSONResponse(content={
                "success": True,
                "valid": result.is_valid,
                "errors": result.errors,
                "warnings": result.warnings,
                "info": result.info,
            })

        except Exception as e:
            logger.exception("POST /api/v2/validate-image failed: %s", e)
            return JSONResponse(
                content={"success": False, "error": str(e)},
                status_code=500,
            )

    logger.info("Animation pipeline routes registered: "
                "/api/analyze-image, /api/animate-prompt, /api/animate-frames, "
                "/api/rembg-frames, /api/encode-animation, /api/animation-health, "
                "/api/v2/animate, /api/v2/animate/preview, /api/v2/validate-image")
