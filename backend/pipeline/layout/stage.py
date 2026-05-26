from __future__ import annotations
import base64, hashlib, io, json, logging, time
from typing import Any, Dict, List, Optional, Tuple
from backend.pipeline.layout.detect import vision_detect, VisionDetectConfig
logger = logging.getLogger(__name__)
try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

# ═══════════════════════════════════════════════════════════════════════
#  §6  Pipeline Stage
# ═══════════════════════════════════════════════════════════════════════

async def stage_omniparser_detect(
    frames_b64: List[str],
    config: Optional[VisionDetectConfig] = None,
    progress=None,
) -> Dict[str, Any]:
    """Pipeline stage: run vision detection on generated frames.

    Slots into pipeline_orchestrator.py as Stage 0.5:
        Gemini gen → THIS → removebg → layers → export
    """
    if config is None:
        config = VisionDetectConfig()

    t0 = time.monotonic()

    # Create AIEngine once for all frames
    ai_engine = None
    try:
        from backend.config import get_settings
        from backend.ai_engine import AIEngine
        ai_engine = AIEngine(get_settings())
    except Exception as e:
        logger.warning("Cannot create AIEngine for vision detect: %s", e)

    layouts = []
    total_elements = 0

    for i, frame_b64 in enumerate(frames_b64):
        if progress:
            progress("omniparser_detect", f"frame {i+1}/{len(frames_b64)}",
                     int(i / len(frames_b64) * 100))

        objects = await vision_detect(frame_b64, config, ai_engine)
        layouts.append(objects)
        total_elements += len(objects)

    elapsed = (time.monotonic() - t0) * 1000
    avg = total_elements / len(frames_b64) if frames_b64 else 0

    if progress:
        progress("omniparser_detect", "complete", 100)

    return {
        "success": total_elements > 0,
        "frames_b64": frames_b64,
        "layouts": layouts,
        "stats": {
            "total_elements": total_elements,
            "avg_per_frame": round(avg, 1),
            "processing_time_ms": round(elapsed, 2),
            "method": "vision_llm",
            "model": config.model or "default",
        },
    }


# ═══════════════════════════════════════════════════════════════════════
#  §7  Availability Check
# ═══════════════════════════════════════════════════════════════════════

def is_omniparser_available() -> Dict[str, Any]:
    """Check if vision detection is available (needs API key)."""
    try:
        from backend.config import get_settings
        s = get_settings()
        has_gemini = bool(s.GEMINI_API_KEY)
        has_openai = bool(s.OPENAI_API_KEY)
        has_claude = bool(s.ANTHROPIC_API_KEY or s.CLAUDE_COMPATIBLE_API_KEY)
        return {
            "available": has_gemini or has_openai or has_claude,
            "gemini": has_gemini,
            "openai": has_openai,
            "claude": has_claude,
            "method": "vision_llm",
        }
    except Exception:
        return {"available": False, "method": "vision_llm"}


# ═══════════════════════════════════════════════════════════════════════
#  §8  API Handler
# ═══════════════════════════════════════════════════════════════════════

async def handle_omniparser_detect(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle /api/omniparser-detect endpoint."""
    image_b64 = request_data.get("image_b64")
    if not image_b64:
        return {"success": False, "error": "image_b64 is required"}

    cfg_raw = request_data.get("config", {})
    config = VisionDetectConfig(
        model=cfg_raw.get("model", ""),
        grid_snap=int(cfg_raw.get("grid_snap", 0)),
        min_element_area=int(cfg_raw.get("min_element_area", 64)),
        max_elements=int(cfg_raw.get("max_elements", 200)),
        temperature=float(cfg_raw.get("temperature", 0.1)),
        cache_enabled=bool(cfg_raw.get("cache", True)),
    )

    t0 = time.monotonic()
    objects = await vision_detect(image_b64, config)
    elapsed = (time.monotonic() - t0) * 1000

    return {
        "success": len(objects) > 0,
        "layout": objects,
        "total_elements": len(objects),
        "stats": {
            "processing_time_ms": round(elapsed, 2),
            **is_omniparser_available(),
        },
    }

