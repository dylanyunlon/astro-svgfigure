from __future__ import annotations
import base64, hashlib, io, json, logging, time
from typing import Any, Dict, List, Optional, Tuple
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
#  §1  Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class VisionDetectConfig:
    """Configuration for vision-LLM UI detection."""
    model: str = ""                   # Empty = auto (prefer Claude for vision)
    grid_snap: int = 0                # Snap bbox to grid (0=off, 4=4px grid)
    min_element_area: int = 64        # Skip elements smaller than 8×8
    max_elements: int = 200           # Cap output count
    temperature: float = 0.1          # Low temp for deterministic coordinates
    cache_enabled: bool = True        # Cache per image hash
    include_text: bool = True         # Detect text elements too
    include_icons: bool = True        # Detect icon/image elements
    include_containers: bool = False  # Detect group/container elements


# ═══════════════════════════════════════════════════════════════════════
#  §2  Prompt (the core of the approach)
# ═══════════════════════════════════════════════════════════════════════

DETECT_SYSTEM_PROMPT = """You are a UI layout analyzer. You receive a screenshot of a user interface and output a JSON array of ALL visible UI elements with their PIXEL-PRECISE bounding boxes.

CRITICAL RULES:
1. Output ONLY a JSON array. No markdown, no explanation, no preamble.
2. Coordinates are in PIXELS relative to the image top-left corner (0,0).
3. Every element gets: id (string), name (descriptive English or original language), bbox (object with x, y, width, height as integers).
4. Detect: icons, buttons, text labels, progress bars, images, indicators, navigation items.
5. name should be descriptive and match what a designer would call it (e.g. "tire_pressure_icon", "speed_gauge", "bluetooth_status", "menu_dock").
6. Be EXHAUSTIVE — detect every visible element, even small indicators and decorative elements.
7. bbox must be TIGHT — match the element boundary exactly, no extra padding.
8. For overlapping elements, report each one separately with its own bbox."""

DETECT_USER_PROMPT = """Analyze this UI screenshot ({width}×{height} pixels).

Output a JSON array of ALL UI elements:
[
  {{"id": "0", "name": "descriptive_name", "bbox": {{"x": 100, "y": 50, "width": 52, "height": 52}}}},
  ...
]

Detect every icon, button, text label, gauge, indicator, progress bar, and image. Be pixel-precise."""


# ═══════════════════════════════════════════════════════════════════════
#  §3  Detection Cache
# ═══════════════════════════════════════════════════════════════════════

_detection_cache: Dict[str, List[Dict[str, Any]]] = {}


def _image_hash(b64: str) -> str:
    return hashlib.md5(b64[:4096].encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════
#  §4  Core: Vision-LLM Detection
# ═══════════════════════════════════════════════════════════════════════

async def vision_detect(
    image_b64: str,
    config: Optional[VisionDetectConfig] = None,
    ai_engine=None,
) -> List[Dict[str, Any]]:
    """Detect UI elements via vision LLM API, return mastergo format.

    Parameters
    ----------
    image_b64 : str
        Base64-encoded screenshot (PNG/JPEG).
    config : VisionDetectConfig
        Detection parameters.
    ai_engine : AIEngine, optional
        Existing AIEngine instance. If None, creates one from settings.

    Returns
    -------
    List of mastergo-format objects:
        [{"id": "0", "name": "icon_name", "bbox": {"x":int,"y":int,"width":int,"height":int}}]
    """
    if config is None:
        config = VisionDetectConfig()

    # Cache check
    if config.cache_enabled:
        key = _image_hash(image_b64)
        if key in _detection_cache:
            logger.info("Vision detect cache hit: %s", key)
            return _detection_cache[key]

    # Strip data URI
    raw_b64 = image_b64
    if raw_b64.startswith("data:"):
        raw_b64 = raw_b64.split(",", 1)[1]

    # Get image dimensions
    if _HAS_PIL:
        try:
            img = Image.open(io.BytesIO(base64.b64decode(raw_b64)))
            img_w, img_h = img.size
        except Exception:
            img_w, img_h = 1024, 600
    else:
        img_w, img_h = 1024, 600

    # Get or create AIEngine
    if ai_engine is None:
        try:
            from backend.config import get_settings
            from backend.ai_engine import AIEngine
            ai_engine = AIEngine(get_settings())
        except Exception as e:
            logger.error("Cannot create AIEngine: %s", e)
            return _fallback_detect(raw_b64, img_w, img_h, config)

    # Build vision message (OpenAI format, converted per-provider)
    data_uri = f"data:image/png;base64,{raw_b64}"
    user_prompt = DETECT_USER_PROMPT.format(width=img_w, height=img_h)

    messages = [
        {"role": "system", "content": DETECT_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": user_prompt},
        ]},
    ]

    # Pick model
    # Pick model: prefer Claude for vision → structured JSON
    # Claude outputs more precise bbox coordinates than Gemini for UI analysis
    if config.model:
        detect_model = config.model
    elif ai_engine._settings.ANTHROPIC_API_KEY or ai_engine._settings.CLAUDE_COMPATIBLE_API_KEY:
        detect_model = ai_engine._settings.ANTHROPIC_DEFAULT_MODEL or "claude-sonnet-4-20250514"
    else:
        detect_model = ai_engine._settings.DEFAULT_MODEL

    t0 = time.monotonic()
    try:
        provider = ai_engine._select_provider(detect_model)
        response = await provider.get_completion(
            messages=messages,
            model=detect_model,
            temperature=config.temperature,
            max_tokens=8192,
        )

        raw_text = response.get("content", "")
        elements = _parse_response(raw_text, img_w, img_h, config)

    except Exception as e:
        logger.warning("Vision detect API call failed: %s, using fallback", e)
        elements = _fallback_detect(raw_b64, img_w, img_h, config)

    elapsed = (time.monotonic() - t0) * 1000
    logger.info("Vision detect: %d elements in %.0fms (%dx%d)", len(elements), elapsed, img_w, img_h)

    # Cache
    if config.cache_enabled and elements:
        _detection_cache[_image_hash(image_b64)] = elements

    return elements


def _parse_response(
    raw_text: str,
    img_w: int,
    img_h: int,
    config: VisionDetectConfig,
) -> List[Dict[str, Any]]:
    """Parse LLM JSON response into mastergo-format objects.

    Handles common LLM output issues:
    - Markdown code fences
    - Trailing commas
    - Partial JSON
    """
    # Strip markdown fences
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    # Try direct parse
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try fixing trailing commas
        import re
        fixed = re.sub(r',\s*([}\]])', r'\1', cleaned)
        try:
            data = json.loads(fixed)
        except json.JSONDecodeError:
            # Try extracting array from text
            match = re.search(r'\[.*\]', cleaned, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.warning("Cannot parse vision detect response")
                    return []
            else:
                return []

    if not isinstance(data, list):
        data = [data] if isinstance(data, dict) else []

    # Convert and validate
    objects = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        bbox = item.get("bbox", {})
        if not isinstance(bbox, dict):
            continue

        try:
            px_x = int(round(float(bbox.get("x", 0))))
            px_y = int(round(float(bbox.get("y", 0))))
            px_w = int(round(float(bbox.get("width", 0))))
            px_h = int(round(float(bbox.get("height", 0))))
        except (TypeError, ValueError):
            continue

        # Grid snap
        if config.grid_snap > 0:
            g = config.grid_snap
            px_x = round(px_x / g) * g
            px_y = round(px_y / g) * g
            px_w = max(g, round(px_w / g) * g)
            px_h = max(g, round(px_h / g) * g)

        # Validate bounds
        if px_w <= 0 or px_h <= 0:
            continue
        if px_w * px_h < config.min_element_area:
            continue
        if px_x < 0 or px_y < 0 or px_x >= img_w or px_y >= img_h:
            continue

        # Clamp to image
        px_x = min(px_x, img_w - 1)
        px_y = min(px_y, img_h - 1)
        px_w = min(px_w, img_w - px_x)
        px_h = min(px_h, img_h - px_y)

        name = str(item.get("name", f"element_{i}")).strip()
        obj_id = str(item.get("id", str(i)))

        objects.append({
            "id": f"vd:{obj_id}",
            "name": name,
            "bbox": {"x": px_x, "y": px_y, "width": px_w, "height": px_h},
        })

    # Sort by area descending
    objects.sort(key=lambda o: o["bbox"]["width"] * o["bbox"]["height"], reverse=True)

    # Limit
    if len(objects) > config.max_elements:
        objects = objects[:config.max_elements]

    return objects


# ═══════════════════════════════════════════════════════════════════════
#  §5  Fallback: CCL when no API available
# ═══════════════════════════════════════════════════════════════════════

def _fallback_detect(
    image_b64: str,
    img_w: int,
    img_h: int,
    config: VisionDetectConfig,
) -> List[Dict[str, Any]]:
    """Fallback when vision API is unavailable. Uses scipy CCL."""
    if not _HAS_NUMPY or not _HAS_PIL:
        return []

    try:
        img = Image.open(io.BytesIO(base64.b64decode(image_b64)))
        gray = np.array(img.convert("L"), dtype=np.float32)

        dx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        dy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        edges = ((dx + dy) > 30).astype(np.uint8)

        from scipy.ndimage import label as scipy_label, find_objects
        labels, num = scipy_label(edges)

        objects = []
        for i, sl in enumerate(find_objects(labels)):
            if sl is None:
                continue
            y_sl, x_sl = sl
            x, y = x_sl.start, y_sl.start
            w, h = x_sl.stop - x, y_sl.stop - y
            if w * h < config.min_element_area:
                continue
            objects.append({
                "id": f"ccl:{i}",
                "name": f"component_{i}",
                "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
            })

        objects.sort(key=lambda o: o["bbox"]["width"] * o["bbox"]["height"], reverse=True)
        return objects[:config.max_elements]

    except Exception as e:
        logger.warning("Fallback CCL failed: %s", e)
        return []

