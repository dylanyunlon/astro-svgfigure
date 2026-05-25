from __future__ import annotations

import json
import os
import queue
import shutil
import signal
import socket
import subprocess
import threading
import time
import uuid
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import asyncio
import logging

from collections import defaultdict
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

# Pipeline imports (Phase 1 backend modules)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backend.config import get_settings
from backend.ai_engine import AIEngine
from backend.schemas import (
    TopologyRequest,
    BeautifyRequest,
    ValidateRequest,
    TopologyResponse,
    BeautifyResponse,
)
from backend.pipeline.topology_gen import generate_topology
from backend.pipeline.scaffold_builder import build_scaffold
from backend.pipeline.nanobanana_bridge import beautify_with_nanobanana
from backend.pipeline.svg_validator import validate_svg as validate_svg_func
from backend.server_animation_routes import register_animation_routes

logger = logging.getLogger(__name__)

# Post-generation pipeline imports (Step 4+: removebg → layers → edges → export)
try:
    from backend.pipeline.removebg_route import handle_removebg, get_removebg_status
    from backend.pipeline.pipeline_orchestrator import run_pipeline
    _PIPELINE_AVAILABLE = True
except ImportError as _e:
    logger.warning(f"Post-generation pipeline modules not fully available: {_e}")
    _PIPELINE_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

PYTHON_EXECUTABLE = os.environ.get("AUTOFIGURE_PYTHON") or sys.executable

DEFAULT_SAM_PROMPT = "icon,person,animal,robot"
DEFAULT_PLACEHOLDER_MODE = "label"
DEFAULT_MERGE_THRESHOLD = 0.01

SVG_EDIT_CANDIDATES = [
    ("vendor/svg-edit/editor/index.html", WEB_DIR / "vendor" / "svg-edit" / "editor" / "index.html"),
    ("vendor/svg-edit/editor.html", WEB_DIR / "vendor" / "svg-edit" / "editor.html"),
    ("vendor/svg-edit/index.html", WEB_DIR / "vendor" / "svg-edit" / "index.html"),
]


def _resolve_svg_edit_path() -> tuple[bool, str | None]:
    for rel, path in SVG_EDIT_CANDIDATES:
        if path.is_file():
            return True, f"/{rel}"
    return False, None


@dataclass
class Job:
    job_id: str
    output_dir: Path
    process: subprocess.Popen
    queue: queue.Queue
    log_path: Path
    log_lock: threading.Lock = field(default_factory=threading.Lock)
    seen: set[str] = field(default_factory=set)
    done: bool = False

    def push(self, event: str, data: dict) -> None:
        self.queue.put({"event": event, "data": data})

    def write_log(self, stream: str, line: str) -> None:
        with self.log_lock:
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(f"[{stream}] {line}\n")


class RunRequest(BaseModel):
    method_text: str = Field(..., min_length=1)
    provider: str = "bianxie"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    image_model: Optional[str] = None
    svg_model: Optional[str] = None
    sam_prompt: Optional[str] = None
    sam_backend: Optional[str] = None
    sam_api_key: Optional[str] = None
    sam_max_masks: Optional[int] = None
    placeholder_mode: Optional[str] = None
    merge_threshold: Optional[float] = None
    optimize_iterations: Optional[int] = None
    reference_image_path: Optional[str] = None


app = FastAPI()

# ── IP Firewall v2 ("French Beach + Underwear Thief") ────────────────
# Layer 0: Whitelist (your own IPs, private ranges)
# Layer 1: Behavior traps (scanner paths → instant 24h ban, zero API cost)
# Layer 2: IP reputation (AbuseIPDB / Scamalytics / proxycheck, parallel max)

_IP_WHITELIST: frozenset[str] = frozenset(
    s.strip() for s in os.environ.get("IP_WHITELIST", "").split(",") if s.strip()
)

# Trap paths — any legitimate user would NEVER request these.
# Hitting one = caught stealing underwear on the beach.
_TRAP_EXACT: frozenset[str] = frozenset({
    "/server", "/server-status", "/server-info",
    "/login.action", "/debug/default/view", "/trace.axd",
    "/info.php", "/phpinfo.php", "/config.json",
    "/telescope/requests", "/.ds_store", "/.vscode/sftp.json",
    "/swagger.json", "/swagger-ui",
})

_TRAP_PREFIXES: tuple[str, ...] = (
    "/wp-json/", "/wp-admin", "/wp-content", "/wp-includes", "/wp-login",
    "/xmlrpc.php",
    "/___proxy_subdomain_whm", "/___proxy_subdomain_cpanel",
    "/cpanel", "/whm",
    "/v2/_catalog",
    "/ecp/", "/owa/", "/autodiscover/",
    "/s/8323",
    "/actuator/",
    "/.env", "/.git/", "/.svn/", "/.hg/",
    "/proc/", "/etc/passwd",
    "/swagger/", "/swagger/v1/", "/api-docs/",
    "/v2/api-docs", "/v3/api-docs", "/webjars/swagger",
    "/graphql",
    "/phpmyadmin", "/adminer", "/pma/",
    "/node_modules/", "/.npmrc",
)

_TRAP_SUBSTRINGS: tuple[str, ...] = (
    "META-INF/", "WEB-INF/", "..%2f", "..%5c", "%00", "pom.properties",
)

def _is_trap_path(path: str) -> bool:
    low = path.lower()
    if low in _TRAP_EXACT:
        return True
    for p in _TRAP_PREFIXES:
        if low.startswith(p):
            return True
    for s in _TRAP_SUBSTRINGS:
        if s in low:
            return True
    return False


class IPFirewall(BaseHTTPMiddleware):
    """Three-layer firewall: whitelist → behavior traps → IP reputation."""

    BLOCK_THRESHOLD = int(os.environ.get("IP_BLOCK_THRESHOLD", "75"))
    CACHE_TTL = int(os.environ.get("IP_CACHE_TTL", "3600"))
    CACHE_MAX = int(os.environ.get("IP_CACHE_MAX", "4096"))
    TRAP_BAN_TTL = 86400  # 24 hours for trap bans

    ABUSEIPDB_KEY = os.environ.get("ABUSEIPDB_API_KEY", "")
    SCAMALYTICS_KEY = os.environ.get("SCAMALYTICS_API_KEY", "")
    SCAMALYTICS_USER = os.environ.get("SCAMALYTICS_USERNAME", "")
    PROXYCHECK_KEY = os.environ.get("PROXYCHECK_API_KEY", "")

    _PRIVATE_PREFIXES = (
        "127.", "10.", "192.168.",
        "172.16.", "172.17.", "172.18.", "172.19.",
        "172.20.", "172.21.", "172.22.", "172.23.",
        "172.24.", "172.25.", "172.26.", "172.27.",
        "172.28.", "172.29.", "172.30.", "172.31.",
        "::1", "fe80:", "0.0.0.0",
    )

    def __init__(self, app):
        super().__init__(app)
        # ip → (expiry_timestamp, score, blocked)
        self._cache: dict[str, tuple[float, int, bool]] = {}
        import httpx
        self._http = httpx.AsyncClient(timeout=3.0)

    def _extract_ip(self, request: Request) -> str:
        for hdr in ("cf-connecting-ip", "x-forwarded-for", "x-real-ip"):
            val = request.headers.get(hdr)
            if val:
                return val.split(",")[0].strip()
        return request.client.host if request.client else "0.0.0.0"

    def _is_trusted(self, ip: str) -> bool:
        if ip in _IP_WHITELIST:
            return True
        return any(ip.startswith(p) for p in self._PRIVATE_PREFIXES)

    def _cache_get(self, ip: str) -> tuple[int, bool] | None:
        entry = self._cache.get(ip)
        if entry is None:
            return None
        expiry, score, blocked = entry
        if time.time() > expiry:
            del self._cache[ip]
            return None
        return score, blocked

    def _cache_set(self, ip: str, score: int, blocked: bool, ttl: int | None = None) -> None:
        if len(self._cache) >= self.CACHE_MAX:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[ip] = (time.time() + (ttl or self.CACHE_TTL), score, blocked)

    # ── Provider queries (unchanged) ────────────────────────────────────

    async def _query_abuseipdb(self, ip: str) -> int | None:
        if not self.ABUSEIPDB_KEY:
            return None
        try:
            r = await self._http.get(
                "https://api.abuseipdb.com/api/v2/check",
                params={"ipAddress": ip, "maxAgeInDays": "90"},
                headers={"Key": self.ABUSEIPDB_KEY, "Accept": "application/json"},
            )
            if r.status_code != 200:
                return None
            score = r.json().get("data", {}).get("abuseConfidenceScore")
            return int(score) if isinstance(score, (int, float)) else None
        except Exception:
            return None

    async def _query_scamalytics(self, ip: str) -> int | None:
        if not self.SCAMALYTICS_KEY or not self.SCAMALYTICS_USER:
            return None
        try:
            r = await self._http.get(
                f"https://api11.scamalytics.com/v3/{self.SCAMALYTICS_USER}",
                params={"key": self.SCAMALYTICS_KEY, "ip": ip},
            )
            if r.status_code != 200:
                return None
            data = r.json()
            score = data.get("score", data.get("risk"))
            if isinstance(score, (int, float)) and 0 <= score <= 100:
                return int(score)
            level = str(data.get("risk", "")).lower()
            return {"very high": 90, "high": 75, "medium": 50, "low": 15}.get(level)
        except Exception:
            return None

    async def _query_proxycheck(self, ip: str) -> int | None:
        if not self.PROXYCHECK_KEY:
            return None
        try:
            r = await self._http.get(
                f"https://proxycheck.io/v2/{ip}",
                params={"key": self.PROXYCHECK_KEY, "risk": "1", "vpn": "1"},
            )
            if r.status_code != 200:
                return None
            entry = r.json().get(ip, {})
            risk = entry.get("risk")
            return int(risk) if isinstance(risk, (int, float)) and 0 <= risk <= 100 else None
        except Exception:
            return None

    async def _get_score(self, ip: str) -> tuple[int, str] | None:
        import asyncio as _aio
        results = await _aio.gather(
            self._query_abuseipdb(ip),
            self._query_scamalytics(ip),
            self._query_proxycheck(ip),
            return_exceptions=True,
        )
        names = ("abuseipdb", "scamalytics", "proxycheck")
        best_score, best_name = -1, ""
        for score, name in zip(results, names):
            if isinstance(score, int) and score > best_score:
                best_score, best_name = score, name
        return (best_score, best_name) if best_score >= 0 else None

    # ── Dispatch: Layer 0 → 1 → 2 ──────────────────────────────────────

    async def dispatch(self, request: Request, call_next):
        ip = self._extract_ip(request)
        path = request.url.path

        # Layer 0: Whitelist
        if self._is_trusted(ip):
            return await call_next(request)

        # Check cache (covers both trap bans and reputation bans)
        cached = self._cache_get(ip)
        if cached is not None:
            _, blocked = cached
            if blocked:
                return Response(status_code=403)
            return await call_next(request)

        # Layer 1: Behavior trap — caught stealing underwear
        if _is_trap_path(path):
            self._cache_set(ip, 100, True, ttl=self.TRAP_BAN_TTL)
            logger.warning(f"[FIREWALL] TRAPPED ip={ip} path={path}")
            return Response(status_code=403)

        # Layer 2: IP reputation
        result = await self._get_score(ip)
        if result is None:
            self._cache_set(ip, 0, False, ttl=300)
            return await call_next(request)

        score, provider = result
        blocked = score >= self.BLOCK_THRESHOLD
        self._cache_set(ip, score, blocked)

        if blocked:
            logger.warning(f"[FIREWALL] BLOCKED ip={ip} score={score} ({provider})")
            return Response(status_code=403)

        return await call_next(request)

# Register firewall BEFORE CORS so blocked IPs never get CORS headers
app.add_middleware(IPFirewall)

# ── CORS (allow Astro frontend at :4321 to call us) ───────────────────
_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.CORS_ORIGINS + ["http://localhost:4321", "http://127.0.0.1:4321"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── AI Engine singleton ───────────────────────────────────────────────
_ai_engine: AIEngine | None = None

def _get_ai_engine() -> AIEngine:
    global _ai_engine
    if _ai_engine is None:
        _ai_engine = AIEngine(_settings)
    return _ai_engine


def _print_config_status():
    """Print AI provider configuration status at startup."""
    print("\n" + "=" * 60)
    print("AI Provider Configuration Status")
    print("=" * 60)
    
    providers_found = []
    providers_missing = []
    
    # Check Gemini
    if _settings.GEMINI_API_KEY:
        base = _settings.GEMINI_API_BASE or "(direct Google API)"
        providers_found.append(f"✓ Gemini: {base}")
    else:
        providers_missing.append("✗ Gemini: GEMINI_API_KEY not set")
    
    # Check OpenAI
    if _settings.OPENAI_API_KEY:
        providers_found.append(f"✓ OpenAI: {_settings.OPENAI_API_BASE}")
    else:
        providers_missing.append("✗ OpenAI: OPENAI_API_KEY not set")
    
    # Check Anthropic
    if _settings.ANTHROPIC_API_KEY:
        providers_found.append(f"✓ Anthropic: {_settings.ANTHROPIC_API_BASE}")
    else:
        providers_missing.append("✗ Anthropic: ANTHROPIC_API_KEY not set")
    
    # Check Claude Compatible
    if _settings.CLAUDE_COMPATIBLE_API_KEY and _settings.CLAUDE_COMPATIBLE_API_BASE:
        providers_found.append(f"✓ Claude-Compatible: {_settings.CLAUDE_COMPATIBLE_API_BASE}")
    
    # Print results
    for p in providers_found:
        print(p)
    for p in providers_missing:
        print(p)
    
    print("-" * 60)
    
    if not providers_found:
        print("⚠️  WARNING: No AI providers configured!")
        print("   Copy .env.example to .env and add your API keys.")
        print("   At least ONE provider (Gemini/OpenAI/Anthropic) is required.")
    else:
        print(f"✓ {len(providers_found)} provider(s) configured")
        
        # Try to initialize AI engine and report available providers
        try:
            engine = _get_ai_engine()
            print(f"✓ Available providers: {engine.available_providers}")
        except Exception as e:
            print(f"⚠️  Failed to initialize AI engine: {e}")
    
    print("=" * 60 + "\n")


# ── Animation pipeline routes ─────────────────────────────────────────
register_animation_routes(app)


# ============================================================================
# Pipeline API Endpoints (fixing 502 error)
# These endpoints are called by Astro frontend at :4321 → proxy → :8000
# ============================================================================

@app.post("/api/topology")
async def api_topology(request_data: dict) -> JSONResponse:
    """
    POST /api/topology — Text → ELK Topology JSON
    Called by: src/pages/api/topology.ts (Astro proxy)
    GitHub: ResearAI/AutoFigure, kieler/elkjs
    """
    try:
        text = request_data.get("text", "").strip()
        if not text:
            return JSONResponse(
                {"error": "text field is required"},
                status_code=400,
            )

        model = request_data.get("model")
        algorithm = request_data.get("algorithm", "layered")
        direction = request_data.get("direction", "DOWN")

        ai_engine = _get_ai_engine()

        from backend.schemas import ElkAlgorithm, ElkDirection
        result = await generate_topology(
            ai_engine=ai_engine,
            text=text,
            model=model,
            algorithm=ElkAlgorithm(algorithm) if algorithm else ElkAlgorithm.LAYERED,
            direction=ElkDirection(direction) if direction else ElkDirection.DOWN,
        )

        if result.success:
            return JSONResponse({
                "success": True,
                "topology": result.topology.model_dump(exclude_none=True) if result.topology else None,
                "model_used": result.model_used,
            })
        else:
            return JSONResponse(
                {"error": result.error or "Topology generation failed"},
                status_code=500,
            )

    except Exception as e:
        logger.exception("api_topology error")
        return JSONResponse(
            {"error": str(e), "hint": "Check GEMINI_API_KEY in .env"},
            status_code=500,
        )


@app.post("/api/beautify")
async def api_beautify(request_data: dict) -> JSONResponse:
    """
    POST /api/beautify — ELK Layouted JSON → NanoBanana SVG
    Called by: src/pages/api/beautify.ts (Astro proxy)
    GitHub: gemini-cli-extensions/nanobanana
    """
    try:
        layouted = request_data.get("layouted")
        if not layouted:
            return JSONResponse(
                {"error": "layouted field is required"},
                status_code=400,
            )

        model = request_data.get("model")
        style = request_data.get("style", "academic")

        ai_engine = _get_ai_engine()

        # Build scaffold from layouted graph
        scaffold = build_scaffold(layouted)

        # Enrich scaffold with SVG icons from Iconify (async, non-blocking)
        try:
            from backend.pipeline.scaffold_builder import enrich_scaffold_with_icons
            scaffold = await enrich_scaffold_with_icons(scaffold, layouted)
        except Exception as icon_err:
            logger.warning(f"Icon enrichment failed (non-fatal): {icon_err}")

        # Generate SVG via NanoBanana bridge
        result = await beautify_with_nanobanana(
            ai_engine=ai_engine,
            layouted=layouted,
            scaffold=scaffold,
            model=model,
            style=style,
        )

        if result.get("success"):
            return JSONResponse(result)
        else:
            return JSONResponse(
                {"error": result.get("error", "Beautify failed")},
                status_code=500,
            )

    except Exception as e:
        logger.exception("api_beautify error")
        return JSONResponse(
            {"error": str(e), "hint": "Check GEMINI_API_KEY in .env"},
            status_code=500,
        )


@app.post("/api/validate")
async def api_validate_svg(request_data: dict) -> JSONResponse:
    """
    POST /api/validate — SVG syntax validation + LLM fix
    GitHub: withastro/astro
    """
    try:
        svg = request_data.get("svg", "")
        if not svg:
            return JSONResponse(
                {"error": "svg field is required"},
                status_code=400,
            )

        auto_fix = request_data.get("auto_fix", True)
        model = request_data.get("model")

        ai_engine = _get_ai_engine() if auto_fix else None
        result = await validate_svg_func(
            svg=svg,
            ai_engine=ai_engine,
            auto_fix=auto_fix,
            model=model,
        )

        # Convert ValidateResponse to dict
        return JSONResponse(result.model_dump() if hasattr(result, 'model_dump') else result)

    except Exception as e:
        logger.exception("api_validate error")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/models")
async def api_models() -> JSONResponse:
    """
    GET /api/models — List available AI models
    """
    return JSONResponse(_settings.AVAILABLE_MODELS)


# ── Step 5: Gemini 3 Pro Image Generation ────────────────────────────────

from backend.pipeline.gemini_image_gen import (
    generate_scientific_figure,
    generate_prompt_with_grok,
    generate_image_with_gemini,
)


@app.post("/api/generate-image")
async def api_generate_image(request_data: dict) -> JSONResponse:
    """
    POST /api/generate-image — Step 5: SVG → Gemini 3 Pro Image

    Flow:
      a) Grok 4 reverse-engineers a professional prompt (optional)
      b) Gemini 3 Pro Image generates the scientific figure

    Request body:
      - svg_content (str): The ELK-generated SVG (required)
      - method_text (str): Paper method description (required)
      - reference_image_b64 (str, optional): Base64 reference image
      - prompt_model (str, optional): Model for prompt engineering (default: grok-4)
      - image_model (str, optional): Gemini image model (default: gemini-3-pro-image-preview)
      - aspect_ratio (str, optional): Image aspect ratio (default: 16:9)
      - image_size (str, optional): Image size (default: 4K)
      - custom_prompt (str, optional): Skip Grok, use this prompt directly
      - elk_graph (dict, optional): Structured graph data from interactive editor
        (nodes with x,y,width,height + edges with sourceId,targetId)

    Returns:
      - success (bool)
      - image_b64 (str): Base64-encoded image
      - mime_type (str): e.g. "image/png"
      - prompt (str): The prompt used
      - prompt_model_used (str)
      - image_model_used (str)
    """
    try:
        svg_content = request_data.get("svg_content", "")
        method_text = request_data.get("method_text", "")

        if not svg_content:
            return JSONResponse(
                {"error": "svg_content is required"}, status_code=400
            )
        if not method_text:
            return JSONResponse(
                {"error": "method_text is required"}, status_code=400
            )

        engine = _get_ai_engine()

        result = await generate_scientific_figure(
            ai_engine=engine,
            method_text=method_text,
            svg_content=svg_content,
            reference_image_b64=request_data.get("reference_image_b64"),
            prompt_model=request_data.get("prompt_model"),
            image_model=request_data.get("image_model", "gemini-3-pro-image-preview"),
            aspect_ratio=request_data.get("aspect_ratio", "16:9"),
            image_size=request_data.get("image_size", "4K"),
            custom_prompt=request_data.get("custom_prompt"),
            elk_graph=request_data.get("elk_graph"),
        )

        return JSONResponse(result)

    except Exception as e:
        logger.exception("api_generate_image error")
        return JSONResponse(
            {"success": False, "error": str(e)}, status_code=500
        )


@app.post("/api/generate-prompt")
async def api_generate_prompt(request_data: dict) -> JSONResponse:
    """
    POST /api/generate-prompt — Grok 4 prompt engineering only

    Use this to get the prompt for review before generating the image.

    Request body:
      - method_text (str): Paper method description
      - svg_content (str): The ELK-generated SVG
      - model (str, optional): Model for prompt engineering

    Returns:
      - success (bool)
      - prompt (str): The generated prompt
    """
    try:
        method_text = request_data.get("method_text", "")
        svg_content = request_data.get("svg_content", "")

        if not method_text or not svg_content:
            return JSONResponse(
                {"error": "method_text and svg_content are required"},
                status_code=400,
            )

        engine = _get_ai_engine()
        result = await generate_prompt_with_grok(
            ai_engine=engine,
            method_text=method_text,
            svg_content=svg_content,
            model=request_data.get("model"),
            reference_image_b64=request_data.get("reference_image_b64"),
            elk_graph=request_data.get("elk_graph"),
        )

        return JSONResponse(result)

    except Exception as e:
        logger.exception("api_generate_prompt error")
        return JSONResponse(
            {"success": False, "error": str(e)}, status_code=500
        )


# ============================================================================
# Post-Generation Pipeline Endpoints (Step 4+)
# ============================================================================
# These endpoints are called by the Astro proxies:
#   src/pages/api/pipeline-run.ts  → /api/pipeline-run
#   src/pages/api/removebg.ts      → /api/removebg
# The pipeline flow:
#   generate-image (Step 3 output) → removebg → layers → edges → export


@app.post("/api/removebg")
async def api_removebg(request_data: dict) -> JSONResponse:
    """
    POST /api/removebg — Background Removal (Tiered Fallback)

    Tier 1: remove-bg.io cloud (best quality, needs API key)
    Tier 2: rembg U2-Net ML (good quality, local)
    Tier 3: Chroma-key HSV (fast, green-screen only)

    Request body:
      - frames (list[str]): base64-encoded frame images
      - method (str, optional): force "removebgio", "rembg", or "chroma"
      - tolerance (int, optional): green-screen tolerance (default 60)
      - edge_blur (float, optional): edge feathering radius
      - despill (bool, optional): green-spill correction
      - api_key (str, optional): remove-bg.io key

    Returns:
      - success (bool)
      - results (list): per-frame {image_b64, method, quality_score}
      - method (str): winning tier name
      - tier (int): 1/2/3
    """
    if not _PIPELINE_AVAILABLE:
        return JSONResponse(
            {"success": False, "error": "Pipeline modules not installed (numpy/PIL required)"},
            status_code=501,
        )

    try:
        frames = request_data.get("frames", [])
        if not frames:
            return JSONResponse(
                {"error": "No frames provided"}, status_code=400
            )

        result = await handle_removebg(
            frames_b64=frames,
            api_key=request_data.get("api_key", ""),
            force_method=request_data.get("method"),
            tolerance=request_data.get("tolerance", 60),
            edge_blur=request_data.get("edge_blur", 1.0),
            despill=request_data.get("despill", True),
        )
        return JSONResponse(result)

    except Exception as e:
        logger.exception("api_removebg error")
        return JSONResponse(
            {"success": False, "error": str(e)}, status_code=500
        )


@app.get("/api/removebg/status")
def api_removebg_status() -> JSONResponse:
    """GET /api/removebg/status — Check available removal methods."""
    if not _PIPELINE_AVAILABLE:
        return JSONResponse({"available": False, "methods": []})

    try:
        status = get_removebg_status()
        return JSONResponse(status)
    except Exception:
        return JSONResponse({"available": False, "methods": []})


# ── Region Layout Processing (Path 2: layout-guided component extraction) ──

@app.post("/api/region-layout")
async def api_region_layout(request_data: dict) -> JSONResponse:
    """
    POST /api/region-layout — Layout-Guided Region Extraction (Path 2)

    Uses structured layout data (mastergo_all_layoutobj.txt format) to
    extract, crop, and optionally remove backgrounds from UI components
    based on precise bounding box coordinates.

    Request body:
      - layout_data (list, required): Array of {id, name, bbox: {x,y,width,height}}
      - image_b64 (str, optional): Base64 source screenshot
      - remove_bg (bool, optional): Run background removal per region (default: true)
      - api_key (str, optional): remove-bg.io API key
      - config (dict, optional): Processing configuration overrides

    Response:
      - success, regions (list), tree (hierarchy), stats, artboard
    """
    try:
        from backend.pipeline.region_layout_processor import handle_region_layout
        result = await handle_region_layout(request_data)
        return JSONResponse(result)
    except ImportError as ie:
        logger.warning("region_layout_processor not available: %s", ie)
        return JSONResponse(
            {"success": False, "error": f"Region layout module not available: {ie}"},
            status_code=501,
        )
    except Exception as e:
        logger.exception("api_region_layout error")
        return JSONResponse(
            {"success": False, "error": str(e)},
            status_code=500,
        )


@app.post("/api/region-layout/prompts")
async def api_region_layout_prompts(request_data: dict) -> JSONResponse:
    """
    POST /api/region-layout/prompts — Generate Per-Region Image Prompts

    For users without a source screenshot: parses layout data and generates
    per-region Gemini/DALL-E prompts based on component names, sizes, and
    positions. Enables "Path 2b": layout → prompts → per-region generation.

    Request body:
      - layout_data (list, required): mastergo-format layout objects
      - artboard_width (int, optional): artboard width (default: 1024)
      - artboard_height (int, optional): artboard height (default: 600)
      - style_context (str, optional): style description for prompts
    """
    try:
        from backend.pipeline.region_layout_processor import (
            parse_layout_objects,
            build_region_tree,
            generate_region_prompts,
            RegionConfig,
        )

        layout_data = request_data.get("layout_data")
        if not layout_data:
            return JSONResponse({"success": False, "error": "layout_data is required"})

        objects = parse_layout_objects(layout_data)
        if not objects:
            return JSONResponse({"success": False, "error": "No valid layout objects"})

        config = RegionConfig()
        objects, by_id = build_region_tree(objects, config)

        prompts = generate_region_prompts(
            objects,
            artboard_width=float(request_data.get("artboard_width", 1024)),
            artboard_height=float(request_data.get("artboard_height", 600)),
            style_context=request_data.get("style_context", ""),
        )

        return JSONResponse({
            "success": True,
            "prompts": prompts,
            "total_regions": len(prompts),
        })

    except Exception as e:
        logger.exception("api_region_layout_prompts error")
        return JSONResponse(
            {"success": False, "error": str(e)},
            status_code=500,
        )


# NOTE: /api/pipeline-run is registered by server_animation_routes.py
# (expects frames_b64 + optional elk_graph). Do NOT re-register here.


# ── Vision-LLM UI Detection (screenshot → mastergo-format layout) ──

@app.post("/api/omniparser-detect")
async def api_omniparser_detect(request_data: dict) -> JSONResponse:
    """
    POST /api/omniparser-detect — Screenshot → Mastergo-Format Layout

    Uses Gemini/Claude/GPT-4o vision API to detect UI elements.
    Returns mastergo-format [{id, name, bbox:{x,y,width,height}}].

    Request: {image_b64: str, config?: {grid_snap, min_element_area, ...}}
    """
    try:
        from backend.pipeline.omniparser_bridge import handle_omniparser_detect
        return JSONResponse(await handle_omniparser_detect(request_data))
    except Exception as e:
        logger.exception("api_omniparser_detect error")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/api/omniparser-detect/status")
def api_omniparser_status() -> JSONResponse:
    """Check vision detection availability (needs API key)."""
    try:
        from backend.pipeline.omniparser_bridge import is_omniparser_available
        return JSONResponse(is_omniparser_available())
    except Exception:
        return JSONResponse({"available": False})


JOBS: dict[str, Job] = {}


@app.get("/api/config")
def get_config() -> JSONResponse:
    available, rel_path = _resolve_svg_edit_path()
    return JSONResponse({"svgEditAvailable": available, "svgEditPath": rel_path})


@app.get("/api/health")
def health_check() -> JSONResponse:
    """
    GET /api/health — Backend health check
    Returns server status, available providers, and model list.
    """
    providers = _get_ai_engine().available_providers if _ai_engine else []
    return JSONResponse({
        "status": "ok",
        "providers": providers,
        "python": PYTHON_EXECUTABLE,
        "timestamp": datetime.now().isoformat(),
    })


@app.post("/api/run")
def run_job(req: RunRequest) -> JSONResponse:
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    output_dir = OUTPUTS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON_EXECUTABLE,
        str(BASE_DIR / "autofigure2.py"),
        "--method_text",
        req.method_text,
        "--output_dir",
        str(output_dir),
        "--provider",
        req.provider,
    ]

    if req.api_key:
        cmd += ["--api_key", req.api_key]
    if req.base_url:
        cmd += ["--base_url", req.base_url]
    if req.image_model:
        cmd += ["--image_model", req.image_model]
    if req.svg_model:
        cmd += ["--svg_model", req.svg_model]

    sam_prompt = req.sam_prompt or DEFAULT_SAM_PROMPT
    placeholder_mode = req.placeholder_mode or DEFAULT_PLACEHOLDER_MODE
    merge_threshold = (
        req.merge_threshold if req.merge_threshold is not None else DEFAULT_MERGE_THRESHOLD
    )

    cmd += ["--sam_prompt", sam_prompt]
    cmd += ["--placeholder_mode", placeholder_mode]
    cmd += ["--merge_threshold", str(merge_threshold)]
    if req.sam_backend:
        cmd += ["--sam_backend", req.sam_backend]
    if req.sam_api_key:
        cmd += ["--sam_api_key", req.sam_api_key]
    if req.sam_max_masks is not None:
        cmd += ["--sam_max_masks", str(req.sam_max_masks)]
    if req.optimize_iterations is not None:
        cmd += ["--optimize_iterations", str(req.optimize_iterations)]

    reference_path = req.reference_image_path
    if reference_path:
        reference_path = (
            str((BASE_DIR / reference_path).resolve())
            if not Path(reference_path).is_absolute()
            else reference_path
        )
        cmd += ["--reference_image_path", reference_path]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    log_path = output_dir / "run.log"
    log_path.write_text(
        f"[meta] python={PYTHON_EXECUTABLE}\n[meta] cmd={' '.join(cmd)}\n",
        encoding="utf-8",
    )

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
        cwd=str(BASE_DIR),
    )

    job = Job(
        job_id=job_id,
        output_dir=output_dir,
        process=process,
        queue=queue.Queue(),
        log_path=log_path,
    )
    JOBS[job_id] = job

    monitor_thread = threading.Thread(target=_monitor_job, args=(job,), daemon=True)
    monitor_thread.start()

    return JSONResponse({"job_id": job_id})


@app.post("/api/upload")
async def upload_reference(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    ext = Path(file.filename).suffix.lower()
    if ext not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}:
        ext = ".png"

    data = await file.read()
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")

    name = f"{uuid.uuid4().hex}{ext}"
    out_path = UPLOADS_DIR / name
    out_path.write_bytes(data)

    rel_path = out_path.relative_to(BASE_DIR).as_posix()
    return JSONResponse(
        {"path": rel_path, "url": f"/api/uploads/{name}", "name": file.filename}
    )


@app.get("/api/events/{job_id}")
def stream_events(job_id: str) -> StreamingResponse:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    def event_stream():
        while True:
            try:
                item = job.queue.get(timeout=1.0)
            except queue.Empty:
                if job.done:
                    break
                continue
            if item.get("event") == "close":
                break
            yield _format_sse(item["event"], item["data"])

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/artifacts/{job_id}/{path:path}")
def get_artifact(job_id: str, path: str) -> FileResponse:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    candidate = (job.output_dir / path).resolve()
    if not str(candidate).startswith(str(job.output_dir.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(candidate)


@app.get("/api/uploads/{filename}")
def get_upload(filename: str) -> FileResponse:
    candidate = (UPLOADS_DIR / filename).resolve()
    if not str(candidate).startswith(str(UPLOADS_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(candidate)


def _format_sse(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=True)
    return f"event: {event}\ndata: {payload}\n\n"


def _monitor_job(job: Job) -> None:
    job.push("status", {"state": "started"})

    stdout_thread = threading.Thread(
        target=_pipe_output, args=(job, job.process.stdout, "stdout"), daemon=True
    )
    stderr_thread = threading.Thread(
        target=_pipe_output, args=(job, job.process.stderr, "stderr"), daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()

    idle_cycles = 0
    while True:
        _scan_artifacts(job)

        if job.process.poll() is not None:
            idle_cycles += 1
        else:
            idle_cycles = 0

        if idle_cycles >= 4:
            break
        time.sleep(0.5)

    _scan_artifacts(job)
    job.push("status", {"state": "finished", "code": job.process.returncode})
    job.push(
        "artifact",
        {
            "kind": "log",
            "name": job.log_path.name,
            "path": job.log_path.relative_to(job.output_dir).as_posix(),
            "url": f"/api/artifacts/{job.job_id}/{job.log_path.name}",
        },
    )
    job.done = True
    job.push("close", {})


def _pipe_output(job: Job, pipe, stream_name: str) -> None:
    if pipe is None:
        return
    for line in iter(pipe.readline, ""):
        text = line.rstrip()
        if text:
            job.write_log(stream_name, text)
            job.push("log", {"stream": stream_name, "line": text})
    pipe.close()


def _scan_artifacts(job: Job) -> None:
    output_dir = job.output_dir
    candidates = [
        output_dir / "figure.png",
        output_dir / "samed.png",
        output_dir / "template.svg",
        output_dir / "final.svg",
    ]

    icons_dir = output_dir / "icons"
    if icons_dir.is_dir():
        candidates.extend(icons_dir.glob("icon_*.png"))

    for path in candidates:
        if not path.is_file():
            continue
        rel_path = path.relative_to(output_dir).as_posix()
        if rel_path in job.seen:
            continue
        job.seen.add(rel_path)

        kind = _classify_artifact(rel_path)
        job.push(
            "artifact",
            {
                "kind": kind,
                "name": path.name,
                "path": rel_path,
                "url": f"/api/artifacts/{job.job_id}/{rel_path}",
            },
        )


def _classify_artifact(rel_path: str) -> str:
    if rel_path == "figure.png":
        return "figure"
    if rel_path == "samed.png":
        return "samed"
    if rel_path.endswith("_nobg.png"):
        return "icon_nobg"
    if rel_path.startswith("icons/") and rel_path.endswith(".png"):
        return "icon_raw"
    if rel_path == "template.svg":
        return "template_svg"
    if rel_path == "final.svg":
        return "final_svg"
    return "artifact"


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            return True
    return False


def _pids_on_port(port: int) -> set[int]:
    pids: set[int] = set()

    if shutil.which("lsof"):
        result = subprocess.run(
            ["lsof", "-t", f"-i:{port}"],
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.isdigit():
                pids.add(int(line))
        return pids

    if shutil.which("ss"):
        result = subprocess.run(
            ["ss", "-lptn", f"sport = :{port}"],
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            if "pid=" in line:
                for part in line.split("pid=")[1:]:
                    pid_str = "".join(ch for ch in part if ch.isdigit())
                    if pid_str:
                        pids.add(int(pid_str))
        return pids

    if shutil.which("netstat"):
        result = subprocess.run(
            ["netstat", "-tlnp"],
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            if f":{port} " not in line or "LISTEN" not in line:
                continue
            fields = line.split()
            if fields and "/" in fields[-1]:
                pid_part = fields[-1].split("/")[0]
                if pid_part.isdigit():
                    pids.add(int(pid_part))

    return pids


def _read_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            data = handle.read()
        parts = [p for p in data.split(b"\x00") if p]
        return " ".join(part.decode(errors="ignore") for part in parts)
    except OSError:
        return ""


def _is_uvicorn_process(pid: int) -> bool:
    cmdline = _read_cmdline(pid)
    if not cmdline:
        return False
    if "uvicorn" not in cmdline:
        return False
    return "server:app" in cmdline or "server.py" in cmdline


def _terminate_pids(pids: set[int], timeout: float = 2.0) -> None:
    current_pid = os.getpid()
    for pid in sorted(pids):
        if pid <= 1 or pid == current_pid:
            continue
        if not _is_uvicorn_process(pid):
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue

    deadline = time.time() + timeout
    while time.time() < deadline:
        alive = False
        for pid in pids:
            if pid <= 1 or pid == current_pid:
                continue
            if not _is_uvicorn_process(pid):
                continue
            try:
                os.kill(pid, 0)
                alive = True
            except ProcessLookupError:
                continue
        if not alive:
            return
        time.sleep(0.1)

    for pid in sorted(pids):
        if pid <= 1 or pid == current_pid:
            continue
        if not _is_uvicorn_process(pid):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue


def _ensure_port_free(port: int) -> None:
    if not _port_in_use(port):
        return
    pids = _pids_on_port(port)
    if not pids:
        return
    _terminate_pids(pids)


# Mount static files only if web/ directory exists (created by `bun run build`)
# During development, Astro dev server (:4321) serves the frontend directly
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static")
else:
    logger.info(
        "web/ directory not found — skipping static file mount. "
        "Run 'bun run build' to generate it, or use 'bun run dev' for Astro dev server."
    )


if __name__ == "__main__":
    import uvicorn

    def find_available_port(start_port: int, max_attempts: int = 100) -> int:
        for port in range(start_port, start_port + max_attempts):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                try:
                    sock.bind(("0.0.0.0", port))
                    return port
                except OSError:
                    print(f"Port {port} is in use, trying next...")
                    continue
        raise IOError(f"No available ports found in range ({start_port} - {start_port + max_attempts})")

    initial_port = 8000
    
    try:
        # Print configuration status before starting
        _print_config_status()
        
        actual_port = find_available_port(initial_port)
        
        print(f"--- Starting Server ---")
        print(f"Local access: http://127.0.0.1:{actual_port}")
        print(f"-----------------------")

        uvicorn.run(
            "server:app",
            host="0.0.0.0",
            port=actual_port,
            reload=False,
            access_log=False,
        )
    except Exception as e:
        print(f"Startup failed: {e}")
        sys.exit(1)