"""
Backend Configuration
=====================
Pydantic-settings based configuration, modeled after skynetCheapBuy/app/config.py.
Loads from environment variables and .env file.

GitHub reference: dylanyunlon/skynetCheapBuy → app/config.py
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Resolve project root (one level up from backend/)
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BACKEND_DIR.parent


class Settings(BaseSettings):
    """
    Central settings object.  Values are read from env vars / .env.
    Naming convention follows skynetCheapBuy for cross-project consistency.
    """

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── OpenAI-compatible provider ──────────────────────────────────────
    OPENAI_API_KEY: str = ""
    OPENAI_API_BASE: str = "https://api.openai.com/v1"
    OPENAI_DEFAULT_MODEL: str = "gpt-4o"

    # ── Anthropic provider ──────────────────────────────────────────────
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_API_BASE: str = "https://api.anthropic.com"
    ANTHROPIC_DEFAULT_MODEL: str = "claude-sonnet-4-20250514"

    # ── Google Gemini provider ──────────────────────────────────────────
    GEMINI_API_KEY: str = ""
    GEMINI_API_BASE: str = ""  # If set, use OpenAI-compatible format via proxy (e.g. tryallai)
    GEMINI_DEFAULT_MODEL: str = "gemini-2.5-flash"

    # ── Claude-Compatible (third-party /v1/messages endpoint) ───────────
    CLAUDE_COMPATIBLE_API_KEY: str = ""
    CLAUDE_COMPATIBLE_API_BASE: str = ""

    # ── Pipeline defaults ───────────────────────────────────────────────
    # 3-Step Pipeline:
    #   Step 1: Topology + ELK Layout (DEFAULT_TOPOLOGY_MODEL)
    #   Step 2: Grok 4 prompt engineering (DEFAULT_PROMPT_MODEL)
    #   Step 3: Gemini 3 image generation (DEFAULT_IMAGE_MODEL)
    DEFAULT_AI_MODEL: str = "claude-opus-4-6"
    DEFAULT_TOPOLOGY_MODEL: str = "claude-opus-4-6"
    DEFAULT_BEAUTIFY_MODEL: str = "grok-4"         # Legacy, kept for backward compat
    DEFAULT_VALIDATOR_MODEL: str = "claude-opus-4-6"

    # ── Step 2+3: Image Generation ───────────────────────────────────────
    DEFAULT_PROMPT_MODEL: str = "grok-4"                    # Grok 4 反推 prompt
    DEFAULT_IMAGE_MODEL: str = "gemini-3-pro-image-preview" # Gemini 3 生成图片

    # ── Server ──────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    CORS_ORIGINS: List[str] = ["http://localhost:4321", "http://localhost:3000"]

    # ── Paths ───────────────────────────────────────────────────────────
    PROJECT_ROOT: Path = _PROJECT_ROOT
    OUTPUTS_DIR: Path = _PROJECT_ROOT / "outputs"
    UPLOADS_DIR: Path = _PROJECT_ROOT / "uploads"

    # ── Limits ──────────────────────────────────────────────────────────
    MAX_TOPOLOGY_TOKENS: int = 8192
    MAX_BEAUTIFY_TOKENS: int = 16384
    MAX_VALIDATE_TOKENS: int = 8192
    SVG_VALIDATOR_MAX_RETRIES: int = 3

    # ── Available models (exposed via /api/models) ──────────────────────
    @property
    def AVAILABLE_MODELS(self) -> Dict[str, List[Dict[str, str]]]:
        """Return models grouped by provider, for frontend model selector."""
        models: Dict[str, List[Dict[str, str]]] = {}

        if self.GEMINI_API_KEY:
            models["gemini"] = [
                {"id": "gemini-3-pro-image-preview", "name": "Gemini 3 Pro Image (figure gen)"},
                {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro (strongest)"},
                {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash (fast)"},
                {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash"},
            ]

        if self.OPENAI_API_KEY:
            models["openai"] = [
                {"id": "grok-4", "name": "Grok 4 (prompt engineering)"},
                {"id": "gpt-4o", "name": "GPT-4o"},
                {"id": "gpt-4o-mini", "name": "GPT-4o Mini"},
                {"id": "o3-mini", "name": "o3-mini"},
            ]

        if self.ANTHROPIC_API_KEY:
            models["anthropic"] = [
                {"id": "claude-sonnet-4-20250514", "name": "Claude Sonnet 4"},
                {"id": "claude-haiku-4-5-20251001", "name": "Claude Haiku 4.5"},
            ]

        if self.CLAUDE_COMPATIBLE_API_KEY:
            models["claude_compatible"] = [
                {"id": "claude-sonnet-4-20250514", "name": "Claude Sonnet 4 (Compatible)"},
            ]

        return models

    def get_api_key_for_model(self, model: str) -> str:
        """Return the appropriate API key for a given model name."""
        m = model.lower()
        if m.startswith("gemini"):
            return self.GEMINI_API_KEY
        if m.startswith(("claude-", "claude_")):
            return self.ANTHROPIC_API_KEY or self.CLAUDE_COMPATIBLE_API_KEY
        if m.startswith(("gpt-", "o1-", "o3-")):
            return self.OPENAI_API_KEY
        # Default: try OpenAI-compatible
        return self.OPENAI_API_KEY

    def get_api_base_for_model(self, model: str) -> str:
        """Return the appropriate API base URL for a given model name."""
        m = model.lower()
        if m.startswith("gemini"):
            return self.GEMINI_API_BASE  # Empty = use SDK; set = use OpenAI-compatible proxy
        if m.startswith(("claude-", "claude_")):
            if self.ANTHROPIC_API_KEY:
                return self.ANTHROPIC_API_BASE
            return self.CLAUDE_COMPATIBLE_API_BASE
        return self.OPENAI_API_BASE


@lru_cache()
def get_settings() -> Settings:
    """Singleton settings instance (cached)."""
    return Settings()