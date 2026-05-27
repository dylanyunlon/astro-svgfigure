"""
removebg_canva_client.py — remove.bg (Canva) API with key pool rotation.

remove.bg (api.remove.bg) 是 Canva 旗下的去背景服务:
  - 每个 key 50 次免费/月
  - 认证方式: X-Api-Key header
  - 输入: multipart/form-data (image_file 或 image_url)
  - 输出: 透明 PNG (raw binary)

通过纹图平台批量获取 key，key 用完 (HTTP 402) 自动轮换到下一个。

Design: key pool 轮换 + 402 自动降级，类似 NCCL 多路径 fallback。
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

REMOVEBG_API_URL = "https://api.remove.bg/v1.0/removebg"


@dataclass
class RemoveBgCanvaResult:
    success: bool
    image_b64: Optional[str] = None
    width: int = 0
    height: int = 0
    processing_time_ms: int = 0
    key_used: str = ""
    error: Optional[str] = None


class RemoveBgCanvaKeyPool:
    """Key pool with automatic rotation on 402 (insufficient credits).

    Keys are tried in order. When a key returns 402, it's marked
    exhausted and the next key is tried. Thread-safe via asyncio.Lock.
    """

    def __init__(self, keys: List[str]):
        self._keys = list(keys)
        self._exhausted: set = set()
        self._current_idx = 0
        self._lock = asyncio.Lock()

    @property
    def available_count(self) -> int:
        return len(self._keys) - len(self._exhausted)

    async def get_key(self) -> Optional[str]:
        async with self._lock:
            for _ in range(len(self._keys)):
                key = self._keys[self._current_idx]
                if key not in self._exhausted:
                    return key
                self._current_idx = (self._current_idx + 1) % len(self._keys)
            return None

    async def mark_exhausted(self, key: str) -> Optional[str]:
        """Mark key as exhausted, return next available key or None."""
        async with self._lock:
            self._exhausted.add(key)
            logger.info(
                "Key ...%s exhausted, %d/%d remaining",
                key[-4:], self.available_count, len(self._keys),
            )
            self._current_idx = (self._current_idx + 1) % len(self._keys)
            for _ in range(len(self._keys)):
                k = self._keys[self._current_idx]
                if k not in self._exhausted:
                    return k
                self._current_idx = (self._current_idx + 1) % len(self._keys)
            return None

    def status(self) -> Dict[str, Any]:
        return {
            "total_keys": len(self._keys),
            "available": self.available_count,
            "exhausted": len(self._exhausted),
        }


class RemoveBgCanvaClient:
    """remove.bg (Canva) API client with key pool rotation.

    Usage:
        client = RemoveBgCanvaClient(keys=["key1", "key2", "key3"])
        result = await client.remove_background(image_bytes)

        # Or from base64
        result = await client.remove_background_b64(b64_string)
    """

    def __init__(
        self,
        keys: Optional[List[str]] = None,
        timeout_s: float = 30,
    ):
        if not keys:
            import os
            env_keys = os.environ.get("REMOVEBG_API_KEYS", "")
            keys = [k.strip() for k in env_keys.split(",") if k.strip()]
        self.pool = RemoveBgCanvaKeyPool(keys) if keys else None
        self.timeout_s = timeout_s

    async def remove_background(
        self,
        image_data: bytes,
        size: str = "auto",
    ) -> RemoveBgCanvaResult:
        if not _HAS_HTTPX:
            return RemoveBgCanvaResult(
                success=False, error="httpx not installed",
            )
        if not self.pool or self.pool.available_count == 0:
            return RemoveBgCanvaResult(
                success=False, error="No remove.bg API keys available",
            )

        t0 = time.monotonic()
        key = await self.pool.get_key()

        while key:
            try:
                async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                    resp = await client.post(
                        REMOVEBG_API_URL,
                        headers={"X-Api-Key": key},
                        files={"image_file": ("image.png", image_data, "image/png")},
                        data={"size": size},
                    )

                if resp.status_code == 200:
                    result_b64 = base64.b64encode(resp.content).decode("ascii")
                    elapsed = int((time.monotonic() - t0) * 1000)

                    w, h = 0, 0
                    try:
                        from PIL import Image
                        img = Image.open(io.BytesIO(resp.content))
                        w, h = img.size
                    except Exception:
                        pass

                    return RemoveBgCanvaResult(
                        success=True,
                        image_b64=result_b64,
                        width=w, height=h,
                        processing_time_ms=elapsed,
                        key_used=f"...{key[-4:]}",
                    )

                elif resp.status_code == 402:
                    # Insufficient credits — rotate to next key
                    key = await self.pool.mark_exhausted(key)
                    continue

                else:
                    error_msg = resp.text[:200]
                    logger.warning("remove.bg error %d: %s", resp.status_code, error_msg)
                    return RemoveBgCanvaResult(
                        success=False,
                        error=f"HTTP {resp.status_code}: {error_msg}",
                        processing_time_ms=int((time.monotonic() - t0) * 1000),
                    )

            except Exception as e:
                logger.warning("remove.bg request failed: %s", e)
                return RemoveBgCanvaResult(
                    success=False,
                    error=str(e),
                    processing_time_ms=int((time.monotonic() - t0) * 1000),
                )

        return RemoveBgCanvaResult(
            success=False,
            error="All remove.bg keys exhausted (402 on every key)",
            processing_time_ms=int((time.monotonic() - t0) * 1000),
        )

    async def remove_background_b64(self, image_b64: str) -> RemoveBgCanvaResult:
        raw = image_b64
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[1]
        image_data = base64.b64decode(raw)
        return await self.remove_background(image_data)

    async def remove_background_batch(
        self,
        frames_b64: List[str],
        concurrency: int = 3,
    ) -> List[RemoveBgCanvaResult]:
        sem = asyncio.Semaphore(concurrency)

        async def _one(b64: str) -> RemoveBgCanvaResult:
            async with sem:
                return await self.remove_background_b64(b64)

        return await asyncio.gather(*[_one(b) for b in frames_b64])
