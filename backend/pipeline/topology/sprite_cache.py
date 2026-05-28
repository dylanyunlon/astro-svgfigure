"""sprite_cache.py — Content-addressed sprite cache (M220).

openai/triton's compiler.py keys its compiled-artifact cache on
ASTSource.hash() — a SHA256 folding EVERY input that affects the output
(fn.cache_key, attrs, sorted_sig, constants_key).  Same key ⇒ same artifact ⇒
zero recompilation.  We apply exactly that to sprites.

M211's SpritePrompt already exposes `cache_key`: a SHA256 over (prompt,
negative, target_size) — the full set of inputs that determine the pixels.
Symbols like ⊗, "feature map", a "ViT encoder block" recur across many paper
figures, so a content-addressed cache turns the second-and-later occurrences
into instant hits and removes them from the generation set entirely.

Design (mirrors triton's on-disk cache + an in-memory LRU on top):
  - persistent layer:   .sprite_cache/<key>.json on disk, survives restarts.
  - hot layer:          an in-process OrderedDict LRU over the same keys.
  - versioning:         a CACHE_VERSION tag is folded into the stored key, so
                        bumping the prompt designer invalidates stale entries
                        without a manual purge (triton does the same with its
                        compiler version).
  - family coherence:   put_family/get_family store a whole family atomically,
                        so a partially-regenerated family never mixes a cached
                        old variant with a fresh one.

A cached entry stores the finished SpriteAsset payload (image_b64, true_bbox,
format, family_id) — i.e. the post-removal, post-vectorize result — so a hit
skips generation AND background removal AND vectorization.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Bump when the sprite *pixels* could change for the same prompt (e.g. the
# prompt designer's style suffix changes, or the vectorizer's output format).
# Folded into every stored key so old entries are transparently ignored.
CACHE_VERSION = "v1"

DEFAULT_CACHE_DIR = os.environ.get(
    "SPRITE_CACHE_DIR", str(Path.home() / ".sprite_cache")
)
DEFAULT_LRU_CAP = 512


# ═══════════════════════════════════════════════════════════════════════════
#  §1  Stats
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    puts: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits, "misses": self.misses, "puts": self.puts,
            "evictions": self.evictions, "hit_rate": round(self.hit_rate, 4),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  §2  The cache
# ═══════════════════════════════════════════════════════════════════════════

class SpriteCache:
    """Two-tier (disk + in-memory LRU) content-addressed sprite store.

    Keys are derived from a SpritePrompt.cache_key plus CACHE_VERSION. Values
    are SpriteAsset-shaped dicts (the finished, background-free, possibly
    vectorized payload).

    Thread-safe for the in-memory tier (a lock guards the OrderedDict); disk
    writes are atomic (temp file + os.replace) so a crash mid-write never
    leaves a corrupt entry — a partial read simply misses and regenerates.
    """

    def __init__(
        self,
        cache_dir: str = DEFAULT_CACHE_DIR,
        lru_capacity: int = DEFAULT_LRU_CAP,
        enable_disk: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.lru_capacity = lru_capacity
        self.enable_disk = enable_disk
        self._lru: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self.stats = CacheStats()
        if self.enable_disk:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("sprite cache dir unavailable (%s); disk off", e)
                self.enable_disk = False

    # ── key derivation ──
    @staticmethod
    def key_for(cache_key: str) -> str:
        """Fold the version tag into the prompt's content key."""
        return f"{CACHE_VERSION}_{cache_key}"

    def _disk_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    # ── get / put for a single sprite ──
    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Return a cached SpriteAsset dict for this content key, or None."""
        key = self.key_for(cache_key)
        with self._lock:
            if key in self._lru:
                self._lru.move_to_end(key)
                self.stats.hits += 1
                return dict(self._lru[key])
        # disk tier
        if self.enable_disk:
            p = self._disk_path(key)
            try:
                if p.exists():
                    data = json.loads(p.read_text(encoding="utf-8"))
                    self._lru_put(key, data)
                    with self._lock:
                        self.stats.hits += 1
                    return dict(data)
            except Exception as e:  # pragma: no cover - corrupt entry → miss
                logger.debug("cache read failed for %s: %s", key, e)
        with self._lock:
            self.stats.misses += 1
        return None

    def put(self, cache_key: str, asset_dict: Dict[str, Any]) -> None:
        """Store a finished SpriteAsset dict under this content key."""
        key = self.key_for(cache_key)
        self._lru_put(key, asset_dict)
        with self._lock:
            self.stats.puts += 1
        if self.enable_disk:
            p = self._disk_path(key)
            try:
                tmp = p.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(asset_dict), encoding="utf-8")
                os.replace(tmp, p)  # atomic
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("cache write failed for %s: %s", key, e)

    def _lru_put(self, key: str, value: Dict[str, Any]) -> None:
        with self._lock:
            if key in self._lru:
                self._lru.move_to_end(key)
            self._lru[key] = dict(value)
            while len(self._lru) > self.lru_capacity:
                self._lru.popitem(last=False)
                self.stats.evictions += 1

    # ── family-atomic helpers ──
    def get_family(self, cache_keys: List[str]) -> Optional[List[Dict[str, Any]]]:
        """All-or-nothing family fetch: returns the list only if EVERY member
        is cached, else None (so we never mix cached + fresh members)."""
        out: List[Dict[str, Any]] = []
        for ck in cache_keys:
            got = self.get(ck)
            if got is None:
                return None
            out.append(got)
        return out

    def put_family(self, items: List[tuple]) -> None:
        """Store a whole family: items is a list of (cache_key, asset_dict)."""
        for ck, asset in items:
            self.put(ck, asset)

    def clear_memory(self) -> None:
        with self._lock:
            self._lru.clear()


# ═══════════════════════════════════════════════════════════════════════════
#  §3  Asset ↔ dict bridge (kept here so callers needn't import the splitter)
# ═══════════════════════════════════════════════════════════════════════════

def asset_to_cache_dict(asset: Any) -> Dict[str, Any]:
    """Serialize a SpriteAsset to a cache-storable dict (only the pixel-bearing
    fields; quality/issues are recomputed-on-demand and not cached)."""
    return {
        "node_id": getattr(asset, "node_id", ""),
        "image_b64": getattr(asset, "image_b64", None),
        "true_bbox": list(getattr(asset, "true_bbox", (0, 0, 0, 0))),
        "format": getattr(asset, "format", "png"),
        "family_id": getattr(asset, "family_id", ""),
    }


def cache_dict_to_asset(d: Dict[str, Any], node_id: str, asset_cls: Any) -> Any:
    """Rebuild a SpriteAsset from a cached dict, re-targeted to `node_id`.

    The cached pixels are concept-keyed, so a hit for one node's ⊗ is reused
    for another node's ⊗ — we just stamp the requesting node_id.
    """
    bb = tuple(d.get("true_bbox", (0, 0, 0, 0)))
    return asset_cls(
        node_id=node_id,
        image_b64=d.get("image_b64"),
        true_bbox=bb,  # type: ignore[arg-type]
        format=d.get("format", "png"),
        dropped=d.get("image_b64") is None,
        family_id=d.get("family_id", ""),
    )


# Process-wide default instance (callers may inject their own).
_DEFAULT_CACHE: Optional[SpriteCache] = None


def get_default_cache() -> SpriteCache:
    global _DEFAULT_CACHE
    if _DEFAULT_CACHE is None:
        _DEFAULT_CACHE = SpriteCache()
    return _DEFAULT_CACHE
