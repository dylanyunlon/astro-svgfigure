"""
backend/pipeline/svg_icon_fetcher.py — Semantic SVG Icon Acquisition

Converts topology node labels / iconHints into production-quality SVG
icons by querying the Iconify API (275k+ open-source icons, MIT/CC0,
no API key, free unlimited usage).

Architecture:
    node.iconHint ("neural network") or node.label ("Database")
      → Iconify search API → ranked icon candidates
      → Iconify SVG endpoint → raw SVG string
      → embedded into ScaffoldElement.icon_svg

Why Iconify over alternatives:
  - 275k+ icons across 200+ collections (Tabler, Lucide, Phosphor…)
  - All MIT / ISC / Apache-2.0 / CC0 — safe for commercial use
  - No API key, no rate-limit registration, no billing
  - Returns clean SVG with `currentColor` fill (themeable)
  - Sub-100ms response times from CDN edge servers

Fail-safe: If Iconify is unreachable or returns no results, the node
renders without an icon (graceful degradation, never blocks the pipeline).

File location: backend/pipeline/svg_icon_fetcher.py
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────

ICONIFY_SEARCH_URL = "https://api.iconify.design/search"
ICONIFY_SVG_URL = "https://api.iconify.design"  # /{prefix}/{name}.svg

# Icon size for embedding (viewBox preserved, width/height set)
ICON_SIZE = 48

# Max time to wait for Iconify per icon (ms)
TIMEOUT_MS = 3000

# Preferred icon collections (higher quality, consistent style)
# Ordered by preference — first match from a preferred set wins
PREFERRED_COLLECTIONS = [
    "tabler",          # 5500+ outline icons, MIT
    "lucide",          # 1000+ minimal icons, ISC
    "phosphor",        # 1200+ with weight variants, MIT
    "carbon",          # IBM Carbon, 2800+ icons, Apache-2.0
    "eos-icons",       # EOS icons, MIT (good for tech/science)
    "heroicons",       # Tailwind team, 292 icons, MIT
    "material-symbols", # Google Material, Apache-2.0
    "mdi",             # Material Design Icons, Apache-2.0
    "fluent",          # Microsoft Fluent, MIT
    "iconoir",         # 1500+ geometric icons, MIT
]

# Mapping from common academic/ML terms to better search queries.
# Loaded from shared/icon-aliases.json — the SINGLE SOURCE OF TRUTH
# also consumed by src/lib/elk/to-svg-icons.ts on the frontend.
# To add a new alias, edit shared/icon-aliases.json only.
import json as _json
from pathlib import Path as _Path

def _load_aliases() -> Dict[str, str]:
    """Load icon aliases from shared JSON. Falls back to empty dict."""
    # Resolve path relative to this file: backend/pipeline/ → ../../shared/
    _candidates = [
        _Path(__file__).resolve().parent.parent.parent / "shared" / "icon-aliases.json",
        _Path.cwd() / "shared" / "icon-aliases.json",
    ]
    for p in _candidates:
        if p.is_file():
            try:
                data = _json.loads(p.read_text(encoding="utf-8"))
                aliases = data.get("aliases", {})
                logger.debug(f"Loaded {len(aliases)} icon aliases from {p}")
                return aliases
            except Exception as e:
                logger.warning(f"Failed to load icon aliases from {p}: {e}")
    logger.warning("shared/icon-aliases.json not found, using empty aliases")
    return {}

TERM_ALIASES: Dict[str, str] = _load_aliases()


@dataclass
class IconResult:
    """Result of an icon search + fetch."""
    query: str
    icon_id: str                 # e.g. "tabler:brain"
    svg: str                     # Raw SVG string
    collection: str              # e.g. "tabler"
    license: str                 # e.g. "MIT"

    @property
    def prefix(self) -> str:
        return self.icon_id.split(":")[0] if ":" in self.icon_id else ""

    @property
    def name(self) -> str:
        return self.icon_id.split(":")[-1] if ":" in self.icon_id else self.icon_id


@dataclass
class IconCache:
    """Simple in-memory cache to avoid redundant API calls."""
    _cache: Dict[str, Optional[IconResult]] = field(default_factory=dict)
    max_size: int = 256

    def get(self, key: str) -> Optional[IconResult]:
        return self._cache.get(key)

    def has(self, key: str) -> bool:
        return key in self._cache

    def put(self, key: str, result: Optional[IconResult]) -> None:
        if len(self._cache) >= self.max_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = result


# Module-level cache (persists across requests within one process)
_icon_cache = IconCache()


def _normalize_query(hint: str) -> str:
    """
    Convert an iconHint or label into a good Iconify search query.

    Strategy:
      1. Check TERM_ALIASES for a direct mapping
      2. Take the first 3 meaningful words (strip articles, prepositions)
      3. Lowercase, strip special chars
    """
    low = hint.strip().lower()

    # Direct alias match (longest match first)
    for term, alias in sorted(TERM_ALIASES.items(), key=lambda x: -len(x[0])):
        if term in low:
            return alias

    # Clean up: remove articles, prepositions, special chars
    cleaned = re.sub(r"[^a-z0-9\s-]", "", low)
    stop_words = {"a", "an", "the", "of", "for", "in", "on", "to", "and", "or", "with", "from"}
    words = [w for w in cleaned.split() if w not in stop_words and len(w) > 1]

    if not words:
        return cleaned.strip() or "box"

    # Take first 2-3 meaningful words
    return " ".join(words[:3])


async def search_icon(
    query: str,
    client: httpx.AsyncClient,
    limit: int = 10,
) -> List[str]:
    """
    Search Iconify for icons matching a query.
    Returns a list of icon IDs like ["tabler:brain", "lucide:brain"].
    """
    try:
        resp = await client.get(
            ICONIFY_SEARCH_URL,
            params={"query": query, "limit": limit},
            timeout=TIMEOUT_MS / 1000,
        )
        if resp.status_code != 200:
            logger.debug(f"Iconify search failed ({resp.status_code}) for '{query}'")
            return []

        data = resp.json()
        return data.get("icons", [])

    except Exception as e:
        logger.debug(f"Iconify search error for '{query}': {e}")
        return []


def _rank_icons(icon_ids: List[str]) -> List[str]:
    """
    Rank icon candidates by preferred collection order.
    Icons from preferred collections appear first.
    """
    def score(icon_id: str) -> int:
        prefix = icon_id.split(":")[0] if ":" in icon_id else ""
        try:
            return PREFERRED_COLLECTIONS.index(prefix)
        except ValueError:
            return len(PREFERRED_COLLECTIONS) + 1  # Unknown → last

    return sorted(icon_ids, key=score)


async def fetch_svg(
    icon_id: str,
    client: httpx.AsyncClient,
    size: int = ICON_SIZE,
) -> Optional[str]:
    """
    Fetch the SVG string for a given Iconify icon ID.
    Returns the raw SVG or None on failure.
    """
    if ":" not in icon_id:
        return None

    prefix, name = icon_id.split(":", 1)
    url = f"{ICONIFY_SVG_URL}/{prefix}/{name}.svg"

    try:
        resp = await client.get(
            url,
            params={"width": str(size), "height": str(size)},
            timeout=TIMEOUT_MS / 1000,
        )
        if resp.status_code != 200:
            return None

        svg = resp.text.strip()
        if not svg.startswith("<svg"):
            return None

        return svg

    except Exception as e:
        logger.debug(f"Iconify SVG fetch error for '{icon_id}': {e}")
        return None


async def get_icon(
    hint: str,
    client: Optional[httpx.AsyncClient] = None,
) -> Optional[IconResult]:
    """
    High-level: given a natural-language hint (iconHint or label),
    search for and return the best matching SVG icon.

    Uses caching to avoid redundant API calls for repeated hints.
    """
    query = _normalize_query(hint)

    # Check cache
    if _icon_cache.has(query):
        return _icon_cache.get(query)

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient()

    try:
        # Search
        candidates = await search_icon(query, client)
        if not candidates:
            # Try with just the first word as fallback
            fallback = query.split()[0] if " " in query else None
            if fallback:
                candidates = await search_icon(fallback, client)

        if not candidates:
            _icon_cache.put(query, None)
            return None

        # Rank by preferred collection
        ranked = _rank_icons(candidates)

        # Fetch the best match
        for icon_id in ranked[:3]:  # Try top 3
            svg = await fetch_svg(icon_id, client)
            if svg:
                prefix = icon_id.split(":")[0]
                result = IconResult(
                    query=query,
                    icon_id=icon_id,
                    svg=svg,
                    collection=prefix,
                    license="MIT",  # All preferred collections are MIT/ISC
                )
                _icon_cache.put(query, result)
                logger.info(f"Icon found: '{hint}' → {icon_id}")
                return result

        _icon_cache.put(query, None)
        return None

    except Exception as e:
        logger.warning(f"Icon fetch failed for '{hint}': {e}")
        _icon_cache.put(query, None)
        return None

    finally:
        if own_client:
            await client.aclose()


async def get_icons_batch(
    hints: List[str],
    concurrency: int = 6,
) -> Dict[str, Optional[IconResult]]:
    """
    Fetch icons for multiple hints in parallel (bounded concurrency).
    Returns a dict mapping hint → IconResult or None.
    """
    results: Dict[str, Optional[IconResult]] = {}
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:
        async def _fetch_one(hint: str) -> None:
            async with sem:
                results[hint] = await get_icon(hint, client)

        await asyncio.gather(*[_fetch_one(h) for h in hints])

    return results