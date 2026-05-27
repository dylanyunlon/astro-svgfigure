"""icon_registry.py — Icon resolution via Iconify API.

The previous version had 150+ hardcoded keyword → icon mappings.
That was wrong — it's doing the LLM's job with worse judgment.

The correct architecture:
  1. LLM outputs natural-language iconHint (e.g., "microprocessor chip")
  2. svg_icon_fetcher.py searches Iconify (275k+ icons, MIT/CC0)
  3. Returns real SVG with currentColor fill (scales to any size)

This module is now a thin bridge: resolve_icon() delegates to
svg_icon_fetcher's normalize_query for synchronous hint cleanup,
and provides the async fetch_icons_for_subgraph() for batch
icon acquisition during the pipeline.

The old ICON_MAP is replaced by svg_icon_fetcher's TERM_ALIASES
(which maps academic/ML jargon to good Iconify search terms).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def resolve_icon_hint(hint: str) -> str:
    """Clean up an iconHint for Iconify search.

    Delegates to svg_icon_fetcher._normalize_query for the heavy
    lifting (TERM_ALIASES, stop-word removal, etc.)

    This is synchronous — for use in prompt templates and tests.
    """
    if not hint:
        return "box"

    try:
        from backend.pipeline.svg_icon_fetcher import _normalize_query
        return _normalize_query(hint)
    except ImportError:
        # Fallback if svg_icon_fetcher not available
        return hint.strip().lower().split()[0] if hint.strip() else "box"


async def fetch_icons_for_subgraph(
    subgraph: Dict[str, Any],
    concurrency: int = 6,
) -> Dict[str, str]:
    """Fetch SVG icons for all nodes with iconHint in a subgraph.

    Walks the ELK subgraph, collects iconHint values, batch-fetches
    from Iconify, returns {node_id: svg_string}.

    Called during the composition stage of the layered pipeline,
    after all regions are generated but before final output.
    """
    hints: Dict[str, str] = {}  # node_id → iconHint

    def _collect(node: Dict[str, Any]) -> None:
        nid = node.get("id", "")
        hint = node.get("iconHint")
        if hint and nid:
            hints[nid] = hint
        for child in node.get("children", []):
            if isinstance(child, dict):
                _collect(child)

    _collect(subgraph)

    if not hints:
        return {}

    try:
        from backend.pipeline.svg_icon_fetcher import get_icons_batch
        unique_hints = list(set(hints.values()))
        results = await get_icons_batch(unique_hints, concurrency)

        # Map back: node_id → svg_string
        svg_map: Dict[str, str] = {}
        for nid, hint in hints.items():
            icon_result = results.get(hint)
            if icon_result and icon_result.svg:
                svg_map[nid] = icon_result.svg

        logger.info(f"Icons fetched: {len(svg_map)}/{len(hints)} resolved")
        return svg_map

    except ImportError:
        logger.warning("svg_icon_fetcher not available, skipping icons")
        return {}
    except Exception as e:
        logger.warning(f"Icon batch fetch failed: {e}")
        return {}


async def fetch_icons_for_canvas(
    elk_graph: Dict[str, Any],
    concurrency: int = 8,
) -> Dict[str, str]:
    """Fetch icons for an entire composed canvas (all regions).

    Walks all region containers in the ELK graph and fetches icons
    in a single batch. More efficient than per-region fetching.
    """
    all_hints: Dict[str, str] = {}

    def _collect_all(node: Dict[str, Any]) -> None:
        nid = node.get("id", "")
        hint = node.get("iconHint")
        if hint and nid:
            all_hints[nid] = hint
        for child in node.get("children", []):
            if isinstance(child, dict):
                _collect_all(child)

    _collect_all(elk_graph)

    if not all_hints:
        return {}

    try:
        from backend.pipeline.svg_icon_fetcher import get_icons_batch
        unique = list(set(all_hints.values()))
        results = await get_icons_batch(unique, concurrency)

        svg_map: Dict[str, str] = {}
        for nid, hint in all_hints.items():
            r = results.get(hint)
            if r and r.svg:
                svg_map[nid] = r.svg

        logger.info(f"Canvas icons: {len(svg_map)}/{len(all_hints)} resolved")
        return svg_map

    except Exception as e:
        logger.warning(f"Canvas icon fetch failed: {e}")
        return {}
