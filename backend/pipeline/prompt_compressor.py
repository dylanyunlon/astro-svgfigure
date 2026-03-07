"""
Prompt Compressor — Verbose Design Spec → Dense Narrative Paragraph
=====================================================================
Knuth-style analysis: The fundamental problem is an IMPEDANCE MISMATCH.

The Grok 4 prompt engineer generates TIER-40/60/80 numbered lists with
thousands of characters of *itemized instructions*.  But Gemini's image
generation model is designed to consume NARRATIVE PARAGRAPHS — "describe
the scene, don't just list keywords" (Google's official best-practice).

Numbered lists of 40-80 items are the WORST possible format for Gemini
image generation:
  1. They trigger the model's text-completion mode (it "reads" the list
     and tries to continue/echo it instead of generating an image)
  2. They bloat the character count far beyond what the image model can
     usefully attend to (~1500 chars of dense narrative >> 6000 chars of
     numbered points)
  3. The proxy (tryallai.com) may not faithfully forward responseModalities
     to the upstream Gemini API, especially when the prompt is very long

The fix is NOT "truncate at 5000 chars" (which is a band-aid that throws
away information).  The fix is COMPRESS: use a local summarizer to fold
the numbered list into a single dense paragraph that Gemini's image model
can efficiently consume.

This module implements three compression strategies:
  A. structural_compress: Parse numbered points, deduplicate, merge
     related points, produce a flowing paragraph (NO LLM needed)
  B. tier_downgrade: If TIER-60/80, forcibly reduce to TIER-20 density
  C. narrative_rewrite: Convert from "Point 1: ... Point 2: ..." to
     a single coherent scene description paragraph

Design principle (Knuth): "Premature optimization is the root of all evil,
but we should not miss our opportunities in the critical 3%."
Prompt compression IS that critical 3% — it determines whether Gemini
generates an image or echoes text.

GitHub: dylanyunlon/astro-svgfigure
"""

from __future__ import annotations

import re
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Gemini image model sweet spot: 800-1500 chars of dense narrative.
# Beyond ~2000 chars, risk of text-echo increases sharply.
# Below ~400 chars, image lacks specificity.
TARGET_NARRATIVE_CHARS = 1200
MAX_NARRATIVE_CHARS = 1800
MIN_NARRATIVE_CHARS = 400

# Categories that we always want represented in the compressed output
ESSENTIAL_CATEGORIES = [
    "canvas",       # background, dimensions, orientation
    "style",        # color palette, typography, overall aesthetic
    "layout",       # spatial arrangement, flow direction
    "components",   # key nodes/modules (names only, not exhaustive)
    "connections",  # arrow types, flow
]


# ============================================================================
# Strategy A: Structural Compression (no LLM needed)
# ============================================================================

def structural_compress(prompt: str, svg_summary: str = "") -> str:
    """
    Parse a TIER-N numbered-point prompt and compress it into a dense
    narrative paragraph suitable for Gemini image generation.

    Algorithm:
      1. Extract tier header and all numbered points
      2. Classify each point into a category
      3. Merge points within each category into one sentence
      4. Compose a single flowing paragraph from merged sentences
      5. Append the spatial layout summary (abbreviated)

    This is a DETERMINISTIC transform — no LLM calls, no network,
    no randomness.  Runs in <1ms for any input.

    Args:
        prompt: The raw Grok-generated TIER-N prompt with numbered points
        svg_summary: Optional abbreviated layout description

    Returns:
        A dense narrative paragraph, typically 800-1500 chars
    """
    # ── Step 1: Extract tier and points ──
    tier = _extract_tier(prompt)
    points = _extract_points(prompt)

    if not points:
        # Not a numbered-point format — return cleaned/truncated original
        logger.warning("No numbered points found — returning cleaned original")
        return _truncate_narrative(prompt, MAX_NARRATIVE_CHARS)

    logger.info(f"Compressing TIER-{tier}: {len(points)} points → narrative paragraph")

    # ── Step 2: Classify points ──
    classified: Dict[str, List[str]] = {cat: [] for cat in ESSENTIAL_CATEGORIES}
    classified["other"] = []

    for point in points:
        category = _classify_point(point)
        classified[category].append(point)

    # ── Step 3: Merge each category into one sentence ──
    sentences: List[str] = []

    # Canvas/background
    if classified["canvas"]:
        merged = _merge_points(classified["canvas"], max_chars=200)
        sentences.append(f"Set a professional scientific figure canvas: {merged}.")

    # Style
    if classified["style"]:
        merged = _merge_points(classified["style"], max_chars=250)
        sentences.append(f"Use {merged}.")

    # Layout
    if classified["layout"]:
        merged = _merge_points(classified["layout"], max_chars=200)
        sentences.append(f"Arrange components in {merged}.")

    # Components — most verbose category, compress aggressively
    if classified["components"]:
        # Extract just the component NAMES, not full descriptions
        component_names = _extract_component_names(classified["components"])
        if component_names:
            names_str = ", ".join(component_names[:15])  # Cap at 15 names
            sentences.append(
                f"Key components include: {names_str}. "
                f"Render each as a rounded rectangle with clean label text."
            )

    # Connections
    if classified["connections"]:
        merged = _merge_points(classified["connections"], max_chars=200)
        sentences.append(f"Connect components with {merged}.")

    # Other — only include if we have room
    remaining_budget = MAX_NARRATIVE_CHARS - sum(len(s) for s in sentences)
    if classified["other"] and remaining_budget > 200:
        merged = _merge_points(classified["other"], max_chars=min(300, remaining_budget))
        sentences.append(merged)

    # ── Step 4: Compose paragraph ──
    paragraph = " ".join(sentences)

    # ── Step 5: Append abbreviated layout if we have room ──
    if svg_summary and len(paragraph) + 200 < MAX_NARRATIVE_CHARS:
        # Take just the first 3 lines of the summary (canvas size + node count)
        summary_lines = svg_summary.strip().split("\n")[:3]
        brief_summary = " ".join(line.strip() for line in summary_lines if line.strip())
        paragraph += f" Layout: {brief_summary}"

    # Final guard
    if len(paragraph) > MAX_NARRATIVE_CHARS:
        paragraph = paragraph[:MAX_NARRATIVE_CHARS - 3] + "..."

    if len(paragraph) < MIN_NARRATIVE_CHARS:
        # Too short — pad with generic quality instructions
        paragraph += (
            " The figure should be immediately understandable and visually striking, "
            "suitable for a top-tier academic conference paper."
        )

    logger.info(f"Compressed: {sum(len(p) for p in points)} chars → {len(paragraph)} chars")
    return paragraph


# ============================================================================
# Strategy B: Tier Downgrade
# ============================================================================

def tier_downgrade(prompt: str, target_tier: int = 20) -> str:
    """
    Forcibly reduce a TIER-60/80 prompt to TIER-20 density by keeping
    only the most important points.

    Selection heuristic (Knuth: "optimize the inner loop"):
      - Always keep: canvas/style points (Point 1-3 typically)
      - Always keep: ONE representative component point
      - Always keep: ONE connection/arrow point
      - Always keep: final summary/quality point
      - Drop: repetitive per-component descriptions

    Args:
        prompt: Raw TIER-N prompt
        target_tier: Target tier (default 20 = 20 points)

    Returns:
        Reduced prompt with ≤target_tier points
    """
    points = _extract_points(prompt)
    if len(points) <= target_tier:
        return prompt  # Already at or below target

    # Keep first 3 (canvas/style), every Nth component, last 2 (summary)
    n_middle = target_tier - 5
    step = max(1, (len(points) - 5) // n_middle)

    kept = points[:3]  # First 3
    kept.extend(points[3:-2][::step][:n_middle])  # Sampled middle
    kept.extend(points[-2:])  # Last 2

    tier_header = f"[TIER-{target_tier}: {len(kept)} design points]"
    numbered = [f"Point {i+1}: {p}" for i, p in enumerate(kept)]

    result = tier_header + "\n" + "\n".join(numbered)
    logger.info(f"Tier downgrade: {len(points)} → {len(kept)} points")
    return result


# ============================================================================
# Strategy C: Full Narrative Rewrite (no LLM, template-based)
# ============================================================================

def to_gemini_narrative(
    prompt: str,
    method_text: str = "",
    svg_summary: str = "",
    component_names: Optional[List[str]] = None,
) -> str:
    """
    Convert ANY prompt format into a Gemini-optimized narrative paragraph.

    This is the RECOMMENDED entry point for the image generation pipeline.
    It produces a single flowing paragraph that:
      1. Starts with "Generate an image of..."  (triggers image mode)
      2. Describes the scene narratively (Google's best practice)
      3. Stays under 1500 chars (sweet spot for image gen)
      4. Never uses numbered lists or "Point N:" format

    Args:
        prompt: Raw design spec (any format)
        method_text: Paper method description (for context)
        svg_summary: SVG spatial layout summary
        component_names: Pre-extracted component names (optional)

    Returns:
        A Gemini-optimized narrative paragraph
    """
    # First, try structural compression
    compressed = structural_compress(prompt, svg_summary)

    # Extract component names from method text if not provided
    if not component_names and method_text:
        # Take key nouns from method text as component hints
        words = method_text.split()
        # Crude but fast: capitalize words are likely component names
        component_names = [w.strip(",.;:()") for w in words if w[0:1].isupper()][:10]

    # Build the narrative with explicit image-generation trigger
    parts = [
        "Generate an image of a publication-ready scientific figure."
    ]

    # Add method context if available
    if method_text:
        # Take first ~200 chars of method text as context
        method_brief = method_text[:200].rsplit(" ", 1)[0] if len(method_text) > 200 else method_text
        parts.append(f"The figure illustrates: {method_brief}.")

    # Add the compressed design spec
    parts.append(compressed)

    # Explicit output instruction (Google recommends being explicit)
    parts.append(
        "Output a single high-quality image suitable for an academic paper. "
        "Professional vector illustration style with clean sans-serif labels."
    )

    result = " ".join(parts)

    # Final trim
    if len(result) > MAX_NARRATIVE_CHARS:
        result = result[:MAX_NARRATIVE_CHARS - 50] + (
            " Apply consistent professional academic style throughout."
        )

    return result


# ============================================================================
# Internal helpers
# ============================================================================

def _extract_tier(prompt: str) -> int:
    """Extract tier number from prompt header."""
    match = re.search(r'\[TIER-(\d+)', prompt)
    return int(match.group(1)) if match else 40  # default assumption

def _extract_points(prompt: str) -> List[str]:
    """Extract numbered points from prompt, stripping 'Point N:' prefix."""
    # Match patterns like "Point 1:", "1.", "1:" etc.
    pattern = re.compile(r'(?:^|\n)\s*(?:Point\s+)?\d+[.:\s：]+(.+?)(?=\n\s*(?:Point\s+)?\d+[.:\s：]|\Z)', re.DOTALL)
    matches = pattern.findall(prompt)
    return [m.strip() for m in matches if m.strip()]

def _classify_point(point: str) -> str:
    """Classify a design point into a category."""
    p_lower = point.lower()

    canvas_keywords = ["canvas", "background", "white background", "landscape", "orientation", "resolution", "dpi", "viewbox"]
    style_keywords = ["color palette", "typography", "font", "sans-serif", "shadow", "gradient", "professional", "academic", "style"]
    layout_keywords = ["position", "arrange", "layout", "flow direction", "top to bottom", "left to right", "spatial", "spacing", "padding"]
    component_keywords = ["node", "module", "component", "block", "rectangle", "rounded", "box", "label", "icon", "illustration"]
    connection_keywords = ["arrow", "connection", "edge", "flow", "dashed", "solid", "bent", "orthogonal", "curved", "line"]

    for kw in canvas_keywords:
        if kw in p_lower:
            return "canvas"
    for kw in style_keywords:
        if kw in p_lower:
            return "style"
    for kw in layout_keywords:
        if kw in p_lower:
            return "layout"
    for kw in connection_keywords:
        if kw in p_lower:
            return "connections"
    for kw in component_keywords:
        if kw in p_lower:
            return "components"

    return "other"

def _merge_points(points: List[str], max_chars: int = 300) -> str:
    """Merge multiple points into a single concise phrase."""
    if not points:
        return ""

    # Strip common verbose prefixes that add noise
    cleaned = []
    strip_prefixes = [
        "Set a ", "Create a ", "Draw a ", "Add a ", "Apply ",
        "Use ", "Position ", "Ensure ", "The ", "Maintain ",
    ]
    for p in points:
        text = p.rstrip(".")
        for prefix in strip_prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):]
                break
        cleaned.append(text)

    # Join all points, then truncate intelligently
    combined = "; ".join(cleaned)

    if len(combined) <= max_chars:
        return combined

    # Truncate at a phrase boundary
    truncated = combined[:max_chars]
    last_sep = max(truncated.rfind(";"), truncated.rfind(","), truncated.rfind("."))
    if last_sep > max_chars // 2:
        truncated = truncated[:last_sep]

    return truncated

def _extract_component_names(component_points: List[str]) -> List[str]:
    """Extract component/module names from design points."""
    names = []
    # Look for quoted strings or capitalized phrases
    for point in component_points:
        # Quoted names: "Encoder", "LSTM Module", etc.
        quoted = re.findall(r'"([^"]{2,40})"', point)
        names.extend(quoted)

        # Also look for "the X module" or "X component" patterns
        named = re.findall(r'(?:the\s+)?([A-Z][a-zA-Z0-9\s]{2,25}?)(?:\s+(?:module|component|block|node|layer))', point)
        names.extend(named)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for name in names:
        name_clean = name.strip()
        if name_clean.lower() not in seen:
            seen.add(name_clean.lower())
            unique.append(name_clean)

    return unique

def _truncate_narrative(text: str, max_chars: int) -> str:
    """Truncate text at a sentence boundary."""
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    # Find last sentence end
    for sep in ["。", ". ", ".\n"]:
        last = truncated.rfind(sep)
        if last > max_chars // 2:
            return truncated[:last + len(sep)]

    return truncated + "..."