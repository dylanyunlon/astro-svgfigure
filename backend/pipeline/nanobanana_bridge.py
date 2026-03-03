"""
NanoBanana Bridge — Step 3 of the Forward Pipeline
====================================================
Takes the ELK layouted graph + JSON scaffold, requests Gemini NanoBanana
to generate academic-quality SVG.

Key insight from plan.md:
  "NanoBanana 生成的图片如此完美以至于让拓扑学家像个小丑"
  ELK.js only provides the skeleton coordinates;
  NanoBanana does the neural-network-level beautiful SVG generation
  using the JSON scaffold as a "json_example_user1" template.

GitHub references:
  - gemini-cli-extensions/nanobanana
  - ZeroLu/awesome-nanobanana-pro (prompt engineering)
  - aaronkwhite/nanobanana-studio-web (self-hosted)
  - GeminiGenAI/Free-Nano-Banana-Pro-API
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from ..ai_engine import AIEngine
from ..schemas import (
    BeautifyResponse,
    ElkGraph,
    NanoBananaScaffold,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Prompt Templates
# ============================================================================

NANOBANANA_SYSTEM_PROMPT = """\
You are an expert SVG generator specializing in academic paper figures.
You create publication-quality SVG diagrams that look professional enough
for top-tier venues (NeurIPS, ICLR, CVPR, etc.).

Style guidelines:
- Clean, minimal design with consistent color palette
- Rounded rectangles for nodes, smooth bezier curves for edges
- Professional fonts (system sans-serif)
- Subtle shadows and gradients (optional, tasteful)
- Arrow markers on edges
- Proper SVG namespace and viewBox
- Text must be readable (min 12px)
- Use the coordinate scaffold provided to place elements precisely

Output ONLY valid SVG markup, starting with <svg and ending with </svg>.
No markdown fences, no explanation, no preamble.
"""

NANOBANANA_USER_PROMPT_TEMPLATE = """\
Generate a publication-quality SVG academic figure based on this layout scaffold.

The scaffold contains precise pixel coordinates computed by ELK.js constraint layout.
Use these coordinates exactly — do not reposition elements.

Style: {style}

JSON Scaffold:
```json
{scaffold_json}
```

Additional rendering hints:
- Canvas size: {width}x{height}
- Total nodes: {node_count}
- Total connections: {edge_count}
- Apply subtle color coding based on node type/position
- Add arrow markers to directed edges
- Include a subtle background grid (optional)

Generate the complete SVG:"""


# ============================================================================
# Main Function
# ============================================================================

async def beautify_with_nanobanana(
    ai_engine: AIEngine,
    layouted,
    scaffold=None,
    model: Optional[str] = None,
    style: str = "academic",
) -> Dict[str, Any]:
    """
    Generate beautiful SVG from ELK layouted graph via NanoBanana (Gemini).

    Args:
        ai_engine: Initialized AIEngine
        layouted: ELK graph with computed coordinates (ElkGraph or dict)
        scaffold: Pre-built scaffold (NanoBananaScaffold or dict or None)
        model: LLM model (defaults to Gemini for NanoBanana)
        style: Visual style hint

    Returns:
        dict with success, svg, error, model_used keys
    """
    # Build scaffold if not provided
    if scaffold is None:
        from .scaffold_builder import build_scaffold
        scaffold = build_scaffold(layouted)

    # Handle both NanoBananaScaffold and dict
    if hasattr(scaffold, "model_dump"):
        scaffold_dict = scaffold.model_dump(by_alias=True)
    elif isinstance(scaffold, dict):
        scaffold_dict = scaffold
    else:
        scaffold_dict = {"elements": [], "connections": [], "canvas": {"width": 800, "height": 600}}

    scaffold_json = json.dumps(scaffold_dict, indent=2, ensure_ascii=False)

    # Extract canvas dimensions
    canvas = scaffold_dict.get("canvas", {})
    elements = scaffold_dict.get("elements", [])
    connections = scaffold_dict.get("connections", [])

    prompt = NANOBANANA_USER_PROMPT_TEMPLATE.format(
        scaffold_json=scaffold_json,
        style=style,
        width=canvas.get("width", 800),
        height=canvas.get("height", 600),
        node_count=len(elements),
        edge_count=len(connections),
    )

    try:
        logger.info(
            f"NanoBanana beautify: {len(scaffold.elements)} elements, "
            f"model={model or 'default'}, style={style}"
        )

        result = await ai_engine.get_completion(
            messages=[
                {"role": "system", "content": NANOBANANA_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=0.5,
            max_tokens=16384,
        )

        raw_svg = result["content"]
        svg = _extract_svg(raw_svg)

        if not svg:
            return {
                "success": False,
                "error": "LLM did not produce valid SVG markup",
                "model_used": result.get("model"),
            }

        logger.info(f"NanoBanana SVG generated: {len(svg)} bytes")

        return {
            "success": True,
            "svg": svg,
            "model_used": result.get("model"),
        }

    except Exception as e:
        logger.error(f"NanoBanana beautify failed: {e}")
        # Fallback: generate skeleton SVG without LLM
        try:
            from .scaffold_builder import build_scaffold
            if scaffold is None:
                scaffold = build_scaffold(layouted)
            fallback_svg = generate_skeleton_svg(scaffold if hasattr(scaffold, 'elements') else NanoBananaScaffold(**scaffold) if isinstance(scaffold, dict) else scaffold)
            return {
                "success": True,
                "svg": fallback_svg,
                "model_used": "skeleton-fallback",
                "warning": f"LLM failed ({str(e)}), using skeleton SVG",
            }
        except Exception as fallback_err:
            return {
                "success": False,
                "error": f"LLM: {str(e)}, Fallback: {str(fallback_err)}",
            }


# ============================================================================
# SVG Extraction Helper
# ============================================================================

def _extract_svg(text: str) -> Optional[str]:
    """
    Extract SVG markup from LLM output.
    Handles cases where the SVG is wrapped in markdown fences or extra text.
    """
    text = text.strip()

    # Remove markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Find SVG boundaries
    svg_start = text.find("<svg")
    svg_end = text.rfind("</svg>")

    if svg_start >= 0 and svg_end > svg_start:
        return text[svg_start : svg_end + 6]

    # Maybe the entire output is SVG
    if text.startswith("<svg") and text.endswith("</svg>"):
        return text

    return None


# ============================================================================
# Fallback: Generate skeleton SVG without LLM
# ============================================================================

def generate_skeleton_svg(scaffold: NanoBananaScaffold) -> str:
    """
    Generate a basic skeleton SVG directly from scaffold without LLM.
    Useful as fallback or for quick preview.
    """
    width = scaffold.canvas.get("width", 800)
    height = scaffold.canvas.get("height", 600)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">',
        '  <defs>',
        '    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">',
        '      <polygon points="0 0, 10 3.5, 0 7" fill="#666"/>',
        '    </marker>',
        '  </defs>',
        f'  <rect width="{width}" height="{height}" fill="#fafafa" rx="4"/>',
    ]

    # Color palette for nodes
    colors = [
        "#E3F2FD", "#E8F5E9", "#FFF3E0", "#F3E5F5",
        "#E0F7FA", "#FBE9E7", "#F1F8E9", "#EDE7F6",
    ]

    # Draw nodes
    for i, elem in enumerate(scaffold.elements):
        color = colors[i % len(colors)]
        border = "#90CAF9" if i % 2 == 0 else "#A5D6A7"
        svg_parts.append(
            f'  <rect x="{elem.x}" y="{elem.y}" width="{elem.width}" height="{elem.height}" '
            f'fill="{color}" stroke="{border}" stroke-width="2" rx="8"/>'
        )
        # Label
        cx = elem.x + elem.width / 2
        cy = elem.y + elem.height / 2
        svg_parts.append(
            f'  <text x="{cx}" y="{cy}" text-anchor="middle" dominant-baseline="middle" '
            f'font-family="system-ui, sans-serif" font-size="13" fill="#333">{elem.label}</text>'
        )

    # Draw edges
    for conn in scaffold.connections:
        if conn.points and len(conn.points) >= 2:
            path_d = f"M {conn.points[0]['x']} {conn.points[0]['y']}"
            for pt in conn.points[1:]:
                path_d += f" L {pt['x']} {pt['y']}"
            svg_parts.append(
                f'  <path d="{path_d}" fill="none" stroke="#666" stroke-width="1.5" '
                f'marker-end="url(#arrowhead)"/>'
            )

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)
