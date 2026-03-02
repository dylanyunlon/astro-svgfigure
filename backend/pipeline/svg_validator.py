"""
SVG Validator — Validation + LLM Auto-Fix
==========================================
Validates SVG using lxml, and optionally auto-fixes issues via LLM.

Ported from autofigure2.py steps 4.5 (lxml validation) and 4.6 (LLM fix).

GitHub references:
  - ResearAI/AutoFigure-Edit (original validation logic)
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from ..ai_engine import AIEngine
from ..schemas import ValidateResponse

logger = logging.getLogger(__name__)


# ============================================================================
# SVG Validation with lxml
# ============================================================================

def validate_svg_syntax(svg: str) -> Tuple[bool, List[str]]:
    """
    Validate SVG syntax using lxml.

    Returns:
        (is_valid, list_of_errors)
    """
    errors: List[str] = []

    # Basic checks
    if not svg or not svg.strip():
        return False, ["SVG content is empty"]

    if "<svg" not in svg:
        return False, ["Missing <svg> root element"]

    if "</svg>" not in svg:
        errors.append("Missing </svg> closing tag")

    # Try lxml parsing
    try:
        from lxml import etree

        # Parse as XML
        parser = etree.XMLParser(recover=True)
        doc = etree.fromstring(svg.encode("utf-8"), parser=parser)

        # Check for parser errors
        if parser.error_log:
            for error in parser.error_log:
                errors.append(f"Line {error.line}: {error.message}")

        # Check root element is <svg>
        if doc is not None:
            tag = doc.tag
            # Handle namespace prefix
            if "}" in tag:
                tag = tag.split("}")[-1]
            if tag != "svg":
                errors.append(f"Root element is <{tag}>, expected <svg>")

            # Check for required attributes
            if doc.get("xmlns") is None and "{http://www.w3.org/2000/svg}" not in doc.tag:
                errors.append("Missing xmlns attribute on <svg>")

            # Check viewBox
            if doc.get("viewBox") is None:
                errors.append("Missing viewBox attribute (recommended)")

    except ImportError:
        logger.warning("lxml not installed, falling back to basic validation")
        # Basic regex validation
        if not re.search(r'<svg[^>]*xmlns', svg):
            errors.append("Missing xmlns attribute on <svg>")

    except Exception as e:
        errors.append(f"XML parse error: {str(e)}")

    is_valid = len(errors) == 0
    return is_valid, errors


# ============================================================================
# LLM Auto-Fix
# ============================================================================

SVG_FIX_SYSTEM_PROMPT = """\
You are an SVG syntax expert. Fix the provided SVG so it is valid XML and renders correctly.

Rules:
1. Fix all XML syntax errors (unclosed tags, invalid attributes, etc.)
2. Ensure proper SVG namespace: xmlns="http://www.w3.org/2000/svg"
3. Ensure viewBox is present and correct
4. Fix any invalid attribute values
5. Preserve ALL visual content — do not remove or change the design
6. Output ONLY the fixed SVG, no explanation, no markdown fences

The fixed SVG must start with <svg and end with </svg>.
"""

SVG_FIX_USER_PROMPT = """\
Fix this SVG. Errors found:
{errors}

SVG to fix:
{svg}

Fixed SVG:"""


async def validate_svg(
    svg: str,
    auto_fix: bool = True,
    ai_engine: Optional[AIEngine] = None,
    model: Optional[str] = None,
    max_retries: int = 3,
) -> ValidateResponse:
    """
    Validate SVG and optionally auto-fix using LLM.

    Args:
        svg: SVG string to validate
        auto_fix: Whether to attempt LLM-based auto-fix
        ai_engine: AIEngine instance (required for auto_fix)
        model: LLM model for fixing
        max_retries: Max fix attempts

    Returns:
        ValidateResponse with validation results
    """
    is_valid, errors = validate_svg_syntax(svg)

    if is_valid:
        return ValidateResponse(valid=True, errors=[], fixed_svg=svg, fix_iterations=0)

    if not auto_fix or ai_engine is None:
        return ValidateResponse(valid=False, errors=errors)

    # Attempt LLM auto-fix
    current_svg = svg
    all_errors = list(errors)
    iterations = 0

    for i in range(max_retries):
        iterations += 1
        logger.info(f"SVG fix attempt {iterations}/{max_retries}: {len(all_errors)} errors")

        try:
            prompt = SVG_FIX_USER_PROMPT.format(
                errors="\n".join(f"- {e}" for e in all_errors),
                svg=current_svg,
            )

            result = await ai_engine.get_completion(
                messages=[
                    {"role": "system", "content": SVG_FIX_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                temperature=0.2,
                max_tokens=16384,
            )

            fixed = result["content"].strip()

            # Extract SVG from response
            svg_start = fixed.find("<svg")
            svg_end = fixed.rfind("</svg>")
            if svg_start >= 0 and svg_end > svg_start:
                fixed = fixed[svg_start : svg_end + 6]
            elif fixed.startswith("```"):
                lines = fixed.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                fixed = "\n".join(lines).strip()

            # Validate the fix
            is_valid, new_errors = validate_svg_syntax(fixed)

            if is_valid:
                logger.info(f"SVG fixed successfully after {iterations} iteration(s)")
                return ValidateResponse(
                    valid=True,
                    errors=[],
                    fixed_svg=fixed,
                    fix_iterations=iterations,
                )

            current_svg = fixed
            all_errors = new_errors

        except Exception as e:
            logger.error(f"SVG fix iteration {iterations} failed: {e}")
            all_errors.append(f"Fix attempt {iterations} error: {str(e)}")

    # All retries exhausted
    return ValidateResponse(
        valid=False,
        errors=all_errors,
        fixed_svg=current_svg,
        fix_iterations=iterations,
    )


# ============================================================================
# Utility: Quick Sanitize
# ============================================================================

def sanitize_svg(svg: str) -> str:
    """
    Quick sanitize SVG without LLM — fixes common issues.
    """
    result = svg.strip()

    # Ensure xmlns
    if "<svg" in result and 'xmlns=' not in result:
        result = result.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1)

    # Ensure closing tag
    if "</svg>" not in result:
        result += "\n</svg>"

    # Remove BOM
    if result.startswith("\ufeff"):
        result = result[1:]

    return result
