"""sprite_export.py — Sprite-aware figure export to self-contained SVG / PDF (M223).

export_components.py::export_batch() already packages pipeline layers into a
download bundle; export_svg() already emits a combined SVG with positioned
<image> elements.  We follow that "package the composited result for download"
pattern, but for the sprite-bearing figure that to-svg.ts renders: a hybrid SVG
that mixes vector skeleton (groups, arrows, operators, vectorized sprites) with
raster sprites embedded as data-URI <image> elements.

Two guarantees a publishable export needs:

  1. self-contained: every raster sprite must be an inline base64 data URI (no
     external file refs that break when the SVG moves). M214 already injects
     sprites as data URIs, so a sprite figure is self-contained by
     construction; this module asserts/repairs that and refuses to emit a file
     with a dangling href.

  2. vector-first PDF: cairosvg renders the SVG to PDF preserving vector paths
     (skeleton + vectorized sprites stay infinitely scalable) and embedding the
     raster sprites. If cairosvg is unavailable, SVG export still works and PDF
     degrades with a clear error rather than a crash (mirrors
     gemini_image_gen.py's cairosvg-optional handling).

Input is the rendered SVG string (the output of elkToSvg). This keeps the
exporter renderer-agnostic: whatever the TS produces, we package.
"""
from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  §1  Result
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ExportResult:
    success: bool
    fmt: str                                  # "svg" | "pdf" | "png"
    data: Optional[bytes] = None              # binary payload (pdf/png)
    text: Optional[str] = None                # svg text payload
    mime_type: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success, "fmt": self.fmt,
            "bytes": len(self.data) if self.data else 0,
            "has_text": self.text is not None,
            "mime_type": self.mime_type,
            "diagnostics": self.diagnostics, "error": self.error,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  §2  Self-containment check / repair
# ═══════════════════════════════════════════════════════════════════════════

# Matches href / xlink:href values on <image> elements.
_HREF_RE = re.compile(r'(?:xlink:)?href\s*=\s*"([^"]*)"')

# The root <svg> from to-svg.ts carries style="max-width:100%;height:auto;".
# Browsers honor it, but cairosvg interprets it as a zero/auto size and
# rasterizes a BLANK canvas. Strip it (root tag only) before rasterizing.
_ROOT_STYLE_RE = re.compile(r'(<svg\b[^>]*?)\s+style="[^"]*"', re.IGNORECASE)


def _prep_for_raster(svg: str) -> str:
    """Make an elkToSvg() string safe for cairosvg rasterization.

    cairosvg renders nothing when the root <svg> has the responsive
    style="max-width:100%;height:auto;" attribute (it resolves to a 0-size
    viewport). The explicit width/height + viewBox are sufficient, so we drop
    the style on the root tag. No-op for SVGs that lack it.
    """
    return _ROOT_STYLE_RE.sub(r"\1", svg, count=1)


def audit_self_contained(svg: str) -> Dict[str, Any]:
    """Report whether every <image> href is an inline data URI.

    Returns {ok, total_images, inline, external_refs[]}. An external_ref is any
    href that is not a data: URI — those break a moved/published SVG.
    """
    hrefs = _HREF_RE.findall(svg)
    external = [h for h in hrefs if not h.startswith("data:")]
    return {
        "ok": len(external) == 0,
        "total_images": len(hrefs),
        "inline": len(hrefs) - len(external),
        "external_refs": external[:10],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  §3  Export entry points
# ═══════════════════════════════════════════════════════════════════════════

def export_svg(svg: str, *, require_self_contained: bool = True) -> ExportResult:
    """Return the SVG as a self-contained text payload.

    With require_self_contained, refuses (success=False) if any <image> points
    at an external file — a publishable figure must carry its sprites inline.
    """
    audit = audit_self_contained(svg)
    if require_self_contained and not audit["ok"]:
        return ExportResult(
            success=False, fmt="svg", diagnostics=audit,
            error=(f"{len(audit['external_refs'])} external image ref(s); "
                   "figure is not self-contained"),
        )
    return ExportResult(
        success=True, fmt="svg", text=svg,
        mime_type="image/svg+xml", diagnostics=audit,
    )


def export_pdf(
    svg: str,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    scale: float = 1.0,
) -> ExportResult:
    """Render the SVG to PDF via cairosvg (vector-preserving).

    Degrades gracefully: if cairosvg is unavailable, returns success=False with
    a clear, actionable error instead of raising.
    """
    audit = audit_self_contained(svg)
    try:
        import cairosvg  # type: ignore
    except Exception:
        return ExportResult(
            success=False, fmt="pdf", diagnostics=audit,
            error=("cairosvg not installed — PDF export unavailable. "
                   "Install cairosvg (in requirements.txt) or export SVG."),
        )
    try:
        kwargs: Dict[str, Any] = {"scale": scale}
        if width:
            kwargs["output_width"] = int(width * scale)
        if height:
            kwargs["output_height"] = int(height * scale)
        pdf_bytes = cairosvg.svg2pdf(bytestring=_prep_for_raster(svg).encode("utf-8"), **kwargs)
        return ExportResult(
            success=bool(pdf_bytes), fmt="pdf", data=pdf_bytes,
            mime_type="application/pdf", diagnostics=audit,
            error=None if pdf_bytes else "cairosvg produced empty PDF",
        )
    except Exception as e:  # pragma: no cover - cairosvg internals
        logger.exception("PDF export failed")
        return ExportResult(success=False, fmt="pdf", diagnostics=audit,
                            error=str(e))


def export_png(
    svg: str,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    scale: float = 2.0,
) -> ExportResult:
    """Render the SVG to a raster PNG via cairosvg (scale 2x default for crisp
    output). Same graceful degradation as export_pdf."""
    audit = audit_self_contained(svg)
    try:
        import cairosvg  # type: ignore
    except Exception:
        return ExportResult(
            success=False, fmt="png", diagnostics=audit,
            error="cairosvg not installed — PNG export unavailable.",
        )
    try:
        kwargs: Dict[str, Any] = {"scale": scale}
        if width:
            kwargs["output_width"] = int(width * scale)
        if height:
            kwargs["output_height"] = int(height * scale)
        png_bytes = cairosvg.svg2png(bytestring=_prep_for_raster(svg).encode("utf-8"), **kwargs)
        return ExportResult(
            success=bool(png_bytes), fmt="png", data=png_bytes,
            mime_type="image/png", diagnostics=audit,
            error=None if png_bytes else "cairosvg produced empty PNG",
        )
    except Exception as e:  # pragma: no cover
        logger.exception("PNG export failed")
        return ExportResult(success=False, fmt="png", diagnostics=audit,
                            error=str(e))


def export_figure(
    svg: str,
    fmt: str = "svg",
    **kw,
) -> ExportResult:
    """Single dispatch entry: fmt in {svg, pdf, png}."""
    fmt = (fmt or "svg").lower()
    if fmt == "svg":
        return export_svg(svg, require_self_contained=kw.get(
            "require_self_contained", True))
    if fmt == "pdf":
        return export_pdf(svg, width=kw.get("width"),
                          height=kw.get("height"), scale=kw.get("scale", 1.0))
    if fmt == "png":
        return export_png(svg, width=kw.get("width"),
                          height=kw.get("height"), scale=kw.get("scale", 2.0))
    return ExportResult(success=False, fmt=fmt,
                        error=f"unsupported format: {fmt}")
