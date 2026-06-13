import os, sys, json, math
from typing import Any, Optional

def _dbg(tag, msg):
    if os.environ.get(f"ASTRO_{tag.replace('-','_')}_VERBOSE", "0") == "1":
        print(f"[{tag}] {msg}", file=sys.stderr)









def _pt_halton(index: int, base: int) -> float:
    """
    Halton low-discrepancy sequence element.
    Mirrors the path tracer's SamplerType=1 (Halton) used when
    CVarPathTracingLockedSamplingPattern is False.  The sequence provides
    better stratification than uniform random for MIS.

    鲁迅式：准随机数是伪装成随机的秩序——比真随机更公平，却不招摇。
    """
    result = 0.0
    f      = 1.0 / base
    i      = index
    while i > 0:
        result += f * (i % base)
        i       = i // base
        f      /= base
    return result

















def _pt_mis_weight(pdf_a: float, pdf_b: float) -> float:
    """
    Power heuristic MIS weight (β=2) — mirrors the balanced MIS combiner
    used in PathTracing.usf when CVarPathTracingMISMode=2.

        w(a) = pdf_a² / (pdf_a² + pdf_b²)

    鲁迅式：权衡是政治，也是物理——两种采样策略各占一半，谁也不独裁。
    """
    a2 = pdf_a * pdf_a
    b2 = pdf_b * pdf_b
    denom = a2 + b2
    return a2 / denom if denom > 1e-12 else 0.5

















def _pt_firefly_clamp(radiance: tuple, max_intensity: float = _PT_MAX_PATH_INTENSITY) -> tuple:
    """
    Per-path intensity clamp — mirrors CVarPathTracingMaxPathIntensity gate.
    Clamps each colour channel independently (not luminance) to keep hue.

    鲁迅式：萤火虫之所以刺眼，是因为它孤立地太亮——统一的上限是公平，不是压制。
    """
    return (
        min(radiance[0], max_intensity),
        min(radiance[1], max_intensity),
        min(radiance[2], max_intensity),
    )

















def _pt_gaussian_filter(radiance: tuple, sigma: float = _PT_FILTER_SIGMA) -> tuple:
    """
    Gaussian reconstruction filter weight for the current sample.
    Mirrors CVarPathTracingFilterWidth (σ = filter_width / (2π)).
    Applied as a scalar weight ∈ (0, 1] on the sample contribution.

    w = exp(-0.5 * r² / σ²), r ≈ 0 for a centred sample → weight ≈ 1.
    For accumulation we treat all samples as centred (no sub-pixel offset
    in the 2-D analogue), so this reduces to a constant σ-dependent scale
    that models the temporal blend decay used in the C++ accumulation buffer.

    鲁迅式：高斯滤波是温柔的遗忘——它让过去的样本随时间淡去，
    而不是被突然的 invalidate 彻底抹除。
    """
    weight = math.exp(-0.5 / max(sigma * sigma, 1e-8))
    return (radiance[0] * weight, radiance[1] * weight, radiance[2] * weight)








