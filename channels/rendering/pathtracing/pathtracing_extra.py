import os, sys, json, math
from typing import Any, Optional

# ── path-tracer constants (referenced by default-argument expressions) ──
_PT_MAX_PATH_INTENSITY: float = 10.0
_PT_FILTER_SIGMA: float = 0.5

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










# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-PATHTRACE] StyleRayMarcher + BounceAccumulator
# Added in M846: style-ray emission from cells, multi-bounce visual blending.
# ═══════════════════════════════════════════════════════════════════════════════

_PT_MAX_BOUNCES: int = 6
_PT_STYLE_RAY_COUNT: int = 8
_PT_TERMINATION_PROB: float = 0.15      # Russian-roulette kill probability

_ASTRO_PT_TAG = "ASTRO-PATHTRACE"


def _dbg_pt(msg: str) -> None:
    """[ASTRO-PATHTRACE] gated debug printer."""
    _dbg(_ASTRO_PT_TAG, msg)


class StyleRayMarcher:
    """
    [ASTRO-PATHTRACE] Emit style rays from a source cell and march them
    through neighbour visual-attribute fields.

    Mirrors the conceptual structure of UE5 PathTracing.usf ray loops:
      - Halton-sampled directions  (SamplerType=1)
      - Per-bounce firefly clamp   (CVarPathTracingMaxPathIntensity)
      - Gaussian reconstruction    (CVarPathTracingFilterWidth)
      - MIS-weighted contribution  (CVarPathTracingMISMode=2)

    鲁迅式：射线不问来路——它只问能走多远，能带回多少光。
    """

    def __init__(
        self,
        cell_id: str,
        origin: tuple,          # (x, y, z) world-space centre
        cell_size: tuple,       # (w, h)
        ray_count: int = _PT_STYLE_RAY_COUNT,
        max_bounces: int = _PT_MAX_BOUNCES,
    ) -> None:
        self.cell_id     = cell_id
        self.origin      = origin
        self.cell_size   = cell_size
        self.ray_count   = max(1, ray_count)
        self.max_bounces = max(1, max_bounces)
        self._samples: list = []          # accumulated (radiance, weight) pairs
        _dbg_pt(
            f"StyleRayMarcher init | cell={cell_id} "
            f"origin={origin} rays={ray_count} bounces={max_bounces}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _direction_from_halton(self, ray_idx: int) -> tuple:
        """
        Map two Halton coordinates (base-2, base-3) to a unit 2-D direction
        then extend to 3-D (z=0) for planar style-space marching.

        鲁迅式：方向是由数字决定的命运——准随机序列给了每条射线不同的命，
        却没有给它们不平等的机会。
        """
        u = _pt_halton(ray_idx + 1, 2)          # ∈ [0, 1)
        v = _pt_halton(ray_idx + 1, 3)          # ∈ [0, 1)
        theta = 2.0 * math.pi * u
        # v biases toward equator — simple cosine-weighted hemisphere in 2-D
        phi = math.acos(max(-1.0, min(1.0, 2.0 * v - 1.0)))
        dx = math.sin(phi) * math.cos(theta)
        dy = math.sin(phi) * math.sin(theta)
        dz = math.cos(phi)
        return (dx, dy, dz)

    def _step_along_ray(
        self, pos: tuple, direction: tuple, step_len: float
    ) -> tuple:
        """Advance position by step_len along direction."""
        return (
            pos[0] + direction[0] * step_len,
            pos[1] + direction[1] * step_len,
            pos[2] + direction[2] * step_len,
        )

    def _sample_neighbour_attribute(
        self,
        pos: tuple,
        neighbour_map: dict,
        step: int,
    ) -> Optional[tuple]:
        """
        Retrieve the visual attribute (colour tuple) of the neighbour cell
        whose bounding region contains `pos`.  Returns None when no neighbour
        covers the point (ray escapes the cell graph).

        鲁迹式：采样是叩门——开门与否，不由射线决定。
        """
        for nbr_id, nbr_data in neighbour_map.items():
            bx, by = nbr_data.get("x", 0.0), nbr_data.get("y", 0.0)
            bw, bh = nbr_data.get("w", 1.0), nbr_data.get("h", 1.0)
            if bx <= pos[0] <= bx + bw and by <= pos[1] <= by + bh:
                colour = nbr_data.get("colour", (0.5, 0.5, 0.5))
                _dbg_pt(
                    f"  ray hit nbr={nbr_id} at step={step} "
                    f"pos=({pos[0]:.2f},{pos[1]:.2f}) colour={colour}"
                )
                return colour
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def march(
        self,
        neighbour_map: dict,
        step_length: Optional[float] = None,
        seed_offset: int = 0,
    ) -> list:
        """
        Fire `self.ray_count` style rays; return list of (radiance, mis_weight)
        pairs — one per ray that gathered a contribution.

        `neighbour_map` is keyed by cell_id and each value is a dict with
        keys x, y, w, h, colour (tuple).  Build it from bbox.json + probe
        data before calling march().

        鲁迅式：行军不需要地图，只需要方向——但归来的光需要地图才能找到家。
        """
        if step_length is None:
            step_length = min(self.cell_size[0], self.cell_size[1]) * 0.5

        self._samples.clear()
        step_length = max(step_length, 1e-3)

        for ray_idx in range(self.ray_count):
            global_idx = ray_idx + seed_offset * self.ray_count
            direction  = self._direction_from_halton(global_idx)
            pos        = self.origin
            throughput = (1.0, 1.0, 1.0)
            radiance   = (0.0, 0.0, 0.0)

            _dbg_pt(
                f"Ray {ray_idx}: dir=({direction[0]:.3f},"
                f"{direction[1]:.3f},{direction[2]:.3f})"
            )

            for bounce in range(self.max_bounces):
                pos = self._step_along_ray(pos, direction, step_length)
                hit_colour = self._sample_neighbour_attribute(
                    pos, neighbour_map, bounce
                )
                if hit_colour is None:
                    _dbg_pt(f"  Ray {ray_idx} escaped at bounce {bounce}")
                    break

                # Accumulate: radiance += throughput * hit_colour
                radiance = (
                    radiance[0] + throughput[0] * hit_colour[0],
                    radiance[1] + throughput[1] * hit_colour[1],
                    radiance[2] + throughput[2] * hit_colour[2],
                )

                # Firefly clamp
                radiance = _pt_firefly_clamp(radiance)

                # Attenuate throughput by a simple albedo proxy (avg channel)
                albedo = (
                    hit_colour[0] * 0.8,
                    hit_colour[1] * 0.8,
                    hit_colour[2] * 0.8,
                )
                throughput = (
                    throughput[0] * albedo[0],
                    throughput[1] * albedo[1],
                    throughput[2] * albedo[2],
                )

                # Russian roulette termination after bounce 2
                if bounce >= 2:
                    luminance = (
                        0.2126 * throughput[0]
                        + 0.7152 * throughput[1]
                        + 0.0722 * throughput[2]
                    )
                    if luminance < _PT_TERMINATION_PROB:
                        _dbg_pt(
                            f"  Ray {ray_idx} RR-terminated at bounce {bounce} "
                            f"lum={luminance:.4f}"
                        )
                        break
                    # Boost surviving paths to maintain unbiasedness
                    inv_survive = 1.0 / max(luminance, 1e-6)
                    throughput = (
                        throughput[0] * inv_survive,
                        throughput[1] * inv_survive,
                        throughput[2] * inv_survive,
                    )

            # Gaussian filter + MIS weight against a unit BSDF PDF
            radiance  = _pt_gaussian_filter(radiance)
            mis_w     = _pt_mis_weight(1.0, 1.0 / max(self.ray_count, 1))
            self._samples.append((radiance, mis_w))
            _dbg_pt(
                f"Ray {ray_idx} done: "
                f"radiance=({radiance[0]:.4f},{radiance[1]:.4f},"
                f"{radiance[2]:.4f}) mis_w={mis_w:.4f}"
            )

        return list(self._samples)

    def average_radiance(self) -> tuple:
        """
        Return the MIS-weighted average of all gathered samples.
        Safe to call after march().

        鲁迅式：平均是民主的幻觉——它把不平等的贡献抹平成一个数字。
        """
        if not self._samples:
            return (0.0, 0.0, 0.0)
        w_total = sum(w for _, w in self._samples)
        if w_total < 1e-12:
            return (0.0, 0.0, 0.0)
        r = sum(rad[0] * w for rad, w in self._samples) / w_total
        g = sum(rad[1] * w for rad, w in self._samples) / w_total
        b = sum(rad[2] * w for rad, w in self._samples) / w_total
        _dbg_pt(
            f"average_radiance cell={self.cell_id} → "
            f"({r:.4f},{g:.4f},{b:.4f}) over {len(self._samples)} samples"
        )
        return (r, g, b)


class BounceAccumulator:
    """
    [ASTRO-PATHTRACE] Multi-bounce style accumulator: fires a StyleRayMarcher
    per cell, gathers per-bounce contributions, and blends them into a final
    style radiance estimate.

    Conceptual mirror of the C++ PathTracingAccumulateToBuffer() path, where
    each bounce layer is weighted by a decay schedule before blending into the
    temporal accumulation buffer.

    鲁迅式：弹射是执念——每一次反弹都带走一点原色，留下一点混合的真相。
    """

    def __init__(
        self,
        max_bounces: int = _PT_MAX_BOUNCES,
        ray_count: int = _PT_STYLE_RAY_COUNT,
        bounce_decay: float = 0.72,
    ) -> None:
        self.max_bounces  = max_bounces
        self.ray_count    = ray_count
        self.bounce_decay = max(0.0, min(1.0, bounce_decay))
        self._cell_results: dict = {}
        _dbg_pt(
            f"BounceAccumulator init | bounces={max_bounces} "
            f"rays={ray_count} decay={bounce_decay}"
        )

    def accumulate(
        self,
        cell_id: str,
        origin: tuple,
        cell_size: tuple,
        neighbour_map: dict,
        seed_offset: int = 0,
    ) -> tuple:
        """
        Fire rays for `cell_id`, accumulate multi-bounce radiance with
        exponential decay weighting, cache and return the blended colour.

        鲁迅式：积累是时间的债务——每一帧都偿还一点，永远还不清。
        """
        marcher = StyleRayMarcher(
            cell_id=cell_id,
            origin=origin,
            cell_size=cell_size,
            ray_count=self.ray_count,
            max_bounces=self.max_bounces,
        )
        samples = marcher.march(neighbour_map, seed_offset=seed_offset)
        avg     = marcher.average_radiance()

        # Decay-weighted blend across virtual bounce layers
        # Layer 0 is the direct hit (weight 1.0), layer i has weight decay^i.
        blended = list(avg)
        decay_w = 1.0
        total_w = 1.0
        for i, (rad, mis_w) in enumerate(samples):
            if i == 0:
                continue                    # first sample already in avg
            decay_w *= self.bounce_decay
            total_w += decay_w
            blended[0] += rad[0] * decay_w * mis_w
            blended[1] += rad[1] * decay_w * mis_w
            blended[2] += rad[2] * decay_w * mis_w

        if total_w > 1e-12:
            blended = [c / total_w for c in blended]

        result = tuple(max(0.0, min(1.0, c)) for c in blended)
        self._cell_results[cell_id] = result
        _dbg_pt(
            f"BounceAccumulator.accumulate cell={cell_id} "
            f"→ {result}  (samples={len(samples)}, total_w={total_w:.3f})"
        )
        return result

    def get_result(self, cell_id: str) -> Optional[tuple]:
        """Return cached result for cell_id, or None if not yet computed."""
        return self._cell_results.get(cell_id)

    def accumulate_all(
        self,
        cell_map: dict,
        seed_offset: int = 0,
    ) -> dict:
        """
        Convenience: accumulate every cell in `cell_map`.
        `cell_map` schema::

            {
                cell_id: {
                    "origin": (x, y, z),
                    "cell_size": (w, h),
                    "neighbours": { nbr_id: {x,y,w,h,colour}, ... }
                }, ...
            }

        Returns dict of {cell_id: blended_colour_tuple}.

        鲁迅式：批量处理是工业的逻辑——逐个的艺术变成了流水线上的零件。
        """
        results: dict = {}
        for cid, data in cell_map.items():
            origin       = data.get("origin", (0.0, 0.0, 0.0))
            cell_size    = data.get("cell_size", (1.0, 1.0))
            neighbours   = data.get("neighbours", {})
            results[cid] = self.accumulate(
                cell_id=cid,
                origin=origin,
                cell_size=cell_size,
                neighbour_map=neighbours,
                seed_offset=seed_offset,
            )
        _dbg_pt(f"accumulate_all complete | {len(results)} cells processed")
        return results
