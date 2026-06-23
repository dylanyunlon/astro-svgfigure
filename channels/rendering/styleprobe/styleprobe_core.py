import os, sys, json, math
from typing import Any, Optional

def _dbg(tag, msg):
    if os.environ.get(f"ASTRO_{tag.replace('-','_')}_VERBOSE", "0") == "1":
        print(f"[{tag}] {msg}", file=sys.stderr)





def build_astro_cell_style_probes(channels_dir: str) -> dict:
    """
    Build one AstroCellStyleProbe per active cell in the registry snapshot.

    Python equivalent of BuildAstroCellStyleProbes() — iterates all published
    cell/*/bbox.json entries, constructs probes, samples surrounding cells,
    returns a dict keyed by cell_id.

    Called from proc() before SVG parameter finalisation so that the probe
    data is ready when adjust_svg_params_for_style_consistency() is called.
    """
    probes: dict = {}

    cell_base = os.path.join(channels_dir, "cell")
    if not os.path.isdir(cell_base):
        return probes

    for cid in os.listdir(cell_base):
        bbox_path = os.path.join(cell_base, cid, "bbox.json")
        if not os.path.isfile(bbox_path):
            continue
        try:
            with open(bbox_path) as _f:
                bbox = json.load(_f)
        except (json.JSONDecodeError, OSError):
            continue

        probe = AstroCellStyleProbe(cid, bbox)
        probe.sample_surrounding_cells(channels_dir)
        probes[cid] = probe

    return probes

# ═══════════════════════════════════════════════
# Channel I/O — Apollo Reader/Writer equivalent
# ═══════════════════════════════════════════════

# ═══════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-STYLEPROBE] NeighborProbe + StyleDiffusion + ProbeCache
# Added in M846: k-NN species sampling, style smoothing diffusion, probe cache.
# ═══════════════════════════════════════════════════════════════════════════════

_SP_DEFAULT_K: int = 6
_SP_DIFFUSION_STEPS: int = 4
_SP_DIFFUSION_ALPHA: float = 0.35        # blend ratio per diffusion step
_SP_CACHE_MAX_SIZE: int = 512

_ASTRO_SP_TAG = "ASTRO-STYLEPROBE"


def _dbg_sp(msg: str) -> None:
    """[ASTRO-STYLEPROBE] gated debug printer."""
    _dbg(_ASTRO_SP_TAG, msg)


def _sp_euclidean_dist(ax: float, ay: float, bx: float, by: float) -> float:
    """2-D Euclidean distance between two cell centres."""
    dx, dy = ax - bx, ay - by
    return math.sqrt(dx * dx + dy * dy)


def _sp_weighted_colour(colours: list, weights: list) -> tuple:
    """
    Return the weighted average of a list of (r,g,b) tuples.

    鲁迅式：权重是偏见的量化——每种颜色都有它的分量，却没有一种是绝对的主角。
    """
    if not colours or not weights:
        return (0.5, 0.5, 0.5)
    total_w = sum(weights)
    if total_w < 1e-12:
        return (0.5, 0.5, 0.5)
    r = sum(c[0] * w for c, w in zip(colours, weights)) / total_w
    g = sum(c[1] * w for c, w in zip(colours, weights)) / total_w
    b = sum(c[2] * w for c, w in zip(colours, weights)) / total_w
    return (max(0.0, min(1.0, r)),
            max(0.0, min(1.0, g)),
            max(0.0, min(1.0, b)))


class NeighborProbe:
    """
    [ASTRO-STYLEPROBE] k-Nearest-Neighbour species sampling probe.

    Given a query cell's world-space centre, scans a flat `cell_registry`
    dict (cell_id → {x, y, w, h, species, colour}) and returns the k closest
    neighbours weighted by inverse-distance.  Used to blend species-specific
    visual attributes from spatial context rather than relying solely on
    cardinal adjacency.

    鲁迅式：近邻是命运，不是选择——距离决定了谁能影响你，而不是谁想影响你。
    """

    def __init__(self, k: int = _SP_DEFAULT_K) -> None:
        self.k = max(1, k)
        _dbg_sp(f"NeighborProbe init | k={self.k}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _candidate_distances(
        self,
        qx: float,
        qy: float,
        cell_registry: dict,
        exclude_id: Optional[str] = None,
    ) -> list:
        """
        Return list of (distance, cell_id, entry) sorted by distance asc.

        鲁迅式：排序是文明的开始——把混乱的邻居按距离排好队，才能谈影响力。
        """
        candidates = []
        for cid, entry in cell_registry.items():
            if cid == exclude_id:
                continue
            cx = entry.get("x", 0.0) + entry.get("w", 0.0) * 0.5
            cy = entry.get("y", 0.0) + entry.get("h", 0.0) * 0.5
            dist = _sp_euclidean_dist(qx, qy, cx, cy)
            candidates.append((dist, cid, entry))
        candidates.sort(key=lambda t: t[0])
        return candidates

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        query_id: str,
        qx: float,
        qy: float,
        cell_registry: dict,
    ) -> dict:
        """
        Return a result dict:
            {
                "neighbours": [(cell_id, distance, weight), ...],  # k entries
                "species_vote": str,          # plurality species name
                "blended_colour": (r, g, b),  # IDW-weighted colour
            }

        Inverse-distance weighting: w_i = 1 / max(dist_i, 1e-3).
        Ties in species vote are broken by total IDW weight per species.

        鲁迅式：投票是多数人的暴政——但在像素世界里，多数才能产生视觉上的和谐。
        """
        candidates = self._candidate_distances(qx, qy, cell_registry, exclude_id=query_id)
        top_k = candidates[: self.k]

        if not top_k:
            _dbg_sp(f"NeighborProbe.query cell={query_id}: no candidates")
            return {
                "neighbours": [],
                "species_vote": "",
                "blended_colour": (0.5, 0.5, 0.5),
            }

        weights     = [1.0 / max(d, 1e-3) for d, _, __ in top_k]
        colours     = [e.get("colour", (0.5, 0.5, 0.5)) for _, __, e in top_k]
        neighbour_list = [
            (cid, dist, w) for (dist, cid, _e), w in zip(top_k, weights)
        ]

        # Species plurality vote weighted by IDW
        species_weights: dict = {}
        for (dist, cid, entry), w in zip(top_k, weights):
            sp = entry.get("species", "")
            species_weights[sp] = species_weights.get(sp, 0.0) + w
        species_vote = max(species_weights, key=species_weights.get) if species_weights else ""

        blended = _sp_weighted_colour(colours, weights)

        _dbg_sp(
            f"NeighborProbe.query cell={query_id} "
            f"k={len(top_k)} species_vote={species_vote!r} "
            f"blended={blended}"
        )
        return {
            "neighbours": neighbour_list,
            "species_vote": species_vote,
            "blended_colour": blended,
        }


class StyleDiffusion:
    """
    [ASTRO-STYLEPROBE] Iterative style-smoothing diffusion over a cell graph.

    Each diffusion step blends every cell's colour toward its k-NN weighted
    average by factor `alpha`.  Running multiple steps spreads smooth style
    gradients across large regions — analogous to a graph Laplacian smooth
    used in texture synthesis / mesh parameterisation.

    鲁迅式：扩散是温柔的同化——它不强迫任何细胞改变，只是让改变在不知不觉中发生。
    """

    def __init__(
        self,
        k: int = _SP_DEFAULT_K,
        alpha: float = _SP_DIFFUSION_ALPHA,
        steps: int = _SP_DIFFUSION_STEPS,
    ) -> None:
        self.probe = NeighborProbe(k=k)
        self.alpha = max(0.0, min(1.0, alpha))
        self.steps = max(1, steps)
        _dbg_sp(
            f"StyleDiffusion init | k={k} alpha={alpha} steps={steps}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lerp_colour(self, a: tuple, b: tuple, t: float) -> tuple:
        """Linear interpolation between two (r,g,b) tuples."""
        t = max(0.0, min(1.0, t))
        return (
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
        )

    def _build_registry_snapshot(self, cell_registry: dict) -> dict:
        """Deep-copy colour fields for a diffusion step snapshot."""
        return {
            cid: dict(entry) for cid, entry in cell_registry.items()
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def diffuse(self, cell_registry: dict) -> dict:
        """
        Run `self.steps` diffusion iterations over `cell_registry`.
        Returns a *new* dict with updated "colour" fields; input is not
        mutated.

        cell_registry schema::

            {
                cell_id: {
                    "x": float, "y": float, "w": float, "h": float,
                    "species": str,
                    "colour": (r, g, b),
                }, ...
            }

        鲁迅式：迭代是改良主义——每一步只改变一点，但走得足够远，面目全非。
        """
        current = self._build_registry_snapshot(cell_registry)

        for step in range(self.steps):
            next_reg = self._build_registry_snapshot(current)
            _dbg_sp(f"StyleDiffusion step {step + 1}/{self.steps}")

            for cid, entry in current.items():
                qx = entry.get("x", 0.0) + entry.get("w", 0.0) * 0.5
                qy = entry.get("y", 0.0) + entry.get("h", 0.0) * 0.5
                result = self.probe.query(cid, qx, qy, current)
                nbr_colour = result["blended_colour"]
                own_colour = entry.get("colour", (0.5, 0.5, 0.5))
                new_colour = self._lerp_colour(own_colour, nbr_colour, self.alpha)
                next_reg[cid]["colour"] = new_colour

                _dbg_sp(
                    f"  cell={cid} own={own_colour} nbr={nbr_colour} "
                    f"→ {new_colour}"
                )

            current = next_reg

        _dbg_sp(
            f"StyleDiffusion.diffuse complete | "
            f"{len(current)} cells, {self.steps} steps"
        )
        return current

    def diffuse_single(
        self,
        target_id: str,
        cell_registry: dict,
    ) -> tuple:
        """
        Run diffusion only for `target_id` (one-shot, no iteration).
        Useful for on-demand re-probe of a single updated cell.

        鲁迅式：单点扩散是外科手术——精准但不彻底，适合更新而非重建。
        """
        entry = cell_registry.get(target_id, {})
        qx = entry.get("x", 0.0) + entry.get("w", 0.0) * 0.5
        qy = entry.get("y", 0.0) + entry.get("h", 0.0) * 0.5
        result = self.probe.query(target_id, qx, qy, cell_registry)
        own_colour = entry.get("colour", (0.5, 0.5, 0.5))
        new_colour = self._lerp_colour(own_colour, result["blended_colour"], self.alpha)
        _dbg_sp(
            f"StyleDiffusion.diffuse_single cell={target_id} "
            f"→ {new_colour}"
        )
        return new_colour


class ProbeCache:
    """
    [ASTRO-STYLEPROBE] LRU-style probe result cache.

    Caches NeighborProbe.query() results keyed by (cell_id, registry_version).
    Avoids redundant k-NN scans when the cell registry hasn't changed between
    successive calls (e.g. multiple style adjustments for the same frame).

    The cache is bounded to _SP_CACHE_MAX_SIZE entries; oldest entries are
    evicted on overflow (FIFO approximation — sufficient for per-frame use).

    鲁迅式：缓存是记忆的偷懒——能不重算就不重算，但要记得什么时候记忆已经过时。
    """

    def __init__(self, max_size: int = _SP_CACHE_MAX_SIZE) -> None:
        self.max_size  = max(1, max_size)
        self._store: dict  = {}            # key → result
        self._order: list  = []            # insertion-order keys for FIFO eviction
        self._hits:  int   = 0
        self._misses: int  = 0
        _dbg_sp(f"ProbeCache init | max_size={max_size}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_key(self, cell_id: str, registry_version: Any) -> str:
        """Produce a stable string cache key."""
        return f"{cell_id}::{registry_version}"

    def _evict_if_needed(self) -> None:
        """Evict the oldest entry when the cache is at capacity."""
        while len(self._store) >= self.max_size and self._order:
            oldest = self._order.pop(0)
            self._store.pop(oldest, None)
            _dbg_sp(f"ProbeCache evict key={oldest!r}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        cell_id: str,
        registry_version: Any,
    ) -> Optional[dict]:
        """
        Return cached probe result or None on miss.

        鲁迅式：命中是侥幸，缺失是常态——但每次命中都省下一次远征。
        """
        key = self._make_key(cell_id, registry_version)
        result = self._store.get(key)
        if result is not None:
            self._hits += 1
            _dbg_sp(
                f"ProbeCache HIT cell={cell_id} ver={registry_version} "
                f"(hits={self._hits} misses={self._misses})"
            )
        else:
            self._misses += 1
            _dbg_sp(
                f"ProbeCache MISS cell={cell_id} ver={registry_version} "
                f"(hits={self._hits} misses={self._misses})"
            )
        return result

    def put(
        self,
        cell_id: str,
        registry_version: Any,
        result: dict,
    ) -> None:
        """
        Store a probe result.  Evicts oldest entry when at capacity.

        鲁迅式：存储是承诺——承诺记住，但不保证永远记住。
        """
        key = self._make_key(cell_id, registry_version)
        if key not in self._store:
            self._evict_if_needed()
            self._order.append(key)
        self._store[key] = result
        _dbg_sp(
            f"ProbeCache PUT cell={cell_id} ver={registry_version} "
            f"size={len(self._store)}"
        )

    def invalidate(self, cell_id: Optional[str] = None) -> int:
        """
        Invalidate all entries for `cell_id`, or the entire cache if None.
        Returns the number of entries removed.

        鲁迅式：失效是诚实——当真相改变，旧的记忆必须清除，哪怕代价高昂。
        """
        if cell_id is None:
            count = len(self._store)
            self._store.clear()
            self._order.clear()
            _dbg_sp(f"ProbeCache invalidate ALL ({count} entries cleared)")
            return count

        prefix = f"{cell_id}::"
        keys_to_remove = [k for k in list(self._store) if k.startswith(prefix)]
        for k in keys_to_remove:
            del self._store[k]
            if k in self._order:
                self._order.remove(k)
        _dbg_sp(
            f"ProbeCache invalidate cell={cell_id}: {len(keys_to_remove)} removed"
        )
        return len(keys_to_remove)

    def query_cached(
        self,
        probe: "NeighborProbe",
        cell_id: str,
        qx: float,
        qy: float,
        cell_registry: dict,
        registry_version: Any,
    ) -> dict:
        """
        Convenience wrapper: return cached result or run probe.query() and cache it.

        鲁迅式：便捷方法是懒惰的礼物——把两件事合并成一件，让调用者少费心。
        """
        cached = self.get(cell_id, registry_version)
        if cached is not None:
            return cached
        result = probe.query(cell_id, qx, qy, cell_registry)
        self.put(cell_id, registry_version, result)
        return result

    @property
    def stats(self) -> dict:
        """Return hit/miss statistics."""
        total = self._hits + self._misses
        rate  = self._hits / total if total else 0.0
        return {
            "hits":       self._hits,
            "misses":     self._misses,
            "total":      total,
            "hit_rate":   rate,
            "cache_size": len(self._store),
            "max_size":   self.max_size,
        }
