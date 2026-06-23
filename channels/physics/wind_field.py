"""
channels/physics/wind_field.py
Wind-field simulation for the cell pub/sub dynamic world.

Computes a 2-D wind velocity at any query point by superimposing four layers:

  1. **Global wind** — uniform directional flow (prevailing breeze).
  2. **Fan emitters** — each cell acts as a radial wind source whose strength
     scales with species buoyancy (from species_physics.json).  High-buoyancy
     cells push harder; heavy cells barely exhale.
  3. **Vortex cores** — placed at community centroids (from cell_groups.json).
     Each vortex imparts a tangential swirl that decays with distance.
  4. **Perlin noise** — two-octave fBm perturbation for organic variation.
     Time parameter lets the field evolve smoothly across epochs.

All outputs are JSON-serialisable dicts of {vx, vy} pairs ready for the
front-end PixiJS particle advection pipeline (no SVG generation).

Upstream references:
  - lygia generative/snoise  (Perlin basis)
  - webgl2-particles          (transform-feedback advection)
  - 3d-force-graph d3-force   (radial decay model)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Perlin noise (CPU-side, matches lygia snoise2 semantics) ───────────────

_PERM = list(range(256))
# Fisher-Yates with fixed seed for reproducibility across epochs.
_rng_state = 42
for i in range(255, 0, -1):
    _rng_state = (_rng_state * 1103515245 + 12345) & 0x7FFFFFFF
    j = _rng_state % (i + 1)
    _PERM[i], _PERM[j] = _PERM[j], _PERM[i]
_PERM *= 2  # double to avoid modulo in lookups

_GRAD2 = [
    (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
    (1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0),
]


def _fade(t: float) -> float:
    """Quintic fade: 6t⁵ − 15t⁴ + 10t³ (Ken Perlin, SIGGRAPH 2002)."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def _dot_grad(ix: int, iy: int, dx: float, dy: float) -> float:
    g = _GRAD2[_PERM[_PERM[ix & 255] + (iy & 255)] % 8]
    return g[0] * dx + g[1] * dy


def perlin2(x: float, y: float) -> float:
    """Classic 2-D Perlin noise in [−1, 1]."""
    xi, yi = int(math.floor(x)), int(math.floor(y))
    xf, yf = x - xi, y - yi
    u, v = _fade(xf), _fade(yf)
    n00 = _dot_grad(xi, yi, xf, yf)
    n10 = _dot_grad(xi + 1, yi, xf - 1.0, yf)
    n01 = _dot_grad(xi, yi + 1, xf, yf - 1.0)
    n11 = _dot_grad(xi + 1, yi + 1, xf - 1.0, yf - 1.0)
    return _lerp(_lerp(n00, n10, u), _lerp(n01, n11, u), v)


def fbm2(x: float, y: float, octaves: int = 2,
         lacunarity: float = 2.0, gain: float = 0.5) -> float:
    """Fractal Brownian Motion built on perlin2."""
    value = 0.0
    amp = 1.0
    freq = 1.0
    for _ in range(octaves):
        value += amp * perlin2(x * freq, y * freq)
        freq *= lacunarity
        amp *= gain
    return value


# ── Data loading helpers ───────────────────────────────────────────────────

_PHYSICS_DIR = Path(__file__).parent


def _load_json(name: str) -> dict[str, Any]:
    path = _PHYSICS_DIR / name
    if path.exists():
        return json.loads(path.read_text())
    return {}


# ── Wind-field configuration ──────────────────────────────────────────────

@dataclass
class WindFieldConfig:
    """Tuneable knobs — safe defaults produce a gentle south-east breeze."""
    # Global uniform wind
    global_angle: float = 2.35   # radians (≈ 135°, blows toward bottom-right)
    global_strength: float = 8.0

    # Fan (cell radial emission)
    fan_radius: float = 200.0    # influence radius in px
    fan_base_strength: float = 12.0

    # Vortex (community centroid swirl)
    vortex_radius: float = 250.0
    vortex_strength: float = 18.0

    # Perlin noise perturbation
    noise_scale: float = 0.006   # spatial frequency (lower = larger swirls)
    noise_strength: float = 5.0
    noise_time_scale: float = 0.1


# ── Core wind-field evaluator ─────────────────────────────────────────────

class WindField:
    """Evaluates a composite wind vector at any (x, y) world position."""

    def __init__(self, cfg: WindFieldConfig | None = None):
        self.cfg = cfg or WindFieldConfig()
        self._cells: dict[str, dict[str, Any]] = {}
        self._species_phys: dict[str, dict[str, float]] = {}
        self._vortices: list[tuple[float, float]] = []
        self._reload()

    # ── data ingestion ────────────────────────────────────────────────────

    def _reload(self) -> None:
        registry = _load_json("cell_registry.json")
        self._cells = registry.get("cells", {})
        self._species_phys = _load_json("species_physics.json")
        groups = _load_json("cell_groups.json")
        self._vortices = []
        for g in groups.get("groups", {}).values():
            c = g.get("centroid", {})
            if "x" in c and "y" in c:
                self._vortices.append((float(c["x"]), float(c["y"])))

    # ── layer: global wind ────────────────────────────────────────────────

    def _global(self) -> tuple[float, float]:
        s = self.cfg.global_strength
        return s * math.cos(self.cfg.global_angle), s * math.sin(self.cfg.global_angle)

    # ── layer: fan emitters (cells as radial wind sources) ────────────────

    def _fan(self, qx: float, qy: float) -> tuple[float, float]:
        vx, vy = 0.0, 0.0
        r_max = self.cfg.fan_radius
        base = self.cfg.fan_base_strength
        for cid, info in self._cells.items():
            bbox = info.get("bbox", {})
            mn, mx = bbox.get("min", [0, 0, 0]), bbox.get("max", [0, 0, 0])
            cx = (mn[0] + mx[0]) * 0.5
            cy = (mn[1] + mx[1]) * 0.5
            dx, dy = qx - cx, qy - cy
            dist = math.hypot(dx, dy)
            if dist < 1.0 or dist > r_max:
                continue
            species = info.get("species", "")
            buoyancy = self._species_phys.get(species, {}).get("buoyancy", 0.5)
            strength = base * buoyancy * (1.0 - dist / r_max) ** 2
            inv = strength / dist
            vx += dx * inv
            vy += dy * inv
        return vx, vy

    # ── layer: vortex swirl at community centroids ────────────────────────

    def _vortex(self, qx: float, qy: float) -> tuple[float, float]:
        vx, vy = 0.0, 0.0
        r_max = self.cfg.vortex_radius
        s = self.cfg.vortex_strength
        for cx, cy in self._vortices:
            dx, dy = qx - cx, qy - cy
            dist = math.hypot(dx, dy)
            if dist < 1.0 or dist > r_max:
                continue
            t = dist / r_max
            decay = t * math.exp(1.0 - t)          # peaks at t=1, smooth falloff
            inv = s * decay / dist
            vx += -dy * inv                         # tangential: perpendicular to radial
            vy += dx * inv
        return vx, vy

    # ── layer: Perlin noise perturbation ──────────────────────────────────

    def _noise(self, qx: float, qy: float, time: float) -> tuple[float, float]:
        sc = self.cfg.noise_scale
        ts = self.cfg.noise_time_scale
        s = self.cfg.noise_strength
        nx = fbm2(qx * sc + 31.7, qy * sc + time * ts)
        ny = fbm2(qx * sc + time * ts, qy * sc + 73.3)
        return nx * s, ny * s

    # ── public: sample wind at a point ────────────────────────────────────

    def sample(self, x: float, y: float, time: float = 0.0) -> dict[str, float]:
        """Return {vx, vy} wind velocity at world position (x, y)."""
        gx, gy = self._global()
        fx, fy = self._fan(x, y)
        wx, wy = self._vortex(x, y)
        nx, ny = self._noise(x, y, time)
        return {
            "vx": round(gx + fx + wx + nx, 4),
            "vy": round(gy + fy + wy + ny, 4),
        }

    # ── public: batch-sample a grid for the front-end texture ─────────────

    def sample_grid(self, x0: float, y0: float, x1: float, y1: float,
                    cols: int = 16, rows: int = 16,
                    time: float = 0.0) -> dict[str, Any]:
        """Return a flat grid of wind samples for GPU texture upload."""
        dx = (x1 - x0) / max(cols - 1, 1)
        dy = (y1 - y0) / max(rows - 1, 1)
        data: list[dict[str, float]] = []
        for r in range(rows):
            for c in range(cols):
                data.append(self.sample(x0 + c * dx, y0 + r * dy, time))
        return {"x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "cols": cols, "rows": rows, "samples": data}

    # ── public: per-cell wind vectors (for particle advection per cell) ───

    def cell_winds(self, time: float = 0.0) -> dict[str, dict[str, float]]:
        """Return wind vector at each cell's centre."""
        result: dict[str, dict[str, float]] = {}
        for cid, info in self._cells.items():
            bbox = info.get("bbox", {})
            mn, mx = bbox.get("min", [0, 0, 0]), bbox.get("max", [0, 0, 0])
            cx = (mn[0] + mx[0]) * 0.5
            cy = (mn[1] + mx[1]) * 0.5
            result[cid] = self.sample(cx, cy, time)
        return result


# ── CLI: dump wind_field.json ──────────────────────────────────────────────

def export(time: float = 0.0) -> dict[str, Any]:
    """Build the full wind-field snapshot for serialisation."""
    wf = WindField()
    return {
        "config": {
            "global_angle": wf.cfg.global_angle,
            "global_strength": wf.cfg.global_strength,
            "fan_radius": wf.cfg.fan_radius,
            "vortex_radius": wf.cfg.vortex_radius,
            "noise_scale": wf.cfg.noise_scale,
        },
        "time": time,
        "cell_winds": wf.cell_winds(time),
        "grid": wf.sample_grid(180, 0, 420, 760, cols=12, rows=16, time=time),
    }


if __name__ == "__main__":
    snapshot = export()
    out = _PHYSICS_DIR / "wind_field.json"
    out.write_text(json.dumps(snapshot, indent=2))
    print(f"Wrote {out}  ({len(snapshot['cell_winds'])} cells)")
    for cid, w in snapshot["cell_winds"].items():
        print(f"  {cid:<14} vx={w['vx']:+8.3f}  vy={w['vy']:+8.3f}")
