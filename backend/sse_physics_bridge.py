"""
backend/sse_physics_bridge.py
─────────────────────────────
Thin bridge that exposes the physics SSE broadcast helpers defined in
server.py so that external modules (physics engine, tests, CLI tools)
can import them without importing the full FastAPI application.

Public API
──────────
  broadcast_physics_step(epoch, force_field, converged)
      Fires a ``physics_step`` SSE event on the shared _cell_event_queues.

  broadcast_physics_collision(collisions, count)
      Fires a ``physics_collision`` SSE event on the shared _cell_event_queues.

  broadcast_composite_params(data=None)
      Fires a ``composite_params`` SSE event containing the full 58-cell
      composite_params payload (cells, edges, canvas, palette).
      If data is None the function reads the latest composite_params.json
      from disk (channels/output/composite_params.json preferred,
      channels/composite_params.json as fallback) before broadcasting.

  register_physics_notifiers()
      Registers DataNotifier callbacks for
        • physics/force_field.json  →  physics_step
        • physics/collision.json    →  physics_collision
        • composite_params.json     →  composite_params   (both root + output/)
      Call this at server startup (already done by server._register_cell_sse_notifier,
      but provided here for testing and for any stand-alone physics process that
      wants to drive SSE without importing the full server module).

  load_qos() → dict
      Read channels/physics/qos.json and return the merged QoS config
      (with defaults filled in for missing keys).

SSE event schemas
─────────────────
physics_step
  {
    "epoch":       <int>    current physics epoch
    "force_field": <dict>   per-cell {dx, dy, dz} displacement vectors
    "converged":   <bool>   whether the layout has converged
    "timestamp":   <str>    ISO-8601
  }

physics_collision
  {
    "collisions":  <list>   collision records from physics/collision.json
    "count":       <int>    number of active collisions
    "timestamp":   <str>    ISO-8601
  }

composite_params
  {
    "cells":      <list>   58 cell descriptors (id, x, y, width, height,
                           species_params, parent_id, children_ids,
                           render_order, z_layer, is_translucent,
                           outgoing_edge_count)
    "edges":      <list>   edge route descriptors (source, target, route_points)
    "canvas":     <dict>   {width, height}
    "palette":    <dict>   {zenith, horizon, nadir}
    "cell_count": <int>    convenience count == len(cells)
    "timestamp":  <str>    ISO-8601
  }
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Root of the repository (two levels up from this file)
_REPO_ROOT = Path(__file__).resolve().parent.parent
_CHANNELS_DIR = _REPO_ROOT / "channels"

# ── QoS defaults ─────────────────────────────────────────────────────────────
_QOS_DEFAULTS: dict = {
    "step_interval_ms": 100,
    "collision_min_delta": 0,
    "max_queue_depth": 256,
    "enabled": True,
}

# Last physics_step broadcast timestamp (per-thread safe via lock)
_last_step_ts: float = 0.0
_last_step_lock = threading.Lock()

# Last known collision count (for collision_min_delta gate)
_last_collision_count: int = -1
_last_collision_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# QoS helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_qos() -> dict:
    """
    Read channels/physics/qos.json and return the merged config dict.
    Missing keys are filled from _QOS_DEFAULTS; file absence is not an error.
    """
    qos_path = _CHANNELS_DIR / "physics" / "qos.json"
    cfg: dict = {}
    if qos_path.exists():
        try:
            cfg = json.loads(qos_path.read_text())
        except Exception as exc:
            logger.warning("[sse_physics_bridge] Failed to read qos.json: %s", exc)
    return {**_QOS_DEFAULTS, **cfg}


# ─────────────────────────────────────────────────────────────────────────────
# Broadcast helpers
# These delegate to server._sse_broadcast so both entry points share the same
# asyncio.Queue list.  We import lazily to avoid circular imports at module
# load time.
# ─────────────────────────────────────────────────────────────────────────────

def _sse_broadcast(event: str, data: dict) -> None:
    """Delegate to server._sse_broadcast (lazy import)."""
    try:
        import server as _srv  # noqa: PLC0415 — intentional lazy import
        _srv._sse_broadcast(event, data)
    except Exception as exc:
        logger.debug("[sse_physics_bridge] _sse_broadcast delegate failed: %s", exc)


def broadcast_physics_step(
    epoch: int,
    force_field: dict,
    converged: bool,
) -> bool:
    """
    Broadcast a ``physics_step`` SSE event, subject to QoS rate-limiting.

    Returns True if the event was actually broadcast, False if it was
    throttled by step_interval_ms or suppressed by enabled=false.
    """
    import time

    qos = load_qos()
    if not qos.get("enabled", True):
        return False

    interval_s = qos.get("step_interval_ms", 100) / 1000.0

    with _last_step_lock:
        global _last_step_ts
        now = time.monotonic()
        if now - _last_step_ts < interval_s:
            return False
        _last_step_ts = now

    _sse_broadcast("physics_step", {
        "epoch": epoch,
        "force_field": force_field,
        "converged": converged,
        "timestamp": datetime.now().isoformat(),
    })
    return True


def broadcast_physics_collision(
    collisions: list,
    count: int,
) -> bool:
    """
    Broadcast a ``physics_collision`` SSE event, subject to QoS gating.

    The event is skipped if the change in collision count is smaller than
    collision_min_delta (default 0 = always fire).

    Returns True if the event was actually broadcast.
    """
    qos = load_qos()
    if not qos.get("enabled", True):
        return False

    min_delta = qos.get("collision_min_delta", 0)

    with _last_collision_lock:
        global _last_collision_count
        delta = abs(count - _last_collision_count)
        if _last_collision_count >= 0 and delta < min_delta:
            return False
        _last_collision_count = count

    _sse_broadcast("physics_collision", {
        "collisions": collisions,
        "count": count,
        "timestamp": datetime.now().isoformat(),
    })
    return True


# ─────────────────────────────────────────────────────────────────────────────
# DataNotifier registration
# ─────────────────────────────────────────────────────────────────────────────

def register_physics_notifiers() -> bool:
    """
    Register DataNotifier callbacks for physics channel files.

    Returns True on success, False if DataNotifier is unavailable
    (e.g. during unit tests that don't have channels/ on sys.path).

    Note: server._register_cell_sse_notifier() already calls the equivalent
    inline callbacks on startup.  Call this function only when you need to
    register from outside the FastAPI server process.
    """
    import sys

    channels_str = str(_CHANNELS_DIR)
    if channels_str not in sys.path:
        sys.path.insert(0, channels_str)

    try:
        from channels.data.notifier import DataNotifier, Notifier  # type: ignore[import]
    except ImportError:
        try:
            from data.notifier import DataNotifier, Notifier  # type: ignore[import]
        except ImportError as exc:
            logger.warning(
                "[sse_physics_bridge] DataNotifier not available: %s", exc
            )
            return False

    notifier = DataNotifier.instance()

    # ── physics/force_field.json → physics_step ───────────────────────────
    def _step_cb() -> None:
        ff_path = _CHANNELS_DIR / "physics" / "force_field.json"
        conv_path = _CHANNELS_DIR / "physics" / "converged.json"
        epoch_path = _CHANNELS_DIR / "skeleton" / "epoch.json"
        try:
            force_field = json.loads(ff_path.read_text()) if ff_path.exists() else {}
        except Exception:
            force_field = {}
        try:
            conv_data = json.loads(conv_path.read_text()) if conv_path.exists() else {}
            converged = bool(conv_data.get("converged", False))
        except Exception:
            converged = False
        try:
            ep_data = json.loads(epoch_path.read_text()) if epoch_path.exists() else {}
            epoch_num = ep_data.get("current", -1)
        except Exception:
            epoch_num = -1
        broadcast_physics_step(epoch_num, force_field, converged)

    notifier.add_notifier("physics/force_field.json", Notifier(_step_cb))

    # ── physics/collision.json → physics_collision ────────────────────────
    def _collision_cb() -> None:
        coll_path = _CHANNELS_DIR / "physics" / "collision.json"
        try:
            coll_data = json.loads(coll_path.read_text()) if coll_path.exists() else {}
            collisions = coll_data.get("collisions", [])
            count = coll_data.get("count", len(collisions))
        except Exception:
            collisions = []
            count = 0
        broadcast_physics_collision(collisions, count)

    notifier.add_notifier("physics/collision.json", Notifier(_collision_cb))

    # ── composite_params.json → composite_params ──────────────────────────
    def _composite_cb() -> None:
        data = _load_composite_params()
        if data is not None:
            broadcast_composite_params(data)

    # Watch both canonical locations: channels/ root and channels/output/
    notifier.add_notifier("composite_params.json", Notifier(_composite_cb))
    notifier.add_notifier("output/composite_params.json", Notifier(_composite_cb))

    logger.info(
        "[sse_physics_bridge] Registered physics_step + physics_collision "
        "+ composite_params DataNotifier callbacks"
    )
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Composite params helpers
# ─────────────────────────────────────────────────────────────────────────────

# Candidate paths for composite_params.json, in priority order.
# 1. channels/output/composite_params.json  — written by assemble_composite_params()
# 2. channels/composite_params.json         — static / pre-seeded fallback
_COMPOSITE_PARAMS_CANDIDATES = [
    _CHANNELS_DIR / "output" / "composite_params.json",
    _CHANNELS_DIR / "composite_params.json",
]


def _load_composite_params() -> dict | None:
    """
    Read the most recent composite_params.json (58 cells).

    Tries candidates in priority order and returns the first valid JSON dict,
    or None if none exist / all fail to parse.
    """
    for path in _COMPOSITE_PARAMS_CANDIDATES:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                cell_count = len(data.get("cells", []))
                logger.debug(
                    "[sse_physics_bridge] Loaded composite_params from %s (%d cells)",
                    path,
                    cell_count,
                )
                return data
            except Exception as exc:
                logger.warning(
                    "[sse_physics_bridge] Failed to parse %s: %s", path, exc
                )
    return None


def broadcast_composite_params(data: dict | None = None) -> bool:
    """
    Broadcast a ``composite_params`` SSE event carrying the full 58-cell
    composite_params payload.

    Parameters
    ----------
    data : dict | None
        If provided, broadcast this dict directly.  If None, the function
        reads the latest composite_params.json from disk before broadcasting.

    Returns True if the event was dispatched, False if no data was available.
    """
    if data is None:
        data = _load_composite_params()
    if data is None:
        logger.warning("[sse_physics_bridge] broadcast_composite_params: no data available")
        return False

    cell_count = len(data.get("cells", []))
    _sse_broadcast("composite_params", {
        "cells":     data.get("cells", []),
        "edges":     data.get("edges", []),
        "canvas":    data.get("canvas", {}),
        "palette":   data.get("palette", {}),
        "cell_count": cell_count,
        "timestamp": datetime.now().isoformat(),
    })
    logger.info(
        "[sse_physics_bridge] broadcast_composite_params: dispatched %d cells", cell_count
    )
    return True
