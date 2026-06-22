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

  register_physics_notifiers()
      Registers DataNotifier callbacks for
        • physics/force_field.json  →  physics_step
        • physics/collision.json    →  physics_collision
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

    logger.info(
        "[sse_physics_bridge] Registered physics_step + physics_collision "
        "DataNotifier callbacks"
    )
    return True
