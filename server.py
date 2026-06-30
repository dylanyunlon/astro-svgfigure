from __future__ import annotations

import json
import os
import queue
import shutil
import signal
import socket
import subprocess
import threading
import time
import uuid
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import asyncio
import logging

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Pipeline imports (Phase 1 backend modules)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backend.config import get_settings
from backend.ai_engine import AIEngine
from backend.schemas import (
    TopologyRequest,
    BeautifyRequest,
    ValidateRequest,
    TopologyResponse,
    BeautifyResponse,
)
from backend.pipeline.topology_gen import generate_topology
from backend.pipeline.scaffold_builder import build_scaffold
from backend.pipeline.nanobanana_bridge import beautify_with_nanobanana
from backend.pipeline.svg_validator import validate_svg as validate_svg_func

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

# ── Cell SSE pub/sub — DataNotifier → browser ────────────────────────────────
# One asyncio.Queue per connected SSE client.  When DataNotifier fires a
# callback we put the event onto every live queue so every browser tab gets it.
#
# Queue items are dicts with an "event" key (the SSE event name) and a "data"
# key (the JSON-serialisable payload).  The event_stream() generator reads
# "event" to emit the correct `event:` line in the SSE frame.
_cell_event_queues: list[asyncio.Queue] = []
_cell_event_queues_lock = threading.Lock()


def _sse_broadcast(event: str, data: dict) -> None:
    """
    Thread-safe push of an arbitrary SSE event to every connected client.

    Parameters
    ----------
    event : str
        SSE event name (e.g. "cell_update", "topology_updated", "epoch_completed").
    data : dict
        JSON-serialisable payload.
    """
    envelope = {"event": event, "data": data}
    with _cell_event_queues_lock:
        for q in _cell_event_queues:
            try:
                q.put_nowait(envelope)
            except asyncio.QueueFull:
                pass  # slow client — drop rather than block


# ── Typed broadcast helpers ──────────────────────────────────────────────────

def _cell_broadcast(cell_id: str, params: dict) -> None:
    """Backward-compat: broadcasts a cell_update event (also used by DataNotifier)."""
    _sse_broadcast("cell_update", {"cell_id": cell_id, "params": params})


def _broadcast_cell_params_updated(cell_id: str, params: dict) -> None:
    """Broadcast cell_params_updated — fired when a cell's params.json is written."""
    _sse_broadcast("cell_params_updated", {"cell_id": cell_id, "params": params})


def _broadcast_topology_updated(topology: dict) -> None:
    """Broadcast topology_updated — fired when skeleton/topology.json changes."""
    _sse_broadcast("topology_updated", {"topology": topology})


def _broadcast_epoch_completed(epoch: int, metrics: dict) -> None:
    """Broadcast epoch_completed — fired at the end of each epoch in the loop."""
    _sse_broadcast("epoch_completed", {"epoch": epoch, **metrics})


def _broadcast_cell_loop_started(max_epochs: int) -> None:
    """Broadcast cell_loop_started — fired when run_loop() begins."""
    _sse_broadcast("cell_loop_started", {
        "max_epochs": max_epochs,
        "timestamp": datetime.now().isoformat(),
    })


def _broadcast_cell_loop_finished(final_epoch: int, converged: bool, cells: dict) -> None:
    """Broadcast cell_loop_finished — fired when run_loop() ends."""
    _sse_broadcast("cell_loop_finished", {
        "final_epoch": final_epoch,
        "converged": converged,
        "cell_count": len(cells),
        "timestamp": datetime.now().isoformat(),
    })


def _broadcast_physics_step(epoch: int, force_field: dict, converged: bool) -> None:
    """
    Broadcast physics_step — fired each time the physics engine writes
    channels/physics/force_field.json (one iteration of the force-directed
    layout loop).

    Parameters
    ----------
    epoch       : current physics epoch index
    force_field : per-cell {dx, dy, dz} displacement vectors
    converged   : whether the engine has converged
    """
    _sse_broadcast("physics_step", {
        "epoch": epoch,
        "force_field": force_field,
        "converged": converged,
        "timestamp": datetime.now().isoformat(),
    })


def _broadcast_physics_collision(collisions: list, count: int) -> None:
    """
    Broadcast physics_collision — fired whenever channels/physics/collision.json
    is updated (a new collision pair is detected or the list is cleared).

    Parameters
    ----------
    collisions : list of collision records produced by the physics engine
    count      : total number of active collisions
    """
    _sse_broadcast("physics_collision", {
        "collisions": collisions,
        "count": count,
        "timestamp": datetime.now().isoformat(),
    })


def _broadcast_geometry_update(cell_id: str, geometry: dict) -> None:
    """
    Broadcast geometry_update — fired when a cell's geometry.json is updated
    by tick-runner or a live cell agent.  The front-end CellGeometryChannel
    ingests this to update the metaball SDF in real time.
    """
    _sse_broadcast("geometry_update", {
        "cell_id": cell_id,
        "geometry": geometry,
        "timestamp": datetime.now().isoformat(),
    })


def _register_cell_sse_notifier() -> None:
    """
    Register DataNotifier callbacks so that pipeline writes to channel files
    propagate to the SSE stream.  Called once at startup.

    Watched channels:
      - cell/{id}/params.json  → cell_update + cell_params_updated
      - skeleton/topology.json → topology_updated
      - skeleton/epoch.json    → epoch_completed
    """
    try:
        from channels.data.notifier import DataNotifier, Notifier

        notifier = DataNotifier.instance()
        cell_ids = [
            "add_norm1", "add_norm2", "ffn",
            "input_embed", "output", "pos_encode", "self_attn",
        ]

        # ── Per-cell params.json watchers ────────────────────────────────────
        for cell_id in cell_ids:
            def _make_callback(cid: str):
                def _cb():
                    params_path = BASE_DIR / "channels" / "cell" / cid / "params.json"
                    try:
                        params = json.loads(params_path.read_text())
                    except Exception:
                        params = {}
                    # Fire both event types so old and new listeners both work
                    _cell_broadcast(cid, params)
                    _broadcast_cell_params_updated(cid, params)
                return _cb

            channel_path = f"cell/{cell_id}/params.json"
            notifier.add_notifier(channel_path, Notifier(_make_callback(cell_id)))

        # ── Topology watcher ─────────────────────────────────────────────────
        def _topology_cb():
            topo_path = BASE_DIR / "channels" / "skeleton" / "topology.json"
            try:
                topology = json.loads(topo_path.read_text())
            except Exception:
                topology = {}
            _broadcast_topology_updated(topology)

        notifier.add_notifier("skeleton/topology.json", Notifier(_topology_cb))

        # ── Epoch watcher (epoch.json written by run_loop each iteration) ────
        def _epoch_cb():
            epoch_path = BASE_DIR / "channels" / "skeleton" / "epoch.json"
            try:
                epoch_data = json.loads(epoch_path.read_text())
            except Exception:
                epoch_data = {}
            epoch_num = epoch_data.get("current", -1)
            _broadcast_epoch_completed(epoch_num, epoch_data)

        notifier.add_notifier("skeleton/epoch.json", Notifier(_epoch_cb))

        # ── Physics force_field watcher (physics_step event) ─────────────
        def _physics_step_cb():
            ff_path = BASE_DIR / "channels" / "physics" / "force_field.json"
            conv_path = BASE_DIR / "channels" / "physics" / "converged.json"
            epoch_path2 = BASE_DIR / "channels" / "skeleton" / "epoch.json"
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
                ep_data = json.loads(epoch_path2.read_text()) if epoch_path2.exists() else {}
                epoch_num = ep_data.get("current", -1)
            except Exception:
                epoch_num = -1
            _broadcast_physics_step(epoch_num, force_field, converged)

        notifier.add_notifier("physics/force_field.json", Notifier(_physics_step_cb))

        # ── Physics collision watcher (physics_collision event) ───────────
        def _physics_collision_cb():
            coll_path = BASE_DIR / "channels" / "physics" / "collision.json"
            try:
                coll_data = json.loads(coll_path.read_text()) if coll_path.exists() else {}
                collisions = coll_data.get("collisions", [])
                count = coll_data.get("count", len(collisions))
            except Exception:
                collisions = []
                count = 0
            _broadcast_physics_collision(collisions, count)

        notifier.add_notifier("physics/collision.json", Notifier(_physics_collision_cb))

        # ── composite_params watcher (composite_params event) ─────────────
        def _composite_params_cb():
            from backend.sse_physics_bridge import _load_composite_params
            data = _load_composite_params()
            if data is not None:
                _sse_broadcast("composite_params", {
                    "cells":      data.get("cells", []),
                    "edges":      data.get("edges", []),
                    "canvas":     data.get("canvas", {}),
                    "palette":    data.get("palette", {}),
                    "cell_count": len(data.get("cells", [])),
                    "timestamp":  datetime.now().isoformat(),
                })

        notifier.add_notifier("composite_params.json",        Notifier(_composite_params_cb))
        notifier.add_notifier("output/composite_params.json", Notifier(_composite_params_cb))

        logger.info(
            "[cell-sse] DataNotifier callbacks registered for %d cells "
            "+ topology + epoch + physics_step + physics_collision "
            "+ composite_params watchers",
            len(cell_ids),
        )
    except Exception as exc:
        logger.warning("[cell-sse] Could not register DataNotifier callbacks: %s", exc)
WEB_DIR = BASE_DIR / "web"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

PYTHON_EXECUTABLE = os.environ.get("AUTOFIGURE_PYTHON") or sys.executable

DEFAULT_SAM_PROMPT = "icon,person,animal,robot"
DEFAULT_PLACEHOLDER_MODE = "label"
DEFAULT_MERGE_THRESHOLD = 0.01

SVG_EDIT_CANDIDATES = [
    ("vendor/svg-edit/editor/index.html", WEB_DIR / "vendor" / "svg-edit" / "editor" / "index.html"),
    ("vendor/svg-edit/editor.html", WEB_DIR / "vendor" / "svg-edit" / "editor.html"),
    ("vendor/svg-edit/index.html", WEB_DIR / "vendor" / "svg-edit" / "index.html"),
]


def _resolve_svg_edit_path() -> tuple[bool, str | None]:
    for rel, path in SVG_EDIT_CANDIDATES:
        if path.is_file():
            return True, f"/{rel}"
    return False, None


@dataclass
class Job:
    job_id: str
    output_dir: Path
    process: subprocess.Popen
    queue: queue.Queue
    log_path: Path
    log_lock: threading.Lock = field(default_factory=threading.Lock)
    seen: set[str] = field(default_factory=set)
    done: bool = False

    def push(self, event: str, data: dict) -> None:
        self.queue.put({"event": event, "data": data})

    def write_log(self, stream: str, line: str) -> None:
        with self.log_lock:
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(f"[{stream}] {line}\n")


class RunRequest(BaseModel):
    method_text: str = Field(..., min_length=1)
    provider: str = "bianxie"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    image_model: Optional[str] = None
    svg_model: Optional[str] = None
    sam_prompt: Optional[str] = None
    sam_backend: Optional[str] = None
    sam_api_key: Optional[str] = None
    sam_max_masks: Optional[int] = None
    placeholder_mode: Optional[str] = None
    merge_threshold: Optional[float] = None
    optimize_iterations: Optional[int] = None
    reference_image_path: Optional[str] = None


from contextlib import asynccontextmanager


@asynccontextmanager
async def _lifespan(application: FastAPI):
    """Register DataNotifier callbacks on startup, optionally start live loop."""
    _register_cell_sse_notifier()

    # ── Manager loop: auto-start when ASTRO_LIVE_AGENTS=1 ────────────────
    if os.environ.get("ASTRO_LIVE_AGENTS", "0") == "1":
        max_epochs = int(os.environ.get("ASTRO_MAX_EPOCHS", "10"))

        def _run_manager_loop():
            import importlib
            _channels_str = str(_CHANNELS_DIR)
            if _channels_str not in sys.path:
                sys.path.insert(0, _channels_str)
            # project root for 'channels.rendering...' imports
            _project_root = str(_CHANNELS_DIR.parent)
            if _project_root not in sys.path:
                sys.path.insert(0, _project_root)

            os.chdir(_channels_str)
            import loop_orchestrator
            importlib.reload(loop_orchestrator)

            logger.info(
                f"[Manager] Live loop started: max_epochs={max_epochs} "
                f"ASTRO_LIVE_AGENTS=1"
            )
            try:
                loop_orchestrator.run_loop(max_epochs=max_epochs)
                logger.info("[Manager] Loop converged")
            except Exception as e:
                logger.error(f"[Manager] Loop error: {e}")

        _manager_thread = threading.Thread(
            target=_run_manager_loop, daemon=True, name="manager-loop"
        )
        _manager_thread.start()
        logger.info(f"[Manager] Dispatching live loop in background (max_epochs={max_epochs})")

    yield


app = FastAPI(lifespan=_lifespan)

# ── CORS (allow Astro frontend at :4321 to call us) ───────────────────
_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.CORS_ORIGINS + ["http://localhost:4321", "http://127.0.0.1:4321"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── AI Engine singleton ───────────────────────────────────────────────
_ai_engine: AIEngine | None = None

def _get_ai_engine() -> AIEngine:
    global _ai_engine
    if _ai_engine is None:
        _ai_engine = AIEngine(_settings)
    return _ai_engine


# ============================================================================
# Cell SSE Endpoint — DataNotifier → Server-Sent Events → browser
# GET /api/cell-events streams real-time cell state to every connected client.
# ============================================================================

@app.get("/api/cell-events")
async def api_cell_events() -> StreamingResponse:
    """
    GET /api/cell-events  →  text/event-stream

    Pushes real-time updates to every connected browser tab.

    SSE event types
    ───────────────
      cell_update          — legacy per-cell state (backward compat)
      cell_params_updated  — cell params.json written
      topology_updated     — skeleton/topology.json changed
      epoch_completed      — one epoch of the convergence loop finished
      cell_loop_started    — run_loop() begins
      cell_loop_finished   — run_loop() ends (converged or max-epoch)
      physics_step         — physics engine wrote force_field.json; payload
                             includes epoch, per-cell {dx,dy,dz} vectors, and
                             converged flag
      physics_collision    — physics engine wrote collision.json; payload
                             includes collisions list and count
      composite_params     — full 58-cell world snapshot; payload includes
                             cells[], edges[], canvas, palette, cell_count

    A keepalive comment (": ping") is sent every 15 s to prevent proxy
    timeouts and to let the client detect a dropped connection quickly.

    Flow:
      pipeline writes a channel file
        → DataNotifier.notify() or explicit _broadcast_*() call
          → _sse_broadcast() puts envelope onto every asyncio.Queue
            → this generator reads the queue and streams SSE to browser
    """
    q: asyncio.Queue = asyncio.Queue(maxsize=256)
    with _cell_event_queues_lock:
        _cell_event_queues.append(q)

    async def _initial_snapshot():
        """Send current state for all cells + topology immediately on connect."""
        # ── Cell params snapshots ────────────────────────────────────────
        cell_dir = BASE_DIR / "channels" / "cell"
        for params_file in sorted(cell_dir.glob("*/params.json")):
            try:
                params = json.loads(params_file.read_text())
                cell_id = params_file.parent.name
                data = json.dumps({"cell_id": cell_id, "params": params}, ensure_ascii=True)
                yield f"event: cell_update\ndata: {data}\n\n"
            except Exception:
                pass

        # ── Topology snapshot ────────────────────────────────────────────
        topo_path = BASE_DIR / "channels" / "skeleton" / "topology.json"
        if topo_path.exists():
            try:
                topology = json.loads(topo_path.read_text())
                data = json.dumps({"topology": topology}, ensure_ascii=True)
                yield f"event: topology_updated\ndata: {data}\n\n"
            except Exception:
                pass

        # ── Current epoch snapshot ───────────────────────────────────────
        epoch_path = BASE_DIR / "channels" / "skeleton" / "epoch.json"
        if epoch_path.exists():
            try:
                epoch_data = json.loads(epoch_path.read_text())
                data = json.dumps(epoch_data, ensure_ascii=True)
                yield f"event: epoch_completed\ndata: {data}\n\n"
            except Exception:
                pass

        # ── composite_params snapshot (58-cell full world state) ─────────
        from backend.sse_physics_bridge import _load_composite_params as _lcp
        cp = _lcp()
        if cp is not None:
            try:
                payload = {
                    "cells":      cp.get("cells", []),
                    "edges":      cp.get("edges", []),
                    "canvas":     cp.get("canvas", {}),
                    "palette":    cp.get("palette", {}),
                    "cell_count": len(cp.get("cells", [])),
                    "timestamp":  datetime.now().isoformat(),
                }
                data = json.dumps(payload, ensure_ascii=True)
                yield f"event: composite_params\ndata: {data}\n\n"
            except Exception:
                pass

    async def event_stream():
        # 1. Initial snapshot so frontend hydrates without separate REST fetches
        async for chunk in _initial_snapshot():
            yield chunk

        # 2. Live updates pushed via _sse_broadcast()
        try:
            while True:
                try:
                    envelope = await asyncio.wait_for(q.get(), timeout=15.0)
                    event_name = envelope.get("event", "cell_update")
                    payload = envelope.get("data", envelope)
                    data = json.dumps(payload, ensure_ascii=True)
                    yield f"event: {event_name}\ndata: {data}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"   # keepalive — invisible to onmessage
        finally:
            with _cell_event_queues_lock:
                try:
                    _cell_event_queues.remove(q)
                except ValueError:
                    pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )


# ============================================================================
# Cell Pub/Sub Endpoint — Apollo CyberRT DataNotifier bridge
# Sub-Claudes POST params here instead of git push.
# Server writes to channels/ and fires DataNotifier callbacks.
# Also triggers _cell_broadcast() so SSE clients get the update instantly.
# ============================================================================

@app.post("/api/cell/publish")
async def api_cell_publish(request_data: dict) -> JSONResponse:
    """
    POST /api/cell/publish — Sub-Claude publishes agent_params to a cell channel.

    This replaces git push for the pub/sub loop. The sub-Claude in its
    claude.hk.cn VM POSTs here instead of pushing to git. The server:
      1. Writes channels/cell/{cell_id}/agent_params.json
      2. Fires DataNotifier.notify("cell/{cell_id}/agent_params.json")
      3. Any subscriber (other cells, physics engine) gets the callback

    This is the bridge between the sub-Claude's isolated VM and the
    Apollo CyberRT DataNotifier running in this Python process.

    Request body:
      {
        "cell_id": "self_attn",
        "agent_params": { "bbox": {...}, "opacity": 0.85, "species_params": {...} },
        "auth": "<token>"  // optional, for verification
      }
    """
    try:
        cell_id = request_data.get("cell_id", "").strip()
        if not cell_id:
            return JSONResponse({"error": "cell_id required"}, status_code=400)

        agent_params = request_data.get("agent_params", {})
        if not agent_params:
            return JSONResponse({"error": "agent_params required"}, status_code=400)

        # ── Write to channel filesystem ──────────────────────────────────────
        channels_dir = os.path.join(os.path.dirname(__file__), "channels")
        cell_dir = os.path.join(channels_dir, "cell", cell_id)
        os.makedirs(cell_dir, exist_ok=True)

        params_path = os.path.join(cell_dir, "agent_params.json")
        with open(params_path, "w") as f:
            json.dump(agent_params, f, indent=2)

        # ── Fire DataNotifier ────────────────────────────────────────────────
        # Import here to avoid circular imports at module level
        try:
            import sys
            sys.path.insert(0, channels_dir)
            from channel_runtime import DataNotifier
            notifier = DataNotifier.instance()
            channel_path = f"cell/{cell_id}/agent_params.json"
            notifier.notify(channel_path)
            notified = True
        except Exception as notify_exc:
            notified = False
            logger.warning(f"DataNotifier.notify failed: {notify_exc}")

        # ── Broadcast to SSE clients ─────────────────────────────────────────
        # agent_params is what the sub-Claude produced; also try to read the
        # canonical params.json if the pipeline has already merged it there.
        try:
            canonical_path = os.path.join(cell_dir, "params.json")
            if os.path.exists(canonical_path):
                sse_params = json.loads(open(canonical_path).read())
            else:
                sse_params = agent_params
            _cell_broadcast(cell_id, sse_params)
            _broadcast_cell_params_updated(cell_id, sse_params)
        except Exception as sse_exc:
            logger.warning(f"_cell_broadcast failed: {sse_exc}")

        logger.info(
            f"[cell/publish] cell_id={cell_id} "
            f"params_keys={list(agent_params.keys())} "
            f"notified={notified} sse_clients={len(_cell_event_queues)}"
        )

        return JSONResponse({
            "success": True,
            "cell_id": cell_id,
            "channel": f"cell/{cell_id}/agent_params.json",
            "notified": notified,
            "sse_clients": len(_cell_event_queues),
        })

    except Exception as e:
        logger.exception("api_cell_publish error")
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================================
# Cell Geometry Endpoint — tick-runner POSTs geometry.json updates here
# Writes to channels/cell/{id}/geometry.json and SSE-broadcasts to browser
# ============================================================================

@app.post("/api/cell/geometry")
async def api_cell_geometry(request_data: dict) -> JSONResponse:
    """
    POST /api/cell/geometry — tick-runner pushes geometry.json for a cell.

    Request body:
      {
        "cell_id": "input",
        "geometry": { ...full geometry.json contents... }
      }

    Or batch mode:
      {
        "cells": [
          {"cell_id": "input", "geometry": {...}},
          {"cell_id": "ffn", "geometry": {...}}
        ]
      }
    """
    try:
        channels_dir = os.path.join(os.path.dirname(__file__), "channels")
        batch = request_data.get("cells", [])
        if not batch:
            # single-cell mode
            cell_id = request_data.get("cell_id", "").strip()
            geometry = request_data.get("geometry", {})
            if not cell_id or not geometry:
                return JSONResponse({"error": "cell_id and geometry required"}, status_code=400)
            batch = [{"cell_id": cell_id, "geometry": geometry}]

        updated = []
        for item in batch:
            cid = item.get("cell_id", "").strip()
            geom = item.get("geometry", {})
            if not cid or not geom:
                continue

            # Write geometry.json to channel filesystem
            cell_dir = os.path.join(channels_dir, "cell", cid)
            if not os.path.isdir(cell_dir):
                continue
            geom_path = os.path.join(cell_dir, "geometry.json")
            with open(geom_path, "w") as f:
                json.dump(geom, f, separators=(",", ":"))

            # SSE broadcast to all connected browsers
            _broadcast_geometry_update(cid, geom)
            updated.append(cid)

        return JSONResponse({
            "success": True,
            "updated": updated,
            "count": len(updated),
            "sse_clients": len(_cell_event_queues),
        })

    except Exception as e:
        logger.exception("api_cell_geometry error")
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================================
# Physics QoS Endpoint — tune SSE event rate / priority for physics events
# POST /api/physics/qos lets callers throttle physics_step and
# physics_collision events so slow clients are not overwhelmed by a high-
# frequency physics loop.
# ============================================================================

@app.post("/api/physics/qos")
async def api_physics_qos(request_data: dict) -> JSONResponse:
    """
    POST /api/physics/qos — Configure QoS for physics SSE events.

    Accepts a JSON body and persists it to
    channels/physics/qos.json so that the physics engine (or any
    bridge that reads that file) knows the desired delivery policy.

    Request body fields (all optional):
      step_interval_ms   (int,   default 100)  — minimum ms between
                         physics_step SSE events; the notifier skips
                         intermediate force_field writes that arrive
                         faster than this cadence.
      collision_min_delta (int,  default 0)   — only broadcast
                         physics_collision when the collision *count*
                         changes by at least this amount.
      max_queue_depth    (int,   default 256)  — per-client SSE queue
                         ceiling; events that arrive when the queue is
                         full are dropped (slow-client protection).
      enabled            (bool,  default true) — master switch; set
                         false to pause all physics SSE events without
                         disconnecting clients.

    Returns:
      {
        "success": true,
        "qos": { ...merged config... },
        "sse_clients": <int>
      }
    """
    try:
        physics_dir = _CHANNELS_DIR / "physics"
        physics_dir.mkdir(parents=True, exist_ok=True)
        qos_path = physics_dir / "qos.json"

        # ── Load existing config (if any) and merge ─────────────────────
        existing: dict = {}
        if qos_path.exists():
            try:
                existing = json.loads(qos_path.read_text())
            except Exception:
                existing = {}

        defaults = {
            "step_interval_ms": 100,
            "collision_min_delta": 0,
            "max_queue_depth": 256,
            "enabled": True,
        }

        # Merge: defaults → existing → request overrides
        merged = {**defaults, **existing}
        for key in ("step_interval_ms", "collision_min_delta", "max_queue_depth"):
            if key in request_data:
                val = request_data[key]
                if not isinstance(val, int) or val < 0:
                    return JSONResponse(
                        {"error": f"{key} must be a non-negative integer"},
                        status_code=400,
                    )
                merged[key] = val
        if "enabled" in request_data:
            merged["enabled"] = bool(request_data["enabled"])

        merged["updated_at"] = datetime.now().isoformat()

        # ── Persist ──────────────────────────────────────────────────────
        qos_path.write_text(json.dumps(merged, indent=2))

        # ── Notify DataNotifier so any watcher picks up the change ───────
        try:
            import sys as _sys
            _channels_str = str(_CHANNELS_DIR)
            if _channels_str not in _sys.path:
                _sys.path.insert(0, _channels_str)
            from channel_runtime import DataNotifier
            DataNotifier.instance().notify("physics/qos.json")
        except Exception as _ne:
            logger.debug("[physics/qos] DataNotifier.notify skipped: %s", _ne)

        logger.info(
            "[physics/qos] updated: step_interval_ms=%d collision_min_delta=%d "
            "max_queue_depth=%d enabled=%s sse_clients=%d",
            merged["step_interval_ms"],
            merged["collision_min_delta"],
            merged["max_queue_depth"],
            merged["enabled"],
            len(_cell_event_queues),
        )

        return JSONResponse({
            "success": True,
            "qos": merged,
            "sse_clients": len(_cell_event_queues),
        })

    except Exception as e:
        logger.exception("api_physics_qos error")
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================================
# Pipeline API Endpoints (fixing 502 error)
# These endpoints are called by Astro frontend at :4321 → proxy → :8000
# ============================================================================

@app.post("/api/topology")
async def api_topology(request_data: dict) -> JSONResponse:
    """
    POST /api/topology — Text → ELK Topology JSON
    Called by: src/pages/api/topology.ts (Astro proxy)
    GitHub: ResearAI/AutoFigure, kieler/elkjs
    """
    try:
        text = request_data.get("text", "").strip()
        if not text:
            return JSONResponse(
                {"error": "text field is required"},
                status_code=400,
            )

        model = request_data.get("model")
        algorithm = request_data.get("algorithm", "layered")
        direction = request_data.get("direction", "DOWN")

        ai_engine = _get_ai_engine()

        from backend.schemas import ElkAlgorithm, ElkDirection
        result = await generate_topology(
            ai_engine=ai_engine,
            text=text,
            model=model,
            algorithm=ElkAlgorithm(algorithm) if algorithm else ElkAlgorithm.LAYERED,
            direction=ElkDirection(direction) if direction else ElkDirection.DOWN,
        )

        if result.success:
            topo_dict = result.topology.model_dump(exclude_none=True) if result.topology else None
            # ── Broadcast topology change to SSE clients ─────────────────
            if topo_dict:
                _broadcast_topology_updated(topo_dict)
            return JSONResponse({
                "success": True,
                "topology": topo_dict,
                "model_used": result.model_used,
            })
        else:
            return JSONResponse(
                {"error": result.error or "Topology generation failed"},
                status_code=500,
            )

    except Exception as e:
        logger.exception("api_topology error")
        return JSONResponse(
            {"error": str(e), "hint": "Check GEMINI_API_KEY in .env"},
            status_code=500,
        )


@app.post("/api/beautify")
async def api_beautify(request_data: dict) -> JSONResponse:
    """
    POST /api/beautify — ELK Layouted JSON → NanoBanana SVG
    Called by: src/pages/api/beautify.ts (Astro proxy)
    GitHub: gemini-cli-extensions/nanobanana
    """
    try:
        layouted = request_data.get("layouted")
        if not layouted:
            return JSONResponse(
                {"error": "layouted field is required"},
                status_code=400,
            )

        model = request_data.get("model")
        style = request_data.get("style", "academic")

        ai_engine = _get_ai_engine()

        # Build scaffold from layouted graph
        scaffold = build_scaffold(layouted)

        # Generate SVG via NanoBanana bridge
        result = await beautify_with_nanobanana(
            ai_engine=ai_engine,
            layouted=layouted,
            scaffold=scaffold,
            model=model,
            style=style,
        )

        if result.get("success"):
            return JSONResponse(result)
        else:
            return JSONResponse(
                {"error": result.get("error", "Beautify failed")},
                status_code=500,
            )

    except Exception as e:
        logger.exception("api_beautify error")
        return JSONResponse(
            {"error": str(e), "hint": "Check GEMINI_API_KEY in .env"},
            status_code=500,
        )


@app.post("/api/validate")
async def api_validate_svg(request_data: dict) -> JSONResponse:
    """
    POST /api/validate — SVG syntax validation + LLM fix
    GitHub: withastro/astro
    """
    try:
        svg = request_data.get("svg", "")
        if not svg:
            return JSONResponse(
                {"error": "svg field is required"},
                status_code=400,
            )

        auto_fix = request_data.get("auto_fix", True)
        model = request_data.get("model")

        ai_engine = _get_ai_engine() if auto_fix else None
        result = await validate_svg_func(
            svg=svg,
            ai_engine=ai_engine,
            auto_fix=auto_fix,
            model=model,
        )

        # Convert ValidateResponse to dict
        return JSONResponse(result.model_dump() if hasattr(result, 'model_dump') else result)

    except Exception as e:
        logger.exception("api_validate error")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/models")
async def api_models() -> JSONResponse:
    """
    GET /api/models — List available AI models
    """
    return JSONResponse(_settings.AVAILABLE_MODELS)


# ── Step 5: Gemini 3 Pro Image Generation ────────────────────────────────

from backend.pipeline.gemini_image_gen import (
    generate_scientific_figure,
    generate_prompt_with_grok,
    generate_image_with_gemini,
)


@app.post("/api/generate-image")
async def api_generate_image(request_data: dict) -> JSONResponse:
    """
    POST /api/generate-image — Step 5: SVG → Gemini 3 Pro Image

    Sends Grok's full design spec + skeleton PNG to Gemini.
    No prompt compression — user skeleton edits are preserved.

    Flow:
      Stage 1: Grok 4 generates verbose design spec
      Stage 2: Strip <think> blocks only — no compression
      Stage 3: Gemini receives FULL design spec + skeleton PNG + layout text

    No retry mechanism. Fail cleanly.
    """
    try:
        svg_content = request_data.get("svg_content", "")
        method_text = request_data.get("method_text", "")

        if not svg_content:
            return JSONResponse(
                {"error": "svg_content is required"}, status_code=400
            )
        if not method_text:
            return JSONResponse(
                {"error": "method_text is required"}, status_code=400
            )

        engine = _get_ai_engine()

        # Generate figure (full prompt, no compression)
        result = await generate_scientific_figure(
            ai_engine=engine,
            method_text=method_text,
            svg_content=svg_content,
            reference_image_b64=request_data.get("reference_image_b64"),
            prompt_model=request_data.get("prompt_model"),
            image_model=request_data.get("image_model", "gemini-3-pro-image-preview"),
            aspect_ratio=request_data.get("aspect_ratio", "16:9"),
            image_size=request_data.get("image_size", "4K"),
            custom_prompt=request_data.get("custom_prompt"),
            elk_graph=request_data.get("elk_graph"),
        )

        return JSONResponse(result)

    except Exception as e:
        logger.exception("api_generate_image error")
        return JSONResponse(
            {"success": False, "error": str(e)}, status_code=500
        )


@app.post("/api/generate-prompt")
async def api_generate_prompt(request_data: dict) -> JSONResponse:
    """
    POST /api/generate-prompt — Grok 4 prompt engineering only

    Use this to get the prompt for review before generating the image.

    Request body:
      - method_text (str): Paper method description
      - svg_content (str): The ELK-generated SVG
      - model (str, optional): Model for prompt engineering

    Returns:
      - success (bool)
      - prompt (str): The generated prompt
    """
    try:
        method_text = request_data.get("method_text", "")
        svg_content = request_data.get("svg_content", "")

        if not method_text or not svg_content:
            return JSONResponse(
                {"error": "method_text and svg_content are required"},
                status_code=400,
            )

        engine = _get_ai_engine()
        result = await generate_prompt_with_grok(
            ai_engine=engine,
            method_text=method_text,
            svg_content=svg_content,
            model=request_data.get("model"),
            reference_image_b64=request_data.get("reference_image_b64"),
            elk_graph=request_data.get("elk_graph"),
        )

        return JSONResponse(result)

    except Exception as e:
        logger.exception("api_generate_prompt error")
        return JSONResponse(
            {"success": False, "error": str(e)}, status_code=500
        )


# ============================================================================
# Cell Loop API endpoints (cell pub/sub loop — channels/)
# ============================================================================

_CHANNELS_DIR = BASE_DIR / "channels"


@app.post("/api/cell-loop")
async def api_cell_loop(request_data: dict) -> JSONResponse:
    """
    POST /api/cell-loop
    Body: { "structured_data": {...}, "max_epochs": 10 }
    Invokes channels/loop_orchestrator.run_loop() and returns all cell params.json.
    """
    try:
        import sys as _sys
        _channels_str = str(_CHANNELS_DIR)
        if _channels_str not in _sys.path:
            _sys.path.insert(0, _channels_str)

        structured_data = request_data.get("structured_data")
        max_epochs = int(request_data.get("max_epochs", 10))

        # Write structured_data to a temp file if provided so topology_to_skeleton
        # can pick it up via the --data path used in loop_orchestrator.__main__.
        if structured_data is not None:
            import tempfile, json as _json
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as _tmp:
                _json.dump(structured_data, _tmp)
                _tmp_path = _tmp.name

            # Pre-populate skeleton from structured_data
            import os as _os
            _os.chdir(_channels_str)
            from topology_to_skeleton import from_structured_data, SKELETON_DIR
            _skel_dir = Path(SKELETON_DIR)
            if _skel_dir.exists():
                for _f in _skel_dir.iterdir():
                    if _f.suffix == ".json":
                        _f.unlink()
            from_structured_data(_tmp_path)
            import os as _os2
            _os2.unlink(_tmp_path)

        import importlib, loop_orchestrator as _lo
        importlib.reload(_lo)

        # ── Broadcast loop start to SSE clients ──────────────────────────
        _broadcast_cell_loop_started(max_epochs)

        output_svg = _lo.run_loop(max_epochs=max_epochs)

        # Collect all cell params.json
        cells_result = {}
        for params_file in _CHANNELS_DIR.glob("cell/*/params.json"):
            cell_id = params_file.parent.name
            cells_result[cell_id] = json.loads(params_file.read_text())

        # ── Determine convergence from epoch.json ────────────────────────
        _epoch_file = _CHANNELS_DIR / "skeleton" / "epoch.json"
        _epoch_info: dict = {}
        if _epoch_file.exists():
            try:
                _epoch_info = json.loads(_epoch_file.read_text())
            except Exception:
                pass
        _final_epoch = _epoch_info.get("current", max_epochs - 1)
        _converged = _epoch_info.get("status") == "converged"

        # ── Broadcast loop finish to SSE clients ─────────────────────────
        _broadcast_cell_loop_finished(_final_epoch, _converged, cells_result)

        return JSONResponse({
            "success": True,
            "output_svg": str(output_svg),
            "cells": cells_result,
        })

    except Exception as e:
        logger.exception("api_cell_loop error")
        # Broadcast loop failure so SSE clients know it stopped
        _broadcast_cell_loop_finished(-1, False, {})
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/api/cells")
def api_cells() -> JSONResponse:
    """
    GET /api/cells
    Returns CellDescriptor[] — one entry per channels/cell/*/params.json.
    Each descriptor is enriched with ``parent_id`` (str | None) and
    ``children`` (list[str]) derived from topology.json edges.
    """
    try:
        # -- Build parent/children maps from topology.json edges -----------
        parent_map: dict[str, str | None] = {}   # cell_id → parent_id
        children_map: dict[str, list[str]] = {}   # cell_id → [child_ids]

        topo_path = _CHANNELS_DIR / "skeleton" / "topology.json"
        if topo_path.exists():
            topo = json.loads(topo_path.read_text())
            # Initialise every node declared in topology children
            for node in topo.get("children", []):
                nid = node.get("id", "")
                if nid:
                    parent_map.setdefault(nid, None)
                    children_map.setdefault(nid, [])
            # Walk non-skip edges to establish hierarchy
            for edge in topo.get("edges", []):
                advanced = edge.get("advanced", {})
                if advanced.get("semanticType") == "skip_connection":
                    continue
                src = edge.get("sources", [None])[0]
                tgt = edge.get("targets", [None])[0]
                if src and tgt:
                    parent_map[tgt] = src
                    children_map.setdefault(src, []).append(tgt)

        # -- Assemble descriptors -----------------------------------------
        descriptors = []
        for params_file in sorted(_CHANNELS_DIR.glob("cell/*/params.json")):
            cell_id = params_file.parent.name
            data = json.loads(params_file.read_text())
            # Merge status if available
            status_file = params_file.parent / "status.json"
            if status_file.exists():
                data["status"] = json.loads(status_file.read_text())
            # Attach hierarchy fields from topology
            data["parent_id"] = parent_map.get(cell_id)
            data["children"] = children_map.get(cell_id, [])
            descriptors.append(data)
        return JSONResponse(descriptors)
    except Exception as e:
        logger.exception("api_cells error")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/api/cell/{cell_id}")
def api_cell(cell_id: str) -> JSONResponse:
    """
    GET /api/cell/{cell_id}
    Returns params.json + status.json for a single cell.
    """
    try:
        cell_dir = _CHANNELS_DIR / "cell" / cell_id
        params_file = cell_dir / "params.json"
        if not params_file.exists():
            raise HTTPException(status_code=404, detail=f"Cell '{cell_id}' not found")

        result = json.loads(params_file.read_text())
        status_file = cell_dir / "status.json"
        if status_file.exists():
            result["status"] = json.loads(status_file.read_text())
        else:
            result["status"] = None
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("api_cell error")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/api/epochs")
def api_epochs() -> JSONResponse:
    """
    GET /api/epochs
    Returns all epoch snapshots from channels/physics/epoch_snapshots/.
    Falls back to returning current skeleton/epoch.json + physics metadata
    when no snapshot directory exists yet.
    """
    try:
        snapshots_dir = _CHANNELS_DIR / "physics" / "epoch_snapshots"
        if snapshots_dir.exists():
            snapshots = []
            for snap_file in sorted(snapshots_dir.glob("*.json")):
                snapshots.append(json.loads(snap_file.read_text()))
            return JSONResponse({"snapshots": snapshots, "count": len(snapshots)})

        # Fallback: current epoch state
        epoch_file = _CHANNELS_DIR / "skeleton" / "epoch.json"
        converged_file = _CHANNELS_DIR / "physics" / "converged.json"
        collision_file = _CHANNELS_DIR / "physics" / "collision.json"

        current: dict = {}
        if epoch_file.exists():
            current["epoch"] = json.loads(epoch_file.read_text())
        if converged_file.exists():
            current["convergence"] = json.loads(converged_file.read_text())
        if collision_file.exists():
            current["collision"] = json.loads(collision_file.read_text())

        return JSONResponse({"snapshots": [current] if current else [], "count": int(bool(current))})
    except Exception as e:
        logger.exception("api_epochs error")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


JOBS: dict[str, Job] = {}


@app.get("/api/config")
def get_config() -> JSONResponse:
    available, rel_path = _resolve_svg_edit_path()
    return JSONResponse({"svgEditAvailable": available, "svgEditPath": rel_path})


@app.get("/api/health")
def health_check() -> JSONResponse:
    """
    GET /api/health — Backend health check
    Returns server status, available providers, and model list.
    """
    providers = _get_ai_engine().available_providers if _ai_engine else []
    return JSONResponse({
        "status": "ok",
        "providers": providers,
        "python": PYTHON_EXECUTABLE,
        "timestamp": datetime.now().isoformat(),
    })


@app.get("/api/world/state")
def api_world_state() -> JSONResponse:
    """
    GET /api/world/state

    Returns the complete world state including the full 58-cell
    composite_params payload plus current epoch and topology metadata.

    Response schema
    ───────────────
    {
      "composite_params": {
        "cells":      [...],   // 58 cell descriptors
        "edges":      [...],   // edge route descriptors
        "canvas":     {...},   // {width, height}
        "palette":    {...},   // {zenith, horizon, nadir}
        "cell_count": <int>
      },
      "epoch":     <int | null>,   // current epoch (-1 = no epoch yet)
      "converged": <bool>,
      "timestamp": <str>           // ISO-8601
    }
    """
    from backend.sse_physics_bridge import _load_composite_params

    # ── composite_params (58 cells) ──────────────────────────────────────────
    cp = _load_composite_params()
    if cp is None:
        return JSONResponse(
            {"error": "composite_params.json not found"},
            status_code=503,
        )

    composite = {
        "cells":      cp.get("cells", []),
        "edges":      cp.get("edges", []),
        "canvas":     cp.get("canvas", {}),
        "palette":    cp.get("palette", {}),
        "cell_count": len(cp.get("cells", [])),
    }

    # ── current epoch ────────────────────────────────────────────────────────
    epoch_num: int | None = None
    converged = False
    epoch_path = BASE_DIR / "channels" / "skeleton" / "epoch.json"
    if epoch_path.exists():
        try:
            ep = json.loads(epoch_path.read_text())
            epoch_num = ep.get("current", -1)
        except Exception:
            pass

    conv_path = BASE_DIR / "channels" / "physics" / "converged.json"
    if conv_path.exists():
        try:
            converged = bool(json.loads(conv_path.read_text()).get("converged", False))
        except Exception:
            pass

    return JSONResponse({
        "composite_params": composite,
        "epoch":     epoch_num,
        "converged": converged,
        "timestamp": datetime.now().isoformat(),
    })


@app.post("/api/run")
def run_job(req: RunRequest) -> JSONResponse:
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    output_dir = OUTPUTS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON_EXECUTABLE,
        str(BASE_DIR / "autofigure2.py"),
        "--method_text",
        req.method_text,
        "--output_dir",
        str(output_dir),
        "--provider",
        req.provider,
    ]

    if req.api_key:
        cmd += ["--api_key", req.api_key]
    if req.base_url:
        cmd += ["--base_url", req.base_url]
    if req.image_model:
        cmd += ["--image_model", req.image_model]
    if req.svg_model:
        cmd += ["--svg_model", req.svg_model]

    sam_prompt = req.sam_prompt or DEFAULT_SAM_PROMPT
    placeholder_mode = req.placeholder_mode or DEFAULT_PLACEHOLDER_MODE
    merge_threshold = (
        req.merge_threshold if req.merge_threshold is not None else DEFAULT_MERGE_THRESHOLD
    )

    cmd += ["--sam_prompt", sam_prompt]
    cmd += ["--placeholder_mode", placeholder_mode]
    cmd += ["--merge_threshold", str(merge_threshold)]
    if req.sam_backend:
        cmd += ["--sam_backend", req.sam_backend]
    if req.sam_api_key:
        cmd += ["--sam_api_key", req.sam_api_key]
    if req.sam_max_masks is not None:
        cmd += ["--sam_max_masks", str(req.sam_max_masks)]
    if req.optimize_iterations is not None:
        cmd += ["--optimize_iterations", str(req.optimize_iterations)]

    reference_path = req.reference_image_path
    if reference_path:
        reference_path = (
            str((BASE_DIR / reference_path).resolve())
            if not Path(reference_path).is_absolute()
            else reference_path
        )
        cmd += ["--reference_image_path", reference_path]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    log_path = output_dir / "run.log"
    log_path.write_text(
        f"[meta] python={PYTHON_EXECUTABLE}\n[meta] cmd={' '.join(cmd)}\n",
        encoding="utf-8",
    )

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
        cwd=str(BASE_DIR),
    )

    job = Job(
        job_id=job_id,
        output_dir=output_dir,
        process=process,
        queue=queue.Queue(),
        log_path=log_path,
    )
    JOBS[job_id] = job

    monitor_thread = threading.Thread(target=_monitor_job, args=(job,), daemon=True)
    monitor_thread.start()

    return JSONResponse({"job_id": job_id})


@app.post("/api/upload")
async def upload_reference(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    ext = Path(file.filename).suffix.lower()
    if ext not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}:
        ext = ".png"

    data = await file.read()
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")

    name = f"{uuid.uuid4().hex}{ext}"
    out_path = UPLOADS_DIR / name
    out_path.write_bytes(data)

    rel_path = out_path.relative_to(BASE_DIR).as_posix()
    return JSONResponse(
        {"path": rel_path, "url": f"/api/uploads/{name}", "name": file.filename}
    )


@app.get("/api/events/{job_id}")
def stream_events(job_id: str) -> StreamingResponse:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    def event_stream():
        while True:
            try:
                item = job.queue.get(timeout=1.0)
            except queue.Empty:
                if job.done:
                    break
                continue
            if item.get("event") == "close":
                break
            yield _format_sse(item["event"], item["data"])

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/artifacts/{job_id}/{path:path}")
def get_artifact(job_id: str, path: str) -> FileResponse:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    candidate = (job.output_dir / path).resolve()
    if not str(candidate).startswith(str(job.output_dir.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(candidate)


@app.get("/api/uploads/{filename}")
def get_upload(filename: str) -> FileResponse:
    candidate = (UPLOADS_DIR / filename).resolve()
    if not str(candidate).startswith(str(UPLOADS_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(candidate)


def _format_sse(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=True)
    return f"event: {event}\ndata: {payload}\n\n"


def _monitor_job(job: Job) -> None:
    job.push("status", {"state": "started"})

    stdout_thread = threading.Thread(
        target=_pipe_output, args=(job, job.process.stdout, "stdout"), daemon=True
    )
    stderr_thread = threading.Thread(
        target=_pipe_output, args=(job, job.process.stderr, "stderr"), daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()

    idle_cycles = 0
    while True:
        _scan_artifacts(job)

        if job.process.poll() is not None:
            idle_cycles += 1
        else:
            idle_cycles = 0

        if idle_cycles >= 4:
            break
        time.sleep(0.5)

    _scan_artifacts(job)
    job.push("status", {"state": "finished", "code": job.process.returncode})
    job.push(
        "artifact",
        {
            "kind": "log",
            "name": job.log_path.name,
            "path": job.log_path.relative_to(job.output_dir).as_posix(),
            "url": f"/api/artifacts/{job.job_id}/{job.log_path.name}",
        },
    )
    job.done = True
    job.push("close", {})


def _pipe_output(job: Job, pipe, stream_name: str) -> None:
    if pipe is None:
        return
    for line in iter(pipe.readline, ""):
        text = line.rstrip()
        if text:
            job.write_log(stream_name, text)
            job.push("log", {"stream": stream_name, "line": text})
    pipe.close()


def _scan_artifacts(job: Job) -> None:
    output_dir = job.output_dir
    candidates = [
        output_dir / "figure.png",
        output_dir / "samed.png",
        output_dir / "template.svg",
        output_dir / "final.svg",
    ]

    icons_dir = output_dir / "icons"
    if icons_dir.is_dir():
        candidates.extend(icons_dir.glob("icon_*.png"))

    for path in candidates:
        if not path.is_file():
            continue
        rel_path = path.relative_to(output_dir).as_posix()
        if rel_path in job.seen:
            continue
        job.seen.add(rel_path)

        kind = _classify_artifact(rel_path)
        job.push(
            "artifact",
            {
                "kind": kind,
                "name": path.name,
                "path": rel_path,
                "url": f"/api/artifacts/{job.job_id}/{rel_path}",
            },
        )


def _classify_artifact(rel_path: str) -> str:
    if rel_path == "figure.png":
        return "figure"
    if rel_path == "samed.png":
        return "samed"
    if rel_path.endswith("_nobg.png"):
        return "icon_nobg"
    if rel_path.startswith("icons/") and rel_path.endswith(".png"):
        return "icon_raw"
    if rel_path == "template.svg":
        return "template_svg"
    if rel_path == "final.svg":
        return "final_svg"
    return "artifact"


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            return True
    return False


def _pids_on_port(port: int) -> set[int]:
    pids: set[int] = set()

    if shutil.which("lsof"):
        result = subprocess.run(
            ["lsof", "-t", f"-i:{port}"],
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.isdigit():
                pids.add(int(line))
        return pids

    if shutil.which("ss"):
        result = subprocess.run(
            ["ss", "-lptn", f"sport = :{port}"],
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            if "pid=" in line:
                for part in line.split("pid=")[1:]:
                    pid_str = "".join(ch for ch in part if ch.isdigit())
                    if pid_str:
                        pids.add(int(pid_str))
        return pids

    if shutil.which("netstat"):
        result = subprocess.run(
            ["netstat", "-tlnp"],
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            if f":{port} " not in line or "LISTEN" not in line:
                continue
            fields = line.split()
            if fields and "/" in fields[-1]:
                pid_part = fields[-1].split("/")[0]
                if pid_part.isdigit():
                    pids.add(int(pid_part))

    return pids


def _read_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            data = handle.read()
        parts = [p for p in data.split(b"\x00") if p]
        return " ".join(part.decode(errors="ignore") for part in parts)
    except OSError:
        return ""


def _is_uvicorn_process(pid: int) -> bool:
    cmdline = _read_cmdline(pid)
    if not cmdline:
        return False
    if "uvicorn" not in cmdline:
        return False
    return "server:app" in cmdline or "server.py" in cmdline


def _terminate_pids(pids: set[int], timeout: float = 2.0) -> None:
    current_pid = os.getpid()
    for pid in sorted(pids):
        if pid <= 1 or pid == current_pid:
            continue
        if not _is_uvicorn_process(pid):
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue

    deadline = time.time() + timeout
    while time.time() < deadline:
        alive = False
        for pid in pids:
            if pid <= 1 or pid == current_pid:
                continue
            if not _is_uvicorn_process(pid):
                continue
            try:
                os.kill(pid, 0)
                alive = True
            except ProcessLookupError:
                continue
        if not alive:
            return
        time.sleep(0.1)

    for pid in sorted(pids):
        if pid <= 1 or pid == current_pid:
            continue
        if not _is_uvicorn_process(pid):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue


def _ensure_port_free(port: int) -> None:
    if not _port_in_use(port):
        return
    pids = _pids_on_port(port)
    if not pids:
        return
    _terminate_pids(pids)


# Mount static files only if web/ directory exists (created by `bun run build`)
# During development, Astro dev server (:4321) serves the frontend directly
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static")
else:
    logger.info(
        "web/ directory not found — skipping static file mount. "
        "Run 'bun run build' to generate it, or use 'bun run dev' for Astro dev server."
    )


if __name__ == "__main__":
    import uvicorn

    def find_available_port(start_port: int, max_attempts: int = 100) -> int:
        for port in range(start_port, start_port + max_attempts):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                try:
                    sock.bind(("0.0.0.0", port))
                    return port
                except OSError:
                    print(f"Port {port} is in use, trying next...")
                    continue
        raise IOError(f"No available ports found in range ({start_port} - {start_port + max_attempts})")

    initial_port = 8000
    
    try:
        actual_port = find_available_port(initial_port)
        
        print(f"--- Starting Server ---")
        print(f"Local access: http://127.0.0.1:{actual_port}")
        print(f"-----------------------")

        uvicorn.run(
            "server:app",
            host="0.0.0.0",
            port=actual_port,
            reload=False,
            access_log=False,
        )
    except Exception as e:
        print(f"Startup failed: {e}")
        sys.exit(1)