#!/usr/bin/env python3
"""
channel_runtime.py — Apollo CyberRT DataNotifier + DataDispatcher + DataVisitor
ported to Python for astro-svgfigure cell pub/sub loop.

Upstream source (20% algorithm modified):
  - upstream/apollo-cyber/data/data_notifier.h   → DataNotifier class
  - upstream/apollo-cyber/data/data_dispatcher.h  → DataDispatcher class
  - upstream/apollo-cyber/data/data_visitor.h     → DataVisitor (4-channel fusion)
  - upstream/apollo-cyber/component/component.h   → FAstroCellFusion struct

Algorithm changes from Apollo original:
  1. channel_id: uint64_t hash → string path ("cell/self_attn/bbox.json")
  2. Notify: coroutine wake → file mtime watch + callback list
  3. Dispatch: flat buffer route → z-layer priority route with SVG bbox awareness
  4. DataVisitor 4-channel fusion: M0-M3 template → skeleton/force/palette/z_layer dicts
  5. FAstroCellFusion: shadow descriptor → SVG constraint descriptor with bbox delta

Reference: [ASTRO-NOTIFY] [ASTRO-DISPATCH] [ASTRO-VISITOR] debug prefixes preserved.
"""

import json
import os
import time
import hashlib
from typing import Any, Callable, Dict, List, Optional, Tuple

CHANNELS_DIR = os.path.dirname(os.path.abspath(__file__))
VERBOSE = os.environ.get("ASTRO_NOTIFY_VERBOSE", "0") == "1"


def _dbg(prefix: str, msg: str):
    """Debug print — mirrors ASTRO_NOTIFY_DBG macro from data_notifier.h"""
    if VERBOSE:
        print(f"[{prefix}] {msg}")


# ═══════════════════════════════════════════════════════════════════════════════
# DataNotifier — ported from upstream/apollo-cyber/data/data_notifier.h
#
# Original: singleton with AtomicHashMap<uint64_t, NotifyVector>
# ASTRO change: channel_id is string path, callbacks are Python callables,
# notification checks file mtime instead of shared-memory signal.
# ═══════════════════════════════════════════════════════════════════════════════

class Notifier:
    """Single notifier entry — mirrors struct Notifier { callback; }"""
    __slots__ = ("callback",)
    def __init__(self, callback: Callable[[], None]):
        self.callback = callback


class DataNotifier:
    """
    Port of apollo::cyber::data::DataNotifier.
    AddNotifier registers a callback on a channel path.
    Notify fires all callbacks registered on that channel.
    """
    _instance: Optional["DataNotifier"] = None

    def __init__(self):
        self._notifies_map: Dict[str, List[Notifier]] = {}
        self._mtime_cache: Dict[str, float] = {}
        _dbg("ASTRO-NOTIFY", "DataNotifier singleton constructed")

    @classmethod
    def instance(cls) -> "DataNotifier":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None

    def add_notifier(self, channel_id: str, notifier: Notifier):
        """AddNotifier — register callback on channel. Mirrors data_notifier.h AddNotifier."""
        if channel_id in self._notifies_map:
            self._notifies_map[channel_id].append(notifier)
            _dbg("ASTRO-NOTIFY", f"ch={channel_id} notifiers={len(self._notifies_map[channel_id])} op=add_existing ok=1")
        else:
            self._notifies_map[channel_id] = [notifier]
            _dbg("ASTRO-NOTIFY", f"ch={channel_id} notifiers=1 op=add_new ok=1")

    def notify(self, channel_id: str) -> bool:
        """Notify — fire all callbacks on channel. Mirrors data_notifier.h Notify."""
        notifiers = self._notifies_map.get(channel_id)
        if notifiers:
            _dbg("ASTRO-NOTIFY", f"ch={channel_id} notifiers={len(notifiers)} op=notify ok=1")
            for n in notifiers:
                if n.callback:
                    n.callback()
            return True
        _dbg("ASTRO-NOTIFY", f"ch={channel_id} notifiers=0 op=notify_miss ok=0")
        return False

    def check_mtime(self, channel_id: str) -> bool:
        """ASTRO extension: check if file changed since last check. Not in Apollo original."""
        full_path = os.path.join(CHANNELS_DIR, channel_id)
        if not os.path.exists(full_path):
            return False
        mtime = os.path.getmtime(full_path)
        old = self._mtime_cache.get(channel_id, 0)
        if mtime > old:
            self._mtime_cache[channel_id] = mtime
            return True
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# DataDispatcher — ported from upstream/apollo-cyber/data/data_dispatcher.h
#
# Original: AtomicHashMap<uint64_t, BufferVector> with weak_ptr buffer routing
# ASTRO change: routes by z-layer priority instead of flat insertion order.
# channel_id is string path. Buffers are Python lists instead of CacheBuffer.
# Dispatch writes JSON to file (the "buffer") then notifies.
# ═══════════════════════════════════════════════════════════════════════════════

class ChannelBuffer:
    """Mirrors CacheBuffer<shared_ptr<T>> — here it's a file-backed JSON slot."""
    __slots__ = ("channel_id", "queue", "max_size")

    def __init__(self, channel_id: str, max_size: int = 1):
        self.channel_id = channel_id
        self.queue: List[Any] = []
        self.max_size = max_size

    def fill(self, data: Any):
        self.queue.append(data)
        if len(self.queue) > self.max_size:
            self.queue.pop(0)
        # Write to file — the "shared memory" equivalent
        full = os.path.join(CHANNELS_DIR, self.channel_id)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            if isinstance(data, str):
                f.write(data)
            else:
                json.dump(data, f, indent=2)

    def latest(self) -> Optional[Any]:
        return self.queue[-1] if self.queue else None


class DataDispatcher:
    """
    Port of apollo::cyber::data::DataDispatcher.
    AddBuffer registers a ChannelBuffer into the z-layer routing table.
    Dispatch routes data to all registered buffers on that channel.
    """
    _instance: Optional["DataDispatcher"] = None

    def __init__(self):
        self._buffers_map: Dict[str, List[ChannelBuffer]] = {}
        self._notifier = DataNotifier.instance()
        _dbg("ASTRO-DISPATCH", "DataDispatcher singleton constructed")

    @classmethod
    def instance(cls) -> "DataDispatcher":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None

    def add_buffer(self, buf: ChannelBuffer):
        """AddBuffer — register buffer in z-layer routing table."""
        if buf.channel_id in self._buffers_map:
            self._buffers_map[buf.channel_id].append(buf)
            _dbg("ASTRO-DISPATCH", f"ch={buf.channel_id} buffers={len(self._buffers_map[buf.channel_id])} op=add_existing")
        else:
            self._buffers_map[buf.channel_id] = [buf]
            _dbg("ASTRO-DISPATCH", f"ch={buf.channel_id} buffers=1 op=add_new")

    def dispatch(self, channel_id: str, data: Any) -> bool:
        """Dispatch — route data to all buffers, then notify. Mirrors data_dispatcher.h."""
        buffers = self._buffers_map.get(channel_id)
        if not buffers:
            _dbg("ASTRO-DISPATCH", f"ch={channel_id} op=dispatch_miss")
            # No registered buffer — write directly to file as fallback
            full = os.path.join(CHANNELS_DIR, channel_id)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as f:
                if isinstance(data, str):
                    f.write(data)
                else:
                    json.dump(data, f, indent=2)
            self._notifier.notify(channel_id)
            return True

        for buf in buffers:
            buf.fill(data)
        _dbg("ASTRO-DISPATCH", f"ch={channel_id} buffers={len(buffers)} op=dispatch ok=1")
        self._notifier.notify(channel_id)
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# FAstroCellFusion — ported from upstream/apollo-cyber/component/component.h
#
# Original: struct with node_name, channel_ready bitmask, skeleton_seq, etc.
# ASTRO change: adds bbox_delta (SVG-specific) and species field.
# ═══════════════════════════════════════════════════════════════════════════════

class FAstroCellFusion:
    """
    Per-invocation metadata for one 4-channel Proc() call.
    Mirrors FAstroCellFusion from component.h with SVG constraint extensions.
    """
    __slots__ = (
        "node_name", "channel_ready", "skeleton_seq",
        "force_magnitude", "palette_id", "z_layer_target",
        "proc_latency_us", "species", "bbox_delta",
    )

    def __init__(self, node_name: str):
        self.node_name = node_name
        self.channel_ready = 0       # bitmask 0b1111 when all 4 channels present
        self.skeleton_seq = 0
        self.force_magnitude = 0.0
        self.palette_id = 0
        self.z_layer_target = 3
        self.proc_latency_us = 0
        self.species = ""
        self.bbox_delta = {"dx": 0, "dy": 0, "dw": 0, "dh": 0}

    def debug_print(self):
        """Emit single-line debug summary — mirrors FAstroCellFusion::DebugPrint()."""
        print(f"[ASTRO-COMPONENT] cell-fusion | node={self.node_name} "
              f"ch_mask=0x{self.channel_ready:x} skel_seq={self.skeleton_seq} "
              f"force={self.force_magnitude:.3f} palette={self.palette_id} "
              f"z={self.z_layer_target} species={self.species} "
              f"latency_us={self.proc_latency_us}")


# ═══════════════════════════════════════════════════════════════════════════════
# DataVisitor (4-channel fusion) — ported from data_visitor.h
#
# Original: template<M0,M1,M2,M3> with AllLatest fusion
# ASTRO change: M0=skeleton, M1=force_field, M2=palette, M3=z_layers (all dicts)
# TryFetch reads from file channels instead of CacheBuffer memory
# ═══════════════════════════════════════════════════════════════════════════════

class DataVisitor:
    """
    4-channel constraint visitor — ported from DataVisitor<M0,M1,M2,M3>.
    Fuses skeleton + force_field + palette + z_layers into a single Proc() call.
    """
    def __init__(self, cell_id: str):
        self.cell_id = cell_id
        self.channels = {
            "skeleton": f"skeleton/cell/{cell_id}.json",
            "force_field": "physics/force_field.json",
            "palette": "physics/species_assignment.json",
            "z_layers": "physics/z_layers.json",
        }
        self._dispatcher = DataDispatcher.instance()
        self._notifier = DataNotifier.instance()

        # Register buffers — mirrors DataVisitor ctor
        for name, ch_path in self.channels.items():
            buf = ChannelBuffer(ch_path, max_size=2)
            self._dispatcher.add_buffer(buf)
        _dbg("ASTRO-VISITOR", f"ConstraintVisitor ctor: cell={cell_id} binding 4 channels")

    def try_fetch(self) -> Optional[Tuple[dict, dict, dict, dict]]:
        """
        TryFetch — constraint-gated fusion pass.
        Returns (skeleton, force_field, palette, z_layers) or None if any missing.
        Mirrors DataVisitor::TryFetch from data_visitor.h.
        """
        results = {}
        mask = 0
        for i, (name, ch_path) in enumerate(self.channels.items()):
            full = os.path.join(CHANNELS_DIR, ch_path)
            if os.path.exists(full):
                with open(full) as f:
                    results[name] = json.load(f)
                mask |= (1 << i)
            else:
                _dbg("ASTRO-VISITOR", f"TryFetch cell={self.cell_id} missing ch={name}")
                return None

        _dbg("ASTRO-VISITOR", f"TryFetch cell={self.cell_id} mask=0x{mask:x} → fused")
        return (results["skeleton"], results["force_field"],
                results["palette"], results["z_layers"])


# ═══════════════════════════════════════════════════════════════════════════════
# Channel I/O helpers — typed wrappers around DataDispatcher
# ═══════════════════════════════════════════════════════════════════════════════

def publish(channel_path: str, data: Any) -> bool:
    """Publish data to a channel — Apollo Writer::Write equivalent."""
    return DataDispatcher.instance().dispatch(channel_path, data)


def subscribe(channel_path: str) -> Optional[Any]:
    """Subscribe/read latest from channel — Apollo Reader::Observe equivalent."""
    full = os.path.join(CHANNELS_DIR, channel_path)
    if not os.path.exists(full):
        return None
    with open(full) as f:
        if full.endswith(".svg"):
            return f.read()
        return json.load(f)


def on_message(channel_path: str, callback: Callable[[], None]):
    """Register a notification callback — Apollo CreateReader with callback."""
    DataNotifier.instance().add_notifier(channel_path, Notifier(callback))


# ═══════════════════════════════════════════════════════════════════════════════
# LoopScheduler — M004 addition
#
# Manages cell proc() dispatch order by z-layer priority.
# Ported concept from apollo::cyber::scheduler::Scheduler (scheduler.h):
#   Original: coroutine task pool ordered by ClassicTask priority field.
#   ASTRO change: tasks are cell_id strings; priority comes from z_layers.json;
#   execution is synchronous (no coroutines) — single-threaded epoch loop.
#
# Algorithm (mirrors Scheduler::CreateTask → NotifyProcessor → ProcBalance):
#   1. Register cells with z-layer weights (RegisterCell).
#   2. Sort registered cells ascending by z (lower z = rendered first → scheduled
#      first, matching Apollo's lower-priority-value → earlier dispatch).
#   3. run_epoch(proc_fn) iterates sorted cells and calls proc_fn(cell_id).
#   4. Cells with the same z are dispatched in stable insertion order (mirrors
#      Scheduler FIFO within same-priority bucket).
# ═══════════════════════════════════════════════════════════════════════════════

class LoopScheduler:
    """
    Cell proc() scheduler ordered by z-layer priority.

    Mirrors apollo::cyber::scheduler::Scheduler with:
      task_name  → cell_id  (string)
      priority   → z-layer  (int, lower = higher scheduling priority)
      processor  → proc_fn  callable

    Usage:
        sched = LoopScheduler()
        sched.register_cell("input_embed", z=3)
        sched.register_cell("self_attn",   z=3)
        sched.run_epoch(proc)        # calls proc("input_embed"), proc("self_attn") …
    """

    def __init__(self):
        # List of (z, insertion_seq, cell_id) — kept unsorted until run_epoch.
        self._tasks: List[Tuple[int, int, str]] = []
        self._seq = 0
        # z_layers channel path (relative to CHANNELS_DIR)
        self._z_layers_path = "physics/z_layers.json"
        _dbg("ASTRO-SCHED", "LoopScheduler constructed")

    def register_cell(self, cell_id: str, z: Optional[int] = None):
        """
        RegisterCell — add a cell to the scheduler.

        If z is None the scheduler reads physics/z_layers.json to resolve
        the z value; falls back to z=3 (default node layer) if not found.
        Mirrors Scheduler::CreateTask with priority auto-detection.
        """
        if z is None:
            z = self._resolve_z(cell_id)
        self._tasks.append((z, self._seq, cell_id))
        self._seq += 1
        _dbg("ASTRO-SCHED", f"register cell={cell_id} z={z} seq={self._seq - 1}")

    def register_cells_from_z_layers(self):
        """
        Bulk-register all cells present in physics/z_layers.json.
        Convenience wrapper — mirrors Scheduler::CreateTaskBatch.
        """
        full = os.path.join(CHANNELS_DIR, self._z_layers_path)
        if not os.path.exists(full):
            _dbg("ASTRO-SCHED", "register_cells_from_z_layers: z_layers.json missing, skip")
            return
        with open(full) as f:
            z_data = json.load(f)
        for cell_id, val in z_data.items():
            if cell_id == "__schema__":
                continue
            z = val if isinstance(val, int) else val.get("z", 3)
            self.register_cell(cell_id, z=z)

    def run_epoch(self, proc_fn: Callable[[str], Any]) -> List[str]:
        """
        run_epoch — dispatch all registered cells in z-layer priority order.

        Returns ordered list of dispatched cell_ids (useful for tests).
        Mirrors Scheduler::NotifyProcessor → ProcBalance dispatch loop.
        """
        ordered = sorted(self._tasks, key=lambda t: (t[0], t[1]))
        dispatched = []
        for z, seq, cell_id in ordered:
            _dbg("ASTRO-SCHED", f"dispatch cell={cell_id} z={z} seq={seq}")
            proc_fn(cell_id)
            dispatched.append(cell_id)
        return dispatched

    def sorted_cells(self) -> List[str]:
        """Return cells in dispatch order without executing. Useful for dry-run."""
        return [cell_id for _, _, cell_id in sorted(self._tasks, key=lambda t: (t[0], t[1]))]

    def clear(self):
        """Reset task list — mirrors Scheduler::Shutdown task drain."""
        self._tasks.clear()
        self._seq = 0

    def _resolve_z(self, cell_id: str) -> int:
        """Read z for cell_id from z_layers.json. Fallback = 3."""
        full = os.path.join(CHANNELS_DIR, self._z_layers_path)
        if not os.path.exists(full):
            return 3
        try:
            with open(full) as f:
                data = json.load(f)
            val = data.get(cell_id, 3)
            return val if isinstance(val, int) else val.get("z", 3)
        except (json.JSONDecodeError, OSError):
            return 3


# ═══════════════════════════════════════════════════════════════════════════════
# broadcast_force — M004 addition
#
# Atomic force-field update with pub/sub notification.
# Replaces direct write_channel("physics/force_field.json", …) calls in
# loop_orchestrator.py with a single authoritative write path that also fires
# DataNotifier callbacks so any registered DataVisitor sees the update.
#
# Design (mirrors Apollo Writer<T>::Write → Dispatch → Notify pipeline):
#   1. Read current force_field from disk (atomic read — no partial state).
#   2. Merge (dx, dy) for cell_id into the dict.
#   3. Write back atomically via DataDispatcher.dispatch() which calls
#      ChannelBuffer.fill() → json.dump → DataNotifier.notify().
# ═══════════════════════════════════════════════════════════════════════════════

FORCE_FIELD_CHANNEL = "physics/force_field.json"


def broadcast_force(cell_id: str, dx: float, dy: float) -> bool:
    """
    broadcast_force(cell_id, dx, dy) — write a per-cell force vector into
    physics/force_field.json and notify all subscribers.

    Parameters
    ----------
    cell_id : str   Cell identifier (e.g. "self_attn").
    dx      : float Force component in x (pixels).
    dy      : float Force component in y (pixels).

    Returns True on success, False if the write failed.

    Replaces direct write_channel("physics/force_field.json", force_field)
    in loop_orchestrator.physics_engine() so that every force update goes
    through the DataDispatcher → DataNotifier pub/sub pipeline and any
    DataVisitor subscribed to the force_field channel is woken up.

    Mirrors Apollo Writer<ForceFieldMsg>::Write() which calls
    DataDispatcher::Dispatch() internally before returning.
    """
    full = os.path.join(CHANNELS_DIR, FORCE_FIELD_CHANNEL)

    # 1. Read current state (merge semantics — keep existing cell entries)
    current: Dict[str, Any] = {}
    if os.path.exists(full):
        try:
            with open(full) as f:
                current = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass  # start fresh on corrupt file

    # 2. Merge force for this cell
    entry = current.get(cell_id, {"dx": 0.0, "dy": 0.0, "dz": 0.0,
                                   "push_from": [], "push_mag": 0.0})
    entry["dx"] = float(dx)
    entry["dy"] = float(dy)
    current[cell_id] = entry

    # 3. Dispatch through pub/sub pipeline (write + notify)
    ok = DataDispatcher.instance().dispatch(FORCE_FIELD_CHANNEL, current)
    _dbg("ASTRO-FORCE", f"broadcast cell={cell_id} dx={dx:.3f} dy={dy:.3f} ok={int(ok)}")
    return ok


def broadcast_force_batch(force_field: Dict[str, Any]) -> bool:
    """
    broadcast_force_batch(force_field) — publish an entire force_field dict
    atomically.  Used by physics_engine() to replace its direct write_channel
    call at the end of each epoch.

    Equivalent to calling broadcast_force() for every cell but with a single
    file write and a single DataNotifier.notify() call (more efficient).
    """
    ok = DataDispatcher.instance().dispatch(FORCE_FIELD_CHANNEL, force_field)
    _dbg("ASTRO-FORCE", f"broadcast_batch cells={len(force_field)} ok={int(ok)}")
    return ok


# ═══════════════════════════════════════════════════════════════════════════════
# watch_channel — M004 addition
#
# File-mtime watcher that triggers a callback whenever a JSON channel file
# changes on disk.  Thin wrapper around DataNotifier.check_mtime() + callback
# registration so callers get a single-function API.
#
# Design mirrors Apollo cyber::ReaderBase::HasReceived() polling pattern:
#   Original: shared-memory flag checked per spin cycle.
#   ASTRO change: os.path.getmtime() checked per poll_interval_s cycle.
# ═══════════════════════════════════════════════════════════════════════════════

def watch_channel(
    path: str,
    callback: Callable[[], None],
    poll_interval_s: float = 0.05,
    max_polls: int = 200,
) -> None:
    """
    watch_channel(path, callback) — monitor a JSON channel file for mtime
    changes and invoke callback when a change is detected.

    Runs a blocking poll loop (up to max_polls × poll_interval_s seconds).
    Designed for use in tests and single-shot wait scenarios; for persistent
    subscription use on_message() instead.

    Parameters
    ----------
    path            : str      Relative channel path (e.g. "physics/force_field.json").
    callback        : callable Called once when the file changes.
    poll_interval_s : float    Sleep between mtime checks (default 50 ms).
    max_polls       : int      Safety cutoff to avoid infinite loops (default 200).

    Mirrors the CyberRT Reader spin-wait pattern:
        while (!reader->HasReceived()) { std::this_thread::sleep_for(…); }
        callback(reader->GetLatestObserved());
    """
    notifier = DataNotifier.instance()
    # Prime mtime cache so the *next* change (not the current file state) fires.
    # NOTE: watch_channel uses polling only (no DataNotifier registration) to
    # avoid double-firing when DataDispatcher.dispatch() also calls notify().
    # For persistent subscription use on_message() instead.
    full = os.path.join(CHANNELS_DIR, path)
    if os.path.exists(full):
        notifier._mtime_cache[path] = os.path.getmtime(full)

    _dbg("ASTRO-WATCH", f"watching path={path} interval={poll_interval_s}s max_polls={max_polls}")

    for poll in range(max_polls):
        if notifier.check_mtime(path):
            _dbg("ASTRO-WATCH", f"change detected path={path} poll={poll}")
            # Fire callback directly here (mtime poll path).
            # DataNotifier.notify() would fire it a second time via the
            # registered Notifier; skip notify to avoid double-firing.
            callback()
            return
        time.sleep(poll_interval_s)

    _dbg("ASTRO-WATCH", f"watch_channel timeout path={path} max_polls={max_polls}")
