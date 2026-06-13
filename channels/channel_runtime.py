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


# ═══════════════════════════════════════════════════════════════════════════════
# AstroCellReader / AstroCellWriter
# Ported from upstream/apollo-cyber/node/reader.h + writer.h
#
# Upstream originals (Apollo CyberRT):
#   Reader<MessageT>  — subscribes a channel, registers CallbackFunc<M>,
#                       maintains a Blocker publish/observe dual-queue,
#                       joins topology via ReceiverManager.
#   Writer<MessageT>  — publishes through a Transmitter, joins topology.
#   ReaderBase        — holds RoleAttributes (channel_name, channel_id, qos).
#   WriterBase        — holds RoleAttributes, mutex-guarded init flag.
#
# Algorithm changes from Apollo original (20 % rule):
#   1. template<MessageT>    → payload type is plain dict (JSON-serialisable).
#   2. protobuf RoleAttributes → simple dataclass with channel_path string.
#   3. channel_id uint64_t hash → string path  ("cell/self_attn/bbox.json").
#   4. Blocker publish/observe dual-queue → deque with configurable capacity.
#   5. ReceiverManager / TransmitterPtr → DataDispatcher + DataNotifier
#      already live in this file; AstroCellReader wires into them directly.
#   6. Topology JoinTheTopology / LeaveTheTopology → lightweight registry
#      dict inside ChannelRegistry singleton (no shared-memory topology map).
#   7. Reader coroutine (RoutineFactory / Scheduler) → synchronous callback
#      list; callbacks are fired inline from AstroCellWriter.write().
#   8. HasWriter / HasReader → ChannelRegistry.has_writer / has_reader queries.
#   9. GetDelaySec uses monotonic wall-clock (time.monotonic) instead of
#      Apollo cyber::Time::Now().ToSecond(), preserving the two-sample delta
#      logic (max of age-since-last and inter-message interval).
#  10. AcquireMessage() on Writer → allocate_message() returns fresh empty dict.
#
# Debug prefix: [ASTRO-CHANNEL] — matches the instrumentation tag in reader.h
# and writer.h so grep across C++ and Python logs produces a unified trace.
# ═══════════════════════════════════════════════════════════════════════════════

import collections
import threading

DEFAULT_PENDING_QUEUE_SIZE: int = 1  # mirrors const uint32_t in reader.h


# ───────────────────────────────────────────────────────────────────────────────
# ChannelRegistry — lightweight topology map
#
# Mirrors the ChannelManager (service_discovery/channel_manager.h) role:
# tracks which writers and readers are currently active on each channel path.
# Apollo uses a distributed shared-memory AtomicHashMap keyed by channel_id
# (uint64_t).  Here we use a plain dict keyed by channel_path (string) guarded
# by a threading.Lock so the registry is safe for multi-threaded test scenarios.
# ───────────────────────────────────────────────────────────────────────────────

class ChannelRegistry:
    """
    Singleton topology registry.

    Maps channel_path → {writers: set[str], readers: set[str]}
    where each entry is the role_id (node_name + channel_path hash).

    Mirrors ChannelManager::Join / Leave / HasWriter / HasReader.
    """
    _instance: Optional["ChannelRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self):
        self._map: Dict[str, Dict[str, set]] = {}   # path → {writers, readers}
        self._mu = threading.Lock()

    @classmethod
    def instance(cls) -> "ChannelRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        with cls._lock:
            cls._instance = None

    def _ensure(self, channel_path: str):
        if channel_path not in self._map:
            self._map[channel_path] = {"writers": set(), "readers": set()}

    def join_writer(self, channel_path: str, role_id: str):
        with self._mu:
            self._ensure(channel_path)
            self._map[channel_path]["writers"].add(role_id)

    def leave_writer(self, channel_path: str, role_id: str):
        with self._mu:
            if channel_path in self._map:
                self._map[channel_path]["writers"].discard(role_id)

    def join_reader(self, channel_path: str, role_id: str):
        with self._mu:
            self._ensure(channel_path)
            self._map[channel_path]["readers"].add(role_id)

    def leave_reader(self, channel_path: str, role_id: str):
        with self._mu:
            if channel_path in self._map:
                self._map[channel_path]["readers"].discard(role_id)

    def has_writer(self, channel_path: str) -> bool:
        with self._mu:
            return bool(self._map.get(channel_path, {}).get("writers"))

    def has_reader(self, channel_path: str) -> bool:
        with self._mu:
            return bool(self._map.get(channel_path, {}).get("readers"))

    def writer_count(self, channel_path: str) -> int:
        with self._mu:
            return len(self._map.get(channel_path, {}).get("writers", set()))

    def reader_count(self, channel_path: str) -> int:
        with self._mu:
            return len(self._map.get(channel_path, {}).get("readers", set()))


# ───────────────────────────────────────────────────────────────────────────────
# AstroBlocker — dual-queue (publish_queue / observe_queue)
#
# Ports blocker::Blocker<MessageT> (cyber/blocker/blocker.h).
# Apollo Blocker keeps two std::list<shared_ptr<T>>:
#   published_ — messages arrive here (Publish); bounded by capacity.
#   observed_  — snapshots here on Observe(); drained by reader callbacks.
# ASTRO change: lists → collections.deque with maxlen for O(1) bounded append.
# ───────────────────────────────────────────────────────────────────────────────

class AstroBlocker:
    """
    Dual-queue message buffer for AstroCellReader.

    publish_queue  — hot side: new messages enqueued here by AstroCellWriter.
    observe_queue  — cold side: snapshot of publish_queue taken on observe().

    Mirrors blocker::Blocker<MessageT> with deque instead of std::list.
    """

    def __init__(self, channel_path: str, capacity: int = DEFAULT_PENDING_QUEUE_SIZE):
        self.channel_path = channel_path
        self._capacity = max(1, capacity)
        self._publish_queue: collections.deque = collections.deque(maxlen=self._capacity)
        self._observe_queue: collections.deque = collections.deque()
        self._mu = threading.Lock()

    # ── publish side ──────────────────────────────────────────────────────────

    def publish(self, msg: Any):
        """Publish — append to publish_queue; oldest drops when at capacity."""
        with self._mu:
            self._publish_queue.append(msg)

    def published_size(self) -> int:
        with self._mu:
            return len(self._publish_queue)

    def is_published_empty(self) -> bool:
        with self._mu:
            return len(self._publish_queue) == 0

    def clear_published(self):
        with self._mu:
            self._publish_queue.clear()

    # ── observe side ─────────────────────────────────────────────────────────

    def observe(self):
        """Observe — snapshot publish_queue into observe_queue. Mirrors Blocker::Observe."""
        with self._mu:
            self._observe_queue = collections.deque(self._publish_queue)

    def is_observed_empty(self) -> bool:
        with self._mu:
            return len(self._observe_queue) == 0

    def get_latest_observed(self) -> Optional[Any]:
        """GetLatestObservedPtr — rightmost element of observe_queue."""
        with self._mu:
            return self._observe_queue[-1] if self._observe_queue else None

    def get_oldest_observed(self) -> Optional[Any]:
        """GetOldestObservedPtr — leftmost element of observe_queue."""
        with self._mu:
            return self._observe_queue[0] if self._observe_queue else None

    def observed_list(self) -> List[Any]:
        """Return a snapshot list of the observe_queue (oldest → newest)."""
        with self._mu:
            return list(self._observe_queue)

    def clear_observed(self):
        with self._mu:
            self._observe_queue.clear()

    # ── capacity ─────────────────────────────────────────────────────────────

    @property
    def capacity(self) -> int:
        return self._capacity

    def set_capacity(self, depth: int):
        with self._mu:
            self._capacity = max(1, depth)
            new_q: collections.deque = collections.deque(self._publish_queue, maxlen=self._capacity)
            self._publish_queue = new_q


# ───────────────────────────────────────────────────────────────────────────────
# AstroCellReader — port of Reader<MessageT>
#
# Lifecycle mirrors Reader:
#   __init__  ≈ Reader(role_attr, reader_func, pending_queue_size)
#   init()    ≈ Reader::Init() — register with DataDispatcher + topology
#   shutdown()≈ Reader::Shutdown() — deregister
#   enqueue() ≈ Reader::Enqueue() — push msg, fire callback
#   observe() ≈ Reader::Observe() — snapshot publish → observe queue
#
# ASTRO algorithm delta:
#   • Coroutine / RoutineFactory → inline synchronous callback from enqueue().
#   • DataVisitor channel_id (uint64) → channel_path string.
#   • Blocker uses AstroBlocker (deque-backed) instead of std::list.
#   • JoinTheTopology → ChannelRegistry.join_reader.
#   • Timestamp uses time.monotonic() (wall-clock, no NTP jumps).
# ───────────────────────────────────────────────────────────────────────────────

class AstroCellReader:
    """
    Subscriber for a single channel path.

    Ports Reader<MessageT> from upstream/apollo-cyber/node/reader.h.

    Parameters
    ----------
    channel_path      : str      Channel path (replaces role_attr.channel_name).
    reader_func       : callable Optional callback(msg: dict) — fires on each enqueue.
    pending_queue_size: int      Blocker publish-queue depth (default 1).
    node_name         : str      Logical node name for topology registry.
    """

    def __init__(
        self,
        channel_path: str,
        reader_func: Optional[Callable[[Any], None]] = None,
        pending_queue_size: int = DEFAULT_PENDING_QUEUE_SIZE,
        node_name: str = "astro_node",
    ):
        # role attributes (flat, no protobuf)
        self.channel_path = channel_path
        self.node_name = node_name
        self._pending_queue_size = pending_queue_size
        self._reader_func = reader_func

        # Blocker — mirrors BlockerPtr in Reader<MessageT>
        self._blocker = AstroBlocker(channel_path, capacity=pending_queue_size)

        # timing — mirrors latest_recv_time_sec_ / second_to_lastest_recv_time_sec_
        self._latest_recv_mono: float = -1.0
        self._second_to_latest_recv_mono: float = -1.0

        # init flag — mirrors std::atomic<bool> init_
        self._init: bool = False
        self._mu = threading.Lock()

        # topology role_id — deterministic string, mirrors role_attr.id (HashValue)
        self._role_id = f"{node_name}::{channel_path}"

        # channel buffer registered with DataDispatcher
        self._channel_buf: Optional[ChannelBuffer] = None

        _dbg("ASTRO-CHANNEL",
             f"AstroCellReader ctor channel={channel_path} node={node_name} "
             f"pending_q={pending_queue_size}")

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def init(self) -> bool:
        """
        Init — mirrors Reader::Init().

        1. Creates a ChannelBuffer and registers it with DataDispatcher.
        2. Registers an on_message (DataNotifier) callback that calls enqueue().
        3. Joins ChannelRegistry as READER.
        """
        with self._mu:
            if self._init:
                return True

            # Register ChannelBuffer — mirrors ReceiverManager::GetReceiver
            self._channel_buf = ChannelBuffer(self.channel_path,
                                              max_size=self._pending_queue_size)
            DataDispatcher.instance().add_buffer(self._channel_buf)

            # Register notification callback — mirrors RoutineFactory / Scheduler::CreateTask.
            # When DataDispatcher.dispatch() fires DataNotifier.notify(), this callback
            # pulls the latest message from the buffer and calls enqueue().
            def _on_notify():
                msg = self._channel_buf.latest() if self._channel_buf else None
                if msg is not None:
                    self.enqueue(msg)

            DataNotifier.instance().add_notifier(
                self.channel_path, Notifier(_on_notify)
            )

            # Join topology
            ChannelRegistry.instance().join_reader(self.channel_path, self._role_id)

            self._init = True

        _dbg("ASTRO-CHANNEL",
             f"AstroCellReader init ok channel={self.channel_path} role={self._role_id}")
        return True

    def shutdown(self):
        """Shutdown — mirrors Reader::Shutdown(). Leaves topology; does not drain queue."""
        with self._mu:
            if not self._init:
                return
            self._init = False

        ChannelRegistry.instance().leave_reader(self.channel_path, self._role_id)
        _dbg("ASTRO-CHANNEL",
             f"AstroCellReader shutdown channel={self.channel_path}")

    # ── message flow ─────────────────────────────────────────────────────────

    def enqueue(self, msg: Any):
        """
        Enqueue — mirrors Reader::Enqueue().

        Updates timing accumulators, pushes msg to Blocker's publish_queue,
        fires reader_func callback if set.

        [ASTRO-CHANNEL] log mirrors the fprintf in reader.h Enqueue().
        """
        self._second_to_latest_recv_mono = self._latest_recv_mono
        self._latest_recv_mono = time.monotonic()

        seq = self._blocker.published_size()   # approx seq from publish depth
        ts_us = int(self._latest_recv_mono * 1e6)

        _dbg("ASTRO-CHANNEL",
             f"Reader enqueued message on channel '{self.channel_path}' "
             f"| seq={seq} | timestamp={ts_us}")

        self._blocker.publish(msg)

        if self._reader_func is not None:
            self._reader_func(msg)

    def observe(self):
        """
        Observe — mirrors Reader::Observe().

        Snapshots publish_queue into observe_queue so callers can iterate
        without racing with concurrent enqueue() calls.

        [ASTRO-CHANNEL] log mirrors the fprintf in reader.h Observe().
        """
        seq = self._blocker.published_size()
        ts_us = int(time.monotonic() * 1e6)
        _dbg("ASTRO-CHANNEL",
             f"Reader observed message on channel '{self.channel_path}' "
             f"| seq={seq} | timestamp={ts_us}")
        self._blocker.observe()

    # ── query interface ───────────────────────────────────────────────────────

    def has_received(self) -> bool:
        """HasReceived — publish_queue non-empty. Mirrors Reader::HasReceived."""
        return not self._blocker.is_published_empty()

    def empty(self) -> bool:
        """Empty — observe_queue is empty. Mirrors Reader::Empty."""
        return self._blocker.is_observed_empty()

    def get_delay_sec(self) -> float:
        """
        GetDelaySec — mirrors Reader::GetDelaySec().

        Returns max(age_since_last, inter_message_interval) so callers can
        detect both stale sources and bursty delivery.
        """
        if self._latest_recv_mono < 0:
            return -1.0
        now = time.monotonic()
        if self._second_to_latest_recv_mono < 0:
            return now - self._latest_recv_mono
        return max(
            now - self._latest_recv_mono,
            self._latest_recv_mono - self._second_to_latest_recv_mono,
        )

    def pending_queue_size(self) -> int:
        """PendingQueueSize — mirrors Reader::PendingQueueSize."""
        return self._pending_queue_size

    def get_latest_observed(self) -> Optional[Any]:
        """GetLatestObserved — mirrors Reader::GetLatestObserved."""
        return self._blocker.get_latest_observed()

    def get_oldest_observed(self) -> Optional[Any]:
        """GetOldestObserved — mirrors Reader::GetOldestObserved."""
        return self._blocker.get_oldest_observed()

    def clear_data(self):
        """ClearData — mirrors Reader::ClearData. Drains both queues."""
        self._blocker.clear_published()
        self._blocker.clear_observed()

    def set_history_depth(self, depth: int):
        """SetHistoryDepth — resize Blocker capacity. Mirrors Reader::SetHistoryDepth."""
        self._blocker.set_capacity(depth)

    def get_history_depth(self) -> int:
        """GetHistoryDepth — mirrors Reader::GetHistoryDepth."""
        return self._blocker.capacity

    def has_writer(self) -> bool:
        """HasWriter — query topology registry. Mirrors Reader::HasWriter."""
        if not self._init:
            return False
        return ChannelRegistry.instance().has_writer(self.channel_path)

    def is_init(self) -> bool:
        return self._init

    def get_channel_name(self) -> str:
        return self.channel_path


# ───────────────────────────────────────────────────────────────────────────────
# AstroCellWriter — port of Writer<MessageT>
#
# Lifecycle mirrors Writer:
#   __init__  ≈ Writer(role_attr)
#   init()    ≈ Writer::Init() — join topology
#   shutdown()≈ Writer::Shutdown()
#   write()   ≈ Writer::Write(msg) — dispatch + notify
#
# ASTRO algorithm delta:
#   • TransmitterPtr → DataDispatcher.dispatch() (file-backed JSON channel).
#   • protobuf serialise → json.dumps (JSON wire format).
#   • sizeof(MessageT) → len(json.dumps(msg)) for payload size logging.
#   • JoinTheTopology → ChannelRegistry.join_writer.
#   • AcquireMessage() → allocate_message() returning empty dict.
# ───────────────────────────────────────────────────────────────────────────────

class AstroCellWriter:
    """
    Publisher for a single channel path.

    Ports Writer<MessageT> from upstream/apollo-cyber/node/writer.h.

    Parameters
    ----------
    channel_path : str  Channel path (replaces role_attr.channel_name).
    node_name    : str  Logical node name for topology registry.
    """

    def __init__(self, channel_path: str, node_name: str = "astro_node"):
        self.channel_path = channel_path
        self.node_name = node_name

        # init flag — mirrors WriterBase bool init_ (mutex-guarded)
        self._init: bool = False
        self._mu = threading.Lock()

        # topology role_id
        self._role_id = f"{node_name}::{channel_path}"

        _dbg("ASTRO-CHANNEL",
             f"AstroCellWriter ctor channel={channel_path} node={node_name}")

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def init(self) -> bool:
        """
        Init — mirrors Writer::Init().

        Registers with ChannelRegistry as WRITER so readers can discover
        this writer via AstroCellReader.has_writer().
        """
        with self._mu:
            if self._init:
                return True
            self._init = True

        ChannelRegistry.instance().join_writer(self.channel_path, self._role_id)
        _dbg("ASTRO-CHANNEL",
             f"AstroCellWriter init ok channel={self.channel_path} role={self._role_id}")
        return True

    def shutdown(self):
        """Shutdown — mirrors Writer::Shutdown. Leaves topology."""
        with self._mu:
            if not self._init:
                return
            self._init = False

        ChannelRegistry.instance().leave_writer(self.channel_path, self._role_id)
        _dbg("ASTRO-CHANNEL",
             f"AstroCellWriter shutdown channel={self.channel_path}")

    # ── publish ───────────────────────────────────────────────────────────────

    def write(self, msg: Any) -> bool:
        """
        Write — mirrors Writer::Write(msg).

        Serialises msg to JSON (replaces protobuf Transmit), routes through
        DataDispatcher (replaces transport::Transmitter::Transmit), which
        writes the file channel and fires DataNotifier callbacks.

        [ASTRO-CHANNEL] log mirrors the fprintf in writer.h Write().
        """
        if not self.is_init():
            return False

        # Approximate payload size — mirrors sizeof(MessageT) in writer.h log
        try:
            payload_bytes = len(json.dumps(msg).encode())
        except (TypeError, ValueError):
            payload_bytes = 0

        _dbg("ASTRO-CHANNEL",
             f"Writer published to channel '{self.channel_path}' "
             f"| size={payload_bytes} bytes")

        return DataDispatcher.instance().dispatch(self.channel_path, msg)

    def acquire_message(self) -> dict:
        """
        acquire_message — mirrors Writer::AcquireMessage().

        Returns a fresh empty dict; callers mutate it then pass to write().
        AcquireMessage in Apollo pre-allocates from a shared-memory pool;
        here we simply return {} since Python heap allocation is trivially cheap.
        """
        return {}

    # ── query interface ───────────────────────────────────────────────────────

    def has_reader(self) -> bool:
        """HasReader — query topology registry. Mirrors Writer::HasReader."""
        if not self.is_init():
            return False
        return ChannelRegistry.instance().has_reader(self.channel_path)

    def is_init(self) -> bool:
        with self._mu:
            return self._init

    def get_channel_name(self) -> str:
        return self.channel_path


# ───────────────────────────────────────────────────────────────────────────────
# Convenience factory — mirrors Node::CreateReader / Node::CreateWriter
# ───────────────────────────────────────────────────────────────────────────────

def create_reader(
    channel_path: str,
    reader_func: Optional[Callable[[Any], None]] = None,
    pending_queue_size: int = DEFAULT_PENDING_QUEUE_SIZE,
    node_name: str = "astro_node",
    auto_init: bool = True,
) -> AstroCellReader:
    """
    create_reader — factory mirroring Node::CreateReader<MessageT>().

    Creates and optionally inits an AstroCellReader in one call.
    """
    r = AstroCellReader(channel_path, reader_func, pending_queue_size, node_name)
    if auto_init:
        r.init()
    return r


def create_writer(
    channel_path: str,
    node_name: str = "astro_node",
    auto_init: bool = True,
) -> AstroCellWriter:
    """
    create_writer — factory mirroring Node::CreateWriter<MessageT>().

    Creates and optionally inits an AstroCellWriter in one call.
    """
    w = AstroCellWriter(channel_path, node_name)
    if auto_init:
        w.init()
    return w


# ═══════════════════════════════════════════════════════════════════════════════
# AstroListenerHandler — ported from
#   upstream/apollo-cyber/transport/message/listener_handler.h
#
# Original: template<MessageT> with Signal<shared_ptr<MessageT>, MessageInfo>
#   • Connect(self_id, listener) — single-cast slot (no oppo filter)
#   • Connect(self_id, oppo_id, listener) — filtered slot per sender
#   • Disconnect(self_id) / Disconnect(self_id, oppo_id)
#   • Run(msg, msg_info) — fires signal_ then per-oppo signals_[oppo_id]
#   • RunFromString(str, msg_info) — parse then Run (proto deserialise)
#
# ASTRO changes (20% algorithm delta):
#   1. MessageT template → duck-typed Python (any dict / str / object).
#   2. Signal<> + base::Connection<> → plain list of callables (simpler).
#   3. RunFromString: proto parse → json.loads (file-channel serialisation).
#   4. oppo_id: uint64 hash → str (channel path or role name).
#   5. AtomicRWLock → threading.Lock (Python GIL provides safety for reads;
#      lock only for structural mutations, matching the write-guard pattern).
#   6. is_raw_message_ flag preserved as _is_raw (bool).
# ═══════════════════════════════════════════════════════════════════════════════

import threading


class AstroListenerHandler:
    """
    Callback manager for one message type on one channel.

    Mirrors ListenerHandler<MessageT> from listener_handler.h.

    Slots are keyed by self_id (str).  Per-sender (oppo_id) slots are stored
    in _oppo_slots: {oppo_id: {self_id: callable}}.

    Usage::

        handler = AstroListenerHandler()
        handler.connect("reader_A", lambda msg, info: print(msg))
        handler.run({"value": 1}, {"sender": "tx_0"})
        handler.disconnect("reader_A")
    """

    def __init__(self, is_raw: bool = False):
        self._is_raw: bool = is_raw          # mirrors is_raw_message_
        self._lock = threading.Lock()
        # mirrors signal_conns_: {self_id: callable}
        self._slots: Dict[str, Callable] = {}
        # mirrors signals_conns_: {oppo_id: {self_id: callable}}
        self._oppo_slots: Dict[str, Dict[str, Callable]] = {}

    # ── connect ────────────────────────────────────────────────────────────────

    def connect(self, self_id: str, listener: Callable) -> None:
        """
        Connect(self_id, listener) — register a broadcast listener.
        Mirrors ListenerHandler::Connect(uint64_t self_id, const Listener&).
        """
        with self._lock:
            self._slots[self_id] = listener
        _dbg("ASTRO-LISTENER",
             f"connect self={self_id} op=broadcast slots={len(self._slots)}")

    def connect_filtered(self, self_id: str, oppo_id: str,
                         listener: Callable) -> None:
        """
        Connect(self_id, oppo_id, listener) — register a sender-filtered slot.
        Mirrors ListenerHandler::Connect(self_id, oppo_id, listener).
        """
        with self._lock:
            if oppo_id not in self._oppo_slots:
                self._oppo_slots[oppo_id] = {}
            self._oppo_slots[oppo_id][self_id] = listener
        _dbg("ASTRO-LISTENER",
             f"connect_filtered self={self_id} oppo={oppo_id} op=filtered")

    # ── disconnect ─────────────────────────────────────────────────────────────

    def disconnect(self, self_id: str) -> None:
        """
        Disconnect(self_id) — remove broadcast slot.
        Mirrors ListenerHandler::Disconnect(uint64_t self_id).
        """
        with self._lock:
            self._slots.pop(self_id, None)
        _dbg("ASTRO-LISTENER",
             f"disconnect self={self_id} op=broadcast remaining={len(self._slots)}")

    def disconnect_filtered(self, self_id: str, oppo_id: str) -> None:
        """
        Disconnect(self_id, oppo_id) — remove sender-filtered slot.
        Mirrors ListenerHandler::Disconnect(self_id, oppo_id).
        """
        with self._lock:
            bucket = self._oppo_slots.get(oppo_id)
            if bucket is not None:
                bucket.pop(self_id, None)
                if not bucket:
                    del self._oppo_slots[oppo_id]
        _dbg("ASTRO-LISTENER",
             f"disconnect_filtered self={self_id} oppo={oppo_id}")

    # ── run ────────────────────────────────────────────────────────────────────

    def run(self, msg: Any, msg_info: Dict) -> None:
        """
        Run(msg, msg_info) — fire broadcast slots then sender-filtered slots.
        Mirrors ListenerHandler::Run(const Message&, const MessageInfo&).

        oppo_id is resolved from msg_info["sender_id"] (str key, not hash).
        """
        with self._lock:
            broadcast = list(self._slots.values())
            sender_id: str = msg_info.get("sender_id", "")
            filtered = list(
                self._oppo_slots.get(sender_id, {}).values()
            )

        for cb in broadcast:
            try:
                cb(msg, msg_info)
            except Exception as exc:  # noqa: BLE001
                _dbg("ASTRO-LISTENER", f"run broadcast exc={exc}")

        for cb in filtered:
            try:
                cb(msg, msg_info)
            except Exception as exc:  # noqa: BLE001
                _dbg("ASTRO-LISTENER", f"run filtered exc={exc}")

    def run_from_string(self, raw: str, msg_info: Dict) -> None:
        """
        RunFromString(str, msg_info) — deserialise then run.
        Mirrors ListenerHandler::RunFromString (proto parse → json.loads here).
        """
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError as exc:
            _dbg("ASTRO-LISTENER", f"run_from_string parse_error={exc}")
            return
        self.run(msg, msg_info)


# ═══════════════════════════════════════════════════════════════════════════════
# AstroIntraDispatcher — ported from
#   upstream/apollo-cyber/transport/dispatcher/intra_dispatcher.h
#   + dispatcher/dispatcher.h  (base class behaviour inlined)
#
# Original: singleton with AtomicHashMap<channel_id→ListenerHandlerBasePtr>
#   and a ChannelChain (multi-type fan-out per channel).
#   AddListener → GetHandler → handler.Connect(self_id, listener)
#   OnMessage   → handler.Run(message, message_info)
#   RemoveListener → handler.Disconnect(self_id)
#
# ASTRO changes (20% algorithm delta):
#   1. channel_id: uint64_t → str (channel path, e.g. "cell/self_attn/bbox.json").
#   2. AtomicHashMap → plain dict protected by threading.Lock.
#   3. ChannelChain (multi-type fan-out) → single AstroListenerHandler per
#      channel (one message type per channel in Astro's file-channel model).
#   4. is_shutdown_ atomic bool → Python threading.Event.
#   5. DECLARE_SINGLETON → class-level _instance + instance() classmethod.
#   6. MessageInfo typed proto → plain dict with "sender_id" / "seq_num" keys.
#   7. SHM and RTPS dispatch paths dropped (INTRA only in this class).
# ═══════════════════════════════════════════════════════════════════════════════

class AstroIntraDispatcher:
    """
    Process-internal message dispatcher — Python port of IntraDispatcher.

    Maintains a per-channel AstroListenerHandler registry.  When a transmitter
    calls on_message() the handler fans the message out to all connected
    receivers in the same process, mirroring the INTRA transport mode.

    Singleton: obtain via AstroIntraDispatcher.instance().

    Lifecycle::

        disp = AstroIntraDispatcher.instance()
        disp.add_listener("cell/self_attn/out.json", "reader_A",
                          lambda msg, info: print("got", msg))
        disp.on_message("cell/self_attn/out.json", {"value": 42},
                        {"sender_id": "tx_0", "seq_num": 1})
        disp.remove_listener("cell/self_attn/out.json", "reader_A")
    """

    _instance: Optional["AstroIntraDispatcher"] = None
    _instance_lock = threading.Lock()

    def __init__(self):
        # mirrors is_shutdown_ std::atomic<bool>
        self._shutdown = threading.Event()
        # mirrors msg_listeners_ AtomicHashMap<channel_id, ListenerHandlerBasePtr>
        self._handlers: Dict[str, AstroListenerHandler] = {}
        self._lock = threading.Lock()
        _dbg("ASTRO-INTRA", "AstroIntraDispatcher constructed")

    # ── singleton ──────────────────────────────────────────────────────────────

    @classmethod
    def instance(cls) -> "AstroIntraDispatcher":
        """DECLARE_SINGLETON equivalent."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Test helper — tear down singleton and release handlers."""
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.shutdown()
            cls._instance = None

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """
        Shutdown() — mark dispatcher as shut down; subsequent calls are no-ops.
        Mirrors Dispatcher::Shutdown().
        """
        self._shutdown.set()
        _dbg("ASTRO-INTRA", "shutdown called")

    def is_shutdown(self) -> bool:
        return self._shutdown.is_set()

    # ── internal handler access ────────────────────────────────────────────────

    def _get_or_create_handler(self, channel_id: str) -> AstroListenerHandler:
        """
        GetHandler<MessageT> equivalent — always returns the channel's handler,
        creating it on first access (mirrors IntraDispatcher::GetHandler).
        """
        with self._lock:
            if channel_id not in self._handlers:
                self._handlers[channel_id] = AstroListenerHandler()
                _dbg("ASTRO-INTRA",
                     f"new handler ch={channel_id} total={len(self._handlers)}")
            return self._handlers[channel_id]

    # ── add_listener ──────────────────────────────────────────────────────────

    def add_listener(self, channel_id: str, self_id: str,
                     listener: Callable) -> None:
        """
        AddListener(self_attr, listener) — broadcast subscription.
        Mirrors IntraDispatcher::AddListener<MessageT>(self_attr, listener).
        """
        if self.is_shutdown():
            return
        handler = self._get_or_create_handler(channel_id)
        handler.connect(self_id, listener)
        _dbg("ASTRO-INTRA",
             f"add_listener ch={channel_id} self={self_id} mode=broadcast")

    def add_listener_filtered(self, channel_id: str, self_id: str,
                              oppo_id: str, listener: Callable) -> None:
        """
        AddListener(self_attr, opposite_attr, listener) — sender-filtered.
        Mirrors IntraDispatcher::AddListener<MessageT>(self_attr, opposite_attr, …).
        """
        if self.is_shutdown():
            return
        handler = self._get_or_create_handler(channel_id)
        handler.connect_filtered(self_id, oppo_id, listener)
        _dbg("ASTRO-INTRA",
             f"add_listener_filtered ch={channel_id} self={self_id} oppo={oppo_id}")

    # ── remove_listener ───────────────────────────────────────────────────────

    def remove_listener(self, channel_id: str, self_id: str) -> None:
        """
        RemoveListener(self_attr) — disconnect broadcast slot.
        Mirrors IntraDispatcher::RemoveListener<MessageT>(self_attr).
        """
        if self.is_shutdown():
            return
        with self._lock:
            handler = self._handlers.get(channel_id)
        if handler:
            handler.disconnect(self_id)
            _dbg("ASTRO-INTRA",
                 f"remove_listener ch={channel_id} self={self_id}")

    def remove_listener_filtered(self, channel_id: str, self_id: str,
                                 oppo_id: str) -> None:
        """
        RemoveListener(self_attr, opposite_attr) — disconnect filtered slot.
        Mirrors IntraDispatcher::RemoveListener<MessageT>(self_attr, opposite_attr).
        """
        if self.is_shutdown():
            return
        with self._lock:
            handler = self._handlers.get(channel_id)
        if handler:
            handler.disconnect_filtered(self_id, oppo_id)
            _dbg("ASTRO-INTRA",
                 f"remove_listener_filtered ch={channel_id} self={self_id} oppo={oppo_id}")

    # ── on_message ────────────────────────────────────────────────────────────

    def on_message(self, channel_id: str, message: Any,
                   msg_info: Optional[Dict] = None) -> None:
        """
        OnMessage<MessageT>(channel_id, message, message_info) — deliver a
        message to all listeners on the channel.

        Mirrors IntraDispatcher::OnMessage which resolves the handler from
        msg_listeners_ and calls handler->Run(message, message_info).

        msg_info dict keys (mirrors MessageInfo proto):
            sender_id : str   — transmitter role name / path
            seq_num   : int   — monotonically increasing sequence number
        """
        if self.is_shutdown():
            return
        if msg_info is None:
            msg_info = {"sender_id": "", "seq_num": 0}

        with self._lock:
            handler = self._handlers.get(channel_id)

        if handler is None:
            _dbg("ASTRO-INTRA",
                 f"on_message ch={channel_id} no_handler — drop")
            return

        _dbg("ASTRO-INTRA",
             f"on_message ch={channel_id} sender={msg_info.get('sender_id','')} "
             f"seq={msg_info.get('seq_num', 0)}")
        handler.run(message, msg_info)

    # ── has_channel ───────────────────────────────────────────────────────────

    def has_channel(self, channel_id: str) -> bool:
        """HasChannel — True if any listener is registered on channel_id."""
        with self._lock:
            return channel_id in self._handlers


# ═══════════════════════════════════════════════════════════════════════════════
# AstroTransmitter / AstroIntraTransmitter — ported from
#   upstream/apollo-cyber/transport/transmitter/transmitter.h
#   (IntraTransmitter specialisation kept; SHM dropped; RTPS → HTTP POST)
#
# Original Transmitter<M>:
#   • seq_num_ uint64, msg_info_ MessageInfo
#   • Enable() / Disable() pure-virtual
#   • Transmit(msg) → sets seq_num, send_time → calls Transmit(msg, msg_info)
#   • AcquireMessage() pure-virtual (arena allocation hook)
#
# ASTRO changes:
#   1. Template type M → duck-typed Python (any dict/str).
#   2. msg_info_ proto → dict with sender_id / seq_num / send_time_us.
#   3. AcquireMessage() → returns {} (no arena; heap-allocated every time).
#   4. IntraTransmitter: Transmit calls AstroIntraDispatcher.on_message().
#   5. RtpsTransmitter: Transmit does HTTP POST to self._endpoint URL.
#   6. ShmTransmitter: not ported (inter-process shared memory not needed).
# ═══════════════════════════════════════════════════════════════════════════════

import urllib.request


class AstroTransmitterBase:
    """
    Abstract base — mirrors Transmitter<M> from transmitter.h.

    Subclasses implement enable(), disable(), transmit_impl(msg, msg_info).
    """

    def __init__(self, channel_id: str, sender_id: str = ""):
        self.channel_id: str = channel_id
        self.sender_id: str = sender_id or channel_id
        self._seq_num: int = 0           # mirrors seq_num_
        self._enabled: bool = False      # mirrors Enable/Disable state

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def enable(self) -> None:
        """Enable() — activate transmitter. Mirrors Transmitter::Enable()."""
        self._enabled = True
        _dbg("ASTRO-TX", f"enable ch={self.channel_id} sender={self.sender_id}")

    def disable(self) -> None:
        """Disable() — deactivate transmitter. Mirrors Transmitter::Disable()."""
        self._enabled = False
        _dbg("ASTRO-TX", f"disable ch={self.channel_id}")

    # ── sequence ───────────────────────────────────────────────────────────────

    def _next_seq(self) -> int:
        """NextSeqNum() — increment and return seq_num_."""
        self._seq_num += 1
        return self._seq_num

    # ── acquire ────────────────────────────────────────────────────────────────

    def acquire_message(self) -> Dict:
        """
        AcquireMessage() — return an empty message container.
        Mirrors the arena-allocation hook; here we just return {}.
        """
        return {}

    # ── transmit ───────────────────────────────────────────────────────────────

    def transmit(self, msg: Any) -> bool:
        """
        Transmit(msg) — stamp msg_info then delegate to transmit_impl.
        Mirrors Transmitter<M>::Transmit(const MessagePtr& msg) which sets
        seq_num, msg_seq_num, send_time before calling Transmit(msg, msg_info).
        """
        if not self._enabled:
            _dbg("ASTRO-TX",
                 f"transmit ch={self.channel_id} disabled — drop")
            return False
        msg_info = {
            "sender_id": self.sender_id,
            "seq_num": self._next_seq(),
            "send_time_us": int(time.time() * 1_000_000),
        }
        _dbg("ASTRO-TX",
             f"transmit ch={self.channel_id} seq={msg_info['seq_num']}")
        return self._transmit_impl(msg, msg_info)

    def _transmit_impl(self, msg: Any, msg_info: Dict) -> bool:
        raise NotImplementedError


class AstroIntraTransmitter(AstroTransmitterBase):
    """
    Intra-process transmitter — mirrors IntraTransmitter<M>.

    Transmit calls AstroIntraDispatcher.on_message(), delivering the message
    directly to in-process listeners without any serialisation.
    """

    def __init__(self, channel_id: str, sender_id: str = ""):
        super().__init__(channel_id, sender_id)
        self._dispatcher = AstroIntraDispatcher.instance()

    def _transmit_impl(self, msg: Any, msg_info: Dict) -> bool:
        self._dispatcher.on_message(self.channel_id, msg, msg_info)
        return True


class AstroRtpsTransmitter(AstroTransmitterBase):
    """
    RTPS-mode transmitter mapped to HTTP POST.

    Original RtpsTransmitter<M> sends via DDS/RTPS participant.
    ASTRO substitution: POST JSON to self._endpoint (configured at init).

    Mirrors the enable/disable participant lifecycle:
        Enable()  → store endpoint URL (participant.start() analogue)
        Disable() → clear endpoint URL  (participant.stop() analogue)
    """

    def __init__(self, channel_id: str, sender_id: str = "",
                 endpoint: str = ""):
        super().__init__(channel_id, sender_id)
        self._endpoint: str = endpoint   # HTTP URL for POST delivery

    def enable(self) -> None:
        super().enable()
        _dbg("ASTRO-RTPS",
             f"enable ch={self.channel_id} endpoint={self._endpoint}")

    def disable(self) -> None:
        super().disable()
        _dbg("ASTRO-RTPS", f"disable ch={self.channel_id}")

    def _transmit_impl(self, msg: Any, msg_info: Dict) -> bool:
        """
        Transmit via HTTP POST — replaces DDS participant.write().
        Payload: {"channel_id": …, "msg_info": …, "data": msg}.
        Returns True on HTTP 2xx, False otherwise (connection errors → False).
        """
        if not self._endpoint:
            _dbg("ASTRO-RTPS",
                 f"transmit ch={self.channel_id} no_endpoint — drop")
            return False
        payload = json.dumps({
            "channel_id": self.channel_id,
            "msg_info": msg_info,
            "data": msg,
        }).encode()
        req = urllib.request.Request(
            self._endpoint, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=2) as resp:
                ok = 200 <= resp.status < 300
            _dbg("ASTRO-RTPS",
                 f"post ch={self.channel_id} status={'ok' if ok else 'err'}")
            return ok
        except Exception as exc:  # noqa: BLE001
            _dbg("ASTRO-RTPS",
                 f"post ch={self.channel_id} exc={exc}")
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# AstroReceiver — ported from
#   upstream/apollo-cyber/transport/receiver/receiver.h
#
# Original Receiver<M>:
#   • Holds MessageListener (callback: (msg, msg_info, role_attr) → void)
#   • Enable() / Disable() pure-virtual (register / deregister from dispatcher)
#   • OnNewMessage(msg, msg_info) → calls msg_listener_
#
# ASTRO changes:
#   1. Template type M → duck-typed Python.
#   2. role_attr proto → dict with "channel_id" / "role_id" keys.
#   3. IntraReceiver: Enable registers with AstroIntraDispatcher;
#      Disable unregisters.
#   4. MessageListener signature: (msg, msg_info, role_attr) → same as C++.
# ═══════════════════════════════════════════════════════════════════════════════

class AstroIntraReceiver:
    """
    Intra-process receiver — mirrors IntraReceiver<M> from receiver.h.

    Enable()  → AstroIntraDispatcher.add_listener(channel_id, role_id, cb)
    Disable() → AstroIntraDispatcher.remove_listener(channel_id, role_id)
    OnNewMessage delivered via AstroListenerHandler.run().

    Usage::

        def my_handler(msg, msg_info, role_attr):
            print("received", msg)

        rx = AstroIntraReceiver("cell/self_attn/out.json",
                                "reader_A", my_handler)
        rx.enable()
        # … transmitter publishes …
        rx.disable()
    """

    def __init__(self, channel_id: str, role_id: str,
                 msg_listener: Callable):
        self.channel_id: str = channel_id
        self.role_id: str = role_id
        self._role_attr: Dict = {"channel_id": channel_id, "role_id": role_id}
        self._dispatcher = AstroIntraDispatcher.instance()
        self._enabled: bool = False

        # Wrap msg_listener to inject role_attr (matches C++ OnNewMessage sig).
        def _cb(msg: Any, msg_info: Dict) -> None:
            try:
                msg_listener(msg, msg_info, self._role_attr)
            except Exception as exc:  # noqa: BLE001
                _dbg("ASTRO-RX", f"listener exc ch={channel_id} exc={exc}")

        self._cb = _cb

    def enable(self) -> None:
        """
        Enable() — register with dispatcher.
        Mirrors IntraReceiver::Enable() which calls IntraDispatcher::AddListener.
        """
        if not self._enabled:
            self._dispatcher.add_listener(
                self.channel_id, self.role_id, self._cb)
            self._enabled = True
            _dbg("ASTRO-RX",
                 f"enable ch={self.channel_id} role={self.role_id}")

    def disable(self) -> None:
        """
        Disable() — unregister from dispatcher.
        Mirrors IntraReceiver::Disable() → IntraDispatcher::RemoveListener.
        """
        if self._enabled:
            self._dispatcher.remove_listener(
                self.channel_id, self.role_id)
            self._enabled = False
            _dbg("ASTRO-RX",
                 f"disable ch={self.channel_id} role={self.role_id}")


# ═══════════════════════════════════════════════════════════════════════════════
# AstroTransport — ported from
#   upstream/apollo-cyber/transport/transport.h
#
# Original Transport singleton:
#   • CreateTransmitter<M>(attr, mode) → Transmitter<M> (INTRA/SHM/RTPS/HYBRID)
#   • CreateReceiver<M>(attr, listener, mode) → Receiver<M>
#   • Holds IntraDispatcher*, ShmDispatcher*, RtpsDispatcher*, Participant*
#
# ASTRO changes (20% algorithm delta):
#   1. Template generics → duck-typed Python factory methods.
#   2. INTRA mode → AstroIntraTransmitter / AstroIntraReceiver.
#   3. SHM mode → dropped (no inter-process SHM in Astro single-process model).
#   4. RTPS mode → AstroRtpsTransmitter (HTTP POST) / no RTPS receiver.
#   5. HYBRID mode → INTRA (default for in-process cell pub/sub).
#   6. Participant (DDS) → optional HTTP endpoint string.
#   7. ASTRO_TRANS_VERBOSE env var preserved for debug logging.
# ═══════════════════════════════════════════════════════════════════════════════

_TRANS_VERBOSE = os.environ.get("ASTRO_TRANS_VERBOSE", "0") == "1"


def _trans_dbg(channel: str, mode: str, role: str, op: str) -> None:
    """
    ASTRO_TRANS_DBG macro equivalent.
    Mirrors: AINFO << "[ASTRO-TRANS] ch=" << ch << " mode=" << mode …
    """
    if _TRANS_VERBOSE:
        print(f"[ASTRO-TRANS] ch={channel} mode={mode} role={role} op={op}")


class AstroTransport:
    """
    Transport factory — Python port of Transport singleton from transport.h.

    Supported modes (OptionalMode enum analogues):
        "INTRA"  — AstroIntraTransmitter / AstroIntraReceiver (default)
        "RTPS"   — AstroRtpsTransmitter / (no receiver — pull model)
        "HYBRID" — treated as INTRA

    SHM is intentionally not ported.

    Usage::

        transport = AstroTransport.instance()
        tx = transport.create_transmitter("cell/self_attn/out.json",
                                          sender_id="tx_A")
        tx.enable()

        rx = transport.create_receiver("cell/self_attn/out.json",
                                       role_id="rx_B",
                                       listener=lambda m, i, a: print(m))
        rx.enable()

        tx.transmit({"value": 1})
    """

    _instance: Optional["AstroTransport"] = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._shutdown = threading.Event()
        # holds refs so GC doesn't collect live transmitters/receivers
        self._transmitters: List[AstroTransmitterBase] = []
        self._receivers: List[AstroIntraReceiver] = []
        _dbg("ASTRO-TRANS", "AstroTransport constructed")

    # ── singleton ──────────────────────────────────────────────────────────────

    @classmethod
    def instance(cls) -> "AstroTransport":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.shutdown()
            cls._instance = None

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """
        Shutdown() — disable all transmitters and receivers.
        Mirrors Transport::Shutdown() which shuts down dispatcher singletons.
        """
        self._shutdown.set()
        for tx in self._transmitters:
            tx.disable()
        for rx in self._receivers:
            rx.disable()
        AstroIntraDispatcher.instance().shutdown()
        _dbg("ASTRO-TRANS", "shutdown complete")

    def is_shutdown(self) -> bool:
        return self._shutdown.is_set()

    # ── create_transmitter ────────────────────────────────────────────────────

    def create_transmitter(
        self,
        channel_id: str,
        sender_id: str = "",
        mode: str = "INTRA",
        rtps_endpoint: str = "",
    ) -> AstroTransmitterBase:
        """
        CreateTransmitter<M>(attr, mode) — factory for transmitters.

        Parameters
        ----------
        channel_id    : str  Channel path (e.g. "cell/self_attn/out.json").
        sender_id     : str  Role identifier (default = channel_id).
        mode          : str  "INTRA" | "RTPS" | "HYBRID" (HYBRID → INTRA).
        rtps_endpoint : str  HTTP URL for RTPS mode.

        Returns an enabled transmitter.
        """
        if self.is_shutdown():
            _dbg("ASTRO-TRANS", "create_transmitter: shutdown — skip")
            raise RuntimeError("AstroTransport is shut down")

        _trans_dbg(channel_id, mode, "tx", "create")

        if mode == "RTPS":
            tx: AstroTransmitterBase = AstroRtpsTransmitter(
                channel_id, sender_id, endpoint=rtps_endpoint)
        else:
            # INTRA or HYBRID → INTRA
            tx = AstroIntraTransmitter(channel_id, sender_id)

        tx.enable()
        _trans_dbg(channel_id, mode, "tx", "enable")
        self._transmitters.append(tx)
        return tx

    # ── create_receiver ───────────────────────────────────────────────────────

    def create_receiver(
        self,
        channel_id: str,
        role_id: str,
        listener: Callable,
        mode: str = "INTRA",
    ) -> AstroIntraReceiver:
        """
        CreateReceiver<M>(attr, listener, mode) — factory for receivers.

        Parameters
        ----------
        channel_id : str      Channel path.
        role_id    : str      Unique role / reader name.
        listener   : callable Signature: (msg, msg_info, role_attr) → None.
        mode       : str      "INTRA" | "HYBRID" (RTPS receive not ported).

        Returns an enabled receiver.
        """
        if self.is_shutdown():
            raise RuntimeError("AstroTransport is shut down")

        _trans_dbg(channel_id, mode, "rx", "create")

        rx = AstroIntraReceiver(channel_id, role_id, listener)
        rx.enable()
        _trans_dbg(channel_id, mode, "rx", "enable")
        self._receivers.append(rx)
        return rx


# ═══════════════════════════════════════════════════════════════════════════════
# AstroCacheBuffer — ported from upstream/apollo-cyber/data/cache_buffer.h
#
# Original C++: CacheBuffer<T> circular ring (head_/tail_ uint64_t, mutex,
#   optional FusionCallback on Fill()).
# ASTRO changes (20%): capacity sentinel → _slot(pos) helper; uint64_t counters
#   → Python int; std::function → Optional[Callable]; std::mutex → threading.Lock;
#   typed template T → Any.  All structural invariants (Head/Tail/Size/Empty/Full)
#   unchanged from Apollo original.
# ═══════════════════════════════════════════════════════════════════════════════

import threading


class AstroCacheBuffer:
    """Epoch-indexed ring buffer — port of apollo::cyber::data::CacheBuffer."""

    def __init__(self, size: int):
        self._capacity: int = size + 1          # C++: capacity_ = size + 1
        self._buffer: List[Any] = [None] * self._capacity
        self._head: int = 0
        self._tail: int = 0
        self._lock = threading.Lock()
        self._fusion_cb: Optional[Callable[[Any], None]] = None

    @property
    def lock(self) -> threading.Lock:
        return self._lock

    def capacity(self) -> int:
        return self._capacity

    def head(self) -> int:
        return self._head + 1

    def tail(self) -> int:
        return self._tail

    def size(self) -> int:
        return self._tail - self._head

    def empty(self) -> bool:
        return self._tail == 0

    def full(self) -> bool:
        return self._capacity - 1 == self._tail - self._head

    def _slot(self, pos: int) -> int:
        """GetIndex — modular slot (pos % capacity)."""
        return pos % self._capacity

    def at(self, pos: int) -> Any:
        return self._buffer[self._slot(pos)]

    def front(self) -> Any:
        return self._buffer[self._slot(self._head + 1)]

    def back(self) -> Any:
        return self._buffer[self._slot(self._tail)]

    def set_fusion_callback(self, cb: Callable[[Any], None]):
        """SetFusionCallback — hook used by AstroAllLatest to intercept Fill()."""
        self._fusion_cb = cb

    def fill(self, value: Any):
        """
        Fill — write to ring or delegate to FusionCallback.

        When full, evicts head (overwrites oldest slot), advancing both
        head_ and tail_.  Mirrors C++ Fill() exactly.
        """
        if self._fusion_cb is not None:
            self._fusion_cb(value)
            return
        if self.full():
            self._buffer[self._slot(self._head)] = value
            self._head += 1
            self._tail += 1
        else:
            self._buffer[self._slot(self._tail + 1)] = value
            self._tail += 1


# ═══════════════════════════════════════════════════════════════════════════════
# AstroChannelBuffer — ported from upstream/apollo-cyber/data/channel_buffer.h
#
# Original C++: ChannelBuffer<T> epoch-aware wrapper (Fetch/Latest/FetchMulti).
# ASTRO changes (20%): channel_id uint64_t → str; shared_ptr → direct ref;
#   *index out-param → (new_index, value) return pair; vector* → List return;
#   lock scope preserved per-method.
# ═══════════════════════════════════════════════════════════════════════════════

class AstroChannelBuffer:
    """
    Epoch-constrained reader over AstroCacheBuffer.

    fetch(index)    -> (new_index, value) | (index, None)
    latest()        -> value | None
    fetch_multi(n)  -> List[value], oldest-first
    """

    def __init__(self, channel_id: str, buf: AstroCacheBuffer):
        self._channel_id = channel_id
        self._buf = buf
        _dbg("ASTRO-BUFFER",
             f"AstroChannelBuffer ctor: channel={channel_id} capacity={buf.capacity()}")

    @property
    def channel_id(self) -> str:
        return self._channel_id

    @property
    def buffer(self) -> AstroCacheBuffer:
        return self._buf

    def fetch(self, index: int) -> Tuple[int, Optional[Any]]:
        """Epoch-indexed sequential read.  Overflow detection → WARN + snap."""
        with self._buf.lock:
            if self._buf.empty():
                _dbg("ASTRO-BUFFER", f"Fetch: empty ch={self._channel_id}")
                return index, None
            if index == 0:
                new_idx = self._buf.tail()
                _dbg("ASTRO-BUFFER", f"Fetch: cold-start snap Tail={new_idx} ch={self._channel_id}")
                return new_idx, self._buf.at(new_idx)
            if index == self._buf.tail() + 1:
                _dbg("ASTRO-BUFFER", f"Fetch: epoch current no new data idx={index} ch={self._channel_id}")
                return index, None
            if index < self._buf.head():
                drop = self._buf.tail() - index
                print(
                    f"[ASTRO-BUFFER] Fetch: epoch overflow on channel[{self._channel_id}] "
                    f"drop_messages=[{drop}] stale_epoch_index=[{index}] "
                    f"current_epoch_tail=[{self._buf.tail()}] — "
                    f"snapping cursor to current epoch boundary"
                )
                new_idx = self._buf.tail()
                return new_idx, self._buf.at(new_idx)
            _dbg("ASTRO-BUFFER", f"Fetch: reading idx={index} ch={self._channel_id}")
            return index, self._buf.at(index)

    def latest(self) -> Optional[Any]:
        """Non-destructive tail peek.  Mirrors ChannelBuffer<T>::Latest()."""
        with self._buf.lock:
            if self._buf.empty():
                _dbg("ASTRO-BUFFER", f"Latest: empty ch={self._channel_id}")
                return None
            _dbg("ASTRO-BUFFER", f"Latest: tail={self._buf.tail()} ch={self._channel_id}")
            return self._buf.back()

    def fetch_multi(self, fetch_size: int) -> List[Any]:
        """Bulk read up to fetch_size entries, oldest-first.  Mirrors FetchMulti()."""
        with self._buf.lock:
            if self._buf.empty():
                _dbg("ASTRO-BUFFER", f"FetchMulti: empty ch={self._channel_id}")
                return []
            num = min(self._buf.size(), fetch_size)
            start = self._buf.tail() - num + 1
            result = [self._buf.at(i) for i in range(start, self._buf.tail() + 1)]
            _dbg("ASTRO-BUFFER", f"FetchMulti: count={len(result)} ch={self._channel_id}")
            return result


# ═══════════════════════════════════════════════════════════════════════════════
# AstroAllLatest — ported from upstream/apollo-cyber/data/fusion/all_latest.h
#
# Original C++: AllLatest<M0[,M1[,M2[,M3]]]> partial-specialisation template.
#   FusionCallback on primary (M0) snapshots Latest from secondaries and pushes
#   the N-tuple into a separate fusion CacheBuffer.
# ASTRO changes (20%): three C++ specialisations → single Python class with
#   List[AstroChannelBuffer]; std::tuple<shared_ptr<Mx>,...> → Python tuple;
#   fusion ring sized to primary.capacity()-1 (unchanged); fprintf→print;
#   bool+out-params → Optional[tuple].
# ═══════════════════════════════════════════════════════════════════════════════

class AstroAllLatest:
    """
    Multi-channel AllLatest fusion — port of apollo::cyber::data::fusion::AllLatest.

    Supports 2-, 3-, and 4-channel fusion (len(buffers) in {2,3,4}).
    buffers[0] is the primary; buffers[1:] are secondaries.

    When primary.buffer.fill(m0) is called the FusionCallback fires:
      - Latest() is called on each secondary.
      - If any secondary is None the tuple is dropped.
      - Otherwise (m0, *secondaries) is pushed to the internal fusion ring.

    fusion(index) reads epoch-indexed fused tuples from the fusion ring.
    """

    def __init__(self, buffers: List[AstroChannelBuffer]):
        if not (2 <= len(buffers) <= 4):
            raise ValueError("AstroAllLatest requires 2-4 channel buffers (M0-M3)")
        self._primary: AstroChannelBuffer = buffers[0]
        self._secondaries: List[AstroChannelBuffer] = buffers[1:]
        self._arity: int = len(buffers)
        # Fusion ring — capacity mirrors C++: primary.capacity()-1
        fusion_cap = self._primary.buffer.capacity() - 1
        self._fusion_buf = AstroChannelBuffer(
            self._primary.channel_id,
            AstroCacheBuffer(fusion_cap),
        )
        self._primary.buffer.set_fusion_callback(self._fusion_callback)
        _dbg("ASTRO-FUSION",
             f"AstroAllLatest ctor: primary={self._primary.channel_id} "
             f"arity={self._arity} fusion_cap={fusion_cap}")

    def _fusion_callback(self, m0: Any):
        """FusionCallback — mirrors AllLatest C++ lambda: Latest() all secondaries, push tuple."""
        vals: List[Any] = []
        for sec in self._secondaries:
            v = sec.latest()
            if v is None:
                _dbg("ASTRO-FUSION", f"fusion_callback: secondary={sec.channel_id} not ready — drop")
                return
            vals.append(v)
        print(
            f"[ASTRO-FUSION] AllLatest fusion triggered | "
            f"primary_channel='{self._primary.channel_id}' | "
            f"secondary_channels={len(self._secondaries)} | "
            f"fused_cells={len(vals)}"
        )
        with self._fusion_buf.buffer.lock:
            self._fusion_buf.buffer.fill(tuple([m0] + vals))

    def fusion(self, index: int) -> Tuple[int, Optional[tuple]]:
        """
        fusion(index) -> (new_index, (m0, m1[, m2[, m3]])) | (index, None)

        Mirrors AllLatest::Fusion() -> buffer_fusion_.Fetch(index, data).
        Caller advances index by 1 after each successful read.
        """
        return self._fusion_buf.fetch(index)

    @property
    def arity(self) -> int:
        return self._arity

    @property
    def primary_channel_id(self) -> str:
        return self._primary.channel_id

    @property
    def secondary_channel_ids(self) -> List[str]:
        return [s.channel_id for s in self._secondaries]

# AstroChannelManager — ported from
#   upstream/apollo-cyber/service_discovery/specific_manager/channel_manager.h
#   upstream/apollo-cyber/service_discovery/specific_manager/channel_manager.cc
#
# 原典：Apollo ChannelManager 作为 Manager 的子类，以 MultiValueWarehouse
# （AtomicHashMap<uint64_t, RolePtr>）为索引，分别维护 channel_writers_ /
# channel_readers_（按 channel_id 索引）和 node_writers_ / node_readers_
# （按 node_id 索引），以及一个有向图 node_graph_（Edge=channel_name）。
#
# 算法改动（20% 规则）：
#   1. channel_id: uint64_t hash → str（channel_path，如 "cell/self_attn/bbox.json"）
#   2. MultiValueWarehouse AtomicHashMap → dict[str, dict[str, RoleRecord]]，
#      外层 key = channel_path，内层 key = role_id；threading.Lock 替代 AtomicRWLock。
#   3. node_graph_ Graph (Edge/Vertice) → dict[str, set[str]]，
#      记录 {node_name → set of downstream_node_names}（写侧→读侧有向边）。
#   4. RoleAttributes proto → RoleRecord dataclass（只保留 node_name/channel_path/
#      role_id/msg_type/timestamp 五个字段；proto_desc 省略）。
#   5. ExemptedMessageTypes unordered_set → frozenset[str]，
#      默认豁免 RawMessage / PyMessageWrap 对应的 Python 标记字符串。
#   6. Join/Leave 通过 ChangeEvent 触发 OnTopoChange 回调（Signal 机制）而非
#      Apollo 的 Manager::Notify() → base::Signal。
#   7. GetChannelNames: dedup 由 set 保证，无需 unordered_set + std::move。
#   8. HasWriter/HasReader: 直接检查 dict 非空，O(1)，无需 Search(key) 间接查询。
#   9. ScanMessageType 保留逻辑但改为 Python logging（无 AERROR 宏）。
#  10. OnTopoModuleLeave: 按 (host_name, process_id) 前缀匹配 role_id 批量清除。
#
# Debug prefix: [ASTRO-CHANMGR] — 对应 channel_manager.cc 中的 ADEBUG 前缀。
# ═══════════════════════════════════════════════════════════════════════════════

import dataclasses
import logging
import time as _time

_chanmgr_log = logging.getLogger("astro.channel_manager")


@dataclasses.dataclass
class RoleRecord:
    """
    轻量角色属性 — 对应 apollo::cyber::proto::RoleAttributes。

    Apollo 原典使用 protobuf RoleAttributes（含 host_name, process_id,
    node_name, node_id, channel_name, channel_id, message_type, proto_desc
    等字段）。此处精简为 Astro 文件通道模型所需的五个核心字段。
    """
    node_name: str       # proto.node_name
    channel_path: str    # proto.channel_name（string path 替代 channel_id uint64）
    role_id: str         # proto.id（确定性字符串 hash：node_name::channel_path）
    msg_type: str = ""   # proto.message_type
    timestamp: float = dataclasses.field(default_factory=_time.monotonic)
    # host_name + process_id — 用于 OnTopoModuleLeave 批量清除
    host_name: str = "localhost"
    process_id: int = 0


@dataclasses.dataclass
class ChangeEvent:
    """
    拓扑变化事件 — 对应 apollo::cyber::proto::ChangeMsg。

    operate_type: "JOIN" | "LEAVE"
    role_type:    "WRITER" | "READER"
    role:         RoleRecord
    timestamp:    float (monotonic)
    """
    operate_type: str    # "JOIN" | "LEAVE"
    role_type: str       # "WRITER" | "READER"
    role: RoleRecord
    timestamp: float = dataclasses.field(default_factory=_time.monotonic)


# 豁免类型集合 — 对应 ExemptedMessageTypes
# Apollo 默认豁免 RawMessage / PyMessageWrap；此处以约定字符串替代 MessageType<T>()
_EXEMPTED_MSG_TYPES: frozenset = frozenset({"__raw__", "__py_wrap__"})


class AstroChannelManager:
    """
    Astro 通道拓扑管理器 — Python port of ChannelManager。

    维护两组索引：
      _channel_writers / _channel_readers : Dict[channel_path, Dict[role_id, RoleRecord]]
      _node_writers    / _node_readers    : Dict[node_name,    Dict[role_id, RoleRecord]]

    以及有向图（node_graph_）和 OnTopoChange 回调列表。

    线程安全：所有结构修改持有 self._mu。
    """

    _instance: Optional["AstroChannelManager"] = None
    _instance_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._mu = threading.Lock()

        # channel_id(str) → {role_id: RoleRecord}
        self._channel_writers: Dict[str, Dict[str, RoleRecord]] = {}
        self._channel_readers: Dict[str, Dict[str, RoleRecord]] = {}

        # node_name → {role_id: RoleRecord}
        self._node_writers: Dict[str, Dict[str, RoleRecord]] = {}
        self._node_readers: Dict[str, Dict[str, RoleRecord]] = {}

        # node_graph_: node_name → set of downstream node_names
        # 有向边：writer_node → reader_node，边 value = channel_path
        # 简化为 {src_node: {dst_node}} — 足够 HasWriter/HasReader 之上的上下游查询
        self._node_graph: Dict[str, set] = {}

        # 拓扑变化回调列表 — 对应 change_signal_
        self._topo_listeners: List[Callable[[ChangeEvent], None]] = []

        _chanmgr_log.debug("[ASTRO-CHANMGR] AstroChannelManager constructed")

    # ── singleton ────────────────────────────────────────────────────────────

    @classmethod
    def instance(cls) -> "AstroChannelManager":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._instance_lock:
            cls._instance = None

    # ── topo-change signal ───────────────────────────────────────────────────

    def add_topo_listener(
        self, callback: Callable[[ChangeEvent], None]
    ) -> Callable[[ChangeEvent], None]:
        """
        AddChangeListener — 注册拓扑变化回调。
        返回 callback 本身，便于调用方持有引用后 remove。
        对应 TopologyManager::AddChangeListener / change_signal_.Connect()。
        """
        with self._mu:
            self._topo_listeners.append(callback)
        _chanmgr_log.debug("[ASTRO-CHANMGR] add_topo_listener registered")
        return callback

    def remove_topo_listener(
        self, callback: Callable[[ChangeEvent], None]
    ) -> None:
        """RemoveChangeListener — 注销拓扑变化回调。"""
        with self._mu:
            try:
                self._topo_listeners.remove(callback)
            except ValueError:
                pass

    def _notify(self, event: ChangeEvent) -> None:
        """
        Notify — 触发所有已注册回调。
        对应 Manager::Notify() → change_signal_(msg)。
        回调在锁外串行调用（避免重入死锁，同 Apollo Signal::operator()）。
        """
        with self._mu:
            cbs = list(self._topo_listeners)
        for cb in cbs:
            try:
                cb(event)
            except Exception as exc:  # noqa: BLE001
                _chanmgr_log.warning(
                    "[ASTRO-CHANMGR] topo_listener exception: %s", exc
                )

    # ── internal warehouse helpers ───────────────────────────────────────────

    @staticmethod
    def _make_role_id(node_name: str, channel_path: str) -> str:
        """确定性 role_id — 对应 proto.id = HashValue(node+channel)。"""
        return f"{node_name}::{channel_path}"

    def _add_role(
        self,
        warehouse_ch: Dict[str, Dict[str, RoleRecord]],
        warehouse_nd: Dict[str, Dict[str, RoleRecord]],
        role: RoleRecord,
    ) -> None:
        """MultiValueWarehouse::Add — 双索引写入（channel 维度 + node 维度）。"""
        # channel 维度
        warehouse_ch.setdefault(role.channel_path, {})[role.role_id] = role
        # node 维度
        warehouse_nd.setdefault(role.node_name, {})[role.role_id] = role

    def _remove_role(
        self,
        warehouse_ch: Dict[str, Dict[str, RoleRecord]],
        warehouse_nd: Dict[str, Dict[str, RoleRecord]],
        role: RoleRecord,
    ) -> None:
        """MultiValueWarehouse::Remove — 双索引删除。"""
        bucket_ch = warehouse_ch.get(role.channel_path)
        if bucket_ch:
            bucket_ch.pop(role.role_id, None)
            if not bucket_ch:
                del warehouse_ch[role.channel_path]

        bucket_nd = warehouse_nd.get(role.node_name)
        if bucket_nd:
            bucket_nd.pop(role.role_id, None)
            if not bucket_nd:
                del warehouse_nd[role.node_name]

    # ── graph helpers ────────────────────────────────────────────────────────

    def _graph_insert_writer(self, node_name: str) -> None:
        """node_graph_.Insert(e.set_src(v)) — 确保 writer 节点存在。"""
        self._node_graph.setdefault(node_name, set())

    def _graph_insert_reader(
        self, writer_node: str, reader_node: str
    ) -> None:
        """node_graph_.Insert(e.set_dst(v)) — 建立 writer→reader 有向边。"""
        self._node_graph.setdefault(writer_node, set()).add(reader_node)

    def _graph_delete_edge(
        self, node_a: str, node_b: str, as_writer: bool
    ) -> None:
        """node_graph_.Delete — 移除有向边或孤立顶点。"""
        if as_writer:
            bucket = self._node_graph.get(node_a)
            if bucket is not None:
                bucket.discard(node_b)
        # reader 离开：从所有 writer 节点的邻居集合中删除 reader_node
        else:
            for adj in self._node_graph.values():
                adj.discard(node_a)

    # ── Join / Leave (DisposeJoin / DisposeLeave) ───────────────────────────

    def join(
        self,
        channel_path: str,
        node_name: str,
        role_type: str,
        msg_type: str = "",
        host_name: str = "localhost",
        process_id: int = 0,
    ) -> None:
        """
        Join — 注册 Writer 或 Reader 到拓扑。
        对应 ChannelManager::DisposeJoin(msg)。

        Parameters
        ----------
        channel_path : str  通道路径（替代 channel_name + channel_id）
        node_name    : str  节点名称
        role_type    : str  "WRITER" | "READER"
        msg_type     : str  消息类型标识（可选）
        host_name    : str  主机名（用于 OnTopoModuleLeave）
        process_id   : int  进程 ID
        """
        role_id = self._make_role_id(node_name, channel_path)
        role = RoleRecord(
            node_name=node_name,
            channel_path=channel_path,
            role_id=role_id,
            msg_type=msg_type,
            timestamp=_time.monotonic(),
            host_name=host_name,
            process_id=process_id,
        )

        with self._mu:
            self._scan_message_type(channel_path, msg_type)

            if role_type == "WRITER":
                self._add_role(
                    self._channel_writers, self._node_writers, role
                )
                self._graph_insert_writer(node_name)
            else:
                self._add_role(
                    self._channel_readers, self._node_readers, role
                )
                # 建立所有当前 writer 节点 → reader 节点的有向边
                for wr in self._channel_writers.get(channel_path, {}).values():
                    self._graph_insert_reader(wr.node_name, node_name)

        _chanmgr_log.debug(
            "[ASTRO-CHANMGR] join ch=%s node=%s role=%s",
            channel_path, node_name, role_type,
        )
        event = ChangeEvent(
            operate_type="JOIN", role_type=role_type, role=role
        )
        self._notify(event)

    def leave(
        self,
        channel_path: str,
        node_name: str,
        role_type: str,
    ) -> None:
        """
        Leave — 从拓扑注销 Writer 或 Reader。
        对应 ChannelManager::DisposeLeave(msg)。
        """
        role_id = self._make_role_id(node_name, channel_path)
        role = RoleRecord(
            node_name=node_name,
            channel_path=channel_path,
            role_id=role_id,
        )

        with self._mu:
            if role_type == "WRITER":
                self._remove_role(
                    self._channel_writers, self._node_writers, role
                )
                self._graph_delete_edge(node_name, "", as_writer=True)
            else:
                self._remove_role(
                    self._channel_readers, self._node_readers, role
                )
                self._graph_delete_edge(node_name, "", as_writer=False)

        _chanmgr_log.debug(
            "[ASTRO-CHANMGR] leave ch=%s node=%s role=%s",
            channel_path, node_name, role_type,
        )
        event = ChangeEvent(
            operate_type="LEAVE", role_type=role_type, role=role
        )
        self._notify(event)

    # ── HasWriter / HasReader ────────────────────────────────────────────────

    def has_writer(self, channel_path: str) -> bool:
        """
        HasWriter — 查询通道是否有活跃 Writer。
        对应 ChannelManager::HasWriter(channel_name)。
        """
        with self._mu:
            return bool(self._channel_writers.get(channel_path))

    def has_reader(self, channel_path: str) -> bool:
        """
        HasReader — 查询通道是否有活跃 Reader。
        对应 ChannelManager::HasReader(channel_name)。
        """
        with self._mu:
            return bool(self._channel_readers.get(channel_path))

    # ── GetChannelNames ──────────────────────────────────────────────────────

    def get_channel_names(self) -> List[str]:
        """
        GetChannelNames — 返回拓扑中所有已知通道名称（去重）。
        对应 ChannelManager::GetChannelNames(std::vector<std::string>*)。

        原典：合并 channel_writers_ + channel_readers_ 中所有 role 的
        channel_name 到 unordered_set，再 move 到 vector。
        此处：两个 dict 的键本身已去重，取并集即可。
        """
        with self._mu:
            names: set = set(self._channel_writers.keys()) | set(
                self._channel_readers.keys()
            )
        result = sorted(names)
        _chanmgr_log.debug(
            "[ASTRO-CHANMGR] get_channel_names count=%d", len(result)
        )
        return result

    # ── GetWriters / GetReaders (按通道 / 按节点) ────────────────────────────

    def get_writers_of_channel(self, channel_path: str) -> List[RoleRecord]:
        """GetWritersOfChannel — 返回通道的所有 Writer RoleRecord 列表。"""
        with self._mu:
            bucket = self._channel_writers.get(channel_path, {})
            return list(bucket.values())

    def get_readers_of_channel(self, channel_path: str) -> List[RoleRecord]:
        """GetReadersOfChannel — 返回通道的所有 Reader RoleRecord 列表。"""
        with self._mu:
            bucket = self._channel_readers.get(channel_path, {})
            return list(bucket.values())

    def get_writers_of_node(self, node_name: str) -> List[RoleRecord]:
        """GetWritersOfNode — 返回节点的所有 Writer RoleRecord 列表。"""
        with self._mu:
            bucket = self._node_writers.get(node_name, {})
            return list(bucket.values())

    def get_readers_of_node(self, node_name: str) -> List[RoleRecord]:
        """GetReadersOfNode — 返回节点的所有 Reader RoleRecord 列表。"""
        with self._mu:
            bucket = self._node_readers.get(node_name, {})
            return list(bucket.values())

    # ── GetUpstream / GetDownstream ──────────────────────────────────────────

    def get_upstream_of_node(self, node_name: str) -> List[str]:
        """
        GetUpstreamOfNode — 返回 node_name 的上游节点名称列表。

        逻辑与 channel_manager.cc 相同：
          1. 找 node_name 订阅的所有 channel。
          2. 找这些 channel 的所有 writer 节点。
          原典返回 RoleAttrVec；此处只返回 node_name 字符串列表（更 Pythonic）。
        """
        with self._mu:
            readers = list(
                self._node_readers.get(node_name, {}).values()
            )
        channels = {r.channel_path for r in readers}
        upstreams: set = set()
        for ch in channels:
            for wr in self.get_writers_of_channel(ch):
                if wr.node_name != node_name:
                    upstreams.add(wr.node_name)
        return sorted(upstreams)

    def get_downstream_of_node(self, node_name: str) -> List[str]:
        """
        GetDownstreamOfNode — 返回 node_name 的下游节点名称列表。
        逻辑对称于 get_upstream_of_node。
        """
        with self._mu:
            writers = list(
                self._node_writers.get(node_name, {}).values()
            )
        channels = {w.channel_path for w in writers}
        downstreams: set = set()
        for ch in channels:
            for rd in self.get_readers_of_channel(ch):
                if rd.node_name != node_name:
                    downstreams.add(rd.node_name)
        return sorted(downstreams)

    # ── OnTopoModuleLeave ────────────────────────────────────────────────────

    def on_topo_module_leave(
        self, host_name: str, process_id: int
    ) -> None:
        """
        OnTopoModuleLeave — 批量清除某进程的所有 Writer/Reader。
        对应 ChannelManager::OnTopoModuleLeave(host_name, process_id)。

        原典：用 RoleAttributes{host_name, process_id} 为 key 批量 Search，
        再对每个匹配角色生成 DisposeLeave + Notify ChangeMsg。
        此处：扫描全表，按 role.host_name + role.process_id 前缀匹配。
        """
        _chanmgr_log.debug(
            "[ASTRO-CHANMGR] on_topo_module_leave host=%s pid=%d",
            host_name, process_id,
        )
        to_remove_writers: List[RoleRecord] = []
        to_remove_readers: List[RoleRecord] = []

        with self._mu:
            for bucket in self._channel_writers.values():
                for role in bucket.values():
                    if (
                        role.host_name == host_name
                        and role.process_id == process_id
                    ):
                        to_remove_writers.append(role)
            for bucket in self._channel_readers.values():
                for role in bucket.values():
                    if (
                        role.host_name == host_name
                        and role.process_id == process_id
                    ):
                        to_remove_readers.append(role)

        for role in to_remove_writers:
            self.leave(role.channel_path, role.node_name, "WRITER")
        for role in to_remove_readers:
            self.leave(role.channel_path, role.node_name, "READER")

    # ── ScanMessageType (internal) ───────────────────────────────────────────

    def _scan_message_type(
        self, channel_path: str, new_msg_type: str
    ) -> None:
        """
        ScanMessageType — 检查新角色的消息类型是否与已有 writer/reader 冲突。
        对应 ChannelManager::ScanMessageType(msg)。

        Apollo 原典在 AERROR 级别打印不匹配警告；此处改为 logging.warning。
        豁免类型（_EXEMPTED_MSG_TYPES）不做检查，与原典 exempted_msg_types_ 一致。
        注意：此方法在持锁状态下调用（join() 内部），不重复加锁。
        """
        if new_msg_type in _EXEMPTED_MSG_TYPES or not new_msg_type:
            return

        for wr in self._channel_writers.get(channel_path, {}).values():
            if (
                wr.msg_type
                and wr.msg_type not in _EXEMPTED_MSG_TYPES
                and wr.msg_type != new_msg_type
            ):
                _chanmgr_log.warning(
                    "[ASTRO-CHANMGR] msg_type mismatch ch=%s "
                    "existing_writer_type=%s new_type=%s",
                    channel_path, wr.msg_type, new_msg_type,
                )

        for rd in self._channel_readers.get(channel_path, {}).values():
            if (
                rd.msg_type
                and rd.msg_type not in _EXEMPTED_MSG_TYPES
                and rd.msg_type != new_msg_type
            ):
                _chanmgr_log.warning(
                    "[ASTRO-CHANMGR] msg_type mismatch ch=%s "
                    "existing_reader_type=%s new_type=%s",
                    channel_path, rd.msg_type, new_msg_type,
                )


# ═══════════════════════════════════════════════════════════════════════════════
# AstroTopologyManager — ported from
#   upstream/apollo-cyber/service_discovery/topology_manager.h
#   upstream/apollo-cyber/service_discovery/topology_manager.cc
#
# 原典：TopologyManager 是单例，持有 NodeManager / ChannelManager /
# ServiceManager 三个子管理器，以及一个 DDS/RTPS Participant 用于跨进程
# 拓扑广播。change_signal_ 向所有注册的 ChangeFunc 广播拓扑变化事件。
# OnParticipantChange 在 RTPS Participant 发现/离开时触发，解析
# participant_name（"host_name+process_id"）后调用三个子管理器的
# OnTopoModuleLeave 并广播 ChangeMsg。
#
# 算法改动（20% 规则）：
#   1. RTPS Participant（eprosima fast-rtps）→ 无实际网络层；
#      跨进程发现由外部调用 notify_participant_leave() 模拟。
#   2. NodeManager / ServiceManager → 仅保留 AstroChannelManager；
#      节点/服务维度的拓扑管理通过 AstroChannelManager 间接覆盖。
#   3. DECLARE_SINGLETON → classmethod instance() + _instance。
#   4. ChangeConnection (base::Connection) → 返回 callback 本身（同 add_topo_listener）。
#   5. participant_names_ GUID→str map → _participant_names: dict[str, str]
#      key 改为 participant_name 字符串（进程内无 GUID）。
#   6. Convert(PartInfo) → _convert_participant_event(host_name, process_id, op)。
#   7. 新增 snapshot() — 返回当前拓扑快照 dict，供调试和测试使用。
#      （原典无此接口；TopologyManager 状态散布在三个子管理器中，快照需聚合。）
#   8. OnTopoChange 回调参数改为 ChangeEvent（Astro 本地类型）而非 protobuf ChangeMsg。
#
# Debug prefix: [ASTRO-TOPOMGR] — 对应 topology_manager.cc 中的 ADEBUG 前缀。
# ═══════════════════════════════════════════════════════════════════════════════


class AstroTopologyManager:
    """
    Astro 拓扑总管理器 — Python port of TopologyManager。

    持有 AstroChannelManager 单例，并提供：
      • OnTopoChange 回调注册 / 注销（对应 AddChangeListener / RemoveChangeListener）
      • 参与者 Join/Leave 事件派发（对应 OnParticipantChange）
      • 拓扑快照（snapshot()）

    Singleton: 通过 AstroTopologyManager.instance() 获取。

    Usage::

        topo = AstroTopologyManager.instance()
        conn = topo.add_change_listener(lambda ev: print(ev))

        topo.channel_manager.join("cell/out.json", "node_A", "WRITER")
        topo.channel_manager.join("cell/out.json", "node_B", "READER")

        snap = topo.snapshot()
        print(snap["channels"])          # {"cell/out.json": {writers:[…], readers:[…]}}
        print(snap["nodes"])             # {"node_A": {writers:[…], readers:[]}, …}

        topo.notify_participant_leave("worker-01", 12345)   # 模拟节点离线
        topo.remove_change_listener(conn)
    """

    _instance: Optional["AstroTopologyManager"] = None
    _instance_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._init: bool = False
        self._mu = threading.Lock()

        # 唯一子管理器 — 对应 channel_manager_ shared_ptr
        self._channel_manager: AstroChannelManager = (
            AstroChannelManager.instance()
        )

        # 拓扑变化信号 — 对应 change_signal_
        # list of (callback, id) tuples；id 用于精确 remove
        self._listeners: List[Callable[[ChangeEvent], None]] = []

        # 已知参与者表 — 对应 participant_names_: GUID → participant_name
        # 此处以 participant_key(str) → bool 标记存活状态
        self._participant_names: Dict[str, str] = {}

        # 将 ChannelManager 的拓扑事件转发到 TopologyManager 的监听器
        self._channel_manager.add_topo_listener(self._on_channel_event)

        self._init = True
        _chanmgr_log.debug("[ASTRO-TOPOMGR] AstroTopologyManager constructed")

    # ── singleton ────────────────────────────────────────────────────────────

    @classmethod
    def instance(cls) -> "AstroTopologyManager":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.shutdown()
            cls._instance = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """
        Shutdown — 停止拓扑发现，断开所有回调。
        对应 TopologyManager::Shutdown()。
        """
        with self._mu:
            if not self._init:
                return
            self._init = False
            self._listeners.clear()
            self._participant_names.clear()
        _chanmgr_log.debug("[ASTRO-TOPOMGR] shutdown complete")

    def is_initialized(self) -> bool:
        with self._mu:
            return self._init

    # ── channel_manager accessor ─────────────────────────────────────────────

    @property
    def channel_manager(self) -> AstroChannelManager:
        """
        channel_manager() — 对应 TopologyManager::channel_manager()。
        返回 AstroChannelManager 单例引用。
        """
        return self._channel_manager

    # ── AddChangeListener / RemoveChangeListener ─────────────────────────────

    def add_change_listener(
        self, callback: Callable[[ChangeEvent], None]
    ) -> Callable[[ChangeEvent], None]:
        """
        AddChangeListener — 注册拓扑变化观察者。
        对应 TopologyManager::AddChangeListener(func) → ChangeConnection。

        返回 callback 自身作为 connection handle（同 remove_change_listener 参数）。
        """
        with self._mu:
            self._listeners.append(callback)
        _chanmgr_log.debug("[ASTRO-TOPOMGR] add_change_listener registered")
        return callback

    def remove_change_listener(
        self, callback: Callable[[ChangeEvent], None]
    ) -> None:
        """
        RemoveChangeListener — 注销拓扑变化观察者。
        对应 TopologyManager::RemoveChangeListener(conn)。
        """
        with self._mu:
            try:
                self._listeners.remove(callback)
            except ValueError:
                pass

    def _fire_listeners(self, event: ChangeEvent) -> None:
        """
        change_signal_(msg) — 向所有监听器广播拓扑事件。
        在锁外调用，避免重入死锁（与 Apollo Signal::operator() 相同策略）。
        """
        with self._mu:
            cbs = list(self._listeners)
        for cb in cbs:
            try:
                cb(event)
            except Exception as exc:  # noqa: BLE001
                _chanmgr_log.warning(
                    "[ASTRO-TOPOMGR] change_listener exception: %s", exc
                )

    # ── OnTopoChange 内部转发 ────────────────────────────────────────────────

    def _on_channel_event(self, event: ChangeEvent) -> None:
        """
        OnTopoChange 内部回调 — 将 AstroChannelManager 的事件转发给
        AstroTopologyManager 的外部监听器。

        对应 TopologyManager::OnParticipantChange 调用 change_signal_(msg)
        的最后一步：无论是 JOIN 还是 LEAVE，都广播给所有注册观察者。

        [ASTRO-TOPOMGR] debug: 仅在 VERBOSE 模式下打印（避免性能影响）。
        """
        _chanmgr_log.debug(
            "[ASTRO-TOPOMGR] on_channel_event op=%s role=%s ch=%s node=%s",
            event.operate_type,
            event.role_type,
            event.role.channel_path,
            event.role.node_name,
        )
        if not self.is_initialized():
            return
        self._fire_listeners(event)

    # ── OnParticipantChange / notify_participant_leave ───────────────────────

    def notify_participant_join(
        self, host_name: str, process_id: int
    ) -> None:
        """
        模拟参与者加入 — 对应 OnParticipantChange(DISCOVERED_RTPSPARTICIPANT)。

        原典：从 PartInfo 解析 participant_name（"host+pid"），
        记入 participant_names_，广播 OPT_JOIN ChangeMsg。
        此处：直接接受 host_name + process_id，生成 JOIN ChangeEvent。
        """
        key = f"{host_name}+{process_id}"
        with self._mu:
            if not self._init:
                return
            self._participant_names[key] = key
        _chanmgr_log.debug(
            "[ASTRO-TOPOMGR] participant_join host=%s pid=%d",
            host_name, process_id,
        )
        # 构造一个虚拟 PARTICIPANT JOIN 事件广播给外部监听器
        synthetic_role = RoleRecord(
            node_name=f"{host_name}:{process_id}",
            channel_path="__participant__",
            role_id=key,
            host_name=host_name,
            process_id=process_id,
        )
        event = ChangeEvent(
            operate_type="JOIN",
            role_type="PARTICIPANT",
            role=synthetic_role,
        )
        self._fire_listeners(event)

    def notify_participant_leave(
        self, host_name: str, process_id: int
    ) -> None:
        """
        模拟参与者离开 — 对应 OnParticipantChange(REMOVED/DROPPED_RTPSPARTICIPANT)。

        原典：
          1. 从 participant_names_ 按 GUID 找到 participant_name。
          2. ParseParticipantName → host_name + process_id。
          3. 调用 node_manager_/channel_manager_/service_manager_
             的 OnTopoModuleLeave(host_name, process_id)。
          4. 广播 OPT_LEAVE ChangeMsg。

        此处：直接调用 channel_manager.on_topo_module_leave，
        然后广播合成 LEAVE 事件。
        """
        key = f"{host_name}+{process_id}"
        with self._mu:
            if not self._init:
                return
            self._participant_names.pop(key, None)

        _chanmgr_log.debug(
            "[ASTRO-TOPOMGR] participant_leave host=%s pid=%d — "
            "cleaning channel registrations",
            host_name, process_id,
        )
        # 通知 ChannelManager 清除该进程的所有角色
        self._channel_manager.on_topo_module_leave(host_name, process_id)

        # 广播合成 PARTICIPANT LEAVE 事件
        synthetic_role = RoleRecord(
            node_name=f"{host_name}:{process_id}",
            channel_path="__participant__",
            role_id=key,
            host_name=host_name,
            process_id=process_id,
        )
        event = ChangeEvent(
            operate_type="LEAVE",
            role_type="PARTICIPANT",
            role=synthetic_role,
        )
        self._fire_listeners(event)

    # ── snapshot ─────────────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        """
        snapshot() — 返回当前拓扑的完整快照。

        原典（TopologyManager）无此接口；拓扑状态散布在三个子管理器中，
        调试时需分别查询。此处聚合为单一 dict，结构如下：

        {
          "channels": {
            "<channel_path>": {
              "writers": [{"node_name": …, "msg_type": …, "role_id": …}, …],
              "readers": [{"node_name": …, "msg_type": …, "role_id": …}, …],
            },
            …
          },
          "nodes": {
            "<node_name>": {
              "writers": [<channel_path>, …],
              "readers": [<channel_path>, …],
              "upstream":   [<node_name>, …],
              "downstream": [<node_name>, …],
            },
            …
          },
          "participants": [<"host+pid">, …],
          "snapshot_ts":  <float monotonic>,
        }

        用途：
          • 单元测试断言拓扑状态（无需逐个调用 has_writer/has_reader）。
          • 调试 / SVG 拓扑可视化的数据源。
        """
        ch_mgr = self._channel_manager

        channel_names = ch_mgr.get_channel_names()

        channels_snap: Dict[str, Any] = {}
        all_nodes: set = set()

        for ch in channel_names:
            writers = ch_mgr.get_writers_of_channel(ch)
            readers = ch_mgr.get_readers_of_channel(ch)
            channels_snap[ch] = {
                "writers": [
                    {
                        "node_name": r.node_name,
                        "msg_type": r.msg_type,
                        "role_id": r.role_id,
                        "host_name": r.host_name,
                        "process_id": r.process_id,
                    }
                    for r in writers
                ],
                "readers": [
                    {
                        "node_name": r.node_name,
                        "msg_type": r.msg_type,
                        "role_id": r.role_id,
                        "host_name": r.host_name,
                        "process_id": r.process_id,
                    }
                    for r in readers
                ],
            }
            for r in writers:
                all_nodes.add(r.node_name)
            for r in readers:
                all_nodes.add(r.node_name)

        nodes_snap: Dict[str, Any] = {}
        for node in sorted(all_nodes):
            wrs = ch_mgr.get_writers_of_node(node)
            rds = ch_mgr.get_readers_of_node(node)
            nodes_snap[node] = {
                "writers": sorted({r.channel_path for r in wrs}),
                "readers": sorted({r.channel_path for r in rds}),
                "upstream": ch_mgr.get_upstream_of_node(node),
                "downstream": ch_mgr.get_downstream_of_node(node),
            }

        with self._mu:
            participants = sorted(self._participant_names.keys())

        _chanmgr_log.debug(
            "[ASTRO-TOPOMGR] snapshot channels=%d nodes=%d participants=%d",
            len(channels_snap), len(nodes_snap), len(participants),
        )

        return {
            "channels": channels_snap,
            "nodes": nodes_snap,
            "participants": participants,
            "snapshot_ts": _time.monotonic(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# AstroScheduler + AstroProcessor — ported from
#   upstream/apollo-cyber/scheduler/scheduler.h + scheduler.cc
#   upstream/apollo-cyber/scheduler/processor.h + processor.cc
#   upstream/apollo-cyber/scheduler/policy/scheduler_classic.h
#   upstream/apollo-cyber/scheduler/policy/classic_context.h
#
# Upstream originals (Apollo CyberRT classic policy):
#   Scheduler:
#     CreateTask(factory/func, name, visitor) — register CRoutine in dispatch table;
#       visitor != nullptr → RegisterNotifyCallback → NotifyProcessor on data arrive.
#     NotifyProcessor(crid) — wake the processor owning the coroutine.
#     DispatchTask(cr) — assign CRoutine to a ClassicContext MULTI_PRIO_QUEUE slot
#       keyed by (group_name, priority).
#     CheckSchedStatus() — per-processor utilisation snapshot.
#     Shutdown() — drain epoch slots; stop processor threads.
#   Processor:
#     Run() — spin loop: NextRoutine() → Resume() → Release(); Wait() on idle.
#     Stop() — set running_=false; join thread.
#     BindContext(ctx) — attach ProcessorContext; start thread via std::call_once.
#
# ASTRO changes (20% algorithm delta):
#   1. CRoutine (stackful coroutine)  → plain Python callable (func: () → None).
#   2. AtomicHashMap<uint64_t, …>     → dict[str, …] (task_name → entry).
#   3. MULTI_PRIO_QUEUE (array[20])   → heapq (z_layer as priority key).
#   4. ClassicContext.NextRoutine()   → AstroScheduler._dequeue_task() under lock.
#   5. Processor thread (std::thread) → ThreadPoolExecutor future.
#   6. crid (uint64_t hash)           → task_name (str), matching channel path style.
#   7. DataVisitorBase.RegisterNotifyCallback → on_message callback on channel_path.
#   8. epoch_index_ advance           → AstroScheduler.advance_epoch() (M126 port).
#   9. Snapshot struct                → AstroSnapshot dataclass (preserved fields).
#  10. CheckSchedStatus fprintf       → [ASTRO-SCHED] debug prefix (matches C++ tag).
#
# Debug prefix: [ASTRO-SCHED] / [ASTRO-PROC] — grep-compatible with C++ log tags.
# ═══════════════════════════════════════════════════════════════════════════════

import heapq
import concurrent.futures
import dataclasses


@dataclasses.dataclass
class AstroSnapshot:
    """Per-processor execution snapshot — mirrors Snapshot struct."""
    processor_id: str = ""
    routine_name: str = ""
    execute_start_mono: float = 0.0   # 0.0 → idle (mirrors execute_start_time=0)


class AstroProcessor:
    """
    Executor thread for the classic-policy scheduler.

    Ports apollo::cyber::scheduler::Processor (processor.h / processor.cc).

    ASTRO deltas:
      • std::thread → ThreadPoolExecutor future (no explicit thread management).
      • CRoutine::Resume() → plain callable task().
      • context_->Wait() (condition_variable) → threading.Event.wait(timeout=0.01).
      • Snapshot fields preserved; execute_start_mono replaces execute_start_time.
    """

    def __init__(self, processor_id: str, scheduler: "AstroScheduler"):
        self._id: str = processor_id
        self._scheduler: "AstroScheduler" = scheduler
        self._running: bool = False
        self._future: Optional[concurrent.futures.Future] = None
        self._snap = AstroSnapshot(processor_id=processor_id)
        self._wake = threading.Event()
        _dbg("ASTRO-PROC", f"AstroProcessor ctor id={processor_id}")

    def bind_and_start(self, executor: concurrent.futures.ThreadPoolExecutor) -> None:
        """
        BindContext + thread start — mirrors Processor::BindContext(ctx) which
        calls std::call_once(thread_flag_, [this]{ thread_ = std::thread(Run, this) }).
        Submits _run() to the provided ThreadPoolExecutor.
        """
        if self._running:
            return
        self._running = True
        _dbg("ASTRO-PROC", f"AstroProcessor bind_and_start id={self._id} ONLINE")
        self._future = executor.submit(self._run)

    def stop(self) -> None:
        """Stop() — mirrors Processor::Stop(). Clears flag, wakes loop, joins."""
        if not self._running:
            return
        self._running = False
        self._wake.set()
        if self._future is not None:
            try:
                self._future.result(timeout=2.0)
            except (concurrent.futures.TimeoutError, Exception):
                pass
        _dbg("ASTRO-PROC", f"AstroProcessor stop id={self._id}")

    @property
    def snapshot(self) -> AstroSnapshot:
        """ProcSnapshot() — return live snapshot reference."""
        return self._snap

    def _run(self) -> None:
        """
        Processor::Run() spin loop — ported to Python.

        Each iteration:
            1. Dequeue the highest-priority ready task from the scheduler.
            2. If found: stamp snapshot, execute task(), clear snapshot.
            3. If none:  wait on threading.Event (mirrors context_->Wait()).
        """
        _dbg("ASTRO-PROC",
             f"[ASTRO-PROCESSOR] Processor::Run epoch executor id={self._id} ONLINE")
        while self._running:
            task_name, task_func = self._scheduler._dequeue_task()
            if task_func is not None:
                self._snap.execute_start_mono = time.monotonic()
                self._snap.routine_name = task_name
                _dbg("ASTRO-PROC",
                     f"[ASTRO-PROCESSOR] epoch tick: cell='{task_name}' proc={self._id}")
                try:
                    task_func()
                except Exception as exc:
                    _dbg("ASTRO-PROC",
                         f"task exc cell={task_name} proc={self._id} exc={exc}")
                finally:
                    self._snap.execute_start_mono = 0.0
                    self._snap.routine_name = ""
            else:
                self._wake.clear()
                self._wake.wait(timeout=0.01)

    def notify(self) -> None:
        """Wake the spin loop — mirrors ClassicContext::Notify → cv_wq_.notify_one()."""
        self._wake.set()


class AstroScheduler:
    """
    Classic-policy task scheduler with ThreadPoolExecutor processor pool.

    Ports apollo::cyber::scheduler::Scheduler + SchedulerClassic (scheduler.h,
    scheduler.cc, policy/scheduler_classic.h, policy/classic_context.h).

    Key design decisions:
      • task registry  : dict[task_name, callable]  (mirrors id_cr_ map)
      • dispatch queue : heapq[(z, seq, task_name)] (mirrors MULTI_PRIO_QUEUE)
      • processors     : list[AstroProcessor]       (mirrors processors_ vector)
      • executor       : ThreadPoolExecutor          (backs AstroProcessor threads)
      • epoch_index    : int                        (mirrors atomic<uint64_t> M126)

    Usage::

        sched = AstroScheduler(num_processors=2)
        sched.create_task(lambda: render("self_attn"), "self_attn", z=3)
        sched.create_task(lambda: render("input_embed"), "input_embed", z=1)
        sched.run_until_done()
        sched.shutdown()
    """

    def __init__(self, num_processors: int = 2):
        self._stop: bool = False
        self._epoch_index: int = 0

        self._tasks: Dict[str, Any] = {}
        self._task_lock = threading.Lock()

        self._queue: List[tuple] = []
        self._queue_lock = threading.Lock()
        self._queue_seq: int = 0

        self._num_processors: int = max(1, num_processors)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._num_processors,
            thread_name_prefix="astro_proc",
        )
        self._processors: List[AstroProcessor] = []
        self._pending: int = 0
        self._pending_lock = threading.Lock()
        self._all_done = threading.Event()
        self._all_done.set()

        for i in range(self._num_processors):
            proc = AstroProcessor(processor_id=f"proc_{i}", scheduler=self)
            proc.bind_and_start(self._executor)
            self._processors.append(proc)

        _dbg("ASTRO-SCHED",
             f"AstroScheduler constructed num_processors={self._num_processors}")

    # ── epoch control (M126 port) ──────────────────────────────────────────────

    def advance_epoch(self) -> int:
        """
        AdvanceEpoch() — mirrors Scheduler::AdvanceEpoch() (M126).
        Increments epoch_index_; returns the new epoch value.
        Called by the cell-pubsub loop at each SVG layout iteration.
        """
        self._epoch_index += 1
        _dbg("ASTRO-SCHED",
             f"[ASTRO-SCHED] epoch_advance epoch={self._epoch_index}")
        return self._epoch_index

    @property
    def current_epoch(self) -> int:
        """CurrentEpoch() — read-only snapshot of epoch_index_."""
        return self._epoch_index

    # ── CreateTask ────────────────────────────────────────────────────────────

    def create_task(
        self,
        func: Callable[[], None],
        task_name: str,
        z: int = 3,
        channel_path: Optional[str] = None,
    ) -> bool:
        """
        CreateTask(func, name, z, channel_path) — mirrors Scheduler::CreateTask.

        Registers the callable under task_name in the task registry, then
        calls _dispatch_task() to enqueue it into the priority heap.

        If channel_path is provided, registers a DataNotifier callback so that
        when data arrives on the channel the task is re-enqueued — mirroring
        visitor->RegisterNotifyCallback([this, task_id]{ NotifyProcessor(task_id) }).
        """
        if self._stop:
            _dbg("ASTRO-SCHED",
                 f"CreateTask: scheduler stopped, cannot create task name={task_name}")
            return False

        with self._task_lock:
            self._tasks[task_name] = func

        _dbg("ASTRO-SCHED",
             f"[ASTRO-SCHEDULER] CreateTask name={task_name} z={z} "
             f"hasChannel={channel_path is not None}")

        self._dispatch_task(task_name, z)

        if channel_path is not None:
            def _notify_cb():
                if not self._stop:
                    self.notify_processor(task_name, z)
            DataNotifier.instance().add_notifier(
                channel_path, Notifier(_notify_cb)
            )

        return True

    # ── NotifyProcessor ───────────────────────────────────────────────────────

    def notify_processor(self, task_name: str, z: int = 3) -> bool:
        """
        NotifyProcessor(task_name) — mirrors SchedulerClassic::NotifyProcessor.

        Re-enqueues the named task into the dispatch heap and wakes a processor.
        Called on incoming data notifications (channel callbacks).
        Returns False if the scheduler is stopped or task unknown.
        """
        if self._stop:
            return False

        with self._task_lock:
            if task_name not in self._tasks:
                _dbg("ASTRO-SCHED",
                     f"[ASTRO-SCHEDULER] NotifyProcessor: unknown task={task_name}")
                return False

        _dbg("ASTRO-SCHED",
             f"[ASTRO-SCHEDULER] NotifyProcessor task={task_name} z={z}")
        self._dispatch_task(task_name, z)
        return True

    # ── internal dispatch + dequeue ───────────────────────────────────────────

    def _dispatch_task(self, task_name: str, z: int) -> None:
        """
        _dispatch_task — DispatchTask analogue.

        Pushes (z, seq, task_name) onto the heapq so the lowest-z task is
        served first, mirroring ClassicContext's MULTI_PRIO_QUEUE ordering.
        Then wakes one processor (ClassicContext::Notify → cv_wq_.notify_one()).
        """
        with self._pending_lock:
            self._pending += 1
            self._all_done.clear()

        with self._queue_lock:
            heapq.heappush(self._queue, (z, self._queue_seq, task_name))
            self._queue_seq += 1

        for proc in self._processors:
            proc.notify()
            break

    def _dequeue_task(self) -> tuple:
        """
        _dequeue_task — ClassicContext::NextRoutine analogue.

        Pops the highest-priority (lowest z) entry from the heap and returns
        (task_name, callable).  Returns ("", None) when the queue is empty.
        Called exclusively by AstroProcessor._run() — no external callers.
        """
        with self._queue_lock:
            if not self._queue:
                return ("", None)
            z, seq, task_name = heapq.heappop(self._queue)

        with self._task_lock:
            func = self._tasks.get(task_name)

        if func is None:
            return ("", None)

        def _wrapped():
            try:
                func()
            finally:
                with self._pending_lock:
                    self._pending -= 1
                    if self._pending <= 0:
                        self._pending = 0
                        self._all_done.set()

        return (task_name, _wrapped)

    # ── run_until_done ────────────────────────────────────────────────────────

    def run_until_done(self, timeout: float = 10.0) -> bool:
        """
        Block until all currently queued tasks have completed.
        Mirrors the epoch-completion wait in the CyberRT node spin loop.
        Returns True when all tasks finished within timeout, False on timeout.
        """
        return self._all_done.wait(timeout=timeout)

    # ── CheckSchedStatus ──────────────────────────────────────────────────────

    def check_sched_status(self) -> str:
        """
        CheckSchedStatus() — mirrors Scheduler::CheckSchedStatus().

        Builds a snapshot string identical in format to the C++ version:
            proc_id:routine_name:elapsed_ms, …, timestamp: <ns>
        Returns the snapshot string (also emits [ASTRO-SCHED] debug log).
        """
        now_mono = time.monotonic()
        now_ns = int(now_mono * 1e9)
        parts = []
        for proc in self._processors:
            snap = proc.snapshot
            if snap.execute_start_mono > 0.0:
                elapsed_ms = int((now_mono - snap.execute_start_mono) * 1000)
                parts.append(f"{snap.processor_id}:{snap.routine_name}:{elapsed_ms}")
            else:
                parts.append(f"{snap.processor_id}:idle")
        snap_info = ", ".join(parts) + f", timestamp: {now_ns}"
        _dbg("ASTRO-SCHED",
             f"[ASTRO-SCHEDULER] CheckSchedStatus epoch={self._epoch_index} "
             f"procs={len(self._processors)} snap={snap_info}")
        return snap_info

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """
        Shutdown() — mirrors Scheduler::Shutdown().

        Sets stop flag, stops all processors, drains the task registry,
        shuts down the ThreadPoolExecutor.
        """
        if self._stop:
            return
        self._stop = True

        _dbg("ASTRO-SCHED",
             f"[ASTRO-SCHEDULER] Shutdown draining {len(self._processors)} processors")

        for proc in self._processors:
            proc.stop()

        with self._task_lock:
            self._tasks.clear()

        with self._queue_lock:
            self._queue.clear()

        self._executor.shutdown(wait=False)
        _dbg("ASTRO-SCHED", "AstroScheduler shutdown complete")


# ═══════════════════════════════════════════════════════════════════════════════
# Apollo Cyber Transport — Python Port
# Ported from upstream/apollo-cyber/transport/  (cell-pubsub-loop branch)
# Classes: AstroIdentity · AstroEndpoint · AstroMessageInfo
#          AstroHistory · AstroShmDispatcher · AstroRtpsDispatcher
#          AstroShmTransmitter · AstroHybridTransmitter
#          AstroShmReceiver  · AstroHybridReceiver
#
# 鲁迅曰：有谁从小康人家而坠入困顿的么，我以为在这途路中，大约可以看见世人
# 的真面目。——同理，一条消息从进程内直走到跨主机，沿途层层剥落的，正是那些
# 平日里藏得最深的锁、序号与身份。
# ═══════════════════════════════════════════════════════════════════════════════

import os
import struct
import threading
import time
import hashlib
import queue
import uuid
from collections import deque
from typing import (
    Any, Callable, Dict, Generic, List, Optional, Set, Tuple, TypeVar
)

_MT = TypeVar("_MT")

# ── 内部调试 ──────────────────────────────────────────────────────────────────
_TRANSPORT_DEBUG = os.environ.get("ASTRO_TRANSPORT_DEBUG", "0") == "1"

def _tdbg(tag: str, msg: str) -> None:
    if _TRANSPORT_DEBUG:
        print(f"[{tag}] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# AstroIdentity
# Port of: upstream/apollo-cyber/transport/common/identity.h
#
# 鲁迅曰：所谓身份，不过是八个字节的自欺欺人。哈希一算，人与人之间的区别
# 也不过如此。
# ══════════════════════════════════════════════════════════════════════════════

_ID_SIZE = 8  # constexpr uint8_t ID_SIZE = 8


class AstroIdentity:
    """
    Mirrors ``apollo::cyber::transport::Identity``.

    Stores an 8-byte raw ID and a precomputed uint64 hash.  When
    *need_generate* is True (default) a random UUID-derived payload is
    generated on construction, matching the C++ auto-generate path.
    """

    __slots__ = ("_data", "_hash_value")

    def __init__(self, need_generate: bool = True,
                 data: Optional[bytes] = None) -> None:
        if data is not None:
            if len(data) != _ID_SIZE:
                raise ValueError(f"Identity data must be {_ID_SIZE} bytes")
            self._data: bytes = bytes(data)
        elif need_generate:
            self._data = uuid.uuid4().bytes[:_ID_SIZE]
        else:
            self._data = b"\x00" * _ID_SIZE
        self._update()

    # ── private ──────────────────────────────────────────────────────────────

    def _update(self) -> None:
        """Recompute hash — mirrors Identity::Update()."""
        val: int = 0
        for b in self._data:
            val = (val * 31 + b) & 0xFFFF_FFFF_FFFF_FFFF
        self._hash_value: int = val

    # ── public API ───────────────────────────────────────────────────────────

    @property
    def data(self) -> bytes:
        return self._data

    def set_data(self, data: bytes) -> None:
        if len(data) != _ID_SIZE:
            raise ValueError(f"Identity data must be {_ID_SIZE} bytes")
        self._data = bytes(data[:_ID_SIZE])
        self._update()

    def hash_value(self) -> int:
        return self._hash_value

    def length(self) -> int:
        return _ID_SIZE

    def to_string(self) -> str:
        return self._data.hex()

    # ── dunder ────────────────────────────────────────────────────────────────

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AstroIdentity) and self._data == other._data

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return self._hash_value

    def __repr__(self) -> str:
        return f"AstroIdentity({self.to_string()})"


# ══════════════════════════════════════════════════════════════════════════════
# AstroEndpoint
# Port of: upstream/apollo-cyber/transport/common/endpoint.h
#
# 鲁迅曰：端点就是端点，无论叫做发送者还是接收者，骨子里不过是一个有名有姓
# 的 enabled 开关罢了。
# ══════════════════════════════════════════════════════════════════════════════

class AstroRoleAttributes:
    """
    Lightweight stand-in for ``proto::RoleAttributes``.

    Carries only the fields actually referenced by the transport layer.
    """

    __slots__ = (
        "channel_name", "channel_id", "host_ip", "process_id",
        "id", "message_type", "qos_durability", "qos_history", "qos_depth",
    )

    DURABILITY_VOLATILE        = 0
    DURABILITY_TRANSIENT_LOCAL = 1

    HISTORY_KEEP_LAST = 0
    HISTORY_KEEP_ALL  = 1

    def __init__(
        self,
        channel_name:  str  = "",
        channel_id:    int  = 0,
        host_ip:       str  = "127.0.0.1",
        process_id:    int  = 0,
        id:            int  = 0,                  # noqa: A002
        message_type:  str  = "",
        qos_durability: int = 0,
        qos_history:   int  = 0,
        qos_depth:     int  = 1,
    ) -> None:
        self.channel_name   = channel_name
        self.channel_id     = channel_id
        self.host_ip        = host_ip
        self.process_id     = process_id
        self.id             = id
        self.message_type   = message_type
        self.qos_durability = qos_durability
        self.qos_history    = qos_history
        self.qos_depth      = qos_depth

    def copy_from(self, src: "AstroRoleAttributes") -> None:
        for s in self.__slots__:
            setattr(self, s, getattr(src, s))

    def __repr__(self) -> str:
        return (f"AstroRoleAttributes(ch={self.channel_name!r}, "
                f"id={self.id}, host={self.host_ip})")


class AstroEndpoint:
    """
    Mirrors ``apollo::cyber::transport::Endpoint``.

    Base class for transmitters and receivers; holds the role attributes
    and a random Identity generated at construction time.
    """

    def __init__(self, attr: AstroRoleAttributes) -> None:
        self.enabled_: bool              = False
        self.id_:      AstroIdentity     = AstroIdentity()
        self.attr_:    AstroRoleAttributes = attr

    # read-only accessors matching C++ getters
    @property
    def id(self) -> AstroIdentity:
        return self.id_

    @property
    def attributes(self) -> AstroRoleAttributes:
        return self.attr_

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id_.to_string()!r})"


# ══════════════════════════════════════════════════════════════════════════════
# AstroMessageInfo
# Port of: upstream/apollo-cyber/transport/message/message_info.h
#
# 鲁迅曰：消息的信封里装着发件人、序号和时间，正如旧社会的契约——写得清清楚楚，
# 却未必有人当真查对。
# ══════════════════════════════════════════════════════════════════════════════

# kSize = sender_id(8) + channel_id(8) + seq_num(8) + spare_id(8) +
#         msg_seq_num(4) + send_time(8)  → 44 bytes
_MSG_INFO_FMT    = "<8s Q Q 8s i Q"
_MSG_INFO_SIZE   = struct.calcsize(_MSG_INFO_FMT)   # == 44


class AstroMessageInfo:
    """
    Mirrors ``apollo::cyber::transport::MessageInfo``.

    Serialises/deserialises to/from a fixed 44-byte binary payload so that
    the layout is compatible with the C++ ``kSize`` constant.
    """

    kSize: int = _MSG_INFO_SIZE

    def __init__(
        self,
        sender_id:   Optional[AstroIdentity] = None,
        seq_num:     int = 0,
        spare_id:    Optional[AstroIdentity] = None,
        channel_id:  int = 0,
        msg_seq_num: int = 0,
        send_time:   int = 0,
    ) -> None:
        self.sender_id_:   AstroIdentity = sender_id  or AstroIdentity(need_generate=False)
        self.channel_id_:  int           = channel_id
        self.seq_num_:     int           = seq_num
        self.spare_id_:    AstroIdentity = spare_id   or AstroIdentity(need_generate=False)
        self.msg_seq_num_: int           = msg_seq_num
        self.send_time_:   int           = send_time

    # ── getters/setters ──────────────────────────────────────────────────────

    @property
    def sender_id(self) -> AstroIdentity:
        return self.sender_id_

    def set_sender_id(self, v: AstroIdentity) -> None:
        self.sender_id_ = v

    @property
    def channel_id(self) -> int:
        return self.channel_id_

    def set_channel_id(self, v: int) -> None:
        self.channel_id_ = v

    @property
    def seq_num(self) -> int:
        return self.seq_num_

    def set_seq_num(self, v: int) -> None:
        self.seq_num_ = v

    @property
    def spare_id(self) -> AstroIdentity:
        return self.spare_id_

    def set_spare_id(self, v: AstroIdentity) -> None:
        self.spare_id_ = v

    @property
    def msg_seq_num(self) -> int:
        return self.msg_seq_num_

    def set_msg_seq_num(self, v: int) -> None:
        self.msg_seq_num_ = v

    @property
    def send_time(self) -> int:
        return self.send_time_

    def set_send_time(self, v: int) -> None:
        self.send_time_ = v

    # ── serialization ────────────────────────────────────────────────────────

    def serialize_to(self) -> bytes:
        """``bool SerializeTo(char* dst, size_t len)``"""
        return struct.pack(
            _MSG_INFO_FMT,
            self.sender_id_.data,
            self.channel_id_ & 0xFFFF_FFFF_FFFF_FFFF,
            self.seq_num_    & 0xFFFF_FFFF_FFFF_FFFF,
            self.spare_id_.data,
            self.msg_seq_num_ & 0xFFFF_FFFF,
            self.send_time_   & 0xFFFF_FFFF_FFFF_FFFF,
        )

    @classmethod
    def deserialize_from(cls, raw: bytes) -> "AstroMessageInfo":
        """``bool DeserializeFrom(const char* src, size_t len)``"""
        if len(raw) < _MSG_INFO_SIZE:
            raise ValueError(
                f"AstroMessageInfo.deserialize_from: need {_MSG_INFO_SIZE} bytes"
            )
        sid_b, ch_id, seq, spa_b, msg_seq, send_t = struct.unpack_from(
            _MSG_INFO_FMT, raw
        )
        return cls(
            sender_id   = AstroIdentity(need_generate=False, data=sid_b),
            channel_id  = ch_id,
            seq_num     = seq,
            spare_id    = AstroIdentity(need_generate=False, data=spa_b),
            msg_seq_num = msg_seq,
            send_time   = send_t,
        )

    # ── dunder ────────────────────────────────────────────────────────────────

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AstroMessageInfo):
            return NotImplemented
        return (
            self.sender_id_  == other.sender_id_  and
            self.channel_id_ == other.channel_id_ and
            self.seq_num_    == other.seq_num_    and
            self.spare_id_   == other.spare_id_
        )

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __repr__(self) -> str:
        return (f"AstroMessageInfo(sender={self.sender_id_.to_string()}, "
                f"seq={self.seq_num_}, send_time={self.send_time_})")


# ══════════════════════════════════════════════════════════════════════════════
# AstroHistory
# Port of: upstream/apollo-cyber/transport/message/history.h
#
# 鲁迅曰：历史是会重演的，消息也是——所以 TRANSIENT_LOCAL 才要把它们一条条
# 存起来，等着晚来的订阅者去翻旧账。
# ══════════════════════════════════════════════════════════════════════════════

_MAX_HISTORY_DEPTH_DEFAULT = 1000


class AstroHistory(Generic[_MT]):
    """
    Mirrors ``apollo::cyber::transport::History<MessageT>``.

    Thread-safe bounded deque of (msg, msg_info) pairs.  When
    *history_policy* is HISTORY_KEEP_ALL the depth is capped at
    *max_depth*.
    """

    HISTORY_KEEP_LAST = 0
    HISTORY_KEEP_ALL  = 1

    class CachedMessage(Generic[_MT]):
        __slots__ = ("msg", "msg_info")

        def __init__(self, msg: _MT, msg_info: AstroMessageInfo) -> None:
            self.msg:      _MT             = msg
            self.msg_info: AstroMessageInfo = msg_info

    def __init__(
        self,
        history_policy: int = 0,
        depth:          int = 1,
        max_depth:      int = _MAX_HISTORY_DEPTH_DEFAULT,
    ) -> None:
        self._enabled:   bool  = False
        self._max_depth: int   = max_depth
        if history_policy == self.HISTORY_KEEP_ALL:
            self._depth: int = max_depth
        else:
            self._depth = min(depth, max_depth)
        self._msgs: deque = deque(maxlen=self._depth)
        self._lock: threading.Lock = threading.Lock()

    # ── enable/disable ────────────────────────────────────────────────────────

    def enable(self)  -> None: self._enabled = True
    def disable(self) -> None: self._enabled = False

    # ── public API ───────────────────────────────────────────────────────────

    def add(self, msg: _MT, msg_info: AstroMessageInfo) -> None:
        """``void Add(const MessagePtr&, const MessageInfo&)``"""
        if not self._enabled:
            return
        with self._lock:
            self._msgs.append(self.CachedMessage(msg, msg_info))

    def clear(self) -> None:
        """``void Clear()``"""
        with self._lock:
            self._msgs.clear()

    def get_cached_message(self) -> List["AstroHistory.CachedMessage"]:
        """``void GetCachedMessage(vector<CachedMessage>*)``"""
        with self._lock:
            return list(self._msgs)

    def get_size(self) -> int:
        """``size_t GetSize()``"""
        with self._lock:
            return len(self._msgs)

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def max_depth(self) -> int:
        return self._max_depth


# ══════════════════════════════════════════════════════════════════════════════
# _AstroDispatcherBase  (internal)
# Thin common base for ShmDispatcher / RtpsDispatcher
# ══════════════════════════════════════════════════════════════════════════════

_ListenerFn = Callable[[Any, AstroMessageInfo], None]


class _AstroDispatcherBase:
    """
    Shared skeleton for Apollo's intra-process listener registry.

    Corresponds roughly to ``apollo::cyber::transport::Dispatcher``.
    Each channel_id maps to a list of (identity_hash, listener_fn) pairs.
    """

    def __init__(self) -> None:
        self._is_shutdown: bool = False
        self._listeners:   Dict[int, List[Tuple[int, _ListenerFn]]] = {}
        self._lock:        threading.RLock = threading.RLock()

    # ── listener management ──────────────────────────────────────────────────

    def add_listener(
        self,
        self_attr:    AstroRoleAttributes,
        listener:     _ListenerFn,
        opposite_attr: Optional[AstroRoleAttributes] = None,
    ) -> None:
        if self._is_shutdown:
            return
        ch = self_attr.channel_id
        role_hash = self_attr.id
        with self._lock:
            bucket = self._listeners.setdefault(ch, [])
            # avoid duplicate registration
            if not any(h == role_hash for h, _ in bucket):
                bucket.append((role_hash, listener))
        _tdbg("DISPATCHER", f"add_listener ch={ch} role={role_hash}")

    def remove_listener(
        self,
        self_attr:    AstroRoleAttributes,
        opposite_attr: Optional[AstroRoleAttributes] = None,
    ) -> None:
        ch        = self_attr.channel_id
        role_hash = self_attr.id
        with self._lock:
            if ch in self._listeners:
                self._listeners[ch] = [
                    (h, fn) for h, fn in self._listeners[ch]
                    if h != role_hash
                ]

    def _dispatch(
        self,
        channel_id: int,
        msg:        Any,
        msg_info:   AstroMessageInfo,
    ) -> None:
        """Deliver *msg* to every registered listener on *channel_id*."""
        with self._lock:
            bucket = list(self._listeners.get(channel_id, []))
        for _, fn in bucket:
            try:
                fn(msg, msg_info)
            except Exception as exc:  # pylint: disable=broad-except
                _tdbg("DISPATCHER", f"listener raised: {exc}")

    def shutdown(self) -> None:
        self._is_shutdown = True
        with self._lock:
            self._listeners.clear()


# ══════════════════════════════════════════════════════════════════════════════
# AstroShmDispatcher
# Port of: upstream/apollo-cyber/transport/dispatcher/shm_dispatcher.h
#
# 鲁迅曰：共享内存就像公共食堂——人人皆可取用，锁却挂在别处，吃完了谁也不
# 承认曾经来过。
# ══════════════════════════════════════════════════════════════════════════════

_SHM_BLOCK_SIZE = 4096  # bytes per simulated SHM slot


class _AstroShmSegment:
    """
    Minimal shared-memory segment simulation.

    In the real Apollo implementation this is backed by POSIX shm_open +
    mmap.  Here we use an in-process bytearray protected by a lock so that
    the ShmDispatcher / ShmTransmitter / ShmReceiver round-trip can be
    exercised without OS-level shared memory.
    """

    def __init__(self, channel_id: int, num_blocks: int = 16) -> None:
        self.channel_id:  int              = channel_id
        self.num_blocks:  int              = num_blocks
        self._data:       List[bytearray]  = [
            bytearray(_SHM_BLOCK_SIZE) for _ in range(num_blocks)
        ]
        self._msg_sizes:  List[int]        = [0] * num_blocks
        self._write_lock: threading.Lock   = threading.Lock()
        self._read_cnts:  List[int]        = [0] * num_blocks
        self._write_idx:  int              = 0

    def acquire_block_to_write(self, msg_size: int) -> Optional[int]:
        with self._write_lock:
            idx = self._write_idx % self.num_blocks
            self._write_idx += 1
            return idx

    def release_written_block(self, idx: int, payload: bytes, msg_info_bytes: bytes) -> None:
        total = len(payload) + len(msg_info_bytes)
        if total > _SHM_BLOCK_SIZE:
            _tdbg("SHM_SEG", f"payload too large ({total} > {_SHM_BLOCK_SIZE}), truncating")
            total = _SHM_BLOCK_SIZE
        self._data[idx][:len(payload)] = payload[:_SHM_BLOCK_SIZE]
        end = min(len(payload) + len(msg_info_bytes), _SHM_BLOCK_SIZE)
        self._data[idx][len(payload):end] = msg_info_bytes[:end - len(payload)]
        self._msg_sizes[idx] = len(payload)

    def read_block(self, idx: int) -> Tuple[bytes, bytes]:
        msg_size = self._msg_sizes[idx]
        payload  = bytes(self._data[idx][:msg_size])
        info_raw = bytes(self._data[idx][msg_size:msg_size + AstroMessageInfo.kSize])
        return payload, info_raw


class AstroShmDispatcher(_AstroDispatcherBase):
    """
    Mirrors ``apollo::cyber::transport::ShmDispatcher`` (singleton).

    Maintains a map of channel_id → _AstroShmSegment and a polling thread
    that mimics the notifier wake-up path in the C++ implementation.
    """

    _instance: Optional["AstroShmDispatcher"] = None
    _inst_lock: threading.Lock = threading.Lock()

    @classmethod
    def instance(cls) -> "AstroShmDispatcher":
        with cls._inst_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        super().__init__()
        self._segments:    Dict[int, _AstroShmSegment] = {}
        self._seg_lock:    threading.RLock              = threading.RLock()
        self._notify_q:    queue.Queue                  = queue.Queue()
        self._thread:      threading.Thread             = threading.Thread(
            target=self._thread_func, daemon=True, name="AstroShmDispatcher"
        )
        self._thread.start()
        _tdbg("SHM_DISP", "AstroShmDispatcher started")

    # ── segment management ────────────────────────────────────────────────────

    def add_segment(self, attr: AstroRoleAttributes) -> None:
        ch = attr.channel_id
        with self._seg_lock:
            if ch not in self._segments:
                self._segments[ch] = _AstroShmSegment(ch)
                _tdbg("SHM_DISP", f"new segment for channel {ch}")

    def get_segment(self, channel_id: int) -> Optional[_AstroShmSegment]:
        with self._seg_lock:
            return self._segments.get(channel_id)

    # ── notifier ─────────────────────────────────────────────────────────────

    def notify(self, channel_id: int, block_index: int) -> None:
        """Called by ShmTransmitter after writing a block."""
        self._notify_q.put((channel_id, block_index))

    def _thread_func(self) -> None:
        """
        Poll loop — mirrors ``ShmDispatcher::ThreadFunc()``.

        Drains the notification queue and dispatches each readable block.
        """
        while not self._is_shutdown:
            try:
                ch_id, blk_idx = self._notify_q.get(timeout=0.05)
            except queue.Empty:
                continue
            self._read_message(ch_id, blk_idx)

    def _read_message(self, channel_id: int, block_index: int) -> None:
        seg = self.get_segment(channel_id)
        if seg is None:
            return
        payload, info_raw = seg.read_block(block_index)
        if not info_raw or len(info_raw) < AstroMessageInfo.kSize:
            return
        try:
            msg_info = AstroMessageInfo.deserialize_from(info_raw)
        except Exception:
            return
        _tdbg("SHM_DISP",
              f"dispatch ch={channel_id} blk={block_index} "
              f"seq={msg_info.seq_num}")
        self._dispatch(channel_id, payload, msg_info)

    # ── override shutdown ─────────────────────────────────────────────────────

    def shutdown(self) -> None:
        super().shutdown()
        self._thread.join(timeout=1.0)
        _tdbg("SHM_DISP", "AstroShmDispatcher shutdown")


# ══════════════════════════════════════════════════════════════════════════════
# AstroRtpsDispatcher
# Port of: upstream/apollo-cyber/transport/dispatcher/rtps_dispatcher.h
#
# 鲁迅曰：RTPS 这条路走得远，跨进程，跨主机；然而消息序列化成字符串之后，
# 凡人皆可偷看——这便是"开放"的代价。
# ══════════════════════════════════════════════════════════════════════════════

class AstroRtpsDispatcher(_AstroDispatcherBase):
    """
    Mirrors ``apollo::cyber::transport::RtpsDispatcher`` (singleton).

    In a real deployment this drives FastRTPS subscribers.  Here we expose
    ``inject_message()`` so that RtpsTransmitters (or tests) can push
    serialised payloads in-process.
    """

    _instance:  Optional["AstroRtpsDispatcher"] = None
    _inst_lock: threading.Lock = threading.Lock()

    @classmethod
    def instance(cls) -> "AstroRtpsDispatcher":
        with cls._inst_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        super().__init__()
        # subscriber registry: channel_id → bool (subscribed)
        self._subs:      Dict[int, bool]       = {}
        self._subs_lock: threading.Lock        = threading.Lock()
        _tdbg("RTPS_DISP", "AstroRtpsDispatcher initialised")

    # ── subscriber lifecycle ─────────────────────────────────────────────────

    def add_subscriber(self, attr: AstroRoleAttributes) -> None:
        ch = attr.channel_id
        with self._subs_lock:
            if ch not in self._subs:
                self._subs[ch] = True
                _tdbg("RTPS_DISP", f"subscriber added for channel {ch}")

    # ── external injection (replaces FastRTPS on_data callback) ──────────────

    def inject_message(
        self,
        channel_id: int,
        msg_str:    bytes,
        msg_info:   AstroMessageInfo,
    ) -> None:
        """
        Simulate an inbound RTPS data indication.

        Mirrors ``RtpsDispatcher::OnMessage(channel_id, msg_str, msg_info)``.
        """
        recv_time_us = int(time.monotonic_ns() // 1_000)
        send_time_us = msg_info.send_time
        if send_time_us > recv_time_us:
            _tdbg("RTPS_DISP", "WARNING: recv earlier than send")
        _tdbg("RTPS_DISP",
              f"inject ch={channel_id} len={len(msg_str)} "
              f"latency_us={recv_time_us - send_time_us}")
        self._dispatch(channel_id, msg_str, msg_info)

    # ── add_listener (with subscriber auto-create) ────────────────────────────

    def add_listener(
        self,
        self_attr:    AstroRoleAttributes,
        listener:     _ListenerFn,
        opposite_attr: Optional[AstroRoleAttributes] = None,
    ) -> None:
        super().add_listener(self_attr, listener, opposite_attr)
        self.add_subscriber(self_attr)


# ══════════════════════════════════════════════════════════════════════════════
# AstroShmTransmitter
# Port of: upstream/apollo-cyber/transport/transmitter/shm_transmitter.h
#
# 鲁迅曰：写进共享内存的消息，是寄给同一屋檐下的人的信——投递迅速，字迹清晰，
# 却只能在这堵墙里流通。
# ══════════════════════════════════════════════════════════════════════════════

_SerializeFn   = Callable[[Any], bytes]
_DeserializeFn = Callable[[bytes], Any]


class AstroShmTransmitter(AstroEndpoint):
    """
    Mirrors ``apollo::cyber::transport::ShmTransmitter<M>``.

    *serialize_fn* converts a message object to bytes; if None, the message
    is expected to already be bytes (raw mode).
    """

    def __init__(
        self,
        attr:         AstroRoleAttributes,
        serialize_fn: Optional[_SerializeFn] = None,
    ) -> None:
        super().__init__(attr)
        self._dispatcher:    AstroShmDispatcher   = AstroShmDispatcher.instance()
        self._serialize_fn:  Optional[_SerializeFn] = serialize_fn
        self._seq_num:       int                    = 0
        self._seq_lock:      threading.Lock         = threading.Lock()

    # ── enable / disable ──────────────────────────────────────────────────────

    def enable(self, opposite_attr: Optional[AstroRoleAttributes] = None) -> None:
        if self.enabled_:
            return
        self._dispatcher.add_segment(self.attr_)
        self.enabled_ = True
        _tdbg("SHM_TX", f"enabled ch={self.attr_.channel_id}")

    def disable(self, opposite_attr: Optional[AstroRoleAttributes] = None) -> None:
        self.enabled_ = False
        _tdbg("SHM_TX", f"disabled ch={self.attr_.channel_id}")

    # ── transmit ──────────────────────────────────────────────────────────────

    def transmit(
        self,
        msg:      Any,
        msg_info: Optional[AstroMessageInfo] = None,
    ) -> bool:
        """
        ``bool Transmit(const MessagePtr&, const MessageInfo&)``

        Serialises *msg*, writes it into the SHM segment, then calls
        ``AstroShmDispatcher.notify()`` with the block index.
        """
        if not self.enabled_:
            _tdbg("SHM_TX", "not enabled — drop")
            return False

        payload: bytes = (
            self._serialize_fn(msg)
            if self._serialize_fn is not None
            else (msg if isinstance(msg, (bytes, bytearray)) else str(msg).encode())
        )

        seg = self._dispatcher.get_segment(self.attr_.channel_id)
        if seg is None:
            return False

        blk_idx = seg.acquire_block_to_write(len(payload))
        if blk_idx is None:
            return False

        if msg_info is None:
            with self._seq_lock:
                self._seq_num += 1
                seq = self._seq_num
            msg_info = AstroMessageInfo(
                sender_id   = self.id_,
                seq_num     = seq,
                channel_id  = self.attr_.channel_id,
                msg_seq_num = seq,
                send_time   = int(time.monotonic_ns() // 1_000),
            )

        info_bytes = msg_info.serialize_to()
        seg.release_written_block(blk_idx, payload, info_bytes)
        self._dispatcher.notify(self.attr_.channel_id, blk_idx)
        _tdbg("SHM_TX",
              f"transmit ch={self.attr_.channel_id} blk={blk_idx} "
              f"seq={msg_info.seq_num}")
        return True


# ══════════════════════════════════════════════════════════════════════════════
# AstroHybridTransmitter
# Port of: upstream/apollo-cyber/transport/transmitter/hybrid_transmitter.h
#
# 鲁迅曰：Hybrid 是骑墙的艺术——进程内用 SHM，跨进程用 RTPS，两不相欠，
# 却又暗中都要经历 History 这道关卡。
# ══════════════════════════════════════════════════════════════════════════════

# Relation constants (mirrors cyber/common/types.h)
SAME_PROC   = "SAME_PROC"
DIFF_PROC   = "DIFF_PROC"
DIFF_HOST   = "DIFF_HOST"
NO_RELATION = "NO_RELATION"

# OptionalMode constants
MODE_INTRA = "INTRA"
MODE_SHM   = "SHM"
MODE_RTPS  = "RTPS"


class AstroHybridTransmitter(AstroEndpoint):
    """
    Mirrors ``apollo::cyber::transport::HybridTransmitter<M>``.

    Selects INTRA / SHM / RTPS based on the spatial relationship between
    sender and each registered receiver.  Also maintains a History for
    TRANSIENT_LOCAL durability.
    """

    def __init__(
        self,
        attr:         AstroRoleAttributes,
        serialize_fn: Optional[_SerializeFn]   = None,
        same_proc_mode: str = MODE_SHM,
        diff_proc_mode: str = MODE_SHM,
        diff_host_mode: str = MODE_RTPS,
    ) -> None:
        super().__init__(attr)
        self._serialize_fn = serialize_fn

        # mapping_table_[relation] → mode
        self._mapping: Dict[str, str] = {
            SAME_PROC: same_proc_mode,
            DIFF_PROC: diff_proc_mode,
            DIFF_HOST: diff_host_mode,
        }

        # sub-transmitters keyed by mode
        self._transmitters: Dict[str, AstroShmTransmitter] = {}
        self._init_transmitters()

        # receivers_[mode] = set of receiver IDs
        self._receivers: Dict[str, Set[int]] = {
            m: set() for m in set(self._mapping.values())
        }

        # history for TRANSIENT_LOCAL
        depth = attr.qos_depth if attr.qos_depth > 0 else 1
        self._history: AstroHistory = AstroHistory(
            history_policy = attr.qos_history,
            depth          = depth,
        )
        if attr.qos_durability == AstroRoleAttributes.DURABILITY_TRANSIENT_LOCAL:
            self._history.enable()

        self._mutex: threading.Lock = threading.Lock()
        self._seq:   int            = 0

    # ── init helpers ─────────────────────────────────────────────────────────

    def _init_transmitters(self) -> None:
        modes = set(self._mapping.values())
        for mode in modes:
            # For this Python port we use ShmTransmitter for all modes
            # (RTPS would need network; use SHM as stand-in for DIFF_HOST too)
            self._transmitters[mode] = AstroShmTransmitter(
                self.attr_, self._serialize_fn
            )

    # ── enable / disable ──────────────────────────────────────────────────────

    def enable(self, opposite_attr: Optional[AstroRoleAttributes] = None) -> None:
        if opposite_attr is None:
            with self._mutex:
                for tx in self._transmitters.values():
                    tx.enable()
            self.enabled_ = True
            return
        relation = self._get_relation(opposite_attr)
        if relation == NO_RELATION:
            return
        mode = self._mapping[relation]
        with self._mutex:
            self._receivers[mode].add(opposite_attr.id)
            self._transmitters[mode].enable(opposite_attr)
            self._transmit_history(opposite_attr, mode)
        self.enabled_ = True

    def disable(self, opposite_attr: Optional[AstroRoleAttributes] = None) -> None:
        if opposite_attr is None:
            with self._mutex:
                for tx in self._transmitters.values():
                    tx.disable()
            self.enabled_ = False
            return
        relation = self._get_relation(opposite_attr)
        if relation == NO_RELATION:
            return
        mode = self._mapping[relation]
        with self._mutex:
            self._receivers[mode].discard(opposite_attr.id)
            self._transmitters[mode].disable(opposite_attr)

    # ── transmit ──────────────────────────────────────────────────────────────

    def transmit(
        self,
        msg:      Any,
        msg_info: Optional[AstroMessageInfo] = None,
    ) -> bool:
        """
        ``bool Transmit(const MessagePtr&, const MessageInfo&)``

        Adds to history, then fans out to all sub-transmitters.
        """
        with self._mutex:
            if msg_info is None:
                self._seq += 1
                msg_info = AstroMessageInfo(
                    sender_id   = self.id_,
                    seq_num     = self._seq,
                    channel_id  = self.attr_.channel_id,
                    msg_seq_num = self._seq,
                    send_time   = int(time.monotonic_ns() // 1_000),
                )
            self._history.add(msg, msg_info)
            for tx in self._transmitters.values():
                tx.transmit(msg, msg_info)
        return True

    # ── history replay ────────────────────────────────────────────────────────

    def _transmit_history(
        self, opposite_attr: AstroRoleAttributes, mode: str
    ) -> None:
        if self.attr_.qos_durability != AstroRoleAttributes.DURABILITY_TRANSIENT_LOCAL:
            return
        cached = self._history.get_cached_message()
        if not cached:
            return
        tx = self._transmitters[mode]

        def _replay() -> None:
            for item in cached:
                tx.transmit(item.msg, item.msg_info)
                time.sleep(0.001)

        t = threading.Thread(target=_replay, daemon=True, name="HybridTx-replay")
        t.start()

    # ── relation helper ───────────────────────────────────────────────────────

    def _get_relation(self, opposite_attr: AstroRoleAttributes) -> str:
        if opposite_attr.channel_name != self.attr_.channel_name:
            return NO_RELATION
        if opposite_attr.host_ip != self.attr_.host_ip:
            return DIFF_HOST
        if opposite_attr.process_id != self.attr_.process_id:
            return DIFF_PROC
        return SAME_PROC


# ══════════════════════════════════════════════════════════════════════════════
# AstroShmReceiver
# Port of: upstream/apollo-cyber/transport/receiver/shm_receiver.h
#
# 鲁迅曰：接收者是沉默的，只管等待那一声通知——仿佛旧时深宅里等信的人，
# 门缝里塞进来什么，便接什么。
# ══════════════════════════════════════════════════════════════════════════════

class AstroShmReceiver(AstroEndpoint):
    """
    Mirrors ``apollo::cyber::transport::ShmReceiver<M>``.

    Registers a typed listener with the singleton AstroShmDispatcher.
    The raw-bytes payload received from the dispatcher is passed through
    *deserialize_fn* before reaching *msg_listener*.
    """

    def __init__(
        self,
        attr:           AstroRoleAttributes,
        msg_listener:   _ListenerFn,
        deserialize_fn: Optional[_DeserializeFn] = None,
    ) -> None:
        super().__init__(attr)
        self._msg_listener:   _ListenerFn                  = msg_listener
        self._deserialize_fn: Optional[_DeserializeFn]     = deserialize_fn
        self._dispatcher:     AstroShmDispatcher            = AstroShmDispatcher.instance()

    # ── enable / disable ──────────────────────────────────────────────────────

    def enable(self, opposite_attr: Optional[AstroRoleAttributes] = None) -> None:
        if self.enabled_:
            return
        if opposite_attr is None:
            self._dispatcher.add_segment(self.attr_)
            self._dispatcher.add_listener(self.attr_, self._on_raw_message)
        else:
            self._dispatcher.add_segment(self.attr_)
            self._dispatcher.add_listener(
                self.attr_, self._on_raw_message, opposite_attr
            )
        self.enabled_ = True
        _tdbg("SHM_RX", f"enabled ch={self.attr_.channel_id}")

    def disable(self, opposite_attr: Optional[AstroRoleAttributes] = None) -> None:
        if not self.enabled_:
            return
        self._dispatcher.remove_listener(self.attr_, opposite_attr)
        self.enabled_ = False
        _tdbg("SHM_RX", f"disabled ch={self.attr_.channel_id}")

    # ── internal callback ─────────────────────────────────────────────────────

    def _on_raw_message(self, raw: Any, msg_info: AstroMessageInfo) -> None:
        """Deserialise raw bytes then forward to the user listener."""
        if self._deserialize_fn is not None and isinstance(raw, (bytes, bytearray)):
            try:
                msg = self._deserialize_fn(raw)
            except Exception as exc:
                _tdbg("SHM_RX", f"deserialize failed: {exc}")
                return
        else:
            msg = raw
        self._msg_listener(msg, msg_info)


# ══════════════════════════════════════════════════════════════════════════════
# AstroHybridReceiver
# Port of: upstream/apollo-cyber/transport/receiver/hybrid_receiver.h
#
# 鲁迅曰：Hybrid 接收者身兼数职，同屋、隔壁、远端，皆有办法应付；然而真正
# 到了需要回溯历史的时候，它才显出那份难得的耐心——开一条线，等到消息不再来
# 为止。
# ══════════════════════════════════════════════════════════════════════════════

class AstroHybridReceiver(AstroEndpoint):
    """
    Mirrors ``apollo::cyber::transport::HybridReceiver<M>``.

    Wraps one AstroShmReceiver per communication mode and routes
    Enable(opposite) calls to the correct sub-receiver based on the spatial
    relationship between self and the opposite role.
    """

    def __init__(
        self,
        attr:           AstroRoleAttributes,
        msg_listener:   _ListenerFn,
        deserialize_fn: Optional[_DeserializeFn] = None,
        same_proc_mode: str = MODE_SHM,
        diff_proc_mode: str = MODE_SHM,
        diff_host_mode: str = MODE_RTPS,
    ) -> None:
        super().__init__(attr)
        self._msg_listener   = msg_listener
        self._deserialize_fn = deserialize_fn

        self._mapping: Dict[str, str] = {
            SAME_PROC: same_proc_mode,
            DIFF_PROC: diff_proc_mode,
            DIFF_HOST: diff_host_mode,
        }

        # one sub-receiver per distinct mode
        modes = set(self._mapping.values())
        self._receivers: Dict[str, AstroShmReceiver] = {
            mode: AstroShmReceiver(attr, msg_listener, deserialize_fn)
            for mode in modes
        }

        # transmitter tracking: mode → {id → RoleAttributes}
        self._transmitters: Dict[str, Dict[int, AstroRoleAttributes]] = {
            mode: {} for mode in modes
        }

        # history for TRANSIENT_LOCAL
        depth = attr.qos_depth if attr.qos_depth > 0 else 1
        self._history: AstroHistory = AstroHistory(
            history_policy = attr.qos_history,
            depth          = depth,
        )
        if attr.qos_durability == AstroRoleAttributes.DURABILITY_TRANSIENT_LOCAL:
            self._history.enable()

        self._mutex: threading.Lock = threading.Lock()

    # ── enable / disable ──────────────────────────────────────────────────────

    def enable(self, opposite_attr: Optional[AstroRoleAttributes] = None) -> None:
        if opposite_attr is None:
            with self._mutex:
                for rx in self._receivers.values():
                    rx.enable()
            self.enabled_ = True
            return

        relation = self._get_relation(opposite_attr)
        if relation == NO_RELATION:
            return
        mode = self._mapping[relation]
        oid  = opposite_attr.id

        with self._mutex:
            if oid not in self._transmitters[mode]:
                self._transmitters[mode][oid] = opposite_attr
                self._receivers[mode].enable(opposite_attr)
                self._receive_history(opposite_attr, mode)
        self.enabled_ = True
        _tdbg("HYBRID_RX", f"enable opposite={oid} relation={relation}")

    def disable(self, opposite_attr: Optional[AstroRoleAttributes] = None) -> None:
        if opposite_attr is None:
            with self._mutex:
                for rx in self._receivers.values():
                    rx.disable()
            self.enabled_ = False
            return

        relation = self._get_relation(opposite_attr)
        if relation == NO_RELATION:
            return
        mode = self._mapping[relation]
        oid  = opposite_attr.id

        with self._mutex:
            if oid in self._transmitters[mode]:
                del self._transmitters[mode][oid]
                self._receivers[mode].disable(opposite_attr)
        _tdbg("HYBRID_RX", f"disable opposite={oid}")

    # ── history receive ───────────────────────────────────────────────────────

    def _receive_history(
        self, opposite_attr: AstroRoleAttributes, mode: str
    ) -> None:
        """
        Mirrors ``HybridReceiver::ReceiveHistoryMsg`` + ``ThreadFunc``.

        Spins a thread that waits for cached messages to arrive then exits.
        In this simulation we merely log the intent; real RTPS re-subscription
        would happen here.
        """
        if opposite_attr.qos_durability != AstroRoleAttributes.DURABILITY_TRANSIENT_LOCAL:
            return

        def _wait_thread() -> None:
            _tdbg("HYBRID_RX",
                  f"history thread started for opposite={opposite_attr.id}")
            deadline = time.monotonic() + 1.0
            last_size = self._history.get_size()
            while time.monotonic() < deadline:
                time.sleep(0.05)
                cur = self._history.get_size()
                if cur != last_size:
                    last_size = cur
                    deadline  = time.monotonic() + 1.0
            _tdbg("HYBRID_RX", "history thread exit")

        t = threading.Thread(
            target=_wait_thread, daemon=True, name="HybridRx-hist"
        )
        t.start()

    # ── relation helper ───────────────────────────────────────────────────────

    def _get_relation(self, opposite_attr: AstroRoleAttributes) -> str:
        if opposite_attr.channel_name != self.attr_.channel_name:
            return NO_RELATION
        if opposite_attr.host_ip != self.attr_.host_ip:
            return DIFF_HOST
        if opposite_attr.process_id != self.attr_.process_id:
            return DIFF_PROC
        return SAME_PROC


# ═══════════════════════════════════════════════════════════════════════════════
# AstroNodeChannelImpl — ported from
#   upstream/apollo-cyber/node/node_channel_impl.h
#
# 原典：NodeChannelImpl 是 Node 的私有 pImpl，持有 node_attr_（protobuf
# RoleAttributes）、is_reality_mode_ 标志以及 NodeManager 共享指针；
# CreateWriter<M> / CreateReader<M> 根据 is_reality_mode_ 选择真实
# 传输路径（Writer<M> / Reader<M>）或进程内 Blocker 路径
# （IntraWriter<M> / IntraReader<M>）；ReaderConfig 封装
# channel_name + QosProfile + pending_queue_size 三元组。
#
# 算法改动（20% 规则）：
#   1. template<MessageT>       → 消息类型固定为 Any（dict / str / bytes）。
#   2. proto::RoleAttributes    → 轻量 ReaderConfig / _WriterAttr dataclass。
#   3. is_reality_mode_ 二值分支 → AstroTransport.mode 参数（"INTRA" / "SHM"）；
#      默认 "INTRA"，等价于原典 is_reality_mode_=false 的 IntraWriter/IntraReader。
#   4. NodeManager::Join / Leave → AstroChannelManager（已在本文件实现）。
#   5. GlobalData::RegisterChannel uint64_t → str hash via hashlib.md5 首8字节。
#   6. FillInAttr<M> 填充 QosProfile 默认值 → _fill_attr() 填充 ReaderConfig 默认值。
#   7. CreateReader 过载四版 → 三个 Python 重载（by_name / by_config / by_attr）。
#   8. pending_queue_size 默认值保持 DEFAULT_PENDING_QUEUE_SIZE（已在本文件定义）。
#
# 鲁迅曰：实现细节藏在 Impl 之后，正如苦衷藏在笑脸之后——外人只见工厂方法，
# 不见那些个 FillInAttr 和 Join 的辛苦。
#
# Debug prefix: [ASTRO-NODE-CH] — 与 C++ ADEBUG 前缀 grep 一致。
# ═══════════════════════════════════════════════════════════════════════════════

import dataclasses as _dc


@_dc.dataclass
class ReaderConfig:
    """
    ReaderConfig — mirrors ``apollo::cyber::ReaderConfig``.

    Bundles channel_name + qos_depth + pending_queue_size.
    QosProfile fields not used by the Astro file-channel model are omitted;
    only the depth (HISTORY_KEEP_LAST) and pending_queue_size survive.
    """
    channel_name:      str = ""
    qos_depth:         int = 1          # QosProfile.depth default
    pending_queue_size: int = DEFAULT_PENDING_QUEUE_SIZE


class AstroNodeChannelImpl:
    """
    Node channel factory — Python port of ``NodeChannelImpl``.

    Creates and lifecycle-manages AstroCellReader / AstroCellWriter pairs
    for one logical node, keeping the same factory semantics as the C++ Impl:

      • create_writer(channel_name)          → AstroCellWriter (init'd)
      • create_reader_by_name(ch, cb)        → AstroCellReader (init'd)
      • create_reader_by_config(cfg, cb)     → AstroCellReader (init'd)
      • create_reader_by_attr(cfg, cb, size) → AstroCellReader (init'd, explicit queue)

    All created objects are also registered with AstroChannelManager so the
    topology snapshot reflects the node's pub/sub graph.

    ASTRO delta from NodeChannelImpl::FillInAttr<M>:
      channel_id is computed as md5(channel_name)[:8].hex() — a deterministic
      8-byte string hash matching GlobalData::RegisterChannel semantics
      without requiring the Apollo globaldata singleton.
    """

    def __init__(
        self,
        node_name: str,
        mode: str = "INTRA",
        host_name: str = "localhost",
        process_id: int = 0,
    ) -> None:
        self.node_name:   str = node_name
        self._mode:       str = mode
        self._host_name:  str = host_name
        self._process_id: int = process_id

        # Join node-level topology — mirrors NodeManager::Join(node_attr_, ROLE_NODE)
        self._ch_mgr: AstroChannelManager = AstroChannelManager.instance()

        # Track created objects so __del__ can leave topology
        self._writers: List[AstroCellWriter] = []
        self._readers: List[AstroCellReader] = []

        _dbg("ASTRO-NODE-CH",
             f"NodeChannelImpl ctor node={node_name} mode={mode}")

    # ── internal attr helper ──────────────────────────────────────────────────

    @staticmethod
    def _register_channel(channel_name: str) -> str:
        """
        FillInAttr<M> channel_id assignment.

        Apollo: uint64_t channel_id = GlobalData::RegisterChannel(channel_name).
        ASTRO:  md5(channel_name)[:8].hex() — deterministic 16-hex-char string.
        """
        import hashlib as _hl
        return _hl.md5(channel_name.encode()).hexdigest()[:16]

    # ── CreateWriter ──────────────────────────────────────────────────────────

    def create_writer(self, channel_name: str) -> Optional[AstroCellWriter]:
        """
        CreateWriter(channel_name) — mirrors NodeChannelImpl::CreateWriter<M>
        (string overload).

        Constructs, inits, and topology-registers an AstroCellWriter.
        Returns None if channel_name is empty (mirrors the AERROR guard).
        """
        if not channel_name:
            _dbg("ASTRO-NODE-CH",
                 "create_writer: empty channel_name — abort")
            return None

        w = AstroCellWriter(channel_name, node_name=self.node_name)
        if not w.init():
            return None

        # Join topology — mirrors FillInAttr + NodeManager topology registration
        self._ch_mgr.join(
            channel_path=channel_name,
            node_name=self.node_name,
            role_type="WRITER",
            host_name=self._host_name,
            process_id=self._process_id,
        )

        self._writers.append(w)
        _dbg("ASTRO-NODE-CH",
             f"create_writer ok node={self.node_name} ch={channel_name} "
             f"ch_id={self._register_channel(channel_name)}")
        return w

    # ── CreateReader overloads ────────────────────────────────────────────────

    def create_reader_by_name(
        self,
        channel_name: str,
        reader_func: Optional[Callable[[Any], None]] = None,
    ) -> Optional[AstroCellReader]:
        """
        CreateReader(channel_name, reader_func) — string overload.

        Mirrors ``NodeChannelImpl::CreateReader<M>(channel_name, reader_func)``.
        pending_queue_size defaults to DEFAULT_PENDING_QUEUE_SIZE.
        """
        if not channel_name:
            return None
        cfg = ReaderConfig(channel_name=channel_name)
        return self.create_reader_by_config(cfg, reader_func)

    def create_reader_by_config(
        self,
        config: ReaderConfig,
        reader_func: Optional[Callable[[Any], None]] = None,
    ) -> Optional[AstroCellReader]:
        """
        CreateReader(config, reader_func) — ReaderConfig overload.

        Mirrors ``NodeChannelImpl::CreateReader<M>(ReaderConfig, reader_func)``.
        Extracts pending_queue_size from config.
        """
        return self.create_reader_by_attr(
            config, reader_func, config.pending_queue_size
        )

    def create_reader_by_attr(
        self,
        config: ReaderConfig,
        reader_func: Optional[Callable[[Any], None]] = None,
        pending_queue_size: int = DEFAULT_PENDING_QUEUE_SIZE,
    ) -> Optional[AstroCellReader]:
        """
        CreateReader(role_attr, reader_func, pending_queue_size) — full overload.

        Mirrors the primary ``NodeChannelImpl::CreateReader<M>`` template.

        ASTRO path (always INTRA-mode):
          • Creates AstroCellReader with pending_queue_size.
          • Calls reader.init() — registers ChannelBuffer + DataNotifier callback.
          • Joins AstroChannelManager as READER for topology tracking.
        """
        channel_name = config.channel_name
        if not channel_name:
            _dbg("ASTRO-NODE-CH",
                 "create_reader_by_attr: empty channel_name — abort")
            return None

        r = AstroCellReader(
            channel_path=channel_name,
            reader_func=reader_func,
            pending_queue_size=pending_queue_size,
            node_name=self.node_name,
        )
        if not r.init():
            return None

        # Join topology — mirrors FillInAttr + ReceiverManager topology join
        self._ch_mgr.join(
            channel_path=channel_name,
            node_name=self.node_name,
            role_type="READER",
            host_name=self._host_name,
            process_id=self._process_id,
        )

        self._readers.append(r)
        _dbg("ASTRO-NODE-CH",
             f"create_reader ok node={self.node_name} ch={channel_name} "
             f"pending_q={pending_queue_size} "
             f"ch_id={self._register_channel(channel_name)}")
        return r

    # ── NodeName ──────────────────────────────────────────────────────────────

    def node_name_str(self) -> str:
        """NodeName() — const std::string& accessor."""
        return self.node_name

    # ── shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """
        Leave topology for all created writers and readers.

        Mirrors NodeChannelImpl destructor:
          node_manager_->Leave(node_attr_, RoleType::ROLE_NODE)
        augmented with per-channel Leave so ChannelManager is consistent.
        """
        for w in self._writers:
            self._ch_mgr.leave(w.channel_path, self.node_name, "WRITER")
            w.shutdown()
        for r in self._readers:
            self._ch_mgr.leave(r.channel_path, self.node_name, "READER")
            r.shutdown()
        self._writers.clear()
        self._readers.clear()
        _dbg("ASTRO-NODE-CH",
             f"NodeChannelImpl shutdown node={self.node_name}")

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:  # noqa: BLE001
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# AstroReaderBase — ported from
#   upstream/apollo-cyber/node/reader_base.h
#
# 原典：ReaderBase 是所有 Reader<MessageT> 的抽象基类，持有 role_attr_
# （protobuf RoleAttributes）和 init_（std::atomic<bool>）；定义八个纯虚
# 接口（Init / Shutdown / ClearData / Observe / Empty / HasReceived /
# GetDelaySec / PendingQueueSize）以及若干非虚查询（GetChannelName /
# ChannelId / QosProfile / IsInit）。
#
# 同文件还定义了 ReceiverManager<MessageT> 单例——负责按 channel_name 去重，
# 保证每个 channel 只创建一个 Receiver；dispatch 回调经由 DataDispatcher<M>
# 把消息推入 ChannelBuffer，再由 DataNotifier 唤醒 DataVisitor。
#
# 算法改动（20% 规则）：
#   1. template<MessageT> ReaderBase → 非模板抽象基类（Python 无模板）。
#   2. proto::RoleAttributes → ReaderConfig dataclass（已在本文件定义）。
#   3. std::atomic<bool> init_ → threading.Event（init_set/init_clear）。
#   4. uint64_t channel_id → str channel_path（文件通道路径）。
#   5. QosProfile proto 字段 → qos_depth int（只保留 HISTORY_KEEP_LAST depth）。
#   6. ReceiverManager<M> 单例 → AstroReceiverManager 无泛型单例，
#      用 dict[channel_path, AstroCellReader] 保证去重。
#   7. DataDispatcher<M>::Dispatch + DataNotifier::Notify 回调由
#      AstroCellReader.init() 已完成注册，ReceiverManager 仅需返回 reader。
#
# 鲁迅曰：基类的存在，是为了让子类可以各行其是，却仍被同一把钥匙锁住——
# 这便是纯虚函数的权威，也是接口的温柔暴力。
#
# Debug prefix: [ASTRO-RBASE] — 与 C++ ADEBUG 前缀 grep 一致。
# ═══════════════════════════════════════════════════════════════════════════════

import abc


class AstroReaderBase(abc.ABC):
    """
    Abstract base class for all Astro channel readers.

    Ports ``apollo::cyber::ReaderBase`` from reader_base.h.

    Subclasses must implement the eight pure-virtual interfaces.
    The non-virtual accessors (get_channel_name, channel_id, is_init)
    are provided here using ReaderConfig and a threading.Event init flag.

    ASTRO delta: role_attr_ proto → ReaderConfig; init_ atomic → Event.
    """

    def __init__(self, config: ReaderConfig) -> None:
        self._config: ReaderConfig = config
        # std::atomic<bool> init_ — True once Init() succeeds
        self._init_event: threading.Event = threading.Event()

    # ── pure virtual interface ────────────────────────────────────────────────

    @abc.abstractmethod
    def init(self) -> bool:
        """Init() — initialise receiver; return True on success."""

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Shutdown() — deregister; release resources."""

    @abc.abstractmethod
    def clear_data(self) -> None:
        """ClearData() — drain all buffered messages."""

    @abc.abstractmethod
    def observe(self) -> None:
        """Observe() — snapshot publish-queue into observe-queue."""

    @abc.abstractmethod
    def empty(self) -> bool:
        """Empty() — True when observe-queue is empty."""

    @abc.abstractmethod
    def has_received(self) -> bool:
        """HasReceived() — True if any message arrived since last clear."""

    @abc.abstractmethod
    def get_delay_sec(self) -> float:
        """
        GetDelaySec() — seconds since last receive.

        Returns -1.0 before first message (mirrors Apollo convention).
        """

    @abc.abstractmethod
    def pending_queue_size(self) -> int:
        """PendingQueueSize() — configured depth of the pending queue."""

    # ── non-virtual accessors (mirror ReaderBase non-virtual methods) ─────────

    def has_writer(self) -> bool:
        """
        HasWriter() — query topology for an active writer.

        Default returns False (mirrors base class default body).
        Concrete subclasses override via ChannelRegistry.
        """
        return False

    def get_channel_name(self) -> str:
        """GetChannelName() — return role_attr_.channel_name()."""
        return self._config.channel_name

    def channel_id(self) -> str:
        """
        ChannelId() — deterministic string hash of channel_name.

        Apollo original: uint64_t from GlobalData::RegisterChannel.
        ASTRO: md5(channel_name)[:16] hex string.
        """
        import hashlib as _hl
        return _hl.md5(self._config.channel_name.encode()).hexdigest()[:16]

    def qos_depth(self) -> int:
        """QosProfile() accessor — return configured qos_depth."""
        return self._config.qos_depth

    def is_init(self) -> bool:
        """IsInit() — mirrors init_.load() atomic read."""
        return self._init_event.is_set()

    # ── init flag helpers (used by concrete subclasses) ───────────────────────

    def _mark_init(self) -> None:
        """Set init flag — call at end of successful init()."""
        self._init_event.set()
        _dbg("ASTRO-RBASE",
             f"init ok ch={self._config.channel_name}")

    def _clear_init(self) -> None:
        """Clear init flag — call in shutdown()."""
        self._init_event.clear()


# ── AstroReceiverManager — port of ReceiverManager<MessageT> singleton ────────

class AstroReceiverManager:
    """
    Per-channel reader de-duplicator — Python port of ReceiverManager<MessageT>.

    Apollo original: singleton template with AtomicHashMap<channel_name, Receiver>
    ensuring that multiple readers on the same channel share one Receiver so
    the DataDispatcher::Dispatch callback fires exactly once per message.

    ASTRO mapping:
      Receiver<M>           → AstroCellReader (already registered with dispatcher)
      receiver_map_[name]   → _map[channel_path]
      GetReceiver(role_attr) → get_or_create(channel_path, node_name, reader_func)

    Returns the existing AstroCellReader if already created for a channel,
    creating and init-ing a new one otherwise.
    """

    _instance: Optional["AstroReceiverManager"] = None
    _inst_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._map: Dict[str, AstroCellReader] = {}
        self._mu:  threading.Lock             = threading.Lock()

    @classmethod
    def instance(cls) -> "AstroReceiverManager":
        """DECLARE_SINGLETON equivalent."""
        if cls._instance is None:
            with cls._inst_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._inst_lock:
            cls._instance = None

    def get_receiver(
        self,
        channel_path: str,
        node_name: str = "astro_node",
        reader_func: Optional[Callable[[Any], None]] = None,
        pending_queue_size: int = DEFAULT_PENDING_QUEUE_SIZE,
    ) -> Optional[AstroCellReader]:
        """
        GetReceiver(role_attr) — return shared reader for channel.

        Mirrors ReceiverManager<M>::GetReceiver: if a reader already exists
        for *channel_path* it is returned as-is (receiver_map_ de-dup).
        Otherwise a new AstroCellReader is created, init'd, and cached.

        reader_func and pending_queue_size are only used on first creation;
        subsequent calls with the same channel_path ignore them.
        """
        with self._mu:
            if channel_path in self._map:
                _dbg("ASTRO-RBASE",
                     f"ReceiverManager: reuse ch={channel_path}")
                return self._map[channel_path]

            r = AstroCellReader(
                channel_path=channel_path,
                reader_func=reader_func,
                pending_queue_size=pending_queue_size,
                node_name=node_name,
            )
            if not r.init():
                _dbg("ASTRO-RBASE",
                     f"ReceiverManager: init failed ch={channel_path}")
                return None

            self._map[channel_path] = r
            _dbg("ASTRO-RBASE",
                 f"ReceiverManager: new receiver ch={channel_path} node={node_name}")
            return r


# ═══════════════════════════════════════════════════════════════════════════════
# AstroWriterBase — ported from
#   upstream/apollo-cyber/node/writer_base.h
#
# 原典：WriterBase 是所有 Writer<MessageT> 的抽象基类，持有 role_attr_
# （proto::RoleAttributes）、init_ bool（std::mutex 保护）；
# 定义三个纯虚接口（Init / Shutdown / HasReader + GetReaders 非纯虚）以及
# 非虚查询（GetChannelName / GetChannelId / IsInit）。
# WriterBase 刻意不含消息发布接口（Write/Transmit）——这些留给 Writer<M> 模板。
#
# 算法改动（20% 规则）：
#   1. proto::RoleAttributes → WriterAttr dataclass（channel_name + node_name）。
#   2. std::mutex + bool init_ → threading.Lock + _init bool（语义完全一致）。
#   3. GetChannelId uint64_t → str hash（md5(channel_name)[:16]，与 ReaderBase 对称）。
#   4. HasReader / GetReaders 非纯虚默认实现 → 委托 ChannelRegistry（已在本文件）。
#   5. 无消息发布接口（同 C++ 设计意图）——发布由 AstroCellWriter.write() 提供。
#
# 鲁迅曰：写者无需知道读者在哪里——它只管写，有没有人读，是 HasReader 的事。
# 这正是发布-订阅的解耦哲学：一方沉默，另一方喧哗，互不相欠。
#
# Debug prefix: [ASTRO-WBASE] — 与 C++ ADEBUG 前缀 grep 一致。
# ═══════════════════════════════════════════════════════════════════════════════

@_dc.dataclass
class WriterAttr:
    """
    Lightweight writer role attributes — mirrors proto::RoleAttributes
    fields used by WriterBase (channel_name, node_name only).
    """
    channel_name: str = ""
    node_name:    str = "astro_node"


class AstroWriterBase(abc.ABC):
    """
    Abstract base class for all Astro channel writers.

    Ports ``apollo::cyber::WriterBase`` from writer_base.h.

    Subclasses must implement Init() and Shutdown().
    HasReader / GetReaders have default non-abstract implementations
    (mirrors C++ non-pure-virtual defaults).

    ASTRO delta:
      role_attr_ proto     → WriterAttr dataclass.
      std::mutex + bool    → threading.Lock + _init bool.
      GetChannelId uint64  → md5 hex string.
    """

    def __init__(self, attr: WriterAttr) -> None:
        self._attr: WriterAttr   = attr
        self._init: bool         = False
        self._lock: threading.Lock = threading.Lock()

    # ── pure virtual ──────────────────────────────────────────────────────────

    @abc.abstractmethod
    def init(self) -> bool:
        """Init() — set up transmitter; return True on success."""

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Shutdown() — tear down transmitter; leave topology."""

    # ── non-virtual (mirrored from C++ non-pure-virtual defaults) ────────────

    def has_reader(self) -> bool:
        """
        HasReader() — query ChannelRegistry for active readers.

        Default returns False when not init'd (mirrors C++ default body).
        """
        if not self.is_init():
            return False
        return ChannelRegistry.instance().has_reader(self._attr.channel_name)

    def get_readers(self) -> List[str]:
        """
        GetReaders(readers*) — return list of reader role_ids on the channel.

        Mirrors WriterBase::GetReaders (vector<RoleAttributes>* out-param).
        ASTRO: returns role_id strings from ChannelRegistry.
        """
        with ChannelRegistry.instance()._mu:
            bucket = (
                ChannelRegistry.instance()
                ._map.get(self._attr.channel_name, {})
                .get("readers", set())
            )
            return sorted(bucket)

    # ── non-virtual accessors ─────────────────────────────────────────────────

    def get_channel_name(self) -> str:
        """GetChannelName() — const std::string& accessor."""
        return self._attr.channel_name

    def get_channel_id(self) -> str:
        """
        GetChannelId() — deterministic hex string channel identifier.

        Apollo: uint64_t from role_attr_.channel_id().
        ASTRO:  md5(channel_name)[:16].
        """
        import hashlib as _hl
        return _hl.md5(self._attr.channel_name.encode()).hexdigest()[:16]

    def is_init(self) -> bool:
        """IsInit() — mutex-guarded read of init_ bool."""
        with self._lock:
            return self._init

    # ── init flag helpers ─────────────────────────────────────────────────────

    def _mark_init(self) -> None:
        """Set init_ under lock — call at end of successful init()."""
        with self._lock:
            self._init = True
        _dbg("ASTRO-WBASE",
             f"init ok ch={self._attr.channel_name} node={self._attr.node_name}")

    def _clear_init(self) -> None:
        """Clear init_ under lock — call in shutdown()."""
        with self._lock:
            self._init = False


# ── Concrete minimal writer backed by AstroCellWriter ─────────────────────────

class AstroConcreteWriter(AstroWriterBase):
    """
    Concrete WriterBase subclass backed by AstroCellWriter.

    Bridges the abstract AstroWriterBase interface with the already-implemented
    AstroCellWriter pub/sub machinery, so callers that depend on the base-class
    interface (e.g. WriterBase::IsInit / HasReader / GetChannelId) work without
    duplicating implementation.

    write(msg) delegates directly to AstroCellWriter.write(msg).
    """

    def __init__(self, channel_name: str, node_name: str = "astro_node") -> None:
        super().__init__(WriterAttr(channel_name=channel_name, node_name=node_name))
        self._impl: AstroCellWriter = AstroCellWriter(channel_name, node_name)

    def init(self) -> bool:
        ok = self._impl.init()
        if ok:
            self._mark_init()
        return ok

    def shutdown(self) -> None:
        self._impl.shutdown()
        self._clear_init()

    def write(self, msg: Any) -> bool:
        """Write(msg) — delegates to AstroCellWriter.write()."""
        return self._impl.write(msg)


# ═══════════════════════════════════════════════════════════════════════════════
# AstroNodeServiceImpl — ported from
#   upstream/apollo-cyber/node/node_service_impl.h
#
# 原典：NodeServiceImpl 是 Node 的私有 pImpl，持有 node_name_、
# RoleAttributes attr_（host_name / process_id / node_name / node_id）、
# service_list_（weak_ptr<ServiceBase> vector）和 client_list_（同上）；
# CreateService<Req,Resp> 构造 Service<Req,Resp>，调用 Init()，
# 注册到 TopologyManager::service_manager()（ROLE_SERVER）；
# CreateClient<Req,Resp> 构造 Client<Req,Resp>，调用 Init()，
# 注册到 TopologyManager::service_manager()（ROLE_CLIENT）。
#
# Astro 中没有独立的 Service / Client 模板（服务通信不在本文件的通道模型内），
# 因此我们用可调用的 Python handler 替代 ServiceCallback，
# 用 Future / threading.Event 替代 Client 的同步等待机制。
#
# 算法改动（20% 规则）：
#   1. template<Request, Response>    → handler: Callable[[Any], Any]（无泛型）。
#   2. Service<Req,Resp> + Init()     → AstroService（内嵌类，无 RTPS 层）。
#   3. Client<Req,Resp> + Init()      → AstroClient（threading.Event + result slot）。
#   4. TopologyManager service_manager → AstroChannelManager（以 service_name 为
#      channel_path，role_type = "WRITER"/"READER" 代替 ROLE_SERVER/ROLE_CLIENT）。
#   5. service_list_ / client_list_ weak_ptr 引用 → 强引用 list（Python GC 无需弱引用）。
#   6. GlobalData::RegisterService uint64_t → md5(service_name)[:16] hex string。
#   7. NodeServiceImpl() = delete      → __init__ 需要 node_name（无默认构造）。
#
# 鲁迅曰：服务与客户端，不过是两个在电话两端互望的人——一个等着被叫，一个拨出去
# 等结果。协议写得再严密，也逃不过那句\"请稍候\"。
#
# Debug prefix: [ASTRO-NODE-SVC] — 与 C++ ADEBUG 前缀 grep 一致。
# ═══════════════════════════════════════════════════════════════════════════════

import concurrent.futures as _cf


class AstroService:
    """
    Lightweight RPC service — replaces ``apollo::cyber::Service<Req,Resp>``.

    Registers a synchronous handler callable and dispatches incoming
    requests via ``call(request)`` → response.

    Topology registration mirrors Service::Init() →
    TopologyManager::service_manager()::Join(attr, ROLE_SERVER).
    ASTRO substitution: AstroChannelManager.join(service_name, ROLE="WRITER").
    """

    def __init__(
        self,
        node_name:    str,
        service_name: str,
        handler:      Callable[[Any], Any],
        host_name:    str = "localhost",
        process_id:   int = 0,
    ) -> None:
        self.service_name: str  = service_name
        self.node_name:    str  = node_name
        self._handler:     Callable[[Any], Any] = handler
        self._init:        bool = False

        # Topology: service server = WRITER role on the service channel
        self._ch_mgr = AstroChannelManager.instance()
        self._ch_mgr.join(
            channel_path=service_name,
            node_name=node_name,
            role_type="WRITER",
            host_name=host_name,
            process_id=process_id,
        )
        self._init = True
        _dbg("ASTRO-NODE-SVC",
             f"AstroService init ok svc={service_name} node={node_name}")

    def call(self, request: Any) -> Any:
        """
        Dispatch a request to the handler synchronously.

        Mirrors Service<Req,Resp>::ServiceCallback invocation inside
        Service::HandleRequest().  Returns the handler's response.
        """
        if not self._init:
            raise RuntimeError(f"AstroService '{self.service_name}' not initialised")
        _dbg("ASTRO-NODE-SVC",
             f"AstroService call svc={self.service_name}")
        return self._handler(request)

    def shutdown(self) -> None:
        """Leave topology — mirrors Service destructor / Shutdown()."""
        if self._init:
            self._ch_mgr.leave(self.service_name, self.node_name, "WRITER")
            self._init = False


class AstroClient:
    """
    Lightweight RPC client — replaces ``apollo::cyber::Client<Req,Resp>``.

    Holds a reference to an AstroService and issues synchronous calls
    via ``send_request(request, timeout_s)``.

    Topology registration mirrors Client::Init() →
    TopologyManager::service_manager()::Join(attr, ROLE_CLIENT).
    ASTRO substitution: AstroChannelManager.join(service_name, ROLE="READER").
    """

    def __init__(
        self,
        node_name:    str,
        service_name: str,
        host_name:    str = "localhost",
        process_id:   int = 0,
    ) -> None:
        self.service_name: str  = service_name
        self.node_name:    str  = node_name
        self._init:        bool = False
        self._service:     Optional[AstroService] = None

        # Topology: service client = READER role on the service channel
        self._ch_mgr = AstroChannelManager.instance()
        self._ch_mgr.join(
            channel_path=service_name,
            node_name=node_name,
            role_type="READER",
            host_name=host_name,
            process_id=process_id,
        )
        self._init = True
        _dbg("ASTRO-NODE-SVC",
             f"AstroClient init ok svc={service_name} node={node_name}")

    def bind_service(self, service: AstroService) -> None:
        """
        Bind a local AstroService — simulates RTPS service discovery.

        In Apollo, Client discovers Service via TopologyManager change callbacks
        and stores the RoleAttributes.  Here we accept a direct reference so
        unit tests can wire client → server without network.
        """
        self._service = service
        _dbg("ASTRO-NODE-SVC",
             f"AstroClient bind svc={self.service_name} node={self.node_name}")

    def send_request(
        self,
        request: Any,
        timeout_s: float = 5.0,
    ) -> Optional[Any]:
        """
        SendRequest(request, timeout) — mirrors Client<Req,Resp>::SendRequest.

        Apollo original: serialises request, sends via RTPS, waits on a
        condition variable for the response future.

        ASTRO path: if a service is bound locally, delegates directly to
        AstroService.call() (same-process fast path).  Returns None on
        timeout or unbound service.
        """
        if not self._init:
            _dbg("ASTRO-NODE-SVC",
                 f"AstroClient.send_request: not init svc={self.service_name}")
            return None

        if self._service is None:
            _dbg("ASTRO-NODE-SVC",
                 f"AstroClient.send_request: no service bound svc={self.service_name}")
            return None

        _dbg("ASTRO-NODE-SVC",
             f"AstroClient.send_request svc={self.service_name} timeout={timeout_s}s")
        try:
            future: _cf.Future = _cf.ThreadPoolExecutor(max_workers=1).submit(
                self._service.call, request
            )
            return future.result(timeout=timeout_s)
        except _cf.TimeoutError:
            _dbg("ASTRO-NODE-SVC",
                 f"AstroClient.send_request timeout svc={self.service_name}")
            return None
        except Exception as exc:  # noqa: BLE001
            _dbg("ASTRO-NODE-SVC",
                 f"AstroClient.send_request exc={exc}")
            return None

    def shutdown(self) -> None:
        """Leave topology — mirrors Client destructor / Shutdown()."""
        if self._init:
            self._ch_mgr.leave(self.service_name, self.node_name, "READER")
            self._init = False


class AstroNodeServiceImpl:
    """
    Node service factory — Python port of ``NodeServiceImpl``.

    Creates and lifecycle-manages AstroService / AstroClient pairs for one
    logical node, keeping the same factory semantics as the C++ Impl:

      • create_service(service_name, handler) → AstroService (init'd)
      • create_client(service_name)           → AstroClient  (init'd)

    All created objects are topology-registered with AstroChannelManager so
    the snapshot() shows the service graph alongside the channel graph.

    ASTRO delta from NodeServiceImpl::CreateService / CreateClient:
      service_id = md5(service_name)[:16] — mirrors GlobalData::RegisterService.
      service_list_ / client_list_ → strong refs (Python GC, no weak_ptr needed).
    """

    def __init__(
        self,
        node_name:    str,
        host_name:    str = "localhost",
        process_id:   int = 0,
    ) -> None:
        if not node_name:
            raise ValueError("AstroNodeServiceImpl requires a non-empty node_name")
        self.node_name:   str = node_name
        self._host_name:  str = host_name
        self._process_id: int = process_id

        self._services: List[AstroService] = []
        self._clients:  List[AstroClient]  = []

        _dbg("ASTRO-NODE-SVC",
             f"NodeServiceImpl ctor node={node_name}")

    # ── CreateService ─────────────────────────────────────────────────────────

    def create_service(
        self,
        service_name: str,
        handler:      Callable[[Any], Any],
    ) -> Optional[AstroService]:
        """
        CreateService<Req,Resp>(service_name, callback) — mirrors NodeServiceImpl
        template method.

        Constructs AstroService (which calls Init() internally), registers with
        AstroChannelManager as ROLE_SERVER (WRITER), appends to service_list_.
        Returns None if service_name is empty.
        """
        if not service_name:
            _dbg("ASTRO-NODE-SVC",
                 "create_service: empty service_name — abort")
            return None

        svc = AstroService(
            node_name=self.node_name,
            service_name=service_name,
            handler=handler,
            host_name=self._host_name,
            process_id=self._process_id,
        )
        self._services.append(svc)

        svc_id = self._register_service(service_name)
        _dbg("ASTRO-NODE-SVC",
             f"create_service ok node={self.node_name} svc={service_name} "
             f"svc_id={svc_id}")
        return svc

    # ── CreateClient ──────────────────────────────────────────────────────────

    def create_client(self, service_name: str) -> Optional[AstroClient]:
        """
        CreateClient<Req,Resp>(service_name) — mirrors NodeServiceImpl template.

        Constructs AstroClient (Init() called internally), registers with
        AstroChannelManager as ROLE_CLIENT (READER), appends to client_list_.
        Returns None if service_name is empty.
        """
        if not service_name:
            _dbg("ASTRO-NODE-SVC",
                 "create_client: empty service_name — abort")
            return None

        cli = AstroClient(
            node_name=self.node_name,
            service_name=service_name,
            host_name=self._host_name,
            process_id=self._process_id,
        )
        self._clients.append(cli)

        svc_id = self._register_service(service_name)
        _dbg("ASTRO-NODE-SVC",
             f"create_client ok node={self.node_name} svc={service_name} "
             f"svc_id={svc_id}")
        return cli

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _register_service(service_name: str) -> str:
        """GlobalData::RegisterService — deterministic md5 hex id."""
        import hashlib as _hl
        return _hl.md5(service_name.encode()).hexdigest()[:16]

    # ── shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """
        Shutdown — mirrors NodeServiceImpl destructor.

        Calls shutdown() on all services and clients so topology is updated.
        """
        for svc in self._services:
            svc.shutdown()
        for cli in self._clients:
            cli.shutdown()
        self._services.clear()
        self._clients.clear()
        _dbg("ASTRO-NODE-SVC",
             f"NodeServiceImpl shutdown node={self.node_name}")

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:  # noqa: BLE001
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# AstroUnderlayMessage — ported from
#   upstream/apollo-cyber/transport/rtps/underlay_message.h
#
# 原典：Apollo RtpsTransmitter / RtpsReceiver 透过 FastCDR 序列化 UnderlayMessage，
# 再由 eprosima fastrtps Participant 在 DDS/RTPS 网络上传输。
# UnderlayMessage 是 IDL 自动生成的 POD 结构体，字段：
#   int32_t m_timestamp — 发送端 epoch 时间戳（秒，截断为 int32）
#   int32_t m_seq       — 每条消息的单调递增序号
#   std::string m_data  — 序列化后的消息正文（任意字节串）
#   std::string m_datatype — 消息类型名（对应 proto MessageDescriptor.full_name）
#
# 鲁迅曰：这消息的信封，薄薄的四个字段，却要走遍 DDS 的山山水水。
# 打开来，不过是一个时间戳、一个序号、一堆字节和一个名字——正如一切旅途
# 的终点，都只剩下问：你是谁，你从哪里来。
#
# 算法改动（20% 规则）：
#   1. FastCDR serialize/deserialize → struct.pack / unpack + base64（Python 无 FastCDR）。
#   2. getMaxCdrSerializedSize / getCdrSerializedSize → max_serialized_size() /
#      serialized_size() 返回 int；两函数逻辑完整保留（变长字段 +4 对齐）。
#   3. isKeyDefined() → 类方法 is_key_defined()（Apollo IDL 总返回 false）。
#   4. serializeKey() → serialize_key() 为空操作（Key 未定义，同 C++ 实现）。
#   5. m_timestamp 改为 float（monotonic_ns() 微秒精度）；int32 语义保留在
#      serialise 路径（截断为 int32）以保持线格式兼容。
# ═══════════════════════════════════════════════════════════════════════════════

import struct as _struct
import base64 as _base64


class AstroUnderlayMessage:
    """
    Python port of ``apollo::cyber::transport::UnderlayMessage``.

    Wire layout (CDR-approximate, little-endian):
        int32  timestamp   (4 bytes, saturated to INT32 range)
        int32  seq         (4 bytes)
        uint32 data_len    (4 bytes) + data_len bytes  (m_data)
        uint32 type_len    (4 bytes) + type_len bytes  (m_datatype)

    ``serialize()`` packs to bytes; ``deserialize()`` restores from bytes.
    Both are used by AstroRtpsTransmitter / AstroRtpsDispatcher ASTRO ports.

    鲁迅曰：序号就是命运——第一条消息序号为一，最后一条也不过是个更大的整数，
    中间的岁月，都压扁在那四个字节里。
    """

    _INT32_MAX: int =  2_147_483_647
    _INT32_MIN: int = -2_147_483_648
    _HDR_FMT: str   = "<ii"
    _HDR_SIZE: int  = _struct.calcsize(_HDR_FMT)

    def __init__(self, timestamp: float = 0.0, seq: int = 0,
                 data: str = "", datatype: str = "") -> None:
        self.timestamp: float = timestamp
        self.seq:       int   = seq
        self.data:      str   = data
        self.datatype:  str   = datatype

    def get_timestamp(self) -> int:
        return self._clamp_int32(int(self.timestamp))
    def set_timestamp(self, v: float) -> None:
        self.timestamp = float(v)
    def get_seq(self) -> int:
        return self._clamp_int32(self.seq)
    def set_seq(self, v: int) -> None:
        self.seq = int(v)
    def get_data(self) -> str:
        return self.data
    def set_data(self, v: str) -> None:
        self.data = v
    def get_datatype(self) -> str:
        return self.datatype
    def set_datatype(self, v: str) -> None:
        self.datatype = v

    @staticmethod
    def max_serialized_size(current_alignment: int = 0) -> int:
        return current_alignment + 4 + 4 + 4 + 65535 + 4 + 65535

    @staticmethod
    def serialized_size(msg: "AstroUnderlayMessage", current_alignment: int = 0) -> int:
        return (current_alignment + 4 + 4
                + 4 + len(msg.data.encode("utf-8"))
                + 4 + len(msg.datatype.encode("utf-8")))

    def serialize(self) -> bytes:
        data_b  = self.data.encode("utf-8")
        dtype_b = self.datatype.encode("utf-8")
        return (
            _struct.pack(self._HDR_FMT,
                         self._clamp_int32(int(self.timestamp)),
                         self._clamp_int32(self.seq))
            + _struct.pack("<I", len(data_b))  + data_b
            + _struct.pack("<I", len(dtype_b)) + dtype_b
        )

    @classmethod
    def deserialize(cls, raw: bytes) -> "AstroUnderlayMessage":
        if len(raw) < cls._HDR_SIZE + 8:
            raise ValueError(f"AstroUnderlayMessage.deserialize: need ≥{cls._HDR_SIZE+8} bytes")
        offset = 0
        ts_i32, seq_i32 = _struct.unpack_from(cls._HDR_FMT, raw, offset); offset += cls._HDR_SIZE
        data_len, = _struct.unpack_from("<I", raw, offset); offset += 4
        data_b    = raw[offset: offset + data_len];          offset += data_len
        dtype_len, = _struct.unpack_from("<I", raw, offset); offset += 4
        dtype_b    = raw[offset: offset + dtype_len]
        return cls(timestamp=float(ts_i32), seq=seq_i32,
                   data=data_b.decode("utf-8", errors="replace"),
                   datatype=dtype_b.decode("utf-8", errors="replace"))

    @staticmethod
    def key_max_serialized_size(current_alignment: int = 0) -> int:
        return current_alignment
    @staticmethod
    def is_key_defined() -> bool:
        return False
    def serialize_key(self) -> bytes:
        return b""

    @classmethod
    def _clamp_int32(cls, v: int) -> int:
        return max(cls._INT32_MIN, min(cls._INT32_MAX, v))

    def __repr__(self) -> str:
        return (f"AstroUnderlayMessage(ts={self.timestamp}, seq={self.seq}, "
                f"datatype={self.datatype!r}, data_len={len(self.data)})")


# ═══════════════════════════════════════════════════════════════════════════════
# AstroWarehouseBase / AstroSingleValueWarehouse / AstroMultiValueWarehouse
# Ported from: warehouse_base.h / single_value_warehouse.h / multi_value_warehouse.h
#
# 鲁迅曰：仓库里堆满了角色，按编号取，按属性找，钥匙和货物之间，
# 隔着一道原子锁——那是文明对混乱最后的体面。
#
# 算法改动（20% 规则）：
#   1. uint64_t key + RolePtr  → str channel_path + RoleRecord。
#   2. AtomicRWLock            → threading.RLock。
#   3. unordered_map           → dict[str, RoleRecord]（SingleValue）。
#   4. unordered_multimap      → dict[str, list[RoleRecord]]（MultiValue）。
#   5. ignore_if_exist         → Add() 参数保留，语义同原典。
#   6. bool* out-param Search  → 返回 Optional[RoleRecord] / List[RoleRecord]。
#   7. GetAllRoles 两重载      → get_all_roles() 合并为一个 Pythonic 接口。
# ═══════════════════════════════════════════════════════════════════════════════

import abc as _abc


class AstroWarehouseBase(_abc.ABC):
    """Abstract warehouse interface — port of WarehouseBase."""

    @_abc.abstractmethod
    def add(self, key: str, role: "RoleRecord", ignore_if_exist: bool = True) -> bool: ...
    @_abc.abstractmethod
    def clear(self) -> None: ...
    @_abc.abstractmethod
    def size(self) -> int: ...
    @_abc.abstractmethod
    def remove_by_key(self, key: str) -> None: ...
    @_abc.abstractmethod
    def remove_by_role(self, key: str, role: "RoleRecord") -> None: ...
    @_abc.abstractmethod
    def remove_by_attr(self, attr: "RoleRecord") -> None: ...
    @_abc.abstractmethod
    def search_key(self, key: str) -> bool: ...
    @_abc.abstractmethod
    def search_first(self, key: str) -> Optional["RoleRecord"]: ...
    @_abc.abstractmethod
    def search_all(self, key: str) -> List["RoleRecord"]: ...
    @_abc.abstractmethod
    def search_attr(self, attr: "RoleRecord") -> bool: ...
    @_abc.abstractmethod
    def search_attr_first(self, attr: "RoleRecord") -> Optional["RoleRecord"]: ...
    @_abc.abstractmethod
    def search_attr_all(self, attr: "RoleRecord") -> List["RoleRecord"]: ...
    @_abc.abstractmethod
    def get_all_roles(self) -> List["RoleRecord"]: ...

    @staticmethod
    def _matches(role: "RoleRecord", attr: "RoleRecord") -> bool:
        if attr.channel_path and role.channel_path != attr.channel_path:
            return False
        if attr.node_name and role.node_name != attr.node_name:
            return False
        return True


class AstroSingleValueWarehouse(AstroWarehouseBase):
    """
    One role per key — mirrors SingleValueWarehouse (unordered_map).

    鲁迅曰：单值仓库是专制的——一把钥匙只开一扇门，
    新来的若想入住，先得把旧的赶走。
    """

    def __init__(self) -> None:
        self._roles: Dict[str, "RoleRecord"] = {}
        self._lock  = threading.RLock()

    def add(self, key: str, role: "RoleRecord", ignore_if_exist: bool = True) -> bool:
        with self._lock:
            if key in self._roles and ignore_if_exist:
                return False
            self._roles[key] = role
        return True

    def clear(self) -> None:
        with self._lock: self._roles.clear()

    def size(self) -> int:
        with self._lock: return len(self._roles)

    def remove_by_key(self, key: str) -> None:
        with self._lock: self._roles.pop(key, None)

    def remove_by_role(self, key: str, role: "RoleRecord") -> None:
        with self._lock:
            if self._roles.get(key) and self._roles[key].role_id == role.role_id:
                del self._roles[key]

    def remove_by_attr(self, attr: "RoleRecord") -> None:
        with self._lock:
            for k in [k for k, r in self._roles.items() if self._matches(r, attr)]:
                del self._roles[k]

    def search_key(self, key: str) -> bool:
        with self._lock: return key in self._roles

    def search_first(self, key: str) -> Optional["RoleRecord"]:
        with self._lock: return self._roles.get(key)

    def search_all(self, key: str) -> List["RoleRecord"]:
        with self._lock:
            r = self._roles.get(key); return [r] if r else []

    def search_attr(self, attr: "RoleRecord") -> bool:
        with self._lock: return any(self._matches(r, attr) for r in self._roles.values())

    def search_attr_first(self, attr: "RoleRecord") -> Optional["RoleRecord"]:
        with self._lock:
            for r in self._roles.values():
                if self._matches(r, attr): return r
            return None

    def search_attr_all(self, attr: "RoleRecord") -> List["RoleRecord"]:
        with self._lock: return [r for r in self._roles.values() if self._matches(r, attr)]

    def get_all_roles(self) -> List["RoleRecord"]:
        with self._lock: return list(self._roles.values())


class AstroMultiValueWarehouse(AstroWarehouseBase):
    """
    Multiple roles per key — mirrors MultiValueWarehouse (unordered_multimap).

    鲁迅曰：多值仓库是民主的——一把钥匙后面站着一排人，
    新来者也可以挤进去，只要还没有人和你同名同姓。
    """

    def __init__(self) -> None:
        self._roles: Dict[str, List["RoleRecord"]] = {}
        self._lock  = threading.RLock()

    def add(self, key: str, role: "RoleRecord", ignore_if_exist: bool = True) -> bool:
        with self._lock:
            bucket = self._roles.setdefault(key, [])
            if ignore_if_exist and any(r.role_id == role.role_id for r in bucket):
                return False
            bucket.append(role)
        return True

    def clear(self) -> None:
        with self._lock: self._roles.clear()

    def size(self) -> int:
        with self._lock: return sum(len(v) for v in self._roles.values())

    def remove_by_key(self, key: str) -> None:
        with self._lock: self._roles.pop(key, None)

    def remove_by_role(self, key: str, role: "RoleRecord") -> None:
        with self._lock:
            if key in self._roles:
                self._roles[key] = [r for r in self._roles[key] if r.role_id != role.role_id]
                if not self._roles[key]: del self._roles[key]

    def remove_by_attr(self, attr: "RoleRecord") -> None:
        with self._lock:
            for key in list(self._roles):
                self._roles[key] = [r for r in self._roles[key] if not self._matches(r, attr)]
                if not self._roles[key]: del self._roles[key]

    def search_key(self, key: str) -> bool:
        with self._lock: return bool(self._roles.get(key))

    def search_first(self, key: str) -> Optional["RoleRecord"]:
        with self._lock:
            b = self._roles.get(key); return b[0] if b else None

    def search_all(self, key: str) -> List["RoleRecord"]:
        with self._lock: return list(self._roles.get(key, []))

    def search_attr(self, attr: "RoleRecord") -> bool:
        with self._lock:
            return any(self._matches(r, attr) for b in self._roles.values() for r in b)

    def search_attr_first(self, attr: "RoleRecord") -> Optional["RoleRecord"]:
        with self._lock:
            for b in self._roles.values():
                for r in b:
                    if self._matches(r, attr): return r
            return None

    def search_attr_all(self, attr: "RoleRecord") -> List["RoleRecord"]:
        with self._lock:
            return [r for b in self._roles.values() for r in b if self._matches(r, attr)]

    def get_all_roles(self) -> List["RoleRecord"]:
        with self._lock:
            return [r for b in self._roles.values() for r in b]


# ═══════════════════════════════════════════════════════════════════════════════
# AstroChoreographyScheduler — ported from
#   upstream/apollo-cyber/scheduler/policy/scheduler_choreography.h
#
# 原典：SchedulerChoreography 继承 Scheduler，引入"编舞"概念：
#   choreography processors — 绑定 cr_confs_ 中指定任务的专用核。
#   pool processors         — 处理其余任务的通用线程池。
#
# 鲁迅曰：编舞者把舞台分成两半——台前的明星各自占据指定的 CPU，
# 台后的杂役共用一个线程池；分工清晰，却都在同一幕戏里表演。
#
# 算法改动（20% 规则）：
#   1. ChoreographyTask proto   → ChoreographyTaskConf dataclass（name/priority/processor_id）。
#   2. CRoutine (coroutine)     → Python callable。
#   3. CPU affinity / OS policy → 仅记录，不实际调用 pthread_setaffinity。
#   4. choreography processors  → _ChoreographyProcessor（私有 heapq/slot）。
#   5. pool processors          → _PoolProcessor（共享 pool heapq）。
#   6. DispatchTask             → 按 cr_confs_ 路由到 chore / pool。
#   7. NotifyProcessor          → chore 任务唤醒专用 processor；pool 任务唤醒任意池 proc。
#   8. CreateProcessor          → __init__ 中按 conf 分配两类 Processor。
#   9. RemoveTask/RemoveCRoutine→ 从 cr_confs_、chore_map、task registry 三处清除。
# ═══════════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class ChoreographyTaskConf:
    """Per-task choreography conf — mirrors proto::ChoreographyTask."""
    name:         str = ""
    priority:     int = 0
    processor_id: int = 0


class _ChoreographyProcessor:
    """Dedicated processor for one choreography slot."""

    def __init__(self, proc_id: str, scheduler: "AstroChoreographyScheduler",
                 slot_idx: int) -> None:
        self._id = proc_id; self._sched = scheduler; self._slot_idx = slot_idx
        self._running = False; self._wake = threading.Event()
        self._snap = AstroSnapshot(processor_id=proc_id)
        self._future: Optional[concurrent.futures.Future] = None

    def bind_and_start(self, ex: concurrent.futures.ThreadPoolExecutor) -> None:
        if self._running: return
        self._running = True; self._future = ex.submit(self._run)

    def stop(self) -> None:
        if not self._running: return
        self._running = False; self._wake.set()
        if self._future:
            try: self._future.result(timeout=2.0)
            except Exception: pass

    @property
    def snapshot(self) -> AstroSnapshot: return self._snap

    def notify(self) -> None: self._wake.set()

    def _run(self) -> None:
        _dbg("ASTRO-PROC", f"[ASTRO-PROCESSOR] ChoreographyProcessor::Run id={self._id} ONLINE")
        while self._running:
            task_name, func = self._sched._dequeue_chore(self._slot_idx)
            if func is not None:
                self._snap.execute_start_mono = time.monotonic()
                self._snap.routine_name = task_name
                _dbg("ASTRO-PROC", f"[ASTRO-PROCESSOR] chore tick: cell='{task_name}' proc={self._id}")
                try: func()
                except Exception as exc:
                    _dbg("ASTRO-PROC", f"chore exc cell={task_name} exc={exc}")
                finally: self._snap.execute_start_mono = 0.0; self._snap.routine_name = ""
            else:
                self._wake.clear(); self._wake.wait(timeout=0.01)


class _PoolProcessor:
    """Pool processor — shares the scheduler's pool heapq."""

    def __init__(self, proc_id: str, scheduler: "AstroChoreographyScheduler") -> None:
        self._id = proc_id; self._sched = scheduler
        self._running = False; self._wake = threading.Event()
        self._snap = AstroSnapshot(processor_id=proc_id)
        self._future: Optional[concurrent.futures.Future] = None

    def bind_and_start(self, ex: concurrent.futures.ThreadPoolExecutor) -> None:
        if self._running: return
        self._running = True; self._future = ex.submit(self._run)

    def stop(self) -> None:
        if not self._running: return
        self._running = False; self._wake.set()
        if self._future:
            try: self._future.result(timeout=2.0)
            except Exception: pass

    @property
    def snapshot(self) -> AstroSnapshot: return self._snap

    def notify(self) -> None: self._wake.set()

    def _run(self) -> None:
        _dbg("ASTRO-PROC", f"[ASTRO-PROCESSOR] PoolProcessor::Run id={self._id} ONLINE")
        while self._running:
            task_name, func = self._sched._dequeue_pool()
            if func is not None:
                self._snap.execute_start_mono = time.monotonic()
                self._snap.routine_name = task_name
                _dbg("ASTRO-PROC", f"[ASTRO-PROCESSOR] pool tick: cell='{task_name}' proc={self._id}")
                try: func()
                except Exception as exc:
                    _dbg("ASTRO-PROC", f"pool exc cell={task_name} exc={exc}")
                finally: self._snap.execute_start_mono = 0.0; self._snap.routine_name = ""
            else:
                self._wake.clear(); self._wake.wait(timeout=0.01)


class AstroChoreographyScheduler:
    """
    Choreography-policy scheduler — Python port of SchedulerChoreography.

    Tasks in *chore_task_confs* → pinned choreography processors.
    All other tasks → shared pool heapq served by pool processors.

    Usage::

        confs = [ChoreographyTaskConf("self_attn", priority=0, processor_id=0)]
        sched = AstroChoreographyScheduler(chore_task_confs=confs,
                                           num_chore_processors=1,
                                           num_pool_processors=1)
        sched.create_task(lambda: render("self_attn"), "self_attn", z=0)
        sched.create_task(lambda: render("norm"),      "norm",      z=3)
        sched.run_until_done()
        sched.shutdown()
    """

    def __init__(
        self,
        chore_task_confs:              Optional[List[ChoreographyTaskConf]] = None,
        num_chore_processors:          int = 1,
        num_pool_processors:           int = 1,
        choreography_affinity:         str = "",
        pool_affinity:                 str = "",
        choreography_processor_policy: str = "SCHED_OTHER",
        pool_processor_policy:         str = "SCHED_OTHER",
        choreography_processor_prio:   int = 0,
        pool_processor_prio:           int = 0,
        choreography_cpuset:           Optional[List[int]] = None,
        pool_cpuset:                   Optional[List[int]] = None,
    ) -> None:
        self._stop: bool = False

        # cr_confs_ — mirrors SchedulerChoreography::cr_confs_
        self._cr_confs: Dict[str, ChoreographyTaskConf] = {
            c.name: c for c in (chore_task_confs or [])
        }

        # OS-level policy fields (stored; not applied in Python)
        self.choreography_affinity         = choreography_affinity
        self.pool_affinity                 = pool_affinity
        self.choreography_processor_policy = choreography_processor_policy
        self.pool_processor_policy         = pool_processor_policy
        self.choreography_processor_prio   = choreography_processor_prio
        self.pool_processor_prio           = pool_processor_prio
        self.choreography_cpuset: List[int] = choreography_cpuset or []
        self.pool_cpuset:         List[int] = pool_cpuset or []

        self._tasks: Dict[str, Any]   = {}
        self._task_lock               = threading.Lock()

        self._num_chore_procs = max(1, num_chore_processors)
        self._chore_proc_queues: List[List[tuple]] = [[] for _ in range(self._num_chore_procs)]
        self._chore_queue_locks: List[threading.Lock] = [threading.Lock() for _ in range(self._num_chore_procs)]
        self._chore_map: Dict[str, int] = {}

        self._num_pool_procs  = max(1, num_pool_processors)
        self._pool_queue: List[tuple] = []
        self._pool_q_lock = threading.Lock()
        self._pool_q_seq  = 0

        self._pending     = 0
        self._pending_lock = threading.Lock()
        self._all_done    = threading.Event()
        self._all_done.set()

        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._num_chore_procs + self._num_pool_procs,
            thread_name_prefix="astro_chore",
        )
        self._chore_processors: List[_ChoreographyProcessor] = []
        self._pool_processors:  List[_PoolProcessor]          = []
        self._create_processors()

        _dbg("ASTRO-CHORE",
             f"AstroChoreographyScheduler constructed "
             f"chore_procs={self._num_chore_procs} pool_procs={self._num_pool_procs} "
             f"chore_tasks={list(self._cr_confs.keys())}")

    def _create_processors(self) -> None:
        """CreateProcessor() — mirrors SchedulerChoreography::CreateProcessor()."""
        for idx in range(self._num_chore_procs):
            p = _ChoreographyProcessor(f"chore_{idx}", self, idx)
            p.bind_and_start(self._executor)
            self._chore_processors.append(p)
            _dbg("ASTRO-CHORE",
                 f"CreateProcessor: chore idx={idx} ONLINE "
                 f"policy={self.choreography_processor_policy} prio={self.choreography_processor_prio}")
        for idx in range(self._num_pool_procs):
            p = _PoolProcessor(f"pool_{idx}", self)
            p.bind_and_start(self._executor)
            self._pool_processors.append(p)
            _dbg("ASTRO-CHORE",
                 f"CreateProcessor: pool  idx={idx} ONLINE "
                 f"policy={self.pool_processor_policy} prio={self.pool_processor_prio}")

    def create_task(self, func: Callable[[], None], task_name: str,
                    z: int = 3, channel_path: Optional[str] = None) -> bool:
        """DispatchTask — route to choreography or pool based on cr_confs_."""
        if self._stop: return False
        with self._task_lock:
            self._tasks[task_name] = func
        if task_name in self._cr_confs:
            conf    = self._cr_confs[task_name]
            proc_id = conf.processor_id % self._num_chore_procs
            self._chore_map[task_name] = proc_id
            self._dispatch_chore(task_name, conf.priority, proc_id)
            _dbg("ASTRO-CHORE", f"DispatchTask CHORE task={task_name} proc={proc_id} prio={conf.priority}")
        else:
            self._dispatch_pool(task_name, z)
            _dbg("ASTRO-CHORE", f"DispatchTask POOL  task={task_name} z={z}")
        if channel_path is not None:
            def _notify_cb() -> None:
                if not self._stop: self.notify_processor(task_name, z)
            DataNotifier.instance().add_notifier(channel_path, Notifier(_notify_cb))
        return True

    def notify_processor(self, task_name: str, z: int = 3) -> bool:
        """NotifyProcessor — mirrors SchedulerChoreography::NotifyProcessor(crid)."""
        if self._stop: return False
        with self._task_lock:
            if task_name not in self._tasks: return False
        if task_name in self._chore_map:
            conf = self._cr_confs.get(task_name)
            self._dispatch_chore(task_name, conf.priority if conf else 0, self._chore_map[task_name])
        else:
            self._dispatch_pool(task_name, z)
        return True

    def remove_task(self, task_name: str) -> bool:
        """RemoveTask — mirrors SchedulerChoreography::RemoveTask."""
        with self._task_lock:
            removed = self._tasks.pop(task_name, None) is not None
        self._cr_confs.pop(task_name, None)
        self._chore_map.pop(task_name, None)
        return removed

    def remove_croutine(self, task_name: str) -> bool:
        """RemoveCRoutine — mirrors SchedulerChoreography::RemoveCRoutine(crid)."""
        return self.remove_task(task_name)

    def _dispatch_chore(self, task_name: str, priority: int, proc_id: int) -> None:
        with self._pending_lock:
            self._pending += 1; self._all_done.clear()
        with self._chore_queue_locks[proc_id]:
            heapq.heappush(self._chore_proc_queues[proc_id], (priority, id(task_name), task_name))
        if proc_id < len(self._chore_processors):
            self._chore_processors[proc_id].notify()

    def _dispatch_pool(self, task_name: str, z: int) -> None:
        with self._pending_lock:
            self._pending += 1; self._all_done.clear()
        with self._pool_q_lock:
            heapq.heappush(self._pool_queue, (z, self._pool_q_seq, task_name))
            self._pool_q_seq += 1
        for p in self._pool_processors: p.notify(); break

    def _dequeue_chore(self, proc_id: int) -> Tuple[str, Optional[Callable]]:
        with self._chore_queue_locks[proc_id]:
            if not self._chore_proc_queues[proc_id]: return "", None
            _, _, task_name = heapq.heappop(self._chore_proc_queues[proc_id])
        with self._task_lock:
            func = self._tasks.get(task_name)
        return ("", None) if func is None else (task_name, self._wrap(func))

    def _dequeue_pool(self) -> Tuple[str, Optional[Callable]]:
        with self._pool_q_lock:
            if not self._pool_queue: return "", None
            _, _, task_name = heapq.heappop(self._pool_queue)
        with self._task_lock:
            func = self._tasks.get(task_name)
        return ("", None) if func is None else (task_name, self._wrap(func))

    def _wrap(self, func: Callable) -> Callable:
        def _wrapped() -> None:
            try: func()
            finally:
                with self._pending_lock:
                    self._pending = max(0, self._pending - 1)
                    if self._pending == 0: self._all_done.set()
        return _wrapped

    def run_until_done(self, timeout: float = 10.0) -> bool:
        return self._all_done.wait(timeout=timeout)

    def check_sched_status(self) -> str:
        """CheckSchedStatus — mirrors SchedulerChoreography status snapshot."""
        now = time.monotonic(); parts: List[str] = []
        for p in self._chore_processors + self._pool_processors:  # type: ignore[operator]
            s = p.snapshot
            if s.execute_start_mono > 0.0:
                parts.append(f"{s.processor_id}:{s.routine_name}:{int((now-s.execute_start_mono)*1000)}ms")
            else:
                parts.append(f"{s.processor_id}:idle")
        snap = ", ".join(parts) + f", timestamp: {int(now*1e9)}"
        _dbg("ASTRO-CHORE", f"[ASTRO-CHORE] CheckSchedStatus snap={snap}")
        return snap

    def shutdown(self) -> None:
        """Shutdown — mirrors SchedulerChoreography::Shutdown()."""
        if self._stop: return
        self._stop = True
        _dbg("ASTRO-CHORE",
             f"[ASTRO-CHORE] Shutdown chore={len(self._chore_processors)} pool={len(self._pool_processors)}")
        for p in self._chore_processors: p.stop()
        for p in self._pool_processors:  p.stop()
        with self._task_lock: self._tasks.clear()
        for q in self._chore_proc_queues: q.clear()
        with self._pool_q_lock: self._pool_queue.clear()
        self._executor.shutdown(wait=False)
        _dbg("ASTRO-CHORE", "AstroChoreographyScheduler shutdown complete")
