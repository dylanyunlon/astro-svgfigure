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
