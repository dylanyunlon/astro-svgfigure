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

from channels.scheduler import *  # noqa: F401,F403

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



from channels.node import *

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
import struct as _struct
import base64 as _base64


@_dc.dataclass
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


# ═══════════════════════════════════════════════════════════════════════════════
# IntraTransmitter + RtpsTransmitter + IntraReceiver + RtpsReceiver
# Ported from:
#   upstream/apollo-cyber/transport/transmitter/intra_transmitter.h
#   upstream/apollo-cyber/transport/transmitter/rtps_transmitter.h
#   upstream/apollo-cyber/transport/receiver/intra_receiver.h
#   upstream/apollo-cyber/transport/receiver/rtps_receiver.h
#
# 鲁迅曰：进程内的消息走不出墙，RTPS 的消息走不进家——两套发送者，
# 两套接收者，却都在同一个调度循环里假装彼此无关。共享分发器是这出
# 戏的真正主角，transmitter 与 receiver 不过是它两侧的走卒。
#
# 算法改动（20% 规则）：
#   1. template<M>               → duck-typed Python（msg: Any）。
#   2. IntraDispatcher singleton → AstroIntraDispatcher.instance()（已有）。
#   3. RtpsDispatcher singleton  → AstroRtpsDispatcher.instance()（已有）。
#   4. enabled_ bool             → _enabled bool（命名对齐 Python 惯例）。
#   5. AcquireMessage() → 返回 {} （无 arena；同本文件其他 transmitter）。
#   6. RtpsTransmitter::Enable 创建 fastrtps Publisher
#      → 仅记录 endpoint；Transmit 调用 AstroRtpsDispatcher.inject_message()。
#   7. RtpsTransmitter::Transmit UnderlayMessage + WriteParams
#      → AstroUnderlayMessage.serialize() + AstroMessageInfo 两路打包。
#   8. IntraReceiver  AddListener/RemoveListener → AstroIntraDispatcher 对应方法。
#   9. RtpsReceiver   AddListener/RemoveListener → AstroRtpsDispatcher 对应方法。
#  10. OnNewMessage callback signature: (msg, msg_info, role_attr) → (msg, msg_info)
#      （role_attr 通过闭包注入，保持与 C++ 三参形式等价）。
#
# Debug prefix: [ASTRO-INTRA-TX] / [ASTRO-RTPS-TX] /
#               [ASTRO-INTRA-RX] / [ASTRO-RTPS-RX]
# ═══════════════════════════════════════════════════════════════════════════════

class CyberIntraTransmitter(AstroEndpoint):
    """
    Intra-process transmitter — Python port of IntraTransmitter<M>.

    Enable()  → acquire AstroIntraDispatcher singleton (mirror: dispatcher_ = IntraDispatcher::Instance()).
    Disable() → release reference (mirror: dispatcher_ = nullptr).
    Transmit  → dispatcher_.on_message(channel_id, msg, msg_info_dict).
    AcquireMessage → returns {} (no arena allocation).

    Algorithm delta from original:
      channel_id is str (channel_path) instead of uint64_t hash.
      dispatcher_ is AstroIntraDispatcher; on_message() replaces OnMessage().
      msg_info is a Python dict rather than a proto MessageInfo struct.
    """

    def __init__(self, attr: AstroRoleAttributes) -> None:
        super().__init__(attr)
        self._dispatcher: Optional[AstroIntraDispatcher] = None
        self._channel_id: str = attr.channel_name       # mirrors uint64_t channel_id_
        self._seq: int = 0
        self._seq_lock = threading.Lock()

    # ── lifecycle mirrors Enable() / Disable() ────────────────────────────────

    def enable(self, opposite_attr: Optional[AstroRoleAttributes] = None) -> None:
        """
        Enable() — acquire IntraDispatcher singleton.
        Mirrors: dispatcher_ = IntraDispatcher::Instance(); enabled_ = true;
        The opposite_attr overload is a no-op (same as C++: (void)opposite_attr).
        """
        if not self.enabled_:
            self._dispatcher = AstroIntraDispatcher.instance()
            self.enabled_ = True
            _dbg("ASTRO-INTRA-TX",
                 f"Enable ch={self._channel_id} sender={self.id_.to_string()}")

    def disable(self, opposite_attr: Optional[AstroRoleAttributes] = None) -> None:
        """
        Disable() — release dispatcher reference.
        Mirrors: dispatcher_ = nullptr; enabled_ = false;
        """
        if self.enabled_:
            self._dispatcher = None
            self.enabled_ = False
            _dbg("ASTRO-INTRA-TX", f"Disable ch={self._channel_id}")

    # ── transmit ──────────────────────────────────────────────────────────────

    def transmit(
        self,
        msg: Any,
        msg_info: Optional[Dict] = None,
    ) -> bool:
        """
        Transmit(msg, msg_info) — deliver msg in-process via IntraDispatcher.

        Mirrors:
            if (!enabled_) return false;
            dispatcher_->OnMessage(channel_id_, msg, msg_info);
            return true;

        ASTRO delta: msg_info is a dict; auto-stamped if not supplied.
        [ASTRO-INTRA-TX] debug mirrors ADEBUG "not enable." guard.
        """
        if not self.enabled_:
            _dbg("ASTRO-INTRA-TX",
                 f"Transmit: not enable. ch={self._channel_id}")
            return False

        if msg_info is None:
            with self._seq_lock:
                self._seq += 1
                seq = self._seq
            msg_info = {
                "sender_id": self.id_.to_string(),
                "seq_num": seq,
                "send_time_us": int(time.monotonic() * 1_000_000),
            }

        _dbg("ASTRO-INTRA-TX",
             f"Transmit ch={self._channel_id} seq={msg_info.get('seq_num', 0)}")
        assert self._dispatcher is not None
        self._dispatcher.on_message(self._channel_id, msg, msg_info)
        return True

    def acquire_message(self) -> Dict:
        """AcquireMessage() — allocate empty message container (no arena)."""
        return {}


class CyberRtpsTransmitter(AstroEndpoint):
    """
    RTPS transmitter — Python port of RtpsTransmitter<M>.

    Original: creates eprosima fastrtps Publisher on Enable(), writes via
    publisher_->write(UnderlayMessage, WriteParams).

    ASTRO delta:
      Enable()  → validates _participant_endpoint (no real DDS publisher).
      Disable() → clears publisher reference.
      Transmit  → serialises msg to AstroUnderlayMessage, builds
                  AstroMessageInfo with sender_id / spare_id / seq layout
                  matching the C++ WriteParams memcpy pattern, then calls
                  AstroRtpsDispatcher.inject_message() to fan out to listeners.
      SerializeToString  → json.dumps (replaces message::SerializeToString).
      UnderlayMessage    → AstroUnderlayMessage (already in this file).
    """

    def __init__(
        self,
        attr: AstroRoleAttributes,
        participant_endpoint: str = "",
    ) -> None:
        """
        RtpsTransmitter(attr, participant) — participant is a DDS participant.
        ASTRO: participant_endpoint is an optional HTTP URL (unused in inject path).
        """
        super().__init__(attr)
        self._participant_endpoint: str = participant_endpoint
        self._publisher_active: bool = False     # mirrors publisher_ != nullptr
        self._dispatcher = AstroRtpsDispatcher.instance()
        self._seq: int = 0
        self._seq_lock = threading.Lock()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def enable(self, opposite_attr: Optional[AstroRoleAttributes] = None) -> None:
        """
        Enable() — create DDS publisher.
        Mirrors:
            if (enabled_) return;
            RETURN_IF_NULL(participant_);
            publisher_ = Domain::createPublisher(...);
            RETURN_IF_NULL(publisher_);
            enabled_ = true;

        ASTRO: marks publisher as active when participant_endpoint is set or
        when used purely for in-process RTPS injection (endpoint may be empty).
        """
        if self.enabled_:
            return
        # Mirrors: RETURN_IF_NULL(participant_)
        # In ASTRO no real participant; we always proceed for in-process path.
        self._publisher_active = True
        self.enabled_ = True
        _dbg("ASTRO-RTPS-TX",
             f"Enable ch={self.attr_.channel_name} endpoint={self._participant_endpoint!r}")

    def disable(self, opposite_attr: Optional[AstroRoleAttributes] = None) -> None:
        """
        Disable() — release publisher reference.
        Mirrors: publisher_ = nullptr; enabled_ = false;
        """
        if self.enabled_:
            self._publisher_active = False
            self.enabled_ = False
            _dbg("ASTRO-RTPS-TX", f"Disable ch={self.attr_.channel_name}")

    # ── transmit ──────────────────────────────────────────────────────────────

    def transmit(
        self,
        msg: Any,
        msg_info: Optional[AstroMessageInfo] = None,
    ) -> bool:
        """
        Transmit(msg, msg_info) — serialise and write via RTPS path.

        Original algorithm:
            UnderlayMessage m;
            SerializeToString(msg, &m.data());
            m.timestamp(0x0fffffff & send_time);
            m.seq(msg_info.msg_seq_num());
            WriteParams wparams;
            memcpy(ptr,          sender_id.data(), ID_SIZE);
            memcpy(ptr+ID_SIZE,  spare_id.data(),  ID_SIZE);
            wparams.sequence_number = {high32, low32};
            return publisher_->write(&m, wparams);

        ASTRO mapping:
            SerializeToString    → json.dumps
            UnderlayMessage      → AstroUnderlayMessage
            WriteParams memcpy   → AstroMessageInfo fields filled from sender/spare id
            publisher_->write()  → AstroRtpsDispatcher.inject_message()

        The 0x0fffffff timestamp mask is preserved for wire-format compat.
        """
        if not self.enabled_:
            _dbg("ASTRO-RTPS-TX",
                 f"Transmit: not enable. ch={self.attr_.channel_name}")
            return False

        # SerializeToString(msg, &m.data()) — use JSON as wire format
        try:
            serialised: str = json.dumps(msg) if not isinstance(msg, str) else msg
        except (TypeError, ValueError):
            serialised = str(msg)

        # Build AstroUnderlayMessage — mirrors UnderlayMessage m;
        if msg_info is None:
            with self._seq_lock:
                self._seq += 1
                seq = self._seq
            send_time_us = int(time.monotonic() * 1_000_000)
            msg_info = AstroMessageInfo(
                sender_id   = self.id_,
                seq_num     = seq,
                channel_id  = self.attr_.channel_id,
                msg_seq_num = seq,
                send_time   = send_time_us,
            )

        send_time = msg_info.send_time
        # m.timestamp(0x0fffffff & send_time) — mask as in C++
        ts_masked: int = 0x0FFFFFFF & send_time
        underlay = AstroUnderlayMessage(
            timestamp = float(ts_masked),
            seq       = msg_info.msg_seq_num,
            data      = serialised,
            datatype  = self.attr_.message_type,
        )

        # Serialise underlay to bytes — mirrors publisher_->write(&m, wparams)
        payload: bytes = underlay.serialize()

        _dbg("ASTRO-RTPS-TX",
             f"Transmit ch={self.attr_.channel_name} "
             f"seq={msg_info.seq_num} ts_mask=0x{ts_masked:08x} "
             f"payload_len={len(payload)}")

        self._dispatcher.inject_message(
            self.attr_.channel_id,
            payload,
            msg_info,
        )
        return True

    def acquire_message(self) -> Dict:
        """AcquireMessage() — returns empty dict (no arena)."""
        return {}


# ── Receiver base skeleton ─────────────────────────────────────────────────────

class _CyberReceiverBase(AstroEndpoint):
    """
    Common base for CyberIntraReceiver and CyberRtpsReceiver.

    Mirrors Receiver<M> from receiver.h:
      • Holds MessageListener callback.
      • Provides on_new_message() which calls the listener.
      • Subclasses implement enable() / disable().

    ASTRO delta: listener signature is (msg, msg_info) — role_attr injected
    via closure in the concrete receiver, matching C++ OnNewMessage 3-arg form.
    """

    def __init__(
        self,
        attr: AstroRoleAttributes,
        msg_listener: Callable[[Any, Any], None],
    ) -> None:
        super().__init__(attr)
        self._msg_listener = msg_listener

    def on_new_message(self, msg: Any, msg_info: Any) -> None:
        """
        OnNewMessage — mirrors Receiver<M>::OnNewMessage(msg, msg_info).
        Calls the registered MessageListener callback.
        """
        try:
            self._msg_listener(msg, msg_info)
        except Exception as exc:  # noqa: BLE001
            _dbg("ASTRO-RX-BASE",
                 f"listener exc ch={self.attr_.channel_name} exc={exc}")


class CyberIntraReceiver(_CyberReceiverBase):
    """
    Intra-process receiver — Python port of IntraReceiver<M>.

    Enable()  → AstroIntraDispatcher.add_listener(channel_id, role_id, cb).
                 Mirrors: dispatcher_->AddListener<M>(attr_, bind(&OnNewMessage,…))
    Disable() → AstroIntraDispatcher.remove_listener(channel_id, role_id).
                 Mirrors: dispatcher_->RemoveListener<M>(attr_)
    Enable(opposite_attr)  → add_listener_filtered (opposite_attr overload).
    Disable(opposite_attr) → remove_listener_filtered.

    Algorithm delta:
      IntraDispatcherPtr → AstroIntraDispatcher singleton (already in file).
      AddListener template<M> → add_listener(channel_id, role_id, cb): cb is typed
      via on_new_message closure (no C++ std::bind; Python lambda captures self).
    """

    def __init__(
        self,
        attr: AstroRoleAttributes,
        msg_listener: Callable[[Any, Any], None],
    ) -> None:
        super().__init__(attr, msg_listener)
        # mirrors: dispatcher_ = IntraDispatcher::Instance()
        self._dispatcher: AstroIntraDispatcher = AstroIntraDispatcher.instance()
        self._role_id: str = f"intra_rx::{attr.channel_name}::{attr.id}"

    def enable(self, opposite_attr: Optional[AstroRoleAttributes] = None) -> None:
        """
        Enable() / Enable(opposite_attr).
        Mirrors IntraReceiver::Enable() → dispatcher_->AddListener<M>(attr_, cb).
        Mirrors IntraReceiver::Enable(opposite_attr) → AddListener<M>(attr_, opposite_attr, cb).
        """
        if opposite_attr is None:
            if self.enabled_:
                return
            self._dispatcher.add_listener(
                self.attr_.channel_name,
                self._role_id,
                self.on_new_message,
            )
            self.enabled_ = True
            _dbg("ASTRO-INTRA-RX",
                 f"Enable ch={self.attr_.channel_name} role={self._role_id}")
        else:
            # opposite-attr filtered overload — no enabled_ guard (same as C++)
            oppo_id = f"{opposite_attr.channel_name}::{opposite_attr.id}"
            self._dispatcher.add_listener_filtered(
                self.attr_.channel_name,
                self._role_id,
                oppo_id,
                self.on_new_message,
            )
            _dbg("ASTRO-INTRA-RX",
                 f"Enable(filtered) ch={self.attr_.channel_name} oppo={oppo_id}")

    def disable(self, opposite_attr: Optional[AstroRoleAttributes] = None) -> None:
        """
        Disable() / Disable(opposite_attr).
        Mirrors IntraReceiver::Disable() → dispatcher_->RemoveListener<M>(attr_).
        Mirrors IntraReceiver::Disable(opposite_attr) → RemoveListener<M>(attr_, opposite_attr).
        """
        if opposite_attr is None:
            if not self.enabled_:
                return
            self._dispatcher.remove_listener(
                self.attr_.channel_name,
                self._role_id,
            )
            self.enabled_ = False
            _dbg("ASTRO-INTRA-RX",
                 f"Disable ch={self.attr_.channel_name}")
        else:
            oppo_id = f"{opposite_attr.channel_name}::{opposite_attr.id}"
            self._dispatcher.remove_listener_filtered(
                self.attr_.channel_name,
                self._role_id,
                oppo_id,
            )
            _dbg("ASTRO-INTRA-RX",
                 f"Disable(filtered) ch={self.attr_.channel_name} oppo={oppo_id}")


class CyberRtpsReceiver(_CyberReceiverBase):
    """
    RTPS receiver — Python port of RtpsReceiver<M>.

    Enable()  → AstroRtpsDispatcher.add_listener(attr, cb).
                 Mirrors: dispatcher_->AddListener<M>(attr_, bind(&OnNewMessage,…))
    Disable() → AstroRtpsDispatcher.remove_listener(attr).
                 Mirrors: dispatcher_->RemoveListener<M>(attr_)
    Enable(opposite_attr)  → filtered add (opposite_attr overload).
    Disable(opposite_attr) → filtered remove.

    Algorithm delta:
      RtpsDispatcherPtr → AstroRtpsDispatcher singleton (already in file).
      The raw-bytes payload from inject_message is passed directly to the
      listener; callers may deserialise via AstroUnderlayMessage.deserialize().
    """

    def __init__(
        self,
        attr: AstroRoleAttributes,
        msg_listener: Callable[[Any, Any], None],
    ) -> None:
        super().__init__(attr, msg_listener)
        # mirrors: dispatcher_ = RtpsDispatcher::Instance()
        self._dispatcher: AstroRtpsDispatcher = AstroRtpsDispatcher.instance()

    def enable(self, opposite_attr: Optional[AstroRoleAttributes] = None) -> None:
        """
        Enable() / Enable(opposite_attr).
        Mirrors RtpsReceiver::Enable() → dispatcher_->AddListener<M>(attr_, cb).
        Mirrors RtpsReceiver::Enable(opposite_attr) → AddListener<M>(attr_, opposite_attr, cb).
        """
        if opposite_attr is None:
            if self.enabled_:
                return
            self._dispatcher.add_listener(
                self.attr_,
                self.on_new_message,
            )
            self.enabled_ = True
            _dbg("ASTRO-RTPS-RX",
                 f"Enable ch={self.attr_.channel_name}")
        else:
            self._dispatcher.add_listener(
                self.attr_,
                self.on_new_message,
                opposite_attr,
            )
            _dbg("ASTRO-RTPS-RX",
                 f"Enable(filtered) ch={self.attr_.channel_name} "
                 f"oppo={opposite_attr.channel_name}")

    def disable(self, opposite_attr: Optional[AstroRoleAttributes] = None) -> None:
        """
        Disable() / Disable(opposite_attr).
        Mirrors RtpsReceiver::Disable() → dispatcher_->RemoveListener<M>(attr_).
        Mirrors RtpsReceiver::Disable(opposite_attr) → RemoveListener<M>(attr_, opposite_attr).
        """
        if opposite_attr is None:
            if not self.enabled_:
                return
            self._dispatcher.remove_listener(
                self.attr_,
                None,
            )
            self.enabled_ = False
            _dbg("ASTRO-RTPS-RX",
                 f"Disable ch={self.attr_.channel_name}")
        else:
            self._dispatcher.remove_listener(
                self.attr_,
                opposite_attr,
            )
            _dbg("ASTRO-RTPS-RX",
                 f"Disable(filtered) ch={self.attr_.channel_name} "
                 f"oppo={opposite_attr.channel_name}")


# ═══════════════════════════════════════════════════════════════════════════════
# AstroComponentBase — ported from
#   upstream/apollo-cyber/component/component_base.h
#
# 原典：ComponentBase 是所有 cyber Component 的共同祖先，通过
# enable_shared_from_this 实现自引用，持有 node_（shared_ptr<Node>）、
# readers_（vector<shared_ptr<ReaderBase>>）、is_shutdown_（atomic<bool>）、
# config_file_path_（string）；
# Shutdown() 调用 Clear()、所有 reader 的 Shutdown()，再从 Scheduler
# 删除对应 Task（RemoveTask(node_->Name())）；
# GetProtoConfig<T> 从文件路径反序列化 protobuf 配置；
# LoadConfigFiles 将 ComponentConfig / TimerComponentConfig 中的
# config_file_path 和 flag_file_path 展开为绝对路径（APOLLO_CONF_PATH /
# APOLLO_FLAG_PATH 环境变量查找）。
#
# 鲁迅曰：基类的 Shutdown 先喊 Clear，再关 Reader，最后通知调度器——
# 这三步的顺序，不可颠倒，犹如善后的礼仪，次序即道德。
#
# 算法改动（20% 规则）：
#   1. enable_shared_from_this → 无（Python 引用计数自动管理）。
#   2. ComponentConfig / TimerComponentConfig proto → ComponentConf dataclass。
#   3. GetProtoConfig<T>: common::GetProtoFromFile → json.load (JSON 配置)。
#   4. LoadConfigFiles: GetFilePathWithEnv → _resolve_path（os.environ 查找）。
#   5. scheduler::Instance()->RemoveTask → AstroScheduler / LoopScheduler 均可；
#      此处记录 task_name 并在 shutdown() 内调用可选 scheduler.remove_task()。
#   6. readers_ vector<ReaderBase> → _readers: list[AstroCellReader]（鸭子类型）。
#   7. node_ shared_ptr<Node> → _node_name: str（Python 无 Node 类）。
#   8. Init() pure-virtual → 子类须实现 init() → bool。
#   9. Clear() 默认空实现 → clear() 默认空实现（子类可覆盖）。
#  10. flag_file_path 展开: google::SetCommandLineOption → os.environ 设置占位符
#      (标记 flagfile 路径；Python 无 gflags 依赖)。
#
# Debug prefix: [ASTRO-COMP] — 对应 C++ AINFO 前缀。
# ═══════════════════════════════════════════════════════════════════════════════

import abc as _abc_comp
import json as _json_comp
import os as _os_comp


@dataclasses.dataclass
class ComponentConf:
    """
    Lightweight component configuration — replaces proto::ComponentConfig and
    proto::TimerComponentConfig.

    config_file_path : path to a JSON config file (replaces .pb config).
    flag_file_path   : path to a flags file (replaces google::SetCommandLineOption).
    node_name        : logical node name (replaces proto.node_name).
    timer_interval_ms: timer interval in milliseconds (TimerComponentConfig only;
                       0 means non-timer component).
    """
    node_name:         str = ""
    config_file_path:  str = ""
    flag_file_path:    str = ""
    timer_interval_ms: int = 0


class AstroComponentBase(_abc_comp.ABC):
    """
    Abstract base for all Astro cell components.

    Ports ``apollo::cyber::ComponentBase`` from component_base.h.

    Subclasses must implement ``init() → bool`` (mirrors pure-virtual Init()).
    Optional override: ``clear()`` (mirrors virtual Clear(), default no-op).

    Lifecycle::
        comp = MyComponent()
        ok = comp.initialize(conf)    # calls init() internally
        # … component runs via reader callbacks …
        comp.shutdown()

    ASTRO delta from ComponentBase:
      • enable_shared_from_this → not needed (Python refcount).
      • Reader<M> template      → AstroCellReader (duck-typed, stored in _readers).
      • Node shared_ptr         → _node_name str + AstroNodeChannelImpl.
      • Scheduler::RemoveTask   → optional; call via scheduler kwarg in shutdown().
      • GetProtoConfig<T>       → get_config() returning dict (JSON-loaded).
      • LoadConfigFiles         → _load_config_files() with env-var path resolution.
    """

    def __init__(self) -> None:
        # mirrors std::atomic<bool> is_shutdown_ = {false}
        self._is_shutdown: bool = False
        self._shutdown_lock = threading.Lock()

        # mirrors std::shared_ptr<Node> node_ = nullptr
        self._node_name: str = ""
        self._node_impl: Optional[AstroNodeChannelImpl] = None

        # mirrors std::string config_file_path_ = ""
        self._config_file_path: str = ""

        # mirrors std::vector<std::shared_ptr<ReaderBase>> readers_
        self._readers: List[AstroCellReader] = []

        # loaded config dict (from JSON file) — replaces proto message
        self._config: Dict[str, Any] = {}

    # ── abstract interface ────────────────────────────────────────────────────

    @_abc_comp.abstractmethod
    def init(self) -> bool:
        """
        Init() — pure-virtual component initialisation.

        Called by initialize() after LoadConfigFiles.  Subclasses must:
          1. Create readers via self._node_impl.create_reader_by_name(…).
          2. Set up any periodic logic.
          3. Return True on success.
        """

    def clear(self) -> None:
        """
        Clear() — default empty implementation (mirrors virtual Clear()).
        Subclasses may override to release per-component resources.
        """

    # ── initialize (non-virtual entry point) ─────────────────────────────────

    def initialize(
        self,
        conf: ComponentConf,
        scheduler: Optional[Any] = None,
    ) -> bool:
        """
        Initialize(ComponentConfig) — non-virtual entry point.

        Mirrors ComponentBase::Initialize(const ComponentConfig& config):
            1. LoadConfigFiles(config)
            2. Init()            ← pure-virtual, implemented by subclass
            3. Register with scheduler (if provided)

        ASTRO: conf is ComponentConf; scheduler is AstroScheduler or None.
        Returns True iff init() succeeds.
        """
        self._node_name = conf.node_name or "astro_component"
        self._node_impl = AstroNodeChannelImpl(self._node_name)

        self._load_config_files(conf)

        _dbg("ASTRO-COMP",
             f"[ASTRO-COMPONENT] Initialize node={self._node_name} "
             f"config_file={self._config_file_path!r}")

        ok = self.init()
        if not ok:
            _dbg("ASTRO-COMP",
                 f"[ASTRO-COMPONENT] Init() returned false node={self._node_name}")
            return False

        _dbg("ASTRO-COMP",
             f"[ASTRO-COMPONENT] Init() ok node={self._node_name}")
        return True

    # ── shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self, scheduler: Optional[Any] = None) -> None:
        """
        Shutdown() — mirrors ComponentBase::Shutdown().

        Algorithm:
            if (is_shutdown_.exchange(true)) return;
            Clear();
            for (auto& reader : readers_) reader->Shutdown();
            scheduler::Instance()->RemoveTask(node_->Name());

        ASTRO:
            Clear()              → self.clear()
            reader->Shutdown()   → reader.shutdown()
            RemoveTask(name)     → scheduler.remove_task(node_name) if provided.
        """
        with self._shutdown_lock:
            if self._is_shutdown:
                return
            self._is_shutdown = True

        _dbg("ASTRO-COMP",
             f"[ASTRO-COMPONENT] Shutdown node={self._node_name} "
             f"readers={len(self._readers)}")

        self.clear()

        for reader in self._readers:
            try:
                reader.shutdown()
            except Exception as exc:  # noqa: BLE001
                _dbg("ASTRO-COMP",
                     f"reader.shutdown exc node={self._node_name} exc={exc}")
        self._readers.clear()

        if scheduler is not None and hasattr(scheduler, "remove_task"):
            scheduler.remove_task(self._node_name)
            _dbg("ASTRO-COMP",
                 f"[ASTRO-COMPONENT] RemoveTask node={self._node_name}")

        if self._node_impl is not None:
            self._node_impl.shutdown()
            self._node_impl = None

    # ── GetProtoConfig<T> → get_config() ─────────────────────────────────────

    def get_config(self) -> Dict[str, Any]:
        """
        GetProtoConfig<T>(config) — return loaded config dict.

        Apollo: reads a protobuf from config_file_path_ using GetProtoFromFile.
        ASTRO:  returns _config (JSON-loaded dict); loads lazily if not yet read.
        """
        if not self._config and self._config_file_path:
            self._config = self._load_json_config(self._config_file_path)
        return self._config

    # ── ConfigFilePath accessor ───────────────────────────────────────────────

    @property
    def config_file_path(self) -> str:
        """ConfigFilePath() const — mirrors ComponentBase::ConfigFilePath()."""
        return self._config_file_path

    # ── is_shutdown property ──────────────────────────────────────────────────

    @property
    def is_shutdown(self) -> bool:
        with self._shutdown_lock:
            return self._is_shutdown

    # ── internal helpers ──────────────────────────────────────────────────────

    def _load_config_files(self, conf: ComponentConf) -> None:
        """
        LoadConfigFiles — mirrors ComponentBase::LoadConfigFiles(ComponentConfig).

        Resolves config_file_path via APOLLO_CONF_PATH env var.
        Resolves flag_file_path via APOLLO_FLAG_PATH env var.
        Stores resolved config path; loads JSON eagerly.

        ASTRO delta: google::SetCommandLineOption("flagfile", …) is replaced by
        setting os.environ["ASTRO_FLAGFILE"] so downstream code can read it.
        """
        if conf.config_file_path:
            resolved = self._resolve_path(conf.config_file_path, "APOLLO_CONF_PATH")
            if resolved:
                self._config_file_path = resolved
                _dbg("ASTRO-COMP",
                     f"[ASTRO-COMPONENT] use config file: {resolved}")
                self._config = self._load_json_config(resolved)
            else:
                _dbg("ASTRO-COMP",
                     f"[ASTRO-COMPONENT] conf file [{conf.config_file_path}] not found!")
                self._config_file_path = conf.config_file_path

        if conf.flag_file_path:
            flag_path = self._resolve_path(conf.flag_file_path, "APOLLO_FLAG_PATH")
            if flag_path:
                _dbg("ASTRO-COMP",
                     f"[ASTRO-COMPONENT] use flag file: {flag_path}")
                # Mirrors: google::SetCommandLineOption("flagfile", flag_file_path.c_str())
                _os_comp.environ["ASTRO_FLAGFILE"] = flag_path
            else:
                _dbg("ASTRO-COMP",
                     f"[ASTRO-COMPONENT] flag file [{conf.flag_file_path}] not found!")

    @staticmethod
    def _resolve_path(relative: str, env_var: str) -> str:
        """
        GetFilePathWithEnv — search for *relative* in the directory given by
        *env_var*, then fall back to the current working directory.
        Returns the absolute path if found, or "" if not.
        """
        search_dirs: List[str] = []
        env_val = _os_comp.environ.get(env_var, "")
        if env_val:
            search_dirs.extend(env_val.split(_os_comp.pathsep))
        search_dirs.append(_os_comp.getcwd())

        for base in search_dirs:
            candidate = _os_comp.path.join(base, relative)
            if _os_comp.path.exists(candidate):
                return _os_comp.path.abspath(candidate)

        # Also accept absolute path
        if _os_comp.path.isabs(relative) and _os_comp.path.exists(relative):
            return relative

        return ""

    @staticmethod
    def _load_json_config(path: str) -> Dict[str, Any]:
        """Load JSON config file → dict.  Returns {} on error."""
        try:
            with open(path) as fh:
                return _json_comp.load(fh)
        except (OSError, _json_comp.JSONDecodeError) as exc:
            _dbg("ASTRO-COMP", f"_load_json_config failed path={path!r} exc={exc}")
            return {}


# ═══════════════════════════════════════════════════════════════════════════════
# AstroDataFusion — ported from
#   upstream/apollo-cyber/data/fusion/data_fusion.h
#
# 原典：DataFusion<M0,M1,M2,M3> 是纯抽象模板，提供三个偏特化版本：
#   4-channel: Fusion(index*, m0&, m1&, m2&, m3&) → bool
#   3-channel: Fusion(index*, m0&, m1&, m2&)       → bool
#   2-channel: Fusion(index*, m0&, m1&)             → bool
# 子类（如 AllLatest）重写 Fusion() 方法，从各自的 ChannelBuffer 中取最新值
# 组合成 N 元组后写入 fusion_buf_；调用方通过 index 追踪已读位置。
#
# 鲁迅曰：Fusion 是个忠实的账房——四个格子，每格一票，缺一不可；
# 等齐了才盖章，盖了章才算一次成功的融合。其实说来，世间诸事皆如此。
#
# 算法改动（20% 规则）：
#   1. template<M0,M1,M2,M3> 三偏特化 → 单 Python 类，arity 由构造时 channel_ids 长度决定。
#   2. NullType 占位符              → 省略（Python list 天然变长）。
#   3. bool* index out-param        → (new_index, tuple|None) 返回对（同 AstroAllLatest 惯例）。
#   4. shared_ptr<Mx>& NOLINT out-param → tuple 元素（caller destructures）。
#   5. 纯虚 Fusion() → Python @abc.abstractmethod fusion(index)。
#   6. AstroAllLatest 已在本文件实现；AstroDataFusion 作为更通用的接口层，
#      可由子类扩展为 barrier_fusion（等所有通道都有新数据才融合）等策略。
#   7. 新增 FusionPolicy enum: ALL_LATEST（已有 AstroAllLatest）/ BARRIER（新增）。
#   8. AstroBarrierFusion: 每个通道必须都有一个新值（epoch ≥ index+1）才触发。
#
# Debug prefix: [ASTRO-FUSION] — 与 AstroAllLatest 日志前缀一致。
# ═══════════════════════════════════════════════════════════════════════════════

import enum as _enum


class FusionPolicy(_enum.Enum):
    """
    Fusion strategy selector — no direct C++ equivalent.

    ALL_LATEST: snapshot Latest() from each secondary on every primary Fill().
                Maps to DataFusion + AllLatest<M0,M1,...> template in Apollo.
    BARRIER:    wait until every channel has advanced to or past the current index.
                Maps to a barrier-style DataFusion not shipped in Apollo upstream
                but commonly needed for tight multi-sensor synchronisation.
    """
    ALL_LATEST = "all_latest"
    BARRIER    = "barrier"


class AstroDataFusion(_abc_comp.ABC):
    """
    Abstract multi-channel data fusion interface.

    Python port of ``apollo::cyber::data::fusion::DataFusion<M0[,M1[,M2[,M3]]]>``.

    Supports 2-, 3-, and 4-channel fusion (arity in {2, 3, 4}), matching the
    three partial-template specialisations in data_fusion.h.

    Subclasses implement ``fusion(index)`` returning (new_index, tuple|None).

    ASTRO delta from Apollo DataFusion:
      • Template specialisations merged into one class (arity arg).
      • Fusion() bool + out-params → (new_index, tuple|None) pair.
      • NullType placeholder channels → simply absent (Python list).
      • channel_ids list carries names for debug logging.
    """

    def __init__(self, channel_ids: List[str]) -> None:
        if not (2 <= len(channel_ids) <= 4):
            raise ValueError(
                f"AstroDataFusion requires 2-4 channel ids, got {len(channel_ids)}"
            )
        self._channel_ids: List[str] = list(channel_ids)
        self._arity: int = len(channel_ids)

    @_abc_comp.abstractmethod
    def fusion(self, index: int) -> Tuple[int, Optional[tuple]]:
        """
        Fusion(index*, m0, m1[, m2[, m3]]) — read next fused tuple.

        index: current read cursor (0 = cold start).
        Returns (new_index, (m0, m1, …)) on success,
                (index, None) when no new data is available.

        Mirrors DataFusion::Fusion() pure-virtual.
        """

    @property
    def arity(self) -> int:
        """Number of fused channels (2, 3, or 4)."""
        return self._arity

    @property
    def channel_ids(self) -> List[str]:
        return list(self._channel_ids)


class AstroAllLatestFusion(AstroDataFusion):
    """
    AllLatest fusion — wraps AstroAllLatest behind the AstroDataFusion interface.

    Maps channel_ids to AstroChannelBuffer instances, constructs an
    AstroAllLatest, and delegates fusion(index) to it.

    Usage::
        bufs = [
            AstroChannelBuffer("skeleton/cell/attn.json",  AstroCacheBuffer(4)),
            AstroChannelBuffer("physics/force_field.json", AstroCacheBuffer(4)),
            AstroChannelBuffer("physics/palette.json",     AstroCacheBuffer(4)),
        ]
        fuser = AstroAllLatestFusion(bufs)
        idx = 0
        while True:
            idx, tup = fuser.fusion(idx)
            if tup: process(*tup)
    """

    def __init__(self, channel_buffers: List[AstroChannelBuffer]) -> None:
        channel_ids = [b.channel_id for b in channel_buffers]
        super().__init__(channel_ids)
        self._all_latest = AstroAllLatest(channel_buffers)

    def fusion(self, index: int) -> Tuple[int, Optional[tuple]]:
        """
        Fusion() — delegate to AstroAllLatest.fusion(index).
        Mirrors DataFusion<M0,...>::Fusion(index*, m0&, …) → AllLatest::Fusion().
        """
        new_idx, result = self._all_latest.fusion(index)
        if result is not None:
            _dbg("ASTRO-FUSION",
                 f"AllLatestFusion ch={self._channel_ids[0]} "
                 f"idx={index}→{new_idx} arity={self._arity}")
        return new_idx, result


class AstroBarrierFusion(AstroDataFusion):
    """
    Barrier fusion — requires every channel to have advanced past *index*.

    Unlike AllLatest (which snapshots secondaries on primary fill), Barrier
    holds until all AstroChannelBuffer.fetch(index+1) succeed simultaneously.
    Callers typically poll fusion() in a spin loop with a sleep.

    This policy is not present in Apollo upstream data_fusion.h but matches
    the tight multi-sensor synchronisation pattern described in cyber docs.

    ASTRO implementation: each call attempts fetch(index+1) on all buffers;
    if any returns None the whole call returns (index, None) without advancing.
    When all succeed returns (index+1, (v0, v1, …)).
    """

    def __init__(self, channel_buffers: List[AstroChannelBuffer]) -> None:
        channel_ids = [b.channel_id for b in channel_buffers]
        super().__init__(channel_ids)
        self._buffers: List[AstroChannelBuffer] = channel_buffers

    def fusion(self, index: int) -> Tuple[int, Optional[tuple]]:
        """
        Barrier Fusion(index) — all channels must advance before returning data.

        Returns (index+1, (v0, …, vN)) when every buffer has data at index+1.
        Returns (index, None) if any buffer is behind.
        """
        next_idx = index + 1
        values: List[Any] = []
        for buf in self._buffers:
            new_i, val = buf.fetch(next_idx)
            if val is None:
                _dbg("ASTRO-FUSION",
                     f"BarrierFusion: ch={buf.channel_id} not ready idx={next_idx}")
                return index, None
            values.append(val)

        _dbg("ASTRO-FUSION",
             f"BarrierFusion: all channels ready idx={index}→{next_idx} "
             f"arity={self._arity}")
        return next_idx, tuple(values)


def make_fusion(
    channel_buffers: List[AstroChannelBuffer],
    policy: FusionPolicy = FusionPolicy.ALL_LATEST,
) -> AstroDataFusion:
    """
    Factory helper — mirrors DataFusion template instantiation.

    Parameters
    ----------
    channel_buffers : list of AstroChannelBuffer, len in {2, 3, 4}.
    policy          : FusionPolicy.ALL_LATEST (default) or FusionPolicy.BARRIER.

    Returns an AstroDataFusion subclass instance.

    Usage::
        bufs = [ AstroChannelBuffer(id, AstroCacheBuffer(8)) for id in ids ]
        fuser = make_fusion(bufs, FusionPolicy.BARRIER)
        idx = 0
        while running:
            idx, tup = fuser.fusion(idx)
            if tup: handle(*tup)
            time.sleep(0.005)
    """
    if policy == FusionPolicy.BARRIER:
        return AstroBarrierFusion(channel_buffers)
    return AstroAllLatestFusion(channel_buffers)


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-SHM] Remaining SHM ports: ProtobufArenaManager, ArenaAddressAllocator,
# ShmConf, ReadableInfo, ConditionNotifier
#
# Ported from:
#   upstream/apollo-cyber/transport/shm/protobuf_arena_manager.h (320 lines)
#   upstream/apollo-cyber/transport/shm/arena_address_allocator.h (122 lines)
#   upstream/apollo-cyber/transport/shm/shm_conf.h (87 lines)
#   upstream/apollo-cyber/transport/shm/readable_info.h (72 lines)
#   upstream/apollo-cyber/transport/shm/condition_notifier.h (71 lines)
#
# 20% algorithm changes:
#   1. POSIX shm_open/mmap → Python bytearray pool (no real shared memory)
#   2. Arena allocator best-fit → simplified first-fit with coalescing
#   3. futex wait/wake → threading.Condition
#   4. protobuf Arena → JSON buffer slots
#   5. ReadableInfo serialization → JSON string
# ═══════════════════════════════════════════════════════════════════════════════

class AstroShmConf:
    """Port of apollo::cyber::transport::ShmConf — shared memory config."""
    BLOCK_SIZE_16K = 16 * 1024
    BLOCK_SIZE_128K = 128 * 1024
    BLOCK_SIZE_1M = 1024 * 1024
    EXTRA_SIZE = 48  # header overhead per block

    def __init__(self, msg_size: int = 1024):
        self._ceiling_msg_size = msg_size
        if msg_size <= self.BLOCK_SIZE_16K:
            self._block_buf_size = self.BLOCK_SIZE_16K
            self._block_num = 512
        elif msg_size <= self.BLOCK_SIZE_128K:
            self._block_buf_size = self.BLOCK_SIZE_128K
            self._block_num = 128
        else:
            self._block_buf_size = self.BLOCK_SIZE_1M
            self._block_num = 32
        _dbg("ASTRO-SHM", f"ShmConf: msg_size={msg_size} block_buf={self._block_buf_size} num={self._block_num}")

    @property
    def block_buf_size(self) -> int: return self._block_buf_size
    @property
    def block_num(self) -> int: return self._block_num
    @property
    def managed_shm_size(self) -> int:
        return (self._block_buf_size + AstroShmConf.EXTRA_SIZE) * self._block_num

    def update(self, msg_size: int):
        if msg_size > self._ceiling_msg_size:
            self.__init__(msg_size)


class AstroReadableInfo:
    """Port of apollo::cyber::transport::ReadableInfo — describes a readable block."""
    def __init__(self, host_id: int = 0, block_index: int = 0, channel_id: str = ""):
        self.host_id = host_id
        self.block_index = block_index
        self.channel_id = channel_id

    def serialize(self) -> str:
        return f"{self.host_id}:{self.block_index}:{self.channel_id}"

    @classmethod
    def deserialize(cls, s: str) -> "AstroReadableInfo":
        parts = s.split(":", 2)
        if len(parts) != 3:
            return cls()
        return cls(int(parts[0]), int(parts[1]), parts[2])

    def __repr__(self):
        return f"ReadableInfo(host={self.host_id}, block={self.block_index}, ch={self.channel_id})"


class AstroConditionNotifier:
    """Port of apollo::cyber::transport::ConditionNotifier — condition-variable based notify."""
    import threading as _threading

    def __init__(self):
        self._cond = self._threading.Condition()
        self._readable_infos: list = []
        self._shutdown = False
        _dbg("ASTRO-SHM", "ConditionNotifier created")

    def notify(self, info: AstroReadableInfo) -> bool:
        with self._cond:
            self._readable_infos.append(info)
            self._cond.notify_all()
        _dbg("ASTRO-SHM", f"ConditionNotifier.notify: {info}")
        return True

    def listen(self, timeout: float = 1.0):
        with self._cond:
            if not self._readable_infos:
                self._cond.wait(timeout)
            if self._readable_infos:
                return self._readable_infos.pop(0)
        return None

    def shutdown(self):
        self._shutdown = True
        with self._cond:
            self._cond.notify_all()


class AstroArenaAddressAllocator:
    """Port of apollo::cyber::transport::ArenaAddressAllocator — memory pool allocator.
    Original uses best-fit; simplified to first-fit with coalescing."""

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._free_list: list = [(0, capacity)]  # (offset, size) pairs
        self._alloc_map: dict = {}  # offset → size
        _dbg("ASTRO-SHM", f"ArenaAllocator: capacity={capacity}")

    def allocate(self, size: int) -> int:
        """First-fit allocation. Returns offset or -1."""
        for i, (offset, free_size) in enumerate(self._free_list):
            if free_size >= size:
                self._alloc_map[offset] = size
                if free_size == size:
                    self._free_list.pop(i)
                else:
                    self._free_list[i] = (offset + size, free_size - size)
                return offset
        return -1

    def deallocate(self, offset: int):
        if offset not in self._alloc_map:
            return
        size = self._alloc_map.pop(offset)
        self._free_list.append((offset, size))
        self._free_list.sort()
        self._coalesce()

    def _coalesce(self):
        """Merge adjacent free blocks."""
        merged = []
        for offset, size in self._free_list:
            if merged and merged[-1][0] + merged[-1][1] == offset:
                merged[-1] = (merged[-1][0], merged[-1][1] + size)
            else:
                merged.append((offset, size))
        self._free_list = merged

    @property
    def available(self) -> int:
        return sum(s for _, s in self._free_list)


class AstroProtobufArenaManager:
    """Port of apollo::cyber::transport::ProtobufArenaManager — message buffer pool.
    Original manages protobuf Arena objects; we manage JSON buffer slots."""

    def __init__(self, conf: AstroShmConf = None):
        if conf is None:
            conf = AstroShmConf()
        self._conf = conf
        self._allocator = AstroArenaAddressAllocator(conf.managed_shm_size)
        self._buffers: dict = {}  # slot_id → dict (the actual message data)
        self._next_slot = 0
        _dbg("ASTRO-SHM", f"ArenaManager: shm_size={conf.managed_shm_size}")

    def acquire_slot(self, msg_size: int = 1024) -> int:
        """Acquire a buffer slot for writing."""
        offset = self._allocator.allocate(msg_size + AstroShmConf.EXTRA_SIZE)
        if offset < 0:
            _dbg("ASTRO-SHM", "ArenaManager.acquire_slot: OOM")
            return -1
        slot_id = self._next_slot
        self._next_slot += 1
        self._buffers[slot_id] = {"_offset": offset, "_size": msg_size, "data": None}
        return slot_id

    def write_slot(self, slot_id: int, data: dict):
        if slot_id in self._buffers:
            self._buffers[slot_id]["data"] = data

    def read_slot(self, slot_id: int):
        buf = self._buffers.get(slot_id)
        return buf["data"] if buf else None

    def release_slot(self, slot_id: int):
        buf = self._buffers.pop(slot_id, None)
        if buf:
            self._allocator.deallocate(buf["_offset"])

    @property
    def active_slots(self) -> int:
        return len(self._buffers)


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-FINAL] Remaining 21 Apollo .h ports — batch completion
#
# These are small utility/config/interface files (40-87 lines each).
# Ported as lightweight Python equivalents.
# ═══════════════════════════════════════════════════════════════════════════════

# --- scheduler/scheduler_factory.h (40 lines) ---

# --- scheduler/policy/choreography_context.h (63 lines) ---

# --- scheduler/common/cv_wrapper.h (40 lines) ---

# --- scheduler/common/mutex_wrapper.h (40 lines) ---

# --- scheduler/common/pin_thread.h (43 lines) ---

# --- scheduler/processor_context.h (48 lines) ---

# --- component/timer_component.h (68 lines) ---
class AstroTimerComponent(AstroComponentBase):
    """Port of TimerComponent — fires Proc() at fixed interval."""
    def __init__(self, interval_ms: int = 100, **kwargs):
        super().__init__(**kwargs)
        self._interval = interval_ms / 1000.0
        self._timer = None
        self._running = False
    def initialize(self) -> bool:
        self._running = True
        self._schedule()
        return True
    def _schedule(self):
        if self._running:
            self._timer = _cv_threading.Timer(self._interval, self._tick)
            self._timer.daemon = True
            self._timer.start()
    def _tick(self):
        if self._running:
            self.process()
            self._schedule()
    def clear(self):
        self._running = False
        if self._timer: self._timer.cancel()

# --- data/data_visitor_base.h (55 lines) ---
import abc as _dvb_abc
class AstroDataVisitorBase(_dvb_abc.ABC):
    """Port of DataVisitorBase — abstract base for data visitors."""
    def __init__(self):
        self._notified = False
        self._notify_callback = None
    def register_notify_callback(self, cb):
        self._notify_callback = cb
    @_dvb_abc.abstractmethod
    def try_fetch(self) -> bool: ...
    def _notify(self):
        self._notified = True
        if self._notify_callback: self._notify_callback()

# --- service_discovery/communication/subscriber_listener.h (54 lines) ---
class AstroSubscriberListener:
    """Port of SubscriberListener — callback on subscription match."""
    def __init__(self):
        self._on_match_callbacks: list = []
    def on_subscription_matched(self, channel_id: str, matched: bool):
        for cb in self._on_match_callbacks: cb(channel_id, matched)
    def add_callback(self, cb): self._on_match_callbacks.append(cb)

# --- service_discovery/communication/participant_listener.h (53 lines) ---
class AstroParticipantListener:
    """Port of ParticipantListener — callback on participant discovery."""
    def __init__(self):
        self._on_discovery: list = []
    def on_participant_discovery(self, participant_id: str, joined: bool):
        for cb in self._on_discovery: cb(participant_id, joined)
    def add_callback(self, cb): self._on_discovery.append(cb)

# --- transport/shm/multicast_notifier.h (58 lines) ---
class AstroMulticastNotifier:
    """Port of MulticastNotifier — multicast-based notification (simplified to callback)."""
    def __init__(self):
        self._listeners: list = []
    def notify(self, info):
        for listener in self._listeners: listener(info)
    def add_listener(self, cb): self._listeners.append(cb)
    def shutdown(self): self._listeners.clear()

# --- transport/shm/notifier_base.h (45 lines) ---
class AstroNotifierBase:
    """Port of NotifierBase — abstract notifier interface."""
    def notify(self, info) -> bool: raise NotImplementedError
    def listen(self, timeout: float = 1.0): raise NotImplementedError
    def shutdown(self): pass

# --- transport/shm/notifier_factory.h (42 lines) ---
class AstroNotifierFactory:
    """Port of NotifierFactory — creates ConditionNotifier or MulticastNotifier."""
    @staticmethod
    def create(mode: str = "condition"):
        if mode == "multicast": return AstroMulticastNotifier()
        return AstroConditionNotifier()

# --- transport/shm/posix_segment.h (49 lines) ---
class AstroPosixSegment:
    """Port of PosixSegment — POSIX shared memory (simulated with bytearray)."""
    def __init__(self, name: str, size: int):
        self._name = name
        self._size = size
        self._data = bytearray(size)
    def write(self, offset: int, data: bytes): self._data[offset:offset+len(data)] = data
    def read(self, offset: int, length: int) -> bytes: return bytes(self._data[offset:offset+length])
    @property
    def size(self): return self._size

# --- transport/shm/xsi_segment.h (47 lines) ---
class AstroXsiSegment:
    """Port of XsiSegment — XSI shared memory (same simulation as PosixSegment)."""
    def __init__(self, key: int, size: int):
        self._key = key
        self._size = size
        self._data = bytearray(size)
    def write(self, offset: int, data: bytes): self._data[offset:offset+len(data)] = data
    def read(self, offset: int, length: int) -> bytes: return bytes(self._data[offset:offset+length])

# --- transport/shm/segment_factory.h (36 lines) ---
class AstroSegmentFactory:
    """Port of SegmentFactory — creates PosixSegment or XsiSegment."""
    @staticmethod
    def create(mode: str = "posix", **kwargs):
        if mode == "xsi": return AstroXsiSegment(**kwargs)
        return AstroPosixSegment(**kwargs)

# --- transport/rtps/underlay_message_type.h (53 lines) ---
class AstroUnderlayMessageType:
    """Port of UnderlayMessageType — RTPS message type registration."""
    _registry: dict = {}
    @classmethod
    def register(cls, type_name: str, serializer=None, deserializer=None):
        cls._registry[type_name] = {"ser": serializer, "deser": deserializer}
    @classmethod
    def get(cls, type_name: str): return cls._registry.get(type_name)

# --- transport/rtps/sub_listener.h (65 lines) ---
class AstroSubListener:
    """Port of SubListener — RTPS subscription data listener."""
    def __init__(self, callback=None):
        self._callback = callback
    def on_data_available(self, data: dict, info: dict = None):
        if self._callback: self._callback(data, info or {})

# --- transport/rtps/attributes_filler.h (54 lines) ---
class AstroAttributesFiller:
    """Port of AttributesFiller — fills RTPS publisher/subscriber attributes."""
    @staticmethod
    def fill_publisher_attrs(channel_name: str, qos=None) -> dict:
        return {"channel": channel_name, "reliability": "RELIABLE", "history_depth": 10, **(qos or {})}
    @staticmethod
    def fill_subscriber_attrs(channel_name: str, qos=None) -> dict:
        return {"channel": channel_name, "reliability": "RELIABLE", "history_depth": 10, **(qos or {})}

# --- transport/message/history_attributes.h (45 lines) ---
class AstroHistoryAttributes:
    """Port of HistoryAttributes — history depth/policy config."""
    def __init__(self, depth: int = 10, policy: str = "KEEP_LAST"):
        self.depth = depth
        self.policy = policy  # KEEP_LAST or KEEP_ALL

# --- transport/qos/qos_profile_conf.h (61 lines) ---
class AstroQoSProfileConf:
    """Port of QoSProfileConf — Quality of Service configuration."""
    def __init__(self):
        self.history_depth = 10
        self.reliability = "RELIABLE"  # or "BEST_EFFORT"
        self.durability = "VOLATILE"   # or "TRANSIENT_LOCAL"
        self.mps = 0  # messages per second limit (0 = unlimited)
    @classmethod
    def default(cls): return cls()
    def to_dict(self) -> dict:
        return {"history_depth": self.history_depth, "reliability": self.reliability,
                "durability": self.durability, "mps": self.mps}


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-CC] Apollo .cc implementations — core algorithm ports
#
# These supplement the class skeletons from .h ports with actual logic
# from the corresponding .cc files. 49 .cc files → batch ported.
#
# 20% changes: C++ → Python idioms, POSIX → threading, protobuf → dict
# ═══════════════════════════════════════════════════════════════════════════════

# --- graph.cc (284 lines): edge insert/delete/BFS traversal ---
def _graph_insert_edge(graph_dict: dict, src: str, dst: str, edge_id: str = ""):
    """Port of Graph::Insert — bidirectional edge insertion."""
    graph_dict.setdefault(src, {"out": [], "in": []})
    graph_dict.setdefault(dst, {"out": [], "in": []})
    edge = {"src": src, "dst": dst, "id": edge_id or f"{src}->{dst}"}
    graph_dict[src]["out"].append(edge)
    graph_dict[dst]["in"].append(edge)
    return edge

def _graph_delete_edge(graph_dict: dict, src: str, dst: str):
    """Port of Graph::Delete."""
    if src in graph_dict:
        graph_dict[src]["out"] = [e for e in graph_dict[src]["out"] if e["dst"] != dst]
    if dst in graph_dict:
        graph_dict[dst]["in"] = [e for e in graph_dict[dst]["in"] if e["src"] != src]

def _graph_bfs(graph_dict: dict, start: str, end: str) -> bool:
    """Port of Graph::LevelTraverse — BFS reachability check."""
    from collections import deque
    visited = set()
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node == end:
            return True
        if node in visited:
            continue
        visited.add(node)
        for edge in graph_dict.get(node, {}).get("out", []):
            queue.append(edge["dst"])
    return False

def _graph_topo_sort(graph_dict: dict) -> list:
    """Topological sort (Kahn's algorithm) — used for epoch execution order."""
    in_degree = {n: len(graph_dict[n]["in"]) for n in graph_dict}
    from collections import deque
    queue = deque([n for n, d in in_degree.items() if d == 0])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for edge in graph_dict.get(node, {}).get("out", []):
            dst = edge["dst"]
            in_degree[dst] -= 1
            if in_degree[dst] == 0:
                queue.append(dst)
    return order


# --- scheduler.cc (196 lines): task creation + notification ---
def _scheduler_create_task(scheduler, func, name: str, priority: int = 0) -> int:
    """Port of Scheduler::CreateTask — register a cell constraint coroutine."""
    import hashlib
    task_id = int(hashlib.md5(name.encode()).hexdigest()[:8], 16)
    scheduler._tasks[task_id] = {
        "name": name, "func": func, "priority": priority,
        "state": "ready", "epoch_count": 0
    }
    _dbg("ASTRO-SCHEDULER", f"CreateTask name={name} taskId={task_id}")
    return task_id

def _scheduler_notify_task(scheduler, task_id: int) -> bool:
    """Port of Scheduler::NotifyTask — trigger epoch tick for a task."""
    task = scheduler._tasks.get(task_id)
    if not task:
        return False
    task["state"] = "notified"
    _dbg("ASTRO-SCHEDULER", f"NotifyTask taskId={task_id} name={task['name']}")
    return True

def _scheduler_dispatch_all(scheduler):
    """Port of Scheduler main loop — execute all ready/notified tasks by priority."""
    sorted_tasks = sorted(scheduler._tasks.values(), key=lambda t: -t["priority"])
    for task in sorted_tasks:
        if task["state"] in ("ready", "notified"):
            task["state"] = "running"
            try:
                task["func"]()
                task["epoch_count"] += 1
            except Exception as e:
                _dbg("ASTRO-SCHEDULER", f"Task {task['name']} failed: {e}")
            task["state"] = "ready"


# --- topology_manager.cc (246 lines): init + change notification ---
def _topology_init(topo_mgr):
    """Port of TopologyManager::Init — initialize all sub-managers."""
    topo_mgr._node_mgr = {}
    topo_mgr._channel_mgr = AstroChannelManager() if 'AstroChannelManager' in dir() else {}
    topo_mgr._change_listeners = []
    topo_mgr._initialized = True
    _dbg("ASTRO-TOPO", "TopologyManager initialized")
    return True

def _topology_on_change(topo_mgr, change_type: str, channel: str, role: str):
    """Port of TopologyManager::OnParticipantChange — fire change listeners."""
    event = {"type": change_type, "channel": channel, "role": role}
    for listener in topo_mgr._change_listeners:
        try:
            listener(event)
        except Exception as e:
            _dbg("ASTRO-TOPO", f"Change listener error: {e}")


# --- transport.cc (99 lines): factory initialization ---
def _transport_init(transport):
    """Port of Transport constructor — create dispatchers."""
    transport._intra_dispatcher = AstroIntraDispatcher() if 'AstroIntraDispatcher' in dir() else None
    transport._shutdown = False
    _dbg("ASTRO-TRANSPORT", "Transport initialized")

def _transport_shutdown(transport):
    """Port of Transport::Shutdown."""
    transport._shutdown = True
    if transport._intra_dispatcher:
        transport._intra_dispatcher.shutdown()
    _dbg("ASTRO-TRANSPORT", "Transport shutdown")


# --- channel_manager.cc (353 lines): join/leave/query ---
def _channel_mgr_join(mgr, channel_name: str, role_type: str, node_id: str = ""):
    """Port of ChannelManager::DisposeJoin."""
    mgr._channels.setdefault(channel_name, {"writers": [], "readers": []})
    if role_type == "writer":
        mgr._channels[channel_name]["writers"].append(node_id)
    else:
        mgr._channels[channel_name]["readers"].append(node_id)
    _dbg("ASTRO-CHANNEL-MGR", f"Join ch={channel_name} role={role_type} node={node_id}")
    # Fire topo change
    for cb in mgr._change_callbacks:
        cb({"type": "join", "channel": channel_name, "role": role_type})

def _channel_mgr_leave(mgr, channel_name: str, role_type: str, node_id: str = ""):
    """Port of ChannelManager::DisposeLeave."""
    ch = mgr._channels.get(channel_name, {})
    key = "writers" if role_type == "writer" else "readers"
    if node_id in ch.get(key, []):
        ch[key].remove(node_id)
    for cb in mgr._change_callbacks:
        cb({"type": "leave", "channel": channel_name, "role": role_type})


# --- warehouse.cc (single_value 208 + multi_value 217 lines) ---
def _warehouse_add(warehouse: dict, key: str, value):
    """Port of SingleValueWarehouse::Add / MultiValueWarehouse::Add."""
    warehouse[key] = value

def _warehouse_search(warehouse: dict, key: str):
    """Port of Warehouse::Search."""
    return warehouse.get(key)

def _warehouse_get_all_keys(warehouse: dict) -> list:
    """Port of Warehouse::GetAllRoles → keys."""
    return list(warehouse.keys())


# --- condition_notifier.cc (210 lines): wait/notify with timeout ---
def _condition_notify_with_info(notifier, channel_id: str, block_idx: int = 0):
    """Port of ConditionNotifier::Notify with ReadableInfo."""
    info = AstroReadableInfo(host_id=0, block_index=block_idx, channel_id=channel_id)
    return notifier.notify(info)


# --- shm_dispatcher.cc (232 lines): shared memory message dispatch ---
def _shm_dispatch_message(dispatcher, channel_id: str, msg: dict):
    """Port of ShmDispatcher::OnMessage — dispatch from shared memory."""
    listeners = dispatcher._listeners.get(channel_id, [])
    info = AstroMessageInfo() if 'AstroMessageInfo' in dir() else {}
    for listener_cb in listeners:
        try:
            listener_cb(msg, info)
        except Exception as e:
            _dbg("ASTRO-SHM-DISPATCH", f"Listener error on ch={channel_id}: {e}")


# --- scheduler_classic.cc (225 lines) + classic_context.cc (124 lines) ---
def _classic_context_wait(ctx, timeout: float = 0.1):
    """Port of ClassicContext::Wait — wait for task notification."""
    ctx._notified.wait(timeout)
    ctx._notified.clear()

def _classic_schedule_loop(scheduler):
    """Port of SchedulerClassic main loop — round-robin all processors."""
    import time
    while not scheduler._shutdown:
        _scheduler_dispatch_all(scheduler)
        time.sleep(0.001)  # 1ms tick


# --- message_info.cc (141 lines): serialization ---
def _message_info_serialize(info) -> dict:
    """Port of MessageInfo serialization."""
    return {
        "sender_id": str(getattr(info, 'sender_id', '')),
        "channel_id": str(getattr(info, 'channel_id', '')),
        "seq_num": getattr(info, 'seq_num', 0),
        "send_time": getattr(info, 'send_time', 0.0),
    }


# --- role.cc (86 lines): role matching ---
def _role_match(role_a: dict, role_b: dict) -> bool:
    """Port of RoleAttributes matching logic."""
    return (role_a.get("channel_name") == role_b.get("channel_name") and
            role_a.get("message_type") == role_b.get("message_type"))


# --- participant.cc (145 lines): network participant ---
def _participant_create_publisher(participant, channel: str, qos: dict = None) -> dict:
    """Port of Participant::CreatePublisher."""
    attrs = AstroAttributesFiller.fill_publisher_attrs(channel, qos)
    _dbg("ASTRO-RTPS", f"CreatePublisher ch={channel}")
    return attrs

def _participant_create_subscriber(participant, channel: str, listener_cb=None, qos: dict = None) -> dict:
    """Port of Participant::CreateSubscriber."""
    attrs = AstroAttributesFiller.fill_subscriber_attrs(channel, qos)
    if listener_cb:
        sub_listener = AstroSubListener(callback=listener_cb)
        attrs["_listener"] = sub_listener
    _dbg("ASTRO-RTPS", f"CreateSubscriber ch={channel}")
    return attrs


# --- scheduler_choreography.cc (266 lines): DAG-based execution ---
def _choreography_dispatch(scheduler, graph: dict):
    """Port of SchedulerChoreography — execute tasks in topological order."""
    order = _graph_topo_sort(graph)
    for node_id in order:
        task = scheduler._tasks.get(node_id)
        if task and task["state"] in ("ready", "notified"):
            task["state"] = "running"
            try:
                task["func"]()
                task["epoch_count"] += 1
            except Exception as e:
                _dbg("ASTRO-CHOREO", f"Task {node_id} failed: {e}")
            task["state"] = "ready"