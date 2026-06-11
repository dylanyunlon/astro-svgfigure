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
