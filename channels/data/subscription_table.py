"""
subscription_table.py — Apollo-faithful channel subscription routing.

Core principle (from the owner's critique):
  "中国跟俄罗斯也是邻居, 中国小米的所有事情需要反馈给俄罗斯吗?"

  NO.  Cells do NOT broadcast to neighbors.
  Cells PUBLISH to a channel_id.  Only cells that SUBSCRIBED to that
  channel_id receive the notification.  This is Apollo CyberRT's design:

    Writer → channel_id → DataDispatcher → [only registered CacheBuffers]
                                         → DataNotifier → [only registered callbacks]

  The subscription is declared at TOPOLOGY time (skeleton/topology.json),
  not at runtime by proximity or adjacency.

Port source:
  Apollo cyber/data/data_notifier.h   → notify_map  (channel_id → [callbacks])
  Apollo cyber/data/data_dispatcher.h → buffer_map  (channel_id → [buffers])
  Apollo cyber/service_discovery/specific_manager/channel_manager.h
                                      → writer/reader routing table

This module adds what the current DataVisitor lacks:
  1. Explicit subscription declarations per cell (not hardcoded 4 channels)
  2. Primary/secondary channel distinction for AllLatest fusion
  3. A routing table that the epoch controller can query:
     "when cell_self_attn publishes, who gets notified?"
     Answer: only cells that subscribed to cell_self_attn's output channel.
"""

import json
import os
import threading
from typing import Callable, Dict, List, Optional, Set, Tuple

CHANNELS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERBOSE = os.environ.get("ASTRO_SUB_TABLE_VERBOSE", "0") == "1"


def _dbg(msg: str):
    if VERBOSE:
        print(f"[ASTRO-SUB-TABLE] {msg}")


class SubscriptionEntry:
    """One cell's subscription to one channel."""
    __slots__ = ("cell_id", "channel_id", "role", "is_primary", "callback")

    def __init__(self, cell_id: str, channel_id: str, role: str,
                 is_primary: bool = False,
                 callback: Optional[Callable] = None):
        self.cell_id = cell_id
        self.channel_id = channel_id
        self.role = role  # "WRITER" or "READER"
        self.is_primary = is_primary  # AllLatest: primary triggers fusion
        self.callback = callback


class SubscriptionTable:
    """
    Global subscription routing table.

    Apollo equivalent: the combination of
      ChannelManager (who writes/reads what)
      DataNotifier   (channel_id → callbacks)
      DataDispatcher (channel_id → buffers)

    Usage:
        table = SubscriptionTable.instance()

        # Declare at topology time:
        table.declare_writer("cell_self_attn", "cell/self_attn/out.json")
        table.declare_reader("cell_add_norm1", "cell/self_attn/out.json",
                             is_primary=True,
                             callback=lambda data: add_norm1.on_attn_output(data))

        # At publish time:
        notified = table.publish("cell/self_attn/out.json", {"bbox": ...})
        # Returns ["cell_add_norm1"] — only subscribers, not all neighbors!

        # Query: what channels does a cell subscribe to?
        table.get_subscriptions("cell_add_norm1")
        # → ["cell/self_attn/out.json", "physics/force_field.json", ...]

        # Query: if cell_self_attn publishes, who is affected?
        table.get_readers("cell/self_attn/out.json")
        # → ["cell_add_norm1"] — NOT cell_ffn, NOT cell_output (they don't sub)
    """
    _instance: Optional["SubscriptionTable"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._mu = threading.Lock()
        # channel_id → [SubscriptionEntry] for readers
        self._readers: Dict[str, List[SubscriptionEntry]] = {}
        # channel_id → [SubscriptionEntry] for writers
        self._writers: Dict[str, List[SubscriptionEntry]] = {}
        # cell_id → [channel_id] quick lookup
        self._cell_subs: Dict[str, Set[str]] = {}
        self._cell_pubs: Dict[str, Set[str]] = {}
        _dbg("SubscriptionTable constructed")

    @classmethod
    def instance(cls) -> "SubscriptionTable":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        with cls._lock:
            cls._instance = None

    # ── Declaration API (topology time) ──────────────────────────────────────

    def declare_writer(self, cell_id: str, channel_id: str):
        """Cell announces it will PUBLISH to this channel."""
        with self._mu:
            entry = SubscriptionEntry(cell_id, channel_id, "WRITER")
            self._writers.setdefault(channel_id, []).append(entry)
            self._cell_pubs.setdefault(cell_id, set()).add(channel_id)
            _dbg(f"declare_writer cell={cell_id} ch={channel_id}")

    def declare_reader(self, cell_id: str, channel_id: str,
                       is_primary: bool = False,
                       callback: Optional[Callable] = None):
        """Cell announces it will SUBSCRIBE to this channel."""
        with self._mu:
            entry = SubscriptionEntry(cell_id, channel_id, "READER",
                                      is_primary=is_primary,
                                      callback=callback)
            self._readers.setdefault(channel_id, []).append(entry)
            self._cell_subs.setdefault(cell_id, set()).add(channel_id)
            _dbg(f"declare_reader cell={cell_id} ch={channel_id} primary={is_primary}")

    # ── Publish API (runtime) ────────────────────────────────────────────────

    def publish(self, channel_id: str, data: dict) -> List[str]:
        """
        Publish data to a channel. Returns list of cell_ids that were notified.

        This is the Apollo DataDispatcher.Dispatch + DataNotifier.Notify combined:
          1. Write data to channel file (buffer fill)
          2. Fire callbacks only for SUBSCRIBED readers
          3. Return who was notified (for epoch controller logging)
        """
        notified = []
        with self._mu:
            readers = self._readers.get(channel_id, [])

        # Write to channel file (DataDispatcher buffer fill)
        full_path = os.path.join(CHANNELS_DIR, channel_id)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            json.dump(data, f, indent=2)

        # Fire callbacks (DataNotifier.Notify)
        for entry in readers:
            if entry.callback:
                try:
                    entry.callback(data)
                except Exception as e:
                    _dbg(f"callback error cell={entry.cell_id} ch={channel_id}: {e}")
            notified.append(entry.cell_id)

        _dbg(f"publish ch={channel_id} → notified {len(notified)} readers: {notified}")
        return notified

    # ── Query API ────────────────────────────────────────────────────────────

    def get_readers(self, channel_id: str) -> List[str]:
        """Who subscribes to this channel? (not 'who is adjacent')"""
        with self._mu:
            return [e.cell_id for e in self._readers.get(channel_id, [])]

    def get_writers(self, channel_id: str) -> List[str]:
        """Who publishes to this channel?"""
        with self._mu:
            return [e.cell_id for e in self._writers.get(channel_id, [])]

    def get_subscriptions(self, cell_id: str) -> Set[str]:
        """What channels does this cell subscribe to?"""
        with self._mu:
            return set(self._cell_subs.get(cell_id, set()))

    def get_publications(self, cell_id: str) -> Set[str]:
        """What channels does this cell publish to?"""
        with self._mu:
            return set(self._cell_pubs.get(cell_id, set()))

    def get_primary_channel(self, cell_id: str) -> Optional[str]:
        """Which channel triggers this cell's AllLatest fusion?"""
        with self._mu:
            for ch_id, entries in self._readers.items():
                for e in entries:
                    if e.cell_id == cell_id and e.is_primary:
                        return ch_id
        return None

    # ── Topology bootstrap ───────────────────────────────────────────────────

    def load_from_topology(self, topology_path: Optional[str] = None):
        """
        Bootstrap subscriptions from skeleton/topology.json.

        The topology JSON has edges: [{source, target, id}]
        Each edge creates:
          - source cell → WRITER on "cell/{source}/out.json"
          - target cell → READER on "cell/{source}/out.json" (primary)

        Additionally, ALL cells subscribe to shared physics channels:
          - physics/force_field.json (READER, secondary)
          - physics/species_assignment.json (READER, secondary)
          - physics/z_layers.json (READER, secondary)
        But they only PUBLISH to their own output channel.
        """
        if topology_path is None:
            topology_path = os.path.join(CHANNELS_DIR, "skeleton", "topology.json")

        if not os.path.exists(topology_path):
            _dbg(f"No topology at {topology_path}, skipping bootstrap")
            return

        with open(topology_path) as f:
            topo = json.load(f)

        edges = topo.get("edges", [])
        nodes = set()
        for edge in edges:
            # ELK format: sources/targets are arrays
            src_list = edge.get("sources", [edge.get("source", "")])
            tgt_list = edge.get("targets", [edge.get("target", "")])
            if isinstance(src_list, str):
                src_list = [src_list]
            if isinstance(tgt_list, str):
                tgt_list = [tgt_list]
            for src in src_list:
                for tgt in tgt_list:
                    if src and tgt:
                        nodes.add(src)
                        nodes.add(tgt)
                        ch = f"cell/{src}/out.json"
                        self.declare_writer(src, ch)
                        self.declare_reader(tgt, ch, is_primary=True)

        # Also collect nodes from children array if present
        for child in topo.get("children", []):
            cid = child.get("id", "")
            if cid:
                nodes.add(cid)

        # Shared physics channels — every cell reads, nobody broadcasts to "neighbors"
        shared_channels = [
            "physics/force_field.json",
            "physics/species_assignment.json",
            "physics/z_layers.json",
        ]
        for node in nodes:
            for ch in shared_channels:
                self.declare_reader(node, ch, is_primary=False)
            # Each cell writes its own skeleton signal
            self.declare_writer(node, f"skeleton/cell/{node}.json")

        _dbg(f"Loaded topology: {len(nodes)} cells, {len(edges)} edges")

    # ── Debug / Introspection ────────────────────────────────────────────────

    def dump_routing_table(self) -> dict:
        """Full routing table dump for debugging."""
        with self._mu:
            return {
                "channels": {
                    ch: {
                        "writers": [e.cell_id for e in self._writers.get(ch, [])],
                        "readers": [e.cell_id for e in self._readers.get(ch, [])],
                    }
                    for ch in set(list(self._readers.keys()) + list(self._writers.keys()))
                },
                "cells": {
                    cell_id: {
                        "subscribes_to": sorted(self._cell_subs.get(cell_id, set())),
                        "publishes_to": sorted(self._cell_pubs.get(cell_id, set())),
                    }
                    for cell_id in set(list(self._cell_subs.keys()) + list(self._cell_pubs.keys()))
                }
            }
