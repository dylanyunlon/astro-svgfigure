# -*- coding: utf-8 -*-
"""Extracted from channel_runtime.py"""

from channels.channel_runtime import _tdbg, _dbg

import threading
from typing import Any, Callable, Dict, List, Optional, Tuple


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

