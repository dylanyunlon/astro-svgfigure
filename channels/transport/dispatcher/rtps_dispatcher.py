# -*- coding: utf-8 -*-
"""Extracted from channel_runtime.py"""

from channels.channel_runtime import _tdbg

from channels.transport.dispatcher.intra_dispatcher import _AstroDispatcherBase
import threading
import time
from typing import Any, Callable, Dict, Optional


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

