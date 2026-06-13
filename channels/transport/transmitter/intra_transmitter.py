# -*- coding: utf-8 -*-
"""Extracted from channel_runtime.py"""

from channels.channel_runtime import _dbg

from channels.channel_runtime import AstroEndpoint
from channels.transport.transmitter.transmitter import AstroTransmitterBase
import struct
import threading
import time
from typing import Any, Dict, Optional


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

