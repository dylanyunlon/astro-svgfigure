# -*- coding: utf-8 -*-
"""Extracted from channel_runtime.py"""

from channels.channel_runtime import _dbg

import urllib.request

import time
from typing import Any, Dict


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

