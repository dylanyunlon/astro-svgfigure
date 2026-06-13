# -*- coding: utf-8 -*-
"""Extracted from channel_runtime.py"""

from channels.channel_runtime import _dbg

from channels.channel_runtime import AstroEndpoint
from channels.transport.transmitter.transmitter import AstroTransmitterBase
import threading
import time
import json
import urllib.request
from typing import Any, Dict, Optional


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

