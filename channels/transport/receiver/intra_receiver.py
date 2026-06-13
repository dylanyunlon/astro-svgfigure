import os, sys, json, threading
from typing import Any, Optional, Generic, TypeVar

_MT = TypeVar('_MT')

def _dbg(tag, msg):
    if os.environ.get(f'ASTRO_{tag.replace("-","_")}_VERBOSE', '0') == '1':
        print(f'[{tag}] {msg}', file=sys.stderr)

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



