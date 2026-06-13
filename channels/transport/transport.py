import os, sys, json, threading
from typing import Any, Optional, Generic, TypeVar

_MT = TypeVar('_MT')

def _dbg(tag, msg):
    if os.environ.get(f'ASTRO_{tag.replace("-","_")}_VERBOSE', '0') == '1':
        print(f'[{tag}] {msg}', file=sys.stderr)



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






# ═══════════════════════════════════════════════════════════════════════════════

import dataclasses
import logging
import time as _time

_chanmgr_log = logging.getLogger("astro.channel_manager")


@dataclasses.dataclass

