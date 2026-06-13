import os, sys, json, threading
from typing import Any, Optional, Generic, TypeVar

_MT = TypeVar('_MT')

def _dbg(tag, msg):
    if os.environ.get(f'ASTRO_{tag.replace("-","_")}_VERBOSE', '0') == '1':
        print(f'[{tag}] {msg}', file=sys.stderr)

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


