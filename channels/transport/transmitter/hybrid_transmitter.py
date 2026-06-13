# -*- coding: utf-8 -*-
"""Extracted from channel_runtime.py"""

from channels.channel_runtime import AstroEndpoint
import threading
import time
from typing import Any, Dict, Optional


class AstroHybridTransmitter(AstroEndpoint):
    """
    Mirrors ``apollo::cyber::transport::HybridTransmitter<M>``.

    Selects INTRA / SHM / RTPS based on the spatial relationship between
    sender and each registered receiver.  Also maintains a History for
    TRANSIENT_LOCAL durability.
    """

    def __init__(
        self,
        attr:         AstroRoleAttributes,
        serialize_fn: Optional[_SerializeFn]   = None,
        same_proc_mode: str = MODE_SHM,
        diff_proc_mode: str = MODE_SHM,
        diff_host_mode: str = MODE_RTPS,
    ) -> None:
        super().__init__(attr)
        self._serialize_fn = serialize_fn

        # mapping_table_[relation] → mode
        self._mapping: Dict[str, str] = {
            SAME_PROC: same_proc_mode,
            DIFF_PROC: diff_proc_mode,
            DIFF_HOST: diff_host_mode,
        }

        # sub-transmitters keyed by mode
        self._transmitters: Dict[str, AstroShmTransmitter] = {}
        self._init_transmitters()

        # receivers_[mode] = set of receiver IDs
        self._receivers: Dict[str, Set[int]] = {
            m: set() for m in set(self._mapping.values())
        }

        # history for TRANSIENT_LOCAL
        depth = attr.qos_depth if attr.qos_depth > 0 else 1
        self._history: AstroHistory = AstroHistory(
            history_policy = attr.qos_history,
            depth          = depth,
        )
        if attr.qos_durability == AstroRoleAttributes.DURABILITY_TRANSIENT_LOCAL:
            self._history.enable()

        self._mutex: threading.Lock = threading.Lock()
        self._seq:   int            = 0

    # ── init helpers ─────────────────────────────────────────────────────────

    def _init_transmitters(self) -> None:
        modes = set(self._mapping.values())
        for mode in modes:
            # For this Python port we use ShmTransmitter for all modes
            # (RTPS would need network; use SHM as stand-in for DIFF_HOST too)
            self._transmitters[mode] = AstroShmTransmitter(
                self.attr_, self._serialize_fn
            )

    # ── enable / disable ──────────────────────────────────────────────────────

    def enable(self, opposite_attr: Optional[AstroRoleAttributes] = None) -> None:
        if opposite_attr is None:
            with self._mutex:
                for tx in self._transmitters.values():
                    tx.enable()
            self.enabled_ = True
            return
        relation = self._get_relation(opposite_attr)
        if relation == NO_RELATION:
            return
        mode = self._mapping[relation]
        with self._mutex:
            self._receivers[mode].add(opposite_attr.id)
            self._transmitters[mode].enable(opposite_attr)
            self._transmit_history(opposite_attr, mode)
        self.enabled_ = True

    def disable(self, opposite_attr: Optional[AstroRoleAttributes] = None) -> None:
        if opposite_attr is None:
            with self._mutex:
                for tx in self._transmitters.values():
                    tx.disable()
            self.enabled_ = False
            return
        relation = self._get_relation(opposite_attr)
        if relation == NO_RELATION:
            return
        mode = self._mapping[relation]
        with self._mutex:
            self._receivers[mode].discard(opposite_attr.id)
            self._transmitters[mode].disable(opposite_attr)

    # ── transmit ──────────────────────────────────────────────────────────────

    def transmit(
        self,
        msg:      Any,
        msg_info: Optional[AstroMessageInfo] = None,
    ) -> bool:
        """
        ``bool Transmit(const MessagePtr&, const MessageInfo&)``

        Adds to history, then fans out to all sub-transmitters.
        """
        with self._mutex:
            if msg_info is None:
                self._seq += 1
                msg_info = AstroMessageInfo(
                    sender_id   = self.id_,
                    seq_num     = self._seq,
                    channel_id  = self.attr_.channel_id,
                    msg_seq_num = self._seq,
                    send_time   = int(time.monotonic_ns() // 1_000),
                )
            self._history.add(msg, msg_info)
            for tx in self._transmitters.values():
                tx.transmit(msg, msg_info)
        return True

    # ── history replay ────────────────────────────────────────────────────────

    def _transmit_history(
        self, opposite_attr: AstroRoleAttributes, mode: str
    ) -> None:
        if self.attr_.qos_durability != AstroRoleAttributes.DURABILITY_TRANSIENT_LOCAL:
            return
        cached = self._history.get_cached_message()
        if not cached:
            return
        tx = self._transmitters[mode]

        def _replay() -> None:
            for item in cached:
                tx.transmit(item.msg, item.msg_info)
                time.sleep(0.001)

        t = threading.Thread(target=_replay, daemon=True, name="HybridTx-replay")
        t.start()

    # ── relation helper ───────────────────────────────────────────────────────

    def _get_relation(self, opposite_attr: AstroRoleAttributes) -> str:
        if opposite_attr.channel_name != self.attr_.channel_name:
            return NO_RELATION
        if opposite_attr.host_ip != self.attr_.host_ip:
            return DIFF_HOST
        if opposite_attr.process_id != self.attr_.process_id:
            return DIFF_PROC
        return SAME_PROC


# ══════════════════════════════════════════════════════════════════════════════
# AstroShmReceiver
# Port of: upstream/apollo-cyber/transport/receiver/shm_receiver.h
#
# 鲁迅曰：接收者是沉默的，只管等待那一声通知——仿佛旧时深宅里等信的人，
# 门缝里塞进来什么，便接什么。
# ══════════════════════════════════════════════════════════════════════════════

