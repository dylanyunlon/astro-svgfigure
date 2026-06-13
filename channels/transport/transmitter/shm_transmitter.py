# -*- coding: utf-8 -*-
"""Extracted from channel_runtime.py"""

from channels.channel_runtime import _tdbg

from channels.channel_runtime import AstroEndpoint
import threading
import time
from typing import Any, Optional


class AstroShmTransmitter(AstroEndpoint):
    """
    Mirrors ``apollo::cyber::transport::ShmTransmitter<M>``.

    *serialize_fn* converts a message object to bytes; if None, the message
    is expected to already be bytes (raw mode).
    """

    def __init__(
        self,
        attr:         AstroRoleAttributes,
        serialize_fn: Optional[_SerializeFn] = None,
    ) -> None:
        super().__init__(attr)
        self._dispatcher:    AstroShmDispatcher   = AstroShmDispatcher.instance()
        self._serialize_fn:  Optional[_SerializeFn] = serialize_fn
        self._seq_num:       int                    = 0
        self._seq_lock:      threading.Lock         = threading.Lock()

    # ── enable / disable ──────────────────────────────────────────────────────

    def enable(self, opposite_attr: Optional[AstroRoleAttributes] = None) -> None:
        if self.enabled_:
            return
        self._dispatcher.add_segment(self.attr_)
        self.enabled_ = True
        _tdbg("SHM_TX", f"enabled ch={self.attr_.channel_id}")

    def disable(self, opposite_attr: Optional[AstroRoleAttributes] = None) -> None:
        self.enabled_ = False
        _tdbg("SHM_TX", f"disabled ch={self.attr_.channel_id}")

    # ── transmit ──────────────────────────────────────────────────────────────

    def transmit(
        self,
        msg:      Any,
        msg_info: Optional[AstroMessageInfo] = None,
    ) -> bool:
        """
        ``bool Transmit(const MessagePtr&, const MessageInfo&)``

        Serialises *msg*, writes it into the SHM segment, then calls
        ``AstroShmDispatcher.notify()`` with the block index.
        """
        if not self.enabled_:
            _tdbg("SHM_TX", "not enabled — drop")
            return False

        payload: bytes = (
            self._serialize_fn(msg)
            if self._serialize_fn is not None
            else (msg if isinstance(msg, (bytes, bytearray)) else str(msg).encode())
        )

        seg = self._dispatcher.get_segment(self.attr_.channel_id)
        if seg is None:
            return False

        blk_idx = seg.acquire_block_to_write(len(payload))
        if blk_idx is None:
            return False

        if msg_info is None:
            with self._seq_lock:
                self._seq_num += 1
                seq = self._seq_num
            msg_info = AstroMessageInfo(
                sender_id   = self.id_,
                seq_num     = seq,
                channel_id  = self.attr_.channel_id,
                msg_seq_num = seq,
                send_time   = int(time.monotonic_ns() // 1_000),
            )

        info_bytes = msg_info.serialize_to()
        seg.release_written_block(blk_idx, payload, info_bytes)
        self._dispatcher.notify(self.attr_.channel_id, blk_idx)
        _tdbg("SHM_TX",
              f"transmit ch={self.attr_.channel_id} blk={blk_idx} "
              f"seq={msg_info.seq_num}")
        return True


# ══════════════════════════════════════════════════════════════════════════════
# AstroHybridTransmitter
# Port of: upstream/apollo-cyber/transport/transmitter/hybrid_transmitter.h
#
# 鲁迅曰：Hybrid 是骑墙的艺术——进程内用 SHM，跨进程用 RTPS，两不相欠，
# 却又暗中都要经历 History 这道关卡。
# ══════════════════════════════════════════════════════════════════════════════

# Relation constants (mirrors cyber/common/types.h)
SAME_PROC   = "SAME_PROC"
DIFF_PROC   = "DIFF_PROC"
DIFF_HOST   = "DIFF_HOST"
NO_RELATION = "NO_RELATION"

# OptionalMode constants
MODE_INTRA = "INTRA"
MODE_SHM   = "SHM"
MODE_RTPS  = "RTPS"

