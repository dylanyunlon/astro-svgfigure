# -*- coding: utf-8 -*-
"""Extracted from channel_runtime.py"""

from channels.channel_runtime import _tdbg

from channels.transport.dispatcher.intra_dispatcher import _AstroDispatcherBase
import threading
import queue
from typing import Dict, List, Optional, Tuple


class _AstroShmSegment:
    """
    Minimal shared-memory segment simulation.

    In the real Apollo implementation this is backed by POSIX shm_open +
    mmap.  Here we use an in-process bytearray protected by a lock so that
    the ShmDispatcher / ShmTransmitter / ShmReceiver round-trip can be
    exercised without OS-level shared memory.
    """

    def __init__(self, channel_id: int, num_blocks: int = 16) -> None:
        self.channel_id:  int              = channel_id
        self.num_blocks:  int              = num_blocks
        self._data:       List[bytearray]  = [
            bytearray(_SHM_BLOCK_SIZE) for _ in range(num_blocks)
        ]
        self._msg_sizes:  List[int]        = [0] * num_blocks
        self._write_lock: threading.Lock   = threading.Lock()
        self._read_cnts:  List[int]        = [0] * num_blocks
        self._write_idx:  int              = 0

    def acquire_block_to_write(self, msg_size: int) -> Optional[int]:
        with self._write_lock:
            idx = self._write_idx % self.num_blocks
            self._write_idx += 1
            return idx

    def release_written_block(self, idx: int, payload: bytes, msg_info_bytes: bytes) -> None:
        total = len(payload) + len(msg_info_bytes)
        if total > _SHM_BLOCK_SIZE:
            _tdbg("SHM_SEG", f"payload too large ({total} > {_SHM_BLOCK_SIZE}), truncating")
            total = _SHM_BLOCK_SIZE
        self._data[idx][:len(payload)] = payload[:_SHM_BLOCK_SIZE]
        end = min(len(payload) + len(msg_info_bytes), _SHM_BLOCK_SIZE)
        self._data[idx][len(payload):end] = msg_info_bytes[:end - len(payload)]
        self._msg_sizes[idx] = len(payload)

    def read_block(self, idx: int) -> Tuple[bytes, bytes]:
        msg_size = self._msg_sizes[idx]
        payload  = bytes(self._data[idx][:msg_size])
        info_raw = bytes(self._data[idx][msg_size:msg_size + AstroMessageInfo.kSize])
        return payload, info_raw


class AstroShmDispatcher(_AstroDispatcherBase):
    """
    Mirrors ``apollo::cyber::transport::ShmDispatcher`` (singleton).

    Maintains a map of channel_id → _AstroShmSegment and a polling thread
    that mimics the notifier wake-up path in the C++ implementation.
    """

    _instance: Optional["AstroShmDispatcher"] = None
    _inst_lock: threading.Lock = threading.Lock()

    @classmethod
    def instance(cls) -> "AstroShmDispatcher":
        with cls._inst_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        super().__init__()
        self._segments:    Dict[int, _AstroShmSegment] = {}
        self._seg_lock:    threading.RLock              = threading.RLock()
        self._notify_q:    queue.Queue                  = queue.Queue()
        self._thread:      threading.Thread             = threading.Thread(
            target=self._thread_func, daemon=True, name="AstroShmDispatcher"
        )
        self._thread.start()
        _tdbg("SHM_DISP", "AstroShmDispatcher started")

    # ── segment management ────────────────────────────────────────────────────

    def add_segment(self, attr: AstroRoleAttributes) -> None:
        ch = attr.channel_id
        with self._seg_lock:
            if ch not in self._segments:
                self._segments[ch] = _AstroShmSegment(ch)
                _tdbg("SHM_DISP", f"new segment for channel {ch}")

    def get_segment(self, channel_id: int) -> Optional[_AstroShmSegment]:
        with self._seg_lock:
            return self._segments.get(channel_id)

    # ── notifier ─────────────────────────────────────────────────────────────

    def notify(self, channel_id: int, block_index: int) -> None:
        """Called by ShmTransmitter after writing a block."""
        self._notify_q.put((channel_id, block_index))

    def _thread_func(self) -> None:
        """
        Poll loop — mirrors ``ShmDispatcher::ThreadFunc()``.

        Drains the notification queue and dispatches each readable block.
        """
        while not self._is_shutdown:
            try:
                ch_id, blk_idx = self._notify_q.get(timeout=0.05)
            except queue.Empty:
                continue
            self._read_message(ch_id, blk_idx)

    def _read_message(self, channel_id: int, block_index: int) -> None:
        seg = self.get_segment(channel_id)
        if seg is None:
            return
        payload, info_raw = seg.read_block(block_index)
        if not info_raw or len(info_raw) < AstroMessageInfo.kSize:
            return
        try:
            msg_info = AstroMessageInfo.deserialize_from(info_raw)
        except Exception:
            return
        _tdbg("SHM_DISP",
              f"dispatch ch={channel_id} blk={block_index} "
              f"seq={msg_info.seq_num}")
        self._dispatch(channel_id, payload, msg_info)

    # ── override shutdown ─────────────────────────────────────────────────────

    def shutdown(self) -> None:
        super().shutdown()
        self._thread.join(timeout=1.0)
        _tdbg("SHM_DISP", "AstroShmDispatcher shutdown")


# ══════════════════════════════════════════════════════════════════════════════
# AstroRtpsDispatcher
# Port of: upstream/apollo-cyber/transport/dispatcher/rtps_dispatcher.h
#
# 鲁迅曰：RTPS 这条路走得远，跨进程，跨主机；然而消息序列化成字符串之后，
# 凡人皆可偷看——这便是"开放"的代价。
# ══════════════════════════════════════════════════════════════════════════════

