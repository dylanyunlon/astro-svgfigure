import os, sys, json, threading
from typing import Any, Optional, Generic, TypeVar

_MT = TypeVar('_MT')

def _dbg(tag, msg):
    if os.environ.get(f'ASTRO_{tag.replace("-","_")}_VERBOSE', '0') == '1':
        print(f'[{tag}] {msg}', file=sys.stderr)

# ══════════════════════════════════════════════════════════════════════════════

class AstroHybridReceiver(AstroEndpoint):
    """
    Mirrors ``apollo::cyber::transport::HybridReceiver<M>``.

    Wraps one AstroShmReceiver per communication mode and routes
    Enable(opposite) calls to the correct sub-receiver based on the spatial
    relationship between self and the opposite role.
    """

    def __init__(
        self,
        attr:           AstroRoleAttributes,
        msg_listener:   _ListenerFn,
        deserialize_fn: Optional[_DeserializeFn] = None,
        same_proc_mode: str = MODE_SHM,
        diff_proc_mode: str = MODE_SHM,
        diff_host_mode: str = MODE_RTPS,
    ) -> None:
        super().__init__(attr)
        self._msg_listener   = msg_listener
        self._deserialize_fn = deserialize_fn

        self._mapping: Dict[str, str] = {
            SAME_PROC: same_proc_mode,
            DIFF_PROC: diff_proc_mode,
            DIFF_HOST: diff_host_mode,
        }

        # one sub-receiver per distinct mode
        modes = set(self._mapping.values())
        self._receivers: Dict[str, AstroShmReceiver] = {
            mode: AstroShmReceiver(attr, msg_listener, deserialize_fn)
            for mode in modes
        }

        # transmitter tracking: mode → {id → RoleAttributes}
        self._transmitters: Dict[str, Dict[int, AstroRoleAttributes]] = {
            mode: {} for mode in modes
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

    # ── enable / disable ──────────────────────────────────────────────────────

    def enable(self, opposite_attr: Optional[AstroRoleAttributes] = None) -> None:
        if opposite_attr is None:
            with self._mutex:
                for rx in self._receivers.values():
                    rx.enable()
            self.enabled_ = True
            return

        relation = self._get_relation(opposite_attr)
        if relation == NO_RELATION:
            return
        mode = self._mapping[relation]
        oid  = opposite_attr.id

        with self._mutex:
            if oid not in self._transmitters[mode]:
                self._transmitters[mode][oid] = opposite_attr
                self._receivers[mode].enable(opposite_attr)
                self._receive_history(opposite_attr, mode)
        self.enabled_ = True
        _tdbg("HYBRID_RX", f"enable opposite={oid} relation={relation}")

    def disable(self, opposite_attr: Optional[AstroRoleAttributes] = None) -> None:
        if opposite_attr is None:
            with self._mutex:
                for rx in self._receivers.values():
                    rx.disable()
            self.enabled_ = False
            return

        relation = self._get_relation(opposite_attr)
        if relation == NO_RELATION:
            return
        mode = self._mapping[relation]
        oid  = opposite_attr.id

        with self._mutex:
            if oid in self._transmitters[mode]:
                del self._transmitters[mode][oid]
                self._receivers[mode].disable(opposite_attr)
        _tdbg("HYBRID_RX", f"disable opposite={oid}")

    # ── history receive ───────────────────────────────────────────────────────

    def _receive_history(
        self, opposite_attr: AstroRoleAttributes, mode: str
    ) -> None:
        """
        Mirrors ``HybridReceiver::ReceiveHistoryMsg`` + ``ThreadFunc``.

        Spins a thread that waits for cached messages to arrive then exits.
        In this simulation we merely log the intent; real RTPS re-subscription
        would happen here.
        """
        if opposite_attr.qos_durability != AstroRoleAttributes.DURABILITY_TRANSIENT_LOCAL:
            return

        def _wait_thread() -> None:
            _tdbg("HYBRID_RX",
                  f"history thread started for opposite={opposite_attr.id}")
            deadline = time.monotonic() + 1.0
            last_size = self._history.get_size()
            while time.monotonic() < deadline:
                time.sleep(0.05)
                cur = self._history.get_size()
                if cur != last_size:
                    last_size = cur
                    deadline  = time.monotonic() + 1.0
            _tdbg("HYBRID_RX", "history thread exit")

        t = threading.Thread(
            target=_wait_thread, daemon=True, name="HybridRx-hist"
        )
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


# ═══════════════════════════════════════════════════════════════════════════════
# AstroNodeChannelImpl — ported from
#   upstream/apollo-cyber/node/node_channel_impl.h
#
# 原典：NodeChannelImpl 是 Node 的私有 pImpl，持有 node_attr_（protobuf
# RoleAttributes）、is_reality_mode_ 标志以及 NodeManager 共享指针；
# CreateWriter<M> / CreateReader<M> 根据 is_reality_mode_ 选择真实
# 传输路径（Writer<M> / Reader<M>）或进程内 Blocker 路径
# （IntraWriter<M> / IntraReader<M>）；ReaderConfig 封装
# channel_name + QosProfile + pending_queue_size 三元组。
#
# 算法改动（20% 规则）：
#   1. template<MessageT>       → 消息类型固定为 Any（dict / str / bytes）。
#   2. proto::RoleAttributes    → 轻量 ReaderConfig / _WriterAttr dataclass。
#   3. is_reality_mode_ 二值分支 → AstroTransport.mode 参数（"INTRA" / "SHM"）；
#      默认 "INTRA"，等价于原典 is_reality_mode_=false 的 IntraWriter/IntraReader。
#   4. NodeManager::Join / Leave → AstroChannelManager（已在本文件实现）。
#   5. GlobalData::RegisterChannel uint64_t → str hash via hashlib.md5 首8字节。
#   6. FillInAttr<M> 填充 QosProfile 默认值 → _fill_attr() 填充 ReaderConfig 默认值。
#   7. CreateReader 过载四版 → 三个 Python 重载（by_name / by_config / by_attr）。
#   8. pending_queue_size 默认值保持 DEFAULT_PENDING_QUEUE_SIZE（已在本文件定义）。
#
# 鲁迅曰：实现细节藏在 Impl 之后，正如苦衷藏在笑脸之后——外人只见工厂方法，
# 不见那些个 FillInAttr 和 Join 的辛苦。
#
# Debug prefix: [ASTRO-NODE-CH] — 与 C++ ADEBUG 前缀 grep 一致。
# ═══════════════════════════════════════════════════════════════════════════════

import dataclasses as _dc
import struct as _struct
import base64 as _base64


@_dc.dataclass

