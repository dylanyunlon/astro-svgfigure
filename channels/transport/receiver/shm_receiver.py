import os, sys, json, threading
from typing import Any, Optional, Generic, TypeVar

_MT = TypeVar('_MT')

def _dbg(tag, msg):
    if os.environ.get(f'ASTRO_{tag.replace("-","_")}_VERBOSE', '0') == '1':
        print(f'[{tag}] {msg}', file=sys.stderr)


class AstroShmReceiver(AstroEndpoint):
    """
    Mirrors ``apollo::cyber::transport::ShmReceiver<M>``.

    Registers a typed listener with the singleton AstroShmDispatcher.
    The raw-bytes payload received from the dispatcher is passed through
    *deserialize_fn* before reaching *msg_listener*.
    """

    def __init__(
        self,
        attr:           AstroRoleAttributes,
        msg_listener:   _ListenerFn,
        deserialize_fn: Optional[_DeserializeFn] = None,
    ) -> None:
        super().__init__(attr)
        self._msg_listener:   _ListenerFn                  = msg_listener
        self._deserialize_fn: Optional[_DeserializeFn]     = deserialize_fn
        self._dispatcher:     AstroShmDispatcher            = AstroShmDispatcher.instance()

    # ── enable / disable ──────────────────────────────────────────────────────

    def enable(self, opposite_attr: Optional[AstroRoleAttributes] = None) -> None:
        if self.enabled_:
            return
        if opposite_attr is None:
            self._dispatcher.add_segment(self.attr_)
            self._dispatcher.add_listener(self.attr_, self._on_raw_message)
        else:
            self._dispatcher.add_segment(self.attr_)
            self._dispatcher.add_listener(
                self.attr_, self._on_raw_message, opposite_attr
            )
        self.enabled_ = True
        _tdbg("SHM_RX", f"enabled ch={self.attr_.channel_id}")

    def disable(self, opposite_attr: Optional[AstroRoleAttributes] = None) -> None:
        if not self.enabled_:
            return
        self._dispatcher.remove_listener(self.attr_, opposite_attr)
        self.enabled_ = False
        _tdbg("SHM_RX", f"disabled ch={self.attr_.channel_id}")

    # ── internal callback ─────────────────────────────────────────────────────

    def _on_raw_message(self, raw: Any, msg_info: AstroMessageInfo) -> None:
        """Deserialise raw bytes then forward to the user listener."""
        if self._deserialize_fn is not None and isinstance(raw, (bytes, bytearray)):
            try:
                msg = self._deserialize_fn(raw)
            except Exception as exc:
                _tdbg("SHM_RX", f"deserialize failed: {exc}")
                return
        else:
            msg = raw
        self._msg_listener(msg, msg_info)


# ══════════════════════════════════════════════════════════════════════════════
# AstroHybridReceiver
# Port of: upstream/apollo-cyber/transport/receiver/hybrid_receiver.h
#
# 鲁迅曰：Hybrid 接收者身兼数职，同屋、隔壁、远端，皆有办法应付；然而真正
# 到了需要回溯历史的时候，它才显出那份难得的耐心——开一条线，等到消息不再来
# 为止。
# ══════════════════════════════════════════════════════════════════════════════


