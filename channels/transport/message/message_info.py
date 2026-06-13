import os, sys, json, threading
from typing import Any, Optional, Generic, TypeVar

_MT = TypeVar('_MT')

def _dbg(tag, msg):
    if os.environ.get(f'ASTRO_{tag.replace("-","_")}_VERBOSE', '0') == '1':
        print(f'[{tag}] {msg}', file=sys.stderr)



class AstroMessageInfo:
    """
    Mirrors ``apollo::cyber::transport::MessageInfo``.

    Serialises/deserialises to/from a fixed 44-byte binary payload so that
    the layout is compatible with the C++ ``kSize`` constant.
    """

    kSize: int = _MSG_INFO_SIZE

    def __init__(
        self,
        sender_id:   Optional[AstroIdentity] = None,
        seq_num:     int = 0,
        spare_id:    Optional[AstroIdentity] = None,
        channel_id:  int = 0,
        msg_seq_num: int = 0,
        send_time:   int = 0,
    ) -> None:
        self.sender_id_:   AstroIdentity = sender_id  or AstroIdentity(need_generate=False)
        self.channel_id_:  int           = channel_id
        self.seq_num_:     int           = seq_num
        self.spare_id_:    AstroIdentity = spare_id   or AstroIdentity(need_generate=False)
        self.msg_seq_num_: int           = msg_seq_num
        self.send_time_:   int           = send_time

    # ── getters/setters ──────────────────────────────────────────────────────

    @property
    def sender_id(self) -> AstroIdentity:
        return self.sender_id_

    def set_sender_id(self, v: AstroIdentity) -> None:
        self.sender_id_ = v

    @property
    def channel_id(self) -> int:
        return self.channel_id_

    def set_channel_id(self, v: int) -> None:
        self.channel_id_ = v

    @property
    def seq_num(self) -> int:
        return self.seq_num_

    def set_seq_num(self, v: int) -> None:
        self.seq_num_ = v

    @property
    def spare_id(self) -> AstroIdentity:
        return self.spare_id_

    def set_spare_id(self, v: AstroIdentity) -> None:
        self.spare_id_ = v

    @property
    def msg_seq_num(self) -> int:
        return self.msg_seq_num_

    def set_msg_seq_num(self, v: int) -> None:
        self.msg_seq_num_ = v

    @property
    def send_time(self) -> int:
        return self.send_time_

    def set_send_time(self, v: int) -> None:
        self.send_time_ = v

    # ── serialization ────────────────────────────────────────────────────────

    def serialize_to(self) -> bytes:
        """``bool SerializeTo(char* dst, size_t len)``"""
        return struct.pack(
            _MSG_INFO_FMT,
            self.sender_id_.data,
            self.channel_id_ & 0xFFFF_FFFF_FFFF_FFFF,
            self.seq_num_    & 0xFFFF_FFFF_FFFF_FFFF,
            self.spare_id_.data,
            self.msg_seq_num_ & 0xFFFF_FFFF,
            self.send_time_   & 0xFFFF_FFFF_FFFF_FFFF,
        )

    @classmethod
    def deserialize_from(cls, raw: bytes) -> "AstroMessageInfo":
        """``bool DeserializeFrom(const char* src, size_t len)``"""
        if len(raw) < _MSG_INFO_SIZE:
            raise ValueError(
                f"AstroMessageInfo.deserialize_from: need {_MSG_INFO_SIZE} bytes"
            )
        sid_b, ch_id, seq, spa_b, msg_seq, send_t = struct.unpack_from(
            _MSG_INFO_FMT, raw
        )
        return cls(
            sender_id   = AstroIdentity(need_generate=False, data=sid_b),
            channel_id  = ch_id,
            seq_num     = seq,
            spare_id    = AstroIdentity(need_generate=False, data=spa_b),
            msg_seq_num = msg_seq,
            send_time   = send_t,
        )

    # ── dunder ────────────────────────────────────────────────────────────────

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AstroMessageInfo):
            return NotImplemented
        return (
            self.sender_id_  == other.sender_id_  and
            self.channel_id_ == other.channel_id_ and
            self.seq_num_    == other.seq_num_    and
            self.spare_id_   == other.spare_id_
        )

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __repr__(self) -> str:
        return (f"AstroMessageInfo(sender={self.sender_id_.to_string()}, "
                f"seq={self.seq_num_}, send_time={self.send_time_})")


# ══════════════════════════════════════════════════════════════════════════════
# AstroHistory
# Port of: upstream/apollo-cyber/transport/message/history.h
#
# 鲁迅曰：历史是会重演的，消息也是——所以 TRANSIENT_LOCAL 才要把它们一条条
# 存起来，等着晚来的订阅者去翻旧账。
# ══════════════════════════════════════════════════════════════════════════════

_MAX_HISTORY_DEPTH_DEFAULT = 1000



