import os, sys, json, threading
from typing import Any, Optional, Generic, TypeVar

_MT = TypeVar('_MT')

def _dbg(tag, msg):
    if os.environ.get(f'ASTRO_{tag.replace("-","_")}_VERBOSE', '0') == '1':
        print(f'[{tag}] {msg}', file=sys.stderr)



class AstroEndpoint:
    """
    Mirrors ``apollo::cyber::transport::Endpoint``.

    Base class for transmitters and receivers; holds the role attributes
    and a random Identity generated at construction time.
    """

    def __init__(self, attr: AstroRoleAttributes) -> None:
        self.enabled_: bool              = False
        self.id_:      AstroIdentity     = AstroIdentity()
        self.attr_:    AstroRoleAttributes = attr

    # read-only accessors matching C++ getters
    @property
    def id(self) -> AstroIdentity:
        return self.id_

    @property
    def attributes(self) -> AstroRoleAttributes:
        return self.attr_

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id_.to_string()!r})"


# ══════════════════════════════════════════════════════════════════════════════
# AstroMessageInfo
# Port of: upstream/apollo-cyber/transport/message/message_info.h
#
# 鲁迅曰：消息的信封里装着发件人、序号和时间，正如旧社会的契约——写得清清楚楚，
# 却未必有人当真查对。
# ══════════════════════════════════════════════════════════════════════════════

# kSize = sender_id(8) + channel_id(8) + seq_num(8) + spare_id(8) +
#         msg_seq_num(4) + send_time(8)  → 44 bytes
_MSG_INFO_FMT    = "<8s Q Q 8s i Q"
_MSG_INFO_SIZE   = struct.calcsize(_MSG_INFO_FMT)   # == 44



