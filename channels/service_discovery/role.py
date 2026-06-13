import os, sys, json, threading
from typing import Any, Optional, Generic, TypeVar

_MT = TypeVar('_MT')

def _dbg(tag, msg):
    if os.environ.get(f'ASTRO_{tag.replace("-","_")}_VERBOSE', '0') == '1':
        print(f'[{tag}] {msg}', file=sys.stderr)



@dataclasses.dataclass
class RoleRecord:
    """
    轻量角色属性 — 对应 apollo::cyber::proto::RoleAttributes。

    Apollo 原典使用 protobuf RoleAttributes（含 host_name, process_id,
    node_name, node_id, channel_name, channel_id, message_type, proto_desc
    等字段）。此处精简为 Astro 文件通道模型所需的五个核心字段。
    """
    node_name: str       # proto.node_name
    channel_path: str    # proto.channel_name（string path 替代 channel_id uint64）
    role_id: str         # proto.id（确定性字符串 hash：node_name::channel_path）
    msg_type: str = ""   # proto.message_type
    timestamp: float = dataclasses.field(default_factory=_time.monotonic)
    # host_name + process_id — 用于 OnTopoModuleLeave 批量清除
    host_name: str = "localhost"
    process_id: int = 0


@dataclasses.dataclass

# ══════════════════════════════════════════════════════════════════════════════

class AstroRoleAttributes:
    """
    Lightweight stand-in for ``proto::RoleAttributes``.

    Carries only the fields actually referenced by the transport layer.
    """

    __slots__ = (
        "channel_name", "channel_id", "host_ip", "process_id",
        "id", "message_type", "qos_durability", "qos_history", "qos_depth",
    )

    DURABILITY_VOLATILE        = 0
    DURABILITY_TRANSIENT_LOCAL = 1

    HISTORY_KEEP_LAST = 0
    HISTORY_KEEP_ALL  = 1

    def __init__(
        self,
        channel_name:  str  = "",
        channel_id:    int  = 0,
        host_ip:       str  = "127.0.0.1",
        process_id:    int  = 0,
        id:            int  = 0,                  # noqa: A002
        message_type:  str  = "",
        qos_durability: int = 0,
        qos_history:   int  = 0,
        qos_depth:     int  = 1,
    ) -> None:
        self.channel_name   = channel_name
        self.channel_id     = channel_id
        self.host_ip        = host_ip
        self.process_id     = process_id
        self.id             = id
        self.message_type   = message_type
        self.qos_durability = qos_durability
        self.qos_history    = qos_history
        self.qos_depth      = qos_depth

    def copy_from(self, src: "AstroRoleAttributes") -> None:
        for s in self.__slots__:
            setattr(self, s, getattr(src, s))

    def __repr__(self) -> str:
        return (f"AstroRoleAttributes(ch={self.channel_name!r}, "
                f"id={self.id}, host={self.host_ip})")



