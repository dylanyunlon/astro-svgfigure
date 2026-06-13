import os, sys, json, threading
from typing import Any, Optional, Generic, TypeVar

_MT = TypeVar('_MT')

def _dbg(tag, msg):
    if os.environ.get(f'ASTRO_{tag.replace("-","_")}_VERBOSE', '0') == '1':
        print(f'[{tag}] {msg}', file=sys.stderr)



@_dc.dataclass
class AstroUnderlayMessage:
    """
    Python port of ``apollo::cyber::transport::UnderlayMessage``.

    Wire layout (CDR-approximate, little-endian):
        int32  timestamp   (4 bytes, saturated to INT32 range)
        int32  seq         (4 bytes)
        uint32 data_len    (4 bytes) + data_len bytes  (m_data)
        uint32 type_len    (4 bytes) + type_len bytes  (m_datatype)

    ``serialize()`` packs to bytes; ``deserialize()`` restores from bytes.
    Both are used by AstroRtpsTransmitter / AstroRtpsDispatcher ASTRO ports.

    鲁迅曰：序号就是命运——第一条消息序号为一，最后一条也不过是个更大的整数，
    中间的岁月，都压扁在那四个字节里。
    """

    _INT32_MAX: int =  2_147_483_647
    _INT32_MIN: int = -2_147_483_648
    _HDR_FMT: str   = "<ii"
    _HDR_SIZE: int  = _struct.calcsize(_HDR_FMT)

    def __init__(self, timestamp: float = 0.0, seq: int = 0,
                 data: str = "", datatype: str = "") -> None:
        self.timestamp: float = timestamp
        self.seq:       int   = seq
        self.data:      str   = data
        self.datatype:  str   = datatype

    def get_timestamp(self) -> int:
        return self._clamp_int32(int(self.timestamp))
    def set_timestamp(self, v: float) -> None:
        self.timestamp = float(v)
    def get_seq(self) -> int:
        return self._clamp_int32(self.seq)
    def set_seq(self, v: int) -> None:
        self.seq = int(v)
    def get_data(self) -> str:
        return self.data
    def set_data(self, v: str) -> None:
        self.data = v
    def get_datatype(self) -> str:
        return self.datatype
    def set_datatype(self, v: str) -> None:
        self.datatype = v

    @staticmethod
    def max_serialized_size(current_alignment: int = 0) -> int:
        return current_alignment + 4 + 4 + 4 + 65535 + 4 + 65535

    @staticmethod
    def serialized_size(msg: "AstroUnderlayMessage", current_alignment: int = 0) -> int:
        return (current_alignment + 4 + 4
                + 4 + len(msg.data.encode("utf-8"))
                + 4 + len(msg.datatype.encode("utf-8")))

    def serialize(self) -> bytes:
        data_b  = self.data.encode("utf-8")
        dtype_b = self.datatype.encode("utf-8")
        return (
            _struct.pack(self._HDR_FMT,
                         self._clamp_int32(int(self.timestamp)),
                         self._clamp_int32(self.seq))
            + _struct.pack("<I", len(data_b))  + data_b
            + _struct.pack("<I", len(dtype_b)) + dtype_b
        )

    @classmethod
    def deserialize(cls, raw: bytes) -> "AstroUnderlayMessage":
        if len(raw) < cls._HDR_SIZE + 8:
            raise ValueError(f"AstroUnderlayMessage.deserialize: need ≥{cls._HDR_SIZE+8} bytes")
        offset = 0
        ts_i32, seq_i32 = _struct.unpack_from(cls._HDR_FMT, raw, offset); offset += cls._HDR_SIZE
        data_len, = _struct.unpack_from("<I", raw, offset); offset += 4
        data_b    = raw[offset: offset + data_len];          offset += data_len
        dtype_len, = _struct.unpack_from("<I", raw, offset); offset += 4
        dtype_b    = raw[offset: offset + dtype_len]
        return cls(timestamp=float(ts_i32), seq=seq_i32,
                   data=data_b.decode("utf-8", errors="replace"),
                   datatype=dtype_b.decode("utf-8", errors="replace"))

    @staticmethod
    def key_max_serialized_size(current_alignment: int = 0) -> int:
        return current_alignment
    @staticmethod
    def is_key_defined() -> bool:
        return False
    def serialize_key(self) -> bytes:
        return b""

    @classmethod
    def _clamp_int32(cls, v: int) -> int:
        return max(cls._INT32_MIN, min(cls._INT32_MAX, v))

    def __repr__(self) -> str:
        return (f"AstroUnderlayMessage(ts={self.timestamp}, seq={self.seq}, "
                f"datatype={self.datatype!r}, data_len={len(self.data)})")


# ═══════════════════════════════════════════════════════════════════════════════
# AstroWarehouseBase / AstroSingleValueWarehouse / AstroMultiValueWarehouse
# Ported from: warehouse_base.h / single_value_warehouse.h / multi_value_warehouse.h
#
# 鲁迅曰：仓库里堆满了角色，按编号取，按属性找，钥匙和货物之间，
# 隔着一道原子锁——那是文明对混乱最后的体面。
#
# 算法改动（20% 规则）：
#   1. uint64_t key + RolePtr  → str channel_path + RoleRecord。
#   2. AtomicRWLock            → threading.RLock。
#   3. unordered_map           → dict[str, RoleRecord]（SingleValue）。
#   4. unordered_multimap      → dict[str, list[RoleRecord]]（MultiValue）。
#   5. ignore_if_exist         → Add() 参数保留，语义同原典。
#   6. bool* out-param Search  → 返回 Optional[RoleRecord] / List[RoleRecord]。
#   7. GetAllRoles 两重载      → get_all_roles() 合并为一个 Pythonic 接口。
# ═══════════════════════════════════════════════════════════════════════════════

import abc as _abc




# --- transport/rtps/underlay_message_type.h (53 lines) ---
class AstroUnderlayMessageType:
    """Port of UnderlayMessageType — RTPS message type registration."""
    _registry: dict = {}
    @classmethod
    def register(cls, type_name: str, serializer=None, deserializer=None):
        cls._registry[type_name] = {"ser": serializer, "deser": deserializer}
    @classmethod
    def get(cls, type_name: str): return cls._registry.get(type_name)

# --- transport/rtps/sub_listener.h (65 lines) ---

