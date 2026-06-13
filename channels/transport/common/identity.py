import os, sys, json, threading
from typing import Any, Optional, Generic, TypeVar

_MT = TypeVar('_MT')

def _dbg(tag, msg):
    if os.environ.get(f'ASTRO_{tag.replace("-","_")}_VERBOSE', '0') == '1':
        print(f'[{tag}] {msg}', file=sys.stderr)



class AstroIdentity:
    """
    Mirrors ``apollo::cyber::transport::Identity``.

    Stores an 8-byte raw ID and a precomputed uint64 hash.  When
    *need_generate* is True (default) a random UUID-derived payload is
    generated on construction, matching the C++ auto-generate path.
    """

    __slots__ = ("_data", "_hash_value")

    def __init__(self, need_generate: bool = True,
                 data: Optional[bytes] = None) -> None:
        if data is not None:
            if len(data) != _ID_SIZE:
                raise ValueError(f"Identity data must be {_ID_SIZE} bytes")
            self._data: bytes = bytes(data)
        elif need_generate:
            self._data = uuid.uuid4().bytes[:_ID_SIZE]
        else:
            self._data = b"\x00" * _ID_SIZE
        self._update()

    # ── private ──────────────────────────────────────────────────────────────

    def _update(self) -> None:
        """Recompute hash — mirrors Identity::Update()."""
        val: int = 0
        for b in self._data:
            val = (val * 31 + b) & 0xFFFF_FFFF_FFFF_FFFF
        self._hash_value: int = val

    # ── public API ───────────────────────────────────────────────────────────

    @property
    def data(self) -> bytes:
        return self._data

    def set_data(self, data: bytes) -> None:
        if len(data) != _ID_SIZE:
            raise ValueError(f"Identity data must be {_ID_SIZE} bytes")
        self._data = bytes(data[:_ID_SIZE])
        self._update()

    def hash_value(self) -> int:
        return self._hash_value

    def length(self) -> int:
        return _ID_SIZE

    def to_string(self) -> str:
        return self._data.hex()

    # ── dunder ────────────────────────────────────────────────────────────────

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AstroIdentity) and self._data == other._data

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return self._hash_value

    def __repr__(self) -> str:
        return f"AstroIdentity({self.to_string()})"


# ══════════════════════════════════════════════════════════════════════════════
# AstroEndpoint
# Port of: upstream/apollo-cyber/transport/common/endpoint.h
#
# 鲁迅曰：端点就是端点，无论叫做发送者还是接收者，骨子里不过是一个有名有姓
# 的 enabled 开关罢了。
# ══════════════════════════════════════════════════════════════════════════════


