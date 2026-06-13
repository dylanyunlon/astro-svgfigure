import os, sys, json, threading
from typing import Any, Optional, Generic, TypeVar

_MT = TypeVar('_MT')

def _dbg(tag, msg):
    if os.environ.get(f'ASTRO_{tag.replace("-","_")}_VERBOSE', '0') == '1':
        print(f'[{tag}] {msg}', file=sys.stderr)


# --- transport/shm/posix_segment.h (49 lines) ---
class AstroPosixSegment:
    """Port of PosixSegment — POSIX shared memory (simulated with bytearray)."""
    def __init__(self, name: str, size: int):
        self._name = name
        self._size = size
        self._data = bytearray(size)
    def write(self, offset: int, data: bytes): self._data[offset:offset+len(data)] = data
    def read(self, offset: int, length: int) -> bytes: return bytes(self._data[offset:offset+length])
    @property
    def size(self): return self._size

# --- transport/shm/xsi_segment.h (47 lines) ---


# --- transport/shm/xsi_segment.h (47 lines) ---
class AstroXsiSegment:
    """Port of XsiSegment — XSI shared memory (same simulation as PosixSegment)."""
    def __init__(self, key: int, size: int):
        self._key = key
        self._size = size
        self._data = bytearray(size)
    def write(self, offset: int, data: bytes): self._data[offset:offset+len(data)] = data
    def read(self, offset: int, length: int) -> bytes: return bytes(self._data[offset:offset+length])

# --- transport/shm/segment_factory.h (36 lines) ---


# --- transport/shm/segment_factory.h (36 lines) ---
class AstroSegmentFactory:
    """Port of SegmentFactory — creates PosixSegment or XsiSegment."""
    @staticmethod
    def create(mode: str = "posix", **kwargs):
        if mode == "xsi": return AstroXsiSegment(**kwargs)
        return AstroPosixSegment(**kwargs)

# --- transport/rtps/underlay_message_type.h (53 lines) ---

