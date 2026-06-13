import os, sys, json, threading
from typing import Any, Optional, Generic, TypeVar

_MT = TypeVar('_MT')

def _dbg(tag, msg):
    if os.environ.get(f'ASTRO_{tag.replace("-","_")}_VERBOSE', '0') == '1':
        print(f'[{tag}] {msg}', file=sys.stderr)



class AstroReadableInfo:
    """Port of apollo::cyber::transport::ReadableInfo — describes a readable block."""
    def __init__(self, host_id: int = 0, block_index: int = 0, channel_id: str = ""):
        self.host_id = host_id
        self.block_index = block_index
        self.channel_id = channel_id

    def serialize(self) -> str:
        return f"{self.host_id}:{self.block_index}:{self.channel_id}"

    @classmethod
    def deserialize(cls, s: str) -> "AstroReadableInfo":
        parts = s.split(":", 2)
        if len(parts) != 3:
            return cls()
        return cls(int(parts[0]), int(parts[1]), parts[2])

    def __repr__(self):
        return f"ReadableInfo(host={self.host_id}, block={self.block_index}, ch={self.channel_id})"



