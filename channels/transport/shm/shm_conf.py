import os, sys, json, threading
from typing import Any, Optional, Generic, TypeVar

_MT = TypeVar('_MT')

def _dbg(tag, msg):
    if os.environ.get(f'ASTRO_{tag.replace("-","_")}_VERBOSE', '0') == '1':
        print(f'[{tag}] {msg}', file=sys.stderr)

# ═══════════════════════════════════════════════════════════════════════════════

class AstroShmConf:
    """Port of apollo::cyber::transport::ShmConf — shared memory config."""
    BLOCK_SIZE_16K = 16 * 1024
    BLOCK_SIZE_128K = 128 * 1024
    BLOCK_SIZE_1M = 1024 * 1024
    EXTRA_SIZE = 48  # header overhead per block

    def __init__(self, msg_size: int = 1024):
        self._ceiling_msg_size = msg_size
        if msg_size <= self.BLOCK_SIZE_16K:
            self._block_buf_size = self.BLOCK_SIZE_16K
            self._block_num = 512
        elif msg_size <= self.BLOCK_SIZE_128K:
            self._block_buf_size = self.BLOCK_SIZE_128K
            self._block_num = 128
        else:
            self._block_buf_size = self.BLOCK_SIZE_1M
            self._block_num = 32
        _dbg("ASTRO-SHM", f"ShmConf: msg_size={msg_size} block_buf={self._block_buf_size} num={self._block_num}")

    @property
    def block_buf_size(self) -> int: return self._block_buf_size
    @property
    def block_num(self) -> int: return self._block_num
    @property
    def managed_shm_size(self) -> int:
        return (self._block_buf_size + AstroShmConf.EXTRA_SIZE) * self._block_num

    def update(self, msg_size: int):
        if msg_size > self._ceiling_msg_size:
            self.__init__(msg_size)



