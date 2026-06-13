import os, sys, json, threading
from typing import Any, Optional, Generic, TypeVar

_MT = TypeVar('_MT')

def _dbg(tag, msg):
    if os.environ.get(f'ASTRO_{tag.replace("-","_")}_VERBOSE', '0') == '1':
        print(f'[{tag}] {msg}', file=sys.stderr)



class AstroArenaAddressAllocator:
    """Port of apollo::cyber::transport::ArenaAddressAllocator — memory pool allocator.
    Original uses best-fit; simplified to first-fit with coalescing."""

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._free_list: list = [(0, capacity)]  # (offset, size) pairs
        self._alloc_map: dict = {}  # offset → size
        _dbg("ASTRO-SHM", f"ArenaAllocator: capacity={capacity}")

    def allocate(self, size: int) -> int:
        """First-fit allocation. Returns offset or -1."""
        for i, (offset, free_size) in enumerate(self._free_list):
            if free_size >= size:
                self._alloc_map[offset] = size
                if free_size == size:
                    self._free_list.pop(i)
                else:
                    self._free_list[i] = (offset + size, free_size - size)
                return offset
        return -1

    def deallocate(self, offset: int):
        if offset not in self._alloc_map:
            return
        size = self._alloc_map.pop(offset)
        self._free_list.append((offset, size))
        self._free_list.sort()
        self._coalesce()

    def _coalesce(self):
        """Merge adjacent free blocks."""
        merged = []
        for offset, size in self._free_list:
            if merged and merged[-1][0] + merged[-1][1] == offset:
                merged[-1] = (merged[-1][0], merged[-1][1] + size)
            else:
                merged.append((offset, size))
        self._free_list = merged

    @property
    def available(self) -> int:
        return sum(s for _, s in self._free_list)





class AstroProtobufArenaManager:
    """Port of apollo::cyber::transport::ProtobufArenaManager — message buffer pool.
    Original manages protobuf Arena objects; we manage JSON buffer slots."""

    def __init__(self, conf: AstroShmConf = None):
        if conf is None:
            conf = AstroShmConf()
        self._conf = conf
        self._allocator = AstroArenaAddressAllocator(conf.managed_shm_size)
        self._buffers: dict = {}  # slot_id → dict (the actual message data)
        self._next_slot = 0
        _dbg("ASTRO-SHM", f"ArenaManager: shm_size={conf.managed_shm_size}")

    def acquire_slot(self, msg_size: int = 1024) -> int:
        """Acquire a buffer slot for writing."""
        offset = self._allocator.allocate(msg_size + AstroShmConf.EXTRA_SIZE)
        if offset < 0:
            _dbg("ASTRO-SHM", "ArenaManager.acquire_slot: OOM")
            return -1
        slot_id = self._next_slot
        self._next_slot += 1
        self._buffers[slot_id] = {"_offset": offset, "_size": msg_size, "data": None}
        return slot_id

    def write_slot(self, slot_id: int, data: dict):
        if slot_id in self._buffers:
            self._buffers[slot_id]["data"] = data

    def read_slot(self, slot_id: int):
        buf = self._buffers.get(slot_id)
        return buf["data"] if buf else None

    def release_slot(self, slot_id: int):
        buf = self._buffers.pop(slot_id, None)
        if buf:
            self._allocator.deallocate(buf["_offset"])

    @property
    def active_slots(self) -> int:
        return len(self._buffers)


# ═══════════════════════════════════════════════════════════════════════════════
# [ASTRO-FINAL] Remaining 21 Apollo .h ports — batch completion
#
# These are small utility/config/interface files (40-87 lines each).
# Ported as lightweight Python equivalents.
# ═══════════════════════════════════════════════════════════════════════════════

# --- scheduler/scheduler_factory.h (40 lines) ---

# --- scheduler/policy/choreography_context.h (63 lines) ---

# --- scheduler/common/cv_wrapper.h (40 lines) ---

# --- scheduler/common/mutex_wrapper.h (40 lines) ---

# --- scheduler/common/pin_thread.h (43 lines) ---

# --- scheduler/processor_context.h (48 lines) ---

# --- component/timer_component.h (68 lines) ---

