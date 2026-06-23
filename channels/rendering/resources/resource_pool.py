
# ─────────────────────────────────────────────────────────────────────────────
# M824: Resource Pool — Texture pool + Buffer allocator + LRU lifecycle
# ─────────────────────────────────────────────────────────────────────────────

import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ResourceEntry:
    """A tracked resource with ref counting and LRU timestamp."""
    key: str
    data: Any
    ref_count: int = 0
    size_bytes: int = 0
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)

    def acquire(self):
        self.ref_count += 1
        self.last_access = time.time()

    def release(self):
        self.ref_count = max(0, self.ref_count - 1)


class TexturePool:
    """
    Shared texture pool — reuses textures across cells with same species.

    When two cil-eye cells need the same SDF texture, they share one GPU
    texture object instead of uploading twice.

    [ASTRO-RESOURCE] Pool key = (species, texture_type, mip_level).
    """

    def __init__(self, max_textures: int = 128, max_bytes: int = 256 * 1024 * 1024):
        self._pool: OrderedDict[str, ResourceEntry] = OrderedDict()
        self.max_textures = max_textures
        self.max_bytes = max_bytes
        self._total_bytes = 0

    def acquire(self, key: str, factory=None, size_bytes: int = 0) -> Any:
        """Get or create a texture. Increments ref count."""
        if key in self._pool:
            entry = self._pool[key]
            entry.acquire()
            self._pool.move_to_end(key)
            print(f"[ASTRO-RESOURCE] texture reuse: {key} refs={entry.ref_count}",
                  file=sys.stderr)
            return entry.data

        # Create new
        if factory:
            data = factory()
        else:
            data = {"placeholder": key}

        self._evict_if_needed(size_bytes)
        entry = ResourceEntry(key=key, data=data, ref_count=1, size_bytes=size_bytes)
        self._pool[key] = entry
        self._total_bytes += size_bytes
        print(f"[ASTRO-RESOURCE] texture create: {key} size={size_bytes} "
              f"pool={len(self._pool)}/{self.max_textures}",
              file=sys.stderr)
        return data

    def release(self, key: str):
        if key in self._pool:
            self._pool[key].release()

    def _evict_if_needed(self, incoming_bytes: int):
        """LRU eviction: remove least-recently-used unreferenced textures."""
        while (len(self._pool) >= self.max_textures or
               self._total_bytes + incoming_bytes > self.max_bytes):
            evicted = False
            for k in list(self._pool.keys()):
                entry = self._pool[k]
                if entry.ref_count <= 0:
                    self._total_bytes -= entry.size_bytes
                    del self._pool[k]
                    print(f"[ASTRO-RESOURCE] texture evict: {k}", file=sys.stderr)
                    evicted = True
                    break
            if not evicted:
                break  # All textures in use

    @property
    def stats(self) -> dict:
        return {
            "count": len(self._pool),
            "total_bytes": self._total_bytes,
            "active": sum(1 for e in self._pool.values() if e.ref_count > 0),
            "evictable": sum(1 for e in self._pool.values() if e.ref_count <= 0),
        }


class BufferAllocator:
    """
    VBO/IBO memory pool — sub-allocates from large pre-allocated buffers.

    [ASTRO-RESOURCE] Reduces GPU buffer creation overhead by pooling.
    """

    def __init__(self, block_size: int = 1024 * 1024):
        self.block_size = block_size
        self._blocks: List[dict] = []
        self._allocations: Dict[str, dict] = {}

    def allocate(self, name: str, size_bytes: int) -> dict:
        # Find block with space
        for block in self._blocks:
            if block["used"] + size_bytes <= block["capacity"]:
                offset = block["used"]
                block["used"] += size_bytes
                alloc = {"block_id": block["id"], "offset": offset,
                         "size": size_bytes, "name": name}
                self._allocations[name] = alloc
                print(f"[ASTRO-RESOURCE] buffer alloc: {name} size={size_bytes} "
                      f"block={block['id']} offset={offset}", file=sys.stderr)
                return alloc

        # Need new block
        block = {"id": len(self._blocks), "capacity": max(self.block_size, size_bytes),
                 "used": size_bytes}
        self._blocks.append(block)
        alloc = {"block_id": block["id"], "offset": 0,
                 "size": size_bytes, "name": name}
        self._allocations[name] = alloc
        print(f"[ASTRO-RESOURCE] buffer new block: id={block['id']} for {name}",
              file=sys.stderr)
        return alloc

    def free(self, name: str):
        if name in self._allocations:
            del self._allocations[name]

    @property
    def stats(self) -> dict:
        total_cap = sum(b["capacity"] for b in self._blocks)
        total_used = sum(b["used"] for b in self._blocks)
        return {
            "blocks": len(self._blocks),
            "total_capacity": total_cap,
            "total_used": total_used,
            "utilization": total_used / total_cap if total_cap > 0 else 0,
            "allocations": len(self._allocations),
        }
