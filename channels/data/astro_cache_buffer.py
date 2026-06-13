import threading
from typing import Any, Callable, List, Optional


class AstroCacheBuffer:
    """Epoch-indexed ring buffer — port of apollo::cyber::data::CacheBuffer."""

    def __init__(self, size: int):
        self._capacity: int = size + 1          # C++: capacity_ = size + 1
        self._buffer: List[Any] = [None] * self._capacity
        self._head: int = 0
        self._tail: int = 0
        self._lock = threading.Lock()
        self._fusion_cb: Optional[Callable[[Any], None]] = None

    @property
    def lock(self) -> threading.Lock:
        return self._lock

    def capacity(self) -> int:
        return self._capacity

    def head(self) -> int:
        return self._head + 1

    def tail(self) -> int:
        return self._tail

    def size(self) -> int:
        return self._tail - self._head

    def empty(self) -> bool:
        return self._tail == 0

    def full(self) -> bool:
        return self._capacity - 1 == self._tail - self._head

    def _slot(self, pos: int) -> int:
        """GetIndex — modular slot (pos % capacity)."""
        return pos % self._capacity

    def at(self, pos: int) -> Any:
        return self._buffer[self._slot(pos)]

    def front(self) -> Any:
        return self._buffer[self._slot(self._head + 1)]

    def back(self) -> Any:
        return self._buffer[self._slot(self._tail)]

    def set_fusion_callback(self, cb: Callable[[Any], None]):
        """SetFusionCallback — hook used by AstroAllLatest to intercept Fill()."""
        self._fusion_cb = cb

    def fill(self, value: Any):
        """
        Fill — write to ring or delegate to FusionCallback.

        When full, evicts head (overwrites oldest slot), advancing both
        head_ and tail_.  Mirrors C++ Fill() exactly.
        """
        if self._fusion_cb is not None:
            self._fusion_cb(value)
            return
        if self.full():
            self._buffer[self._slot(self._head)] = value
            self._head += 1
            self._tail += 1
        else:
            self._buffer[self._slot(self._tail + 1)] = value
            self._tail += 1
