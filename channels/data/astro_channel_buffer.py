from typing import Any, List, Optional, Tuple

from channels.data.astro_cache_buffer import AstroCacheBuffer
from channels.data.notifier import _dbg


class AstroChannelBuffer:
    """
    Epoch-constrained reader over AstroCacheBuffer.

    fetch(index)    -> (new_index, value) | (index, None)
    latest()        -> value | None
    fetch_multi(n)  -> List[value], oldest-first
    """

    def __init__(self, channel_id: str, buf: AstroCacheBuffer):
        self._channel_id = channel_id
        self._buf = buf
        _dbg("ASTRO-BUFFER",
             f"AstroChannelBuffer ctor: channel={channel_id} capacity={buf.capacity()}")

    @property
    def channel_id(self) -> str:
        return self._channel_id

    @property
    def buffer(self) -> AstroCacheBuffer:
        return self._buf

    def fetch(self, index: int) -> Tuple[int, Optional[Any]]:
        """Epoch-indexed sequential read.  Overflow detection → WARN + snap."""
        with self._buf.lock:
            if self._buf.empty():
                _dbg("ASTRO-BUFFER", f"Fetch: empty ch={self._channel_id}")
                return index, None
            if index == 0:
                new_idx = self._buf.tail()
                _dbg("ASTRO-BUFFER", f"Fetch: cold-start snap Tail={new_idx} ch={self._channel_id}")
                return new_idx, self._buf.at(new_idx)
            if index == self._buf.tail() + 1:
                _dbg("ASTRO-BUFFER", f"Fetch: epoch current no new data idx={index} ch={self._channel_id}")
                return index, None
            if index < self._buf.head():
                drop = self._buf.tail() - index
                print(
                    f"[ASTRO-BUFFER] Fetch: epoch overflow on channel[{self._channel_id}] "
                    f"drop_messages=[{drop}] stale_epoch_index=[{index}] "
                    f"current_epoch_tail=[{self._buf.tail()}] — "
                    f"snapping cursor to current epoch boundary"
                )
                new_idx = self._buf.tail()
                return new_idx, self._buf.at(new_idx)
            _dbg("ASTRO-BUFFER", f"Fetch: reading idx={index} ch={self._channel_id}")
            return index, self._buf.at(index)

    def latest(self) -> Optional[Any]:
        """Non-destructive tail peek.  Mirrors ChannelBuffer<T>::Latest()."""
        with self._buf.lock:
            if self._buf.empty():
                _dbg("ASTRO-BUFFER", f"Latest: empty ch={self._channel_id}")
                return None
            _dbg("ASTRO-BUFFER", f"Latest: tail={self._buf.tail()} ch={self._channel_id}")
            return self._buf.back()

    def fetch_multi(self, fetch_size: int) -> List[Any]:
        """Bulk read up to fetch_size entries, oldest-first.  Mirrors FetchMulti()."""
        with self._buf.lock:
            if self._buf.empty():
                _dbg("ASTRO-BUFFER", f"FetchMulti: empty ch={self._channel_id}")
                return []
            num = min(self._buf.size(), fetch_size)
            start = self._buf.tail() - num + 1
            result = [self._buf.at(i) for i in range(start, self._buf.tail() + 1)]
            _dbg("ASTRO-BUFFER", f"FetchMulti: count={len(result)} ch={self._channel_id}")
            return result
