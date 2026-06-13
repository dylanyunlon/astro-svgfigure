from typing import Any, List, Optional, Tuple

from channels.data.astro_cache_buffer import AstroCacheBuffer
from channels.data.astro_channel_buffer import AstroChannelBuffer
from channels.data.notifier import _dbg


class AstroAllLatest:
    """
    Multi-channel AllLatest fusion — port of apollo::cyber::data::fusion::AllLatest.

    Supports 2-, 3-, and 4-channel fusion (len(buffers) in {2,3,4}).
    buffers[0] is the primary; buffers[1:] are secondaries.

    When primary.buffer.fill(m0) is called the FusionCallback fires:
      - Latest() is called on each secondary.
      - If any secondary is None the tuple is dropped.
      - Otherwise (m0, *secondaries) is pushed to the internal fusion ring.

    fusion(index) reads epoch-indexed fused tuples from the fusion ring.
    """

    def __init__(self, buffers: List[AstroChannelBuffer]):
        if not (2 <= len(buffers) <= 4):
            raise ValueError("AstroAllLatest requires 2-4 channel buffers (M0-M3)")
        self._primary: AstroChannelBuffer = buffers[0]
        self._secondaries: List[AstroChannelBuffer] = buffers[1:]
        self._arity: int = len(buffers)
        # Fusion ring — capacity mirrors C++: primary.capacity()-1
        fusion_cap = self._primary.buffer.capacity() - 1
        self._fusion_buf = AstroChannelBuffer(
            self._primary.channel_id,
            AstroCacheBuffer(fusion_cap),
        )
        self._primary.buffer.set_fusion_callback(self._fusion_callback)
        _dbg("ASTRO-FUSION",
             f"AstroAllLatest ctor: primary={self._primary.channel_id} "
             f"arity={self._arity} fusion_cap={fusion_cap}")

    def _fusion_callback(self, m0: Any):
        """FusionCallback — mirrors AllLatest C++ lambda: Latest() all secondaries, push tuple."""
        vals: List[Any] = []
        for sec in self._secondaries:
            v = sec.latest()
            if v is None:
                _dbg("ASTRO-FUSION", f"fusion_callback: secondary={sec.channel_id} not ready — drop")
                return
            vals.append(v)
        print(
            f"[ASTRO-FUSION] AllLatest fusion triggered | "
            f"primary_channel='{self._primary.channel_id}' | "
            f"secondary_channels={len(self._secondaries)} | "
            f"fused_cells={len(vals)}"
        )
        with self._fusion_buf.buffer.lock:
            self._fusion_buf.buffer.fill(tuple([m0] + vals))

    def fusion(self, index: int) -> Tuple[int, Optional[tuple]]:
        """
        fusion(index) -> (new_index, (m0, m1[, m2[, m3]])) | (index, None)

        Mirrors AllLatest::Fusion() -> buffer_fusion_.Fetch(index, data).
        Caller advances index by 1 after each successful read.
        """
        return self._fusion_buf.fetch(index)

    @property
    def arity(self) -> int:
        return self._arity

    @property
    def primary_channel_id(self) -> str:
        return self._primary.channel_id

    @property
    def secondary_channel_ids(self) -> List[str]:
        return [s.channel_id for s in self._secondaries]
