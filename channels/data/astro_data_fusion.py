import abc as _abc_comp
from typing import List, Optional, Tuple

from channels.data.astro_all_latest import AstroAllLatest
from channels.data.astro_channel_buffer import AstroChannelBuffer
from channels.data.fusion_policy import FusionPolicy
from channels.data.notifier import _dbg


class AstroDataFusion(_abc_comp.ABC):
    """
    Abstract multi-channel data fusion interface.

    Python port of ``apollo::cyber::data::fusion::DataFusion<M0[,M1[,M2[,M3]]]>``.

    Supports 2-, 3-, and 4-channel fusion (arity in {2, 3, 4}), matching the
    three partial-template specialisations in data_fusion.h.

    Subclasses implement ``fusion(index)`` returning (new_index, tuple|None).

    ASTRO delta from Apollo DataFusion:
      • Template specialisations merged into one class (arity arg).
      • Fusion() bool + out-params → (new_index, tuple|None) pair.
      • NullType placeholder channels → simply absent (Python list).
      • channel_ids list carries names for debug logging.
    """

    def __init__(self, channel_ids: List[str]) -> None:
        if not (2 <= len(channel_ids) <= 4):
            raise ValueError(
                f"AstroDataFusion requires 2-4 channel ids, got {len(channel_ids)}"
            )
        self._channel_ids: List[str] = list(channel_ids)
        self._arity: int = len(channel_ids)

    @_abc_comp.abstractmethod
    def fusion(self, index: int) -> Tuple[int, Optional[tuple]]:
        """
        Fusion(index*, m0, m1[, m2[, m3]]) — read next fused tuple.

        index: current read cursor (0 = cold start).
        Returns (new_index, (m0, m1, …)) on success,
                (index, None) when no new data is available.

        Mirrors DataFusion::Fusion() pure-virtual.
        """

    @property
    def arity(self) -> int:
        """Number of fused channels (2, 3, or 4)."""
        return self._arity

    @property
    def channel_ids(self) -> List[str]:
        return list(self._channel_ids)


class AstroAllLatestFusion(AstroDataFusion):
    """
    AllLatest fusion — wraps AstroAllLatest behind the AstroDataFusion interface.

    Maps channel_ids to AstroChannelBuffer instances, constructs an
    AstroAllLatest, and delegates fusion(index) to it.
    """

    def __init__(self, channel_buffers: List[AstroChannelBuffer]) -> None:
        channel_ids = [b.channel_id for b in channel_buffers]
        super().__init__(channel_ids)
        self._all_latest = AstroAllLatest(channel_buffers)

    def fusion(self, index: int) -> Tuple[int, Optional[tuple]]:
        """
        Fusion() — delegate to AstroAllLatest.fusion(index).
        Mirrors DataFusion<M0,...>::Fusion(index*, m0&, …) → AllLatest::Fusion().
        """
        new_idx, result = self._all_latest.fusion(index)
        if result is not None:
            _dbg("ASTRO-FUSION",
                 f"AllLatestFusion ch={self._channel_ids[0]} "
                 f"idx={index}→{new_idx} arity={self._arity}")
        return new_idx, result


class AstroBarrierFusion(AstroDataFusion):
    """
    Barrier fusion — requires every channel to have advanced past *index*.

    Unlike AllLatest (which snapshots secondaries on primary fill), Barrier
    holds until all AstroChannelBuffer.fetch(index+1) succeed simultaneously.
    Callers typically poll fusion() in a spin loop with a sleep.
    """

    def __init__(self, channel_buffers: List[AstroChannelBuffer]) -> None:
        channel_ids = [b.channel_id for b in channel_buffers]
        super().__init__(channel_ids)
        self._buffers: List[AstroChannelBuffer] = channel_buffers

    def fusion(self, index: int) -> Tuple[int, Optional[tuple]]:
        """
        Barrier Fusion(index) — all channels must advance before returning data.

        Returns (index+1, (v0, …, vN)) when every buffer has data at index+1.
        Returns (index, None) if any buffer is behind.
        """
        next_idx = index + 1
        values = []
        for buf in self._buffers:
            new_i, val = buf.fetch(next_idx)
            if val is None:
                _dbg("ASTRO-FUSION",
                     f"BarrierFusion: ch={buf.channel_id} not ready idx={next_idx}")
                return index, None
            values.append(val)

        _dbg("ASTRO-FUSION",
             f"BarrierFusion: all channels ready idx={index}→{next_idx} "
             f"arity={self._arity}")
        return next_idx, tuple(values)


def make_fusion(
    channel_buffers: List[AstroChannelBuffer],
    policy: FusionPolicy = FusionPolicy.ALL_LATEST,
) -> AstroDataFusion:
    """
    Factory helper — mirrors DataFusion template instantiation.
    """
    if policy == FusionPolicy.BARRIER:
        return AstroBarrierFusion(channel_buffers)
    return AstroAllLatestFusion(channel_buffers)
