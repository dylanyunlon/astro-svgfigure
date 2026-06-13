import enum as _enum


class FusionPolicy(_enum.Enum):
    """
    Fusion strategy selector — no direct C++ equivalent.

    ALL_LATEST: snapshot Latest() from each secondary on every primary Fill().
                Maps to DataFusion + AllLatest<M0,M1,...> template in Apollo.
    BARRIER:    wait until every channel has advanced to or past the current index.
                Maps to a barrier-style DataFusion not shipped in Apollo upstream
                but commonly needed for tight multi-sensor synchronisation.
    """
    ALL_LATEST = "all_latest"
    BARRIER    = "barrier"
