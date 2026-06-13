import os, sys, json, threading
from typing import Any, Optional, Generic, TypeVar

_MT = TypeVar('_MT')

def _dbg(tag, msg):
    if os.environ.get(f'ASTRO_{tag.replace("-","_")}_VERBOSE', '0') == '1':
        print(f'[{tag}] {msg}', file=sys.stderr)



class AstroHistory(Generic[_MT]):
    """
    Mirrors ``apollo::cyber::transport::History<MessageT>``.

    Thread-safe bounded deque of (msg, msg_info) pairs.  When
    *history_policy* is HISTORY_KEEP_ALL the depth is capped at
    *max_depth*.
    """

    HISTORY_KEEP_LAST = 0
    HISTORY_KEEP_ALL  = 1

    class CachedMessage(Generic[_MT]):
        __slots__ = ("msg", "msg_info")

        def __init__(self, msg: _MT, msg_info: AstroMessageInfo) -> None:
            self.msg:      _MT             = msg
            self.msg_info: AstroMessageInfo = msg_info

    def __init__(
        self,
        history_policy: int = 0,
        depth:          int = 1,
        max_depth:      int = _MAX_HISTORY_DEPTH_DEFAULT,
    ) -> None:
        self._enabled:   bool  = False
        self._max_depth: int   = max_depth
        if history_policy == self.HISTORY_KEEP_ALL:
            self._depth: int = max_depth
        else:
            self._depth = min(depth, max_depth)
        self._msgs: deque = deque(maxlen=self._depth)
        self._lock: threading.Lock = threading.Lock()

    # ── enable/disable ────────────────────────────────────────────────────────

    def enable(self)  -> None: self._enabled = True
    def disable(self) -> None: self._enabled = False

    # ── public API ───────────────────────────────────────────────────────────

    def add(self, msg: _MT, msg_info: AstroMessageInfo) -> None:
        """``void Add(const MessagePtr&, const MessageInfo&)``"""
        if not self._enabled:
            return
        with self._lock:
            self._msgs.append(self.CachedMessage(msg, msg_info))

    def clear(self) -> None:
        """``void Clear()``"""
        with self._lock:
            self._msgs.clear()

    def get_cached_message(self) -> List["AstroHistory.CachedMessage"]:
        """``void GetCachedMessage(vector<CachedMessage>*)``"""
        with self._lock:
            return list(self._msgs)

    def get_size(self) -> int:
        """``size_t GetSize()``"""
        with self._lock:
            return len(self._msgs)

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def max_depth(self) -> int:
        return self._max_depth


# ══════════════════════════════════════════════════════════════════════════════
# _AstroDispatcherBase  (internal)
# Thin common base for ShmDispatcher / RtpsDispatcher
# ══════════════════════════════════════════════════════════════════════════════

_ListenerFn = Callable[[Any, AstroMessageInfo], None]



# --- transport/message/history_attributes.h (45 lines) ---
class AstroHistoryAttributes:
    """Port of HistoryAttributes — history depth/policy config."""
    def __init__(self, depth: int = 10, policy: str = "KEEP_LAST"):
        self.depth = depth
        self.policy = policy  # KEEP_LAST or KEEP_ALL

# --- transport/qos/qos_profile_conf.h (61 lines) ---

